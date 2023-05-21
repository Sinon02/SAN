import argparse
import paddle
from paddle import nn
import numpy as np
import random

import torch
import torch.optim.lr_scheduler as lr_scheduler
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Backbone import Backbone as Backbone_paddle
from models_torch.Backbone import Backbone as Backbone_torch
from utils import init

# set device
device = "cpu"  # you can also set it as "cpu"
torch_device = torch.device("cuda:0" if "gpu" in device else "cpu")
paddle.set_device(device)


def gen_iter_data(params, data_loader, data_size=20):
    total_data = []
    # random_indices = random.sample(
    #     range(len(data_loader)), data_size // params['batch_size']
    # )
    random_indices = [0, 1, 2]
    random_indices.sort(reverse=True)
    cur_index = random_indices.pop()
    for i in range(len(data_loader)):
        single_data = next(data_loader())
        if cur_index == i:
            # images, image_masks, labels, labels_masks = single_data
            total_data.append(single_data)
            if len(random_indices) == 0:
                break
            else:
                cur_index = random_indices.pop()
    return total_data


def train_one_epoch_paddle(inputs, model, optimizer, max_iter, reprod_logger):
    for idx in range(max_iter):
        print(idx)
        print('')
        for j, data in enumerate(inputs):
            images, image_masks, labels, label_masks = data
            # images, image_masks, labels, label_masks = (
            #     paddle.to_tensor(images),
            #     paddle.to_tensor(image_masks),
            #     paddle.to_tensor(labels),
            #     paddle.to_tensor(label_masks),
            # )

            probs, loss = model(images, image_masks, labels, label_masks, reprod_logger)

            word_loss, struct_loss, parent_loss, kl_loss = loss
            loss = word_loss + struct_loss + parent_loss + kl_loss
            reprod_logger.add(f"loss_{idx}_{j}", loss.cpu().detach().numpy())
            reprod_logger.add(f"word_loss_{idx}_{j}", word_loss.cpu().detach().numpy())
            reprod_logger.add(f"struct_loss_{idx}_{j}", struct_loss.cpu().detach().numpy())
            reprod_logger.add(f"parent_loss_{idx}_{j}", parent_loss.cpu().detach().numpy())
            reprod_logger.add(f"kl_loss_{idx}_{j}", kl_loss.cpu().detach().numpy())
            # reprod_logger.add(f"lr_{idx}_{j}", np.array(optimizer.get_lr()))

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

    reprod_logger.save("./result/losses_paddle.npy")


def train_one_epoch_torch(inputs, model, optimizer, max_iter, reprod_logger):
    for idx in range(max_iter):
        print(idx)
        print('')
        for j, data in enumerate(inputs):
            device = torch_device
            torch_data = []
            for item in data:
                if str(item.dtype) == 'paddle.int64':
                    torch_data.append(torch.LongTensor(item.numpy()).to(device))
                elif str(item.dtype) == 'paddle.bool':
                    torch_data.append(torch.Tensor(item.numpy()).bool().to(device))
                else:
                    torch_data.append(torch.Tensor(item.numpy()).to(device))
            images, image_masks, labels, label_masks = torch_data
            # images, image_masks, labels, label_masks = (
            #     torch.Tensor(images).to(device),
            #     torch.Tensor(image_masks).to(device),
            #     torch.Tensor(labels).to(device),
            #     torch.Tensor(label_masks).to(device),
            # )
            # import pdb; pdb.set_trace()


            probs, loss = model(images, image_masks, labels, label_masks, reprod_logger)

            word_loss, struct_loss, parent_loss, kl_loss = loss
            loss = word_loss + struct_loss + parent_loss + kl_loss

            reprod_logger.add(f"loss_{idx}_{j}", loss.cpu().detach().numpy())
            reprod_logger.add(f"word_loss_{idx}_{j}", word_loss.cpu().detach().numpy())
            reprod_logger.add(f"struct_loss_{idx}_{j}", struct_loss.cpu().detach().numpy())
            reprod_logger.add(f"parent_loss_{idx}_{j}", parent_loss.cpu().detach().numpy())
            reprod_logger.add(f"kl_loss_{idx}_{j}", kl_loss.cpu().detach().numpy())
            # reprod_logger.add("lr_{}".format(idx), np.array(optimizer.get_lr()))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient'])
            optimizer.step()
        # lr_scheduler.step()

    reprod_logger.save("./result/losses_ref.npy")


def test_backward(params, inputs):
    max_iter = 5
    # set determinnistic flag
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # FLAGS_cudnn_deterministic = True

    # set random seed
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    paddle.seed(params['seed'])

    # load paddle model
    paddle_model = Backbone_paddle(params)
    paddle_model.eval()
    # paddle_state_dict = paddle.load("./checkpoints/SAN_decoder/best_paddle.pdparams")
    # paddle_model.set_dict(paddle_state_dict['model'])

    # load torch model
    torch_model = Backbone_torch(params)
    torch_model.eval()

    # torch_state_dict = torch.load(
    #     "./checkpoints/SAN_decoder/best.pth", map_location=torch_device
    # )
    # torch_model.load_state_dict(torch_state_dict['model'])
    torch_model.to(torch_device)

    # init optimizer
    opt_paddle = getattr(paddle.optimizer, 'Momentum')(
        learning_rate=1e-3,
        epsilon=float(params['eps']),
        parameters=paddle_model.parameters(),
        weight_decay=float(params['weight_decay']),
        grad_clip=nn.ClipGradByGlobalNorm(params['gradient']),
    )

    opt_torch = getattr(torch.optim, 'SGD')(
        torch_model.parameters(),
        lr=1e-3,
        eps=float(params['eps']),
        momentum=0.9,
        weight_decay=float(params['weight_decay']),
    )

    # prepare logger & load data
    reprod_logger = ReprodLogger()
    # inputs = np.load("./result/test_data.npy")

    train_one_epoch_paddle(
        inputs,
        paddle_model,
        opt_paddle,
        max_iter,
        reprod_logger,
    )

    train_one_epoch_torch(
        inputs,
        torch_model,
        opt_torch,
        max_iter,
        reprod_logger,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HYB Tree')
    parser.add_argument(
        '--config', default='config.yaml', type=str, help='path to config file'
    )
    parser.add_argument('--check', action='store_true', help='only for code check')
    parser.add_argument(
        '--gpu',
        type=int,
        help='Use which GPU to train model, -1 means use CPU',
        default=-1,
    )
    parser.add_argument(
        '--multi_gpu', help='whether use multi gpu', action='store_true'
    )
    args = parser.parse_args()

    params, train_loader, eval_loader = init(args)

    params['dropout'] = False
    params['densenet']['use_dropout'] = False


    # if not os.path.exists("./result/test_data.npy"):
    inputs = gen_iter_data(params, train_loader, 50)

    test_backward(params, inputs)

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/losses_ref.npy")
    paddle_info = diff_helper.load_info("./result/losses_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/backward_diff.log", diff_threshold=1e-3)
