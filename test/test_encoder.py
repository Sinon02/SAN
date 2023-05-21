import argparse
import torch
import paddle
import numpy as np
import random
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Backbone import Backbone as Backbone_paddle
from models_torch.Backbone import Backbone as Backbone_torch
from utils import init

device = "gpu:0"  # you can also set it as "cpu"
torch_device = torch.device("cuda:0" if "gpu" in device else "cpu")
paddle.set_device(device)

def test_forward(params, test_data):
    # get inference data
    test_paddle_data = test_data
    if str(test_paddle_data.dtype) == 'paddle.int64':
        test_torch_data = (torch.LongTensor(test_paddle_data.numpy()).to(torch_device))
    elif str(test_paddle_data.dtype) == 'paddle.bool':
        test_torch_data = (torch.Tensor(test_paddle_data.numpy()).bool().to(torch_device))
    else:
        test_torch_data = (torch.Tensor(test_paddle_data.numpy()).to(torch_device))

    # load paddle model
    paddle_model = Backbone_paddle(params)
    paddle_model.eval()
    paddle_state_dict = paddle.load("./checkpoints/SAN_decoder/best_paddle.pdparams")
    paddle_model.set_dict(paddle_state_dict['model'])

    # load torch model
    torch_model = Backbone_torch(params)
    torch_model.eval()

    torch_state_dict = torch.load(
        "./checkpoints/SAN_decoder/best.pth", map_location=torch_device
    )
    torch_model.load_state_dict(torch_state_dict['model'])
    torch_model.to(torch_device)

    # save the paddle output
    reprod_logger = ReprodLogger()
    paddle_out = paddle_model.encoder(test_paddle_data)
    reprod_logger.add("image_feaures", paddle_out.cpu().detach().numpy())
    reprod_logger.save("./result/forward_paddle.npy")

    # save the torch output
    torch_out = torch_model.encoder(test_torch_data)
    reprod_logger.add("image_feaures", torch_out.cpu().detach().numpy())
    reprod_logger.save("./result/forward_ref.npy")


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
        default=0,
    )
    parser.add_argument(
        '--multi_gpu', help='whether use multi gpu', action='store_true'
    )
    args = parser.parse_args()

    params, train_loader, eval_loader = init(args)

    params['dropout'] = False
    params['densenet']['use_dropout'] = False

    for i in range(50):
        test_data = next(eval_loader())

    test_forward(params, test_data[0])

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/forward_ref.npy")
    paddle_info = diff_helper.load_info("./result/forward_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/forward_diff.log", diff_threshold=1e-5)
