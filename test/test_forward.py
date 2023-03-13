import argparse
import torch
import paddle
import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Backbone import Backbone as Backbone_paddle
from models_torch.Backbone import Backbone as Backbone_torch
from utils import init
from dataset import get_dataset



def test_forward(params):
    device = "cpu"  # you can also set it as "cpu"
    torch_device = torch.device("cuda:0" if "gpu" in device else "cpu")
    paddle.set_device(device)

    # load paddle model
    paddle_model = Backbone_paddle(params)
    paddle_model.eval()
    paddle_state_dict = paddle.load("./checkpoints/SAN_decoder/best_paddle.pdparams")
    paddle_model.set_dict(paddle_state_dict)

    # load torch model
    torch_model = Backbone_torch(params)
    torch_model.eval()

    torch_state_dict = torch.load("./checkpoints/SAN_decoder/best.pth", map_location=torch_device)
    torch_model.load_state_dict(torch_state_dict['model'])
    torch_model.to(torch_device)

    # load data
    inputs = np.random.random((1, 1, 320, 320))

    # save the paddle output
    reprod_logger = ReprodLogger()
    paddle_out = paddle_model(paddle.to_tensor(inputs, dtype="float32"))
    reprod_logger.add("logits", paddle_out.cpu().detach().numpy())
    reprod_logger.save("./result/forward_paddle.npy")

    # save the torch output
    torch_out = torch_model(
        torch.tensor(
            inputs, dtype=torch.float32).to(torch_device))
    reprod_logger.add("logits", torch_out.cpu().detach().numpy())
    reprod_logger.save("./result/forward_ref.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HYB Tree')
    parser.add_argument(
        '--config', default='config.yaml', type=str, help='path to config file'
    )
    parser.add_argument('--check', action='store_true', help='only for code check')
    parser.add_argument('--gpu', type=int, help='Use which GPU to train model, -1 means use CPU', default=-1)
    args = parser.parse_args()

    params, train_loader, eval_loader = init(args)

    test_forward(params)

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/forward_ref.npy")
    paddle_info = diff_helper.load_info("./result/forward_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="./result/log/forward_diff.log", diff_threshold=1e-5)