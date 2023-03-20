import os
import time
import argparse
import paddle
from paddle import nn

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_checkpoint, init
from models.Backbone import Backbone
from training import eval

if __name__ == '__main__':
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
    args = parser.parse_args()

    if not args.config:
        print('please provide config yaml')
        exit(-1)

    params, train_loader, eval_loader = init(args)

    model = Backbone(params)
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    model.name = (
        f'{params["experiment"]}_{now}_Encoder-{params["encoder"]["net"]}_Decoder-{params["decoder"]["net"]}_'
        f'max_size-{params["image_height"]}-{params["image_width"]}'
    )
    print(model.name)
    writer = None

    optimizer = getattr(paddle.optimizer, params['optimizer'])(
        learning_rate=float(params['lr']),
        epsilon=float(params['eps']),
        parameters=model.parameters(),
        weight_decay=float(params['weight_decay']),
        grad_clip=nn.ClipGradByGlobalNorm(params['gradient']),
    )

    load_checkpoint(
        model,
        optimizer,
        os.path.join(params['checkpoint_dir'], 'SAN_decoder/best_paddle.pdparams'),
    )

    min_score = 0
    min_step = 0
    epoch = 0

    eval_loss, eval_word_score, eval_node_score, eval_expRate = eval(
        params, model, epoch, eval_loader, writer=writer
    )

    print(
        f'Epoch: {epoch+1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  struct score: {eval_node_score:.4f} '
        f'ExpRate: {eval_expRate:.4f}'
    )
