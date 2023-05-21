import os
import time
import argparse
import paddle
from paddle import nn
import paddle.distributed as dist
import numpy as np
from tensorboardX import SummaryWriter

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import save_checkpoint, load_checkpoint, init
from models.Backbone import Backbone
from training import train, eval

parser = argparse.ArgumentParser(description='HYB Tree')
parser.add_argument(
    '--config', default='config.yaml', type=str, help='path to config file'
)
parser.add_argument('--check', action='store_true', help='only for code check')
parser.add_argument('--gpu', type=int, help='Use which GPU to train model, -1 means use CPU', default=0)
parser.add_argument('--multi_gpu', help='whether use multi gpu', action='store_true')
args = parser.parse_args()

if not args.config:
    print('please provide config yaml')
    exit(-1)
if args.multi_gpu:
    dist.init_parallel_env()

print(f'rank: {dist.get_rank()}')
print(f'world size: {dist.get_world_size()}')

params, train_loader, eval_loader = init(args)


paddle.device.set_device(params['device'])

model = Backbone(params)
if args.multi_gpu:
    model = paddle.DataParallel(model)

now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = (
    f'{params["experiment"]}_{now}_Encoder-{params["encoder"]["net"]}_Decoder-{params["decoder"]["net"]}_'
    f'max_size-{params["image_height"]}-{params["image_width"]}'
)
print(model.name)

if args.check or not dist.get_rank() == 0:
    writer = None
else:
    writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

optimizer = getattr(paddle.optimizer, params['optimizer'])(
    learning_rate=float(params['lr']),
    # epsilon=float(params['eps']),
    parameters=model.parameters(),
    # weight_decay=float(params['weight_decay']),
    grad_clip=nn.ClipGradByNorm(params['gradient']),
)

if params['finetune']:
    print('loading pretrain model weight')
    print(f'pretrain model: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'])

if not args.check and dist.get_rank() == 0:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(
        f'cp {args.config} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml'
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

