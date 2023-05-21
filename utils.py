import os
import yaml
import math
import paddle
import random
import numpy as np
from dataset import get_dataset
from difflib import SequenceMatcher
import paddle.distributed as dist
from paddle.optimizer.lr import LRScheduler, LinearWarmup



def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('try UTF-8 encoding')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

    if not params['experiment']:
        print('expriment name cannot be empty!')
        exit(-1)

    if not params['train_image_path']:
        print('training images cannot be empty!')
        exit(-1)

    if not params['train_label_path']:
        print('training labels cannot be empty!')
        exit(-1)

    if not params['eval_image_path']:
        print('test images cannot be empty!')
        exit(-1)

    if not params['eval_label_path']:
        print('test labels cannot be empty!')
        exit(-1)
    if not params['word_path']:

        print('word dict cannot be empty')
        exit(-1)
    return params


def init(args):
    """config"""
    params = load_config(args.config)

    """random seed"""
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    paddle.seed(params['seed'])

    if args.multi_gpu:
        params['device'] = 'gpu'
    else:
        if args.gpu >= 0:
            device = f'gpu:{args.gpu}'
        else:
            # use cpu
            device = 'cpu'
        params['device'] = device

    train_loader, eval_loader = get_dataset(args, params)
    return params, train_loader, eval_loader

class TwoStepCosineDecay(LRScheduler):
    def __init__(self,
                 learning_rate,
                 T_max1,
                 T_max2,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(T_max1, int):
            raise TypeError(
                "The type of 'T_max1' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max1))
        if not isinstance(T_max2, int):
            raise TypeError(
                "The type of 'T_max2' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max2))
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingDecay' must be 'float, int', but received %s."
                % type(eta_min))
        assert T_max1 > 0 and isinstance(
            T_max1, int), " 'T_max1' must be a positive integer."
        assert T_max2 > 0 and isinstance(
            T_max2, int), " 'T_max1' must be a positive integer."
        self.T_max1 = T_max1
        self.T_max2 = T_max2
        self.eta_min = float(eta_min)
        super(TwoStepCosineDecay, self).__init__(learning_rate, last_epoch,
                                                   verbose)

    def get_lr(self):
        if self.last_epoch <= self.T_max1:
            if self.last_epoch == 0:
                return self.base_lr
            elif (self.last_epoch - 1 - self.T_max1) % (2 * self.T_max1) == 0:
                return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                    math.pi / self.T_max1)) / 2

            return (1 + math.cos(math.pi * self.last_epoch / self.T_max1)) / (
                1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max1)) * (
                    self.last_lr - self.eta_min) + self.eta_min
        else:
            if (self.last_epoch - 1 - self.T_max2) % (2 * self.T_max2) == 0:
                return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                    math.pi / self.T_max2)) / 2

            return (1 + math.cos(math.pi * self.last_epoch / self.T_max2)) / (
                    1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max2)) * (
                           self.last_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self):
        if self.last_epoch <= self.T_max1:
            return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.last_epoch / self.T_max1)) / 2
        else:
            return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.last_epoch / self.T_max2)) / 2

class TwoStepCosineLR(object):
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs1,
                 epochs2,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(TwoStepCosineLR, self).__init__()
        self.learning_rate = learning_rate
        self.T_max1 = step_each_epoch * epochs1
        self.T_max2 = step_each_epoch * epochs2
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = TwoStepCosineDecay(
            learning_rate=self.learning_rate,
            T_max1=self.T_max1,
            T_max2=self.T_max2,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


def updata_lr(optimizer, current_epoch, current_step, steps, epoches, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)

    else:
        new_lr = (
            0.5
            * (
                1
                + math.cos(
                    (current_step + 1 + (current_epoch - 1) * steps)
                    * math.pi
                    / (epoches * steps)
                )
            )
            * initial_lr
        )

    optimizer.set_lr(new_lr)


def save_checkpoint(
    model,
    optimizer,
    word_score,
    struct_score,
    ExpRate_score,
    epoch,
    optimizer_save=False,
    path='checkpoints',
    multi_gpu=False,
    local_rank=0,
):
    filename = f'{os.path.join(path, model.name)}/{model.name}_WordRate-{word_score:.4f}_structRate-{struct_score:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pth'

    if optimizer_save:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    else:
        state = {'model': model.state_dict()}

    paddle.save(state, filename)
    print(f'Save checkpoint: {filename}\n')
    return filename


def load_checkpoint(model, optimizer, path):
    state = paddle.load(path)

    if 'optimizer' in state:
        optimizer.set_state_dict(state['optimizer'])
    else:
        print(f'No optimizer in the pretrained model')

    model.set_state_dict(state['model'])


class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num


def cal_score(probs, labels, mask):
    batch_size = probs[0].shape[0]
    word_probs, struct_probs = probs
    word_label, struct_label = labels[:, :, 1], labels[:, :, 4:]
    struct_label = struct_label.reshape((batch_size, -1))
    line_right = 0
    word_pred = word_probs.argmax(2)

    struct_mask = mask[:, :, 1]
    struct_probs = struct_probs * struct_mask[:, :, None]
    struct_probs = struct_probs.reshape((batch_size, -1))
    struct_pred = struct_probs > 0.5

    word_scores = [
        SequenceMatcher(
            None, s1[: int(np.sum(s3))], s2[: int(np.sum(s3))], autojunk=False
        ).ratio()
        * (len(s1[: int(np.sum(s3))]) + len(s2[: int(np.sum(s3))]))
        / len(s1[: int(np.sum(s3))])
        / 2
        for s1, s2, s3 in zip(
            word_label.cpu().detach().numpy(),
            word_pred.cpu().detach().numpy(),
            mask.cpu().detach().numpy(),
        )
    ]
    struct_scores = [
        SequenceMatcher(
            None, s1[: int(np.sum(s3))], s2[: int(np.sum(s3))], autojunk=False
        ).ratio()
        * (len(s1[: int(np.sum(s3))]) + len(s2[: int(np.sum(s3))]))
        / len(s1[: int(np.sum(s3))])
        / 2
        for s1, s2, s3 in zip(
            struct_label.cpu().detach().numpy(),
            struct_pred.cpu().detach().numpy(),
            mask.cpu().detach().numpy(),
        )
    ]

    batch_size = len(word_scores) if word_probs is not None else len(struct_scores)

    for i in range(batch_size):
        if struct_mask[i].sum() > 0:
            if word_scores[i] == 1 and struct_scores[i] == 1:
                line_right += 1
        else:
            if word_scores[i] == 1:
                line_right += 1

    ExpRate = line_right / batch_size

    word_scores = np.mean(word_scores) if word_probs is not None else 0
    struct_scores = np.mean(struct_scores) if struct_probs is not None else 0
    return word_scores, struct_scores, ExpRate

