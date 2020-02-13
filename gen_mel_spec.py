import os
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from data_utils import TextMelLoader, TextMelCollate
from hparams import create_hparams

def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    testset = TextMelLoader(hparams)


def gen_mel_spec(hparams):
    prepare_dataloaders(hparams)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-o', '--output_directory', type=str,
    #                     help='directory to save checkpoints')
    # parser.add_argument('-l', '--log_directory', type=str,
    #                     help='directory to save tensorboard logs')
    # parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
    #                     required=False, help='checkpoint path')
    # parser.add_argument('--warm_start', action='store_true',
    #                     help='load model weights only, ignore specified layers')
    # parser.add_argument('--n_gpus', type=int, default=1,
    #                     required=False, help='number of gpus')
    # parser.add_argument('--rank', type=int, default=0,
    #                     required=False, help='rank of current gpu')
    # parser.add_argument('--group_name', type=str, default='group_name',
    #                     required=False, help='Distributed group name')
    # parser.add_argument('--hparams', type=str,
    #                     required=False, help='comma separated name=value pairs')

    # args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    gen_mel_spec(hparams)
    # torch.backends.cudnn.enabled = hparams.cudnn_enabled
    # torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

