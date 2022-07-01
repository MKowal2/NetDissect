import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as tf
from torch import Tensor, nn
from typing import Optional, Tuple
from torchvision.io import read_video
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torch.utils.data.dataloader import DataLoader


def generateflow(args):

    if args.dataset == 'dtdb':
        categories = ['dynamics']
        data = dtdb_flow.DTDB(data_root='/home/m2kowal/data/DTDB',
                                 categories=categories)

    data_loader = DataLoader(data, batch_size=1, num_workers=0)


    # for idx,

    print('done!')



if __name__ == '__main__':
    # general imports
    import argparse

    # dataset imports
    import dtdb_flow


    parser = argparse.ArgumentParser(
        description='Generating optical data_processing.')
    parser.add_argument(
            '--save_size',
            type=int, default=112,
            help='pixel size for output videos')
    parser.add_argument(
        '--dataset',
        type=str, default='dtdb',
        help='dataset to generate data_processing for')
    args = parser.parse_args()
    generateflow(args)

