import glob
import os.path
from PIL import Image
import torch.utils.data as data
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as tf
from torch import Tensor, nn
from typing import Tuple


class OpticalFlow(nn.Module):
    def forward(self, img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(img1, Tensor):
            img1 = F.pil_to_tensor(img1)
        if not isinstance(img2, Tensor):
            img2 = F.pil_to_tensor(img2)

        img1 = F.convert_image_dtype(img1, torch.float)
        img2 = F.convert_image_dtype(img2, torch.float)

        # map [0, 1] into [-1, 1]
        img1 = F.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img2 = F.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            "The images are rescaled to ``[-1.0, 1.0]``."
        )

class A2D(data.Dataset):
    def __init__(self, data_root='/home/m2kowal/data/a2d_dataset'):
        # need to replace this with datalist of all images, with corresponding styles and class labels
        self.data_root = data_root
        self.data = self.get_data()
        self.transforms = OpticalFlow()


    def get_data(self):
        annotation_path = os.path.join(self.data_root, 'Release/videoset.csv')
        with open(annotation_path, 'r') as f:
            annotations = list(csv.reader(f))

        print('Constructing paired data list...')
        data = []
        for idx, video in enumerate(annotations):
            for frame_id in range(1, int(video[-3])):
                frame1_path = os.path.join(self.data_root + '/frames',video[0]) + "/{:05d}.png".format(frame_id)
                frame2_path = os.path.join(self.data_root + '/frames',video[0]) + "/{:05d}.png".format(frame_id+1)
                data.append({'video_id': video[0],
                    'frame1': frame1_path,
                    'frame2': frame2_path})

        return data

    def __getitem__(self, i):
        video = self.data[i]
        frame1 = video['frame1']
        save_path = frame1.replace('.png', '_flow.npy')
        frame2 = video['frame2']

        frame1 = Image.open(frame1)
        frame2 = Image.open(frame2)

        frame1 = F.resize(frame1, size=[320, 600])
        frame2 = F.resize(frame2, size=[320, 600])

        frame1, frame2 = self.transforms(frame1, frame2)

        return frame1, frame2, save_path

    def __len__(self):
        return len(self.data)
