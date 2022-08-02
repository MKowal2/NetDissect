import glob
import torch.utils.data as data
import json
from PIL import Image
from torch import Tensor, nn
from typing import Tuple
import torchvision.transforms.functional as F
import torch
from tqdm import tqdm

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

def get_dtdb_data(data_root):
    correspondance_path = data_root + '/app_dyn_correspondance.json'
    with open(correspondance_path) as file:
        data = json.load(file)
    return data

class DTDB(data.Dataset):
    def __init__(self, data_root):
        # need to replace this with datalist of all images, with corresponding styles and class labels
        self.data_root = data_root
        self.data = self.get_data()
        self.transforms = OpticalFlow()

    def get_data(self):
        print('Constructing paired data list...')
        data = []
        videos = glob.glob(self.data_root + '/frames/*')
        for i, vid_path in enumerate(tqdm(videos)):
            frames = glob.glob(vid_path + '/*')
            for frame_id in range(1, int(len(frames))):
                frame1_path = vid_path + "/{:06d}.png".format(frame_id)
                frame2_path = vid_path + "/{:06d}.png".format(frame_id+1)
                data.append({'video_id': vid_path,
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
