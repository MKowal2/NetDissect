import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import pdb
import numpy as np
from torchvision.utils import flow_to_image
import torch
from tqdm import tqdm


def read_cur_imgs(root, files, flow=False, dims=None):
    imgs = []
    for f in range(len(files)):
        if flow:
            img = np.load(os.path.join(root, files[f]))
        else:
            img = cv2.resize(cv2.imread(os.path.join(root, files[f])), dims)
        imgs.append(img)
    return imgs

def read_imgs(imgs_dir, imgs_prefix, flow=False, dims=None):
    imgs = []

    out_prefix = 'outputs/'

    for root, dirs, files in os.walk(imgs_dir):
        files = sorted(files)
        if flow:
            files = [f for f in files if '.npy' in f]
        else:
            files = [f for f in files if '.png' in f]
        imgs += read_cur_imgs(root, files, flow, dims)

        if len(imgs) > 500:
            # trim to first 500 frames
            imgs = imgs[:500]

        if flow:
            imgs = [np.array(flow_to_image(torch.tensor(flow)).permute(1,2,0)) for flow in imgs]

    return imgs

def gen_video(imgs, vidfile, fps):

    # Different ways to generate video, need to look up
    # fourcc= cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fourcc= cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    try:
        video= cv2.VideoWriter(vidfile, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
    except:
        print(imgs[0].shape)
    for i in range(len(imgs)):
        video.write(imgs[i])
    video.release()

if __name__=="__main__":
    imgs = []
    root = '/home/m2kowal/data/DTDB/frames/'
    vid_names =  [
        # 'Dominant_rigid_g2_c26',
        # 'Dominant_rigid_g2_c37',
        # 'Dominant_rigid_g2_c38',
        # 'Pluming_g1_c60',
        # 'Pluming_g1_c64',
        # 'Pluming_g1_c67',
        'Rotary_motion_g5_c18',
        'Rotary_motion_g5_c112',
        'Rotary_motion_g4_c37',
        'Rotary_motion_g4_c28',

                  ]


    # root = '/home/m2kowal/data/a2d_dataset/frames/'
    # vid_names =  [
    #     'Mf7WX8sEq2M',
    #     'qEVpJK7pEPE',
    #     'GxcQTYlL6TQ',
    #     'c2yuTg-G9HI',
    #     'on7MOeFewcc',
    #     'EadxBPmQvtg',
    #     'rhHUc5qEnuc',
    #     '0lb6hfFvsZw',
    #     '02WkK5o25ws',
    #               ]


    dims = (112, 112)
    for vid_name in tqdm(vid_names):
        imgs_dir = root + vid_name
        imgs_prefix = ''

        flow = True

        # CHANGE THIS IF FRAMES ARE TOO SLOW/FAST
        fps = 5

        # for dir_ in os.listdir(imgs_dir):
        # IGNORE EVERYTHING THAT ISNT AN MP4

        imgs = read_imgs(imgs_dir, imgs_prefix, flow=False, dims=dims)
        if flow:
            flow_imgs = read_imgs(imgs_dir, imgs_prefix, flow)
        # gen_video(imgs, os.path.join(imgs_dir, dir_+'.avi'))

        gen_video(imgs, 'output/' +'{}.mp4'.format(vid_name), fps)
        if flow:
            gen_video(flow_imgs, 'output/' + '{}_FLOW.mp4'.format(vid_name), fps)


