import glob
import json
import sys
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_vid_info(filename):
    video = cv2.VideoCapture(filename)

    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)

    return duration, frame_count, fps, (height, width)

def get_img_info(dir):
    frames = glob.glob(dir + '/*')
    video = cv2.VideoCapture(dir)

    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)

    return duration, frame_count, fps, (height, width)

def gen_dtdb_json(data_path, use_img=True):


    # file locations
    data_root = data_path + '/'
    file_path =  data_root + 'Conversion_scripts/'
    dyn_files = ['Dyn2App_correspondence_TEST.csv', 'Dyn2App_correspondence_TRAIN.csv']
    app_files = ['App2Dyn_correspondence_TEST.csv', 'App2Dyn_correspondence_TRAIN.csv']

    # create list of unique video id's

    vid_dict = {} # each id with both types of labels has a dyn value and an app value
    # dynamics first
    for file in dyn_files:
        with open(file_path + file) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                video_id = line.split(', ')[0]
                dyn_label = line.split(', ')[1][:-1]
                if 'TEST' in file:
                    dyn_subset = 'test'
                    dyn_subset_cap = 'TEST'
                else:
                    dyn_subset = 'train'
                    dyn_subset_cap = 'TRAIN'

                cls = dyn_label.split('_g')[0]

                if not use_img:
                    vid_path = data_root + 'BY_DYNAMIC_FINAL/' + dyn_subset_cap + '/' + cls + '/' + dyn_label
                    duration, frame_count, fps, size = get_vid_info(vid_path)
                else:
                    vid_path = data_root + 'frames/' + dyn_label.split('.')[0]
                    duration, frame_count, fps, size = get_img_info(vid_path)
                if not os.path.exists(vid_path):
                    print('Video doesnt exist: {}'.format(vid_path))
                    continue
                vid_dict[video_id] = {'duration': duration, 'frame_count': frame_count, 'fps':fps, 'size': size, 'dynamic': dyn_label, 'dyn_subset': dyn_subset}

    #app second
    for file in app_files:
        with open(file_path + file) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                video_id = line.split(', ')[0]
                app_label = line.split(', ')[1][:-1]
                if 'TEST' in file:
                    app_subset = 'test'
                    app_subset_cap = 'TEST'
                else:
                    app_subset = 'train'
                    app_subset_cap = 'TRAIN'

                cls = app_label.split('_g')[0]

                if cls == 'Sliding':
                    cls = 'Sliding_gate'

                vid_path = data_root + 'BY_APPEARANCE_FINAL/' + app_subset_cap + '/' + cls + '/' + app_label
                if not os.path.exists(vid_path):
                    continue
                vid_dict[video_id]['appearance'] = app_label
                vid_dict[video_id]['app_subset'] = app_subset



    with open(data_root + 'app_dyn_correspondance.json', 'w') as f:
        json.dump(vid_dict, f)



# gen_dtdb_json('/home/m2kowal/data/DTDB', use_img=True)