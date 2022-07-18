import glob
import torch.utils.data as data
import json
import os
import numpy as np
import colorname
import dtdb_utils
from data_processing.bin_flow import bin_flow
from data_processing.bin_flow import flow_names1
from loadseg import AbstractSegmentation
from PIL import Image
import os.path
import torch.utils.data as data
import csv
import mat73

category_dict = {
'Background'	:0,
'adult-climbing':	11,
'adult-crawling':	12,
'adult-eating'	:13,
'adult-flying'	:14,
'adult-jumping'	:15,
'adult-rolling'	:16,
'adult-running'	:17,
'adult-walking	':18,
'adult-none	':19,
'baby-climbing'	:21,
'baby-crawling'	:22,
'baby-eating'	:23,
'baby-flying'	:24,
'baby-jumping':	25,
'baby-rolling':	26,
'baby-running':	27,
'baby-walking':	28,
'baby-none'	:29,
'ball-climbing'	:31,
'ball-crawling'	:32,
'ball-eating'	:33,
'ball-flying	':34,
'ball-jumping':	35,
'ball-rolling':	36,
'ball-running':	37,
'ball-walking':	38,
'ball-none'	:39,
'bird-climbing'	:41,
'bird-crawling'	:42	,
'bird-eating'	:43	,
'bird-flying'	:44	,
'bird-jumping':	45,
'bird-rolling':	46,
'bird-running':	47,
'bird-walking':	48,
'bird-none'	:49,
'car-climbing'	:51,
'car-crawling'	:52,
'car-eating'	:53	,
'car-flying'	:54	,
'car-jumping' :55	,
'car-rolling':	56	,
'car-running':57	,
'car-walking':	58	,
'car-none'	:59	,
'cat-climbing'	:61,
'cat-crawling'	:62,
'cat-eating'	:63,
'cat-flying'	:64,
'cat-jumping'	:65,
'cat-rolling'	:66,
'cat-running'	:67,
'cat-walking'	:68,
'cat-none':	69,
'dog-climbing':	71,
'dog-crawling':	72,
'dog-eating' :73,
'dog-flying':	74,
'dog-jumping':	75,
'dog-rolling':	76,
'dog-running':	77,
'dog-walking':	78,
'dog-none':	79}

class A2D(data.Dataset):
    def __init__(self, data_root, categories, min_video_frame_length=100,center_crop=128):
        # need to replace this with datalist of all images, with corresponding styles and class labels
        self.data_root = data_root
        # todo: decide on pseudo-label technique
        # todo: format get_data() like dtdb
        self.data = self.get_data()

        self.appearances, self.dynamics = self._get_app_dyn()
        self.dynamics_map = dict((t, i) for i, t in enumerate(self.dynamics))
        self.appearance_map = dict((t, i) for i, t in enumerate(self.appearances))
        self.categories = categories

    def get_data(self):
        annotation_path = os.path.join(self.data_root, 'Release/videoset.csv')
        with open(annotation_path, 'r') as f:
            annotations = list(csv.reader(f))


        print('Constructing paired data list...')
        data = []
        for idx, video in enumerate(annotations):
            for frame_id in range(1, int(video[-3])):
                print('only video labels plz')
                mat1 = self.data_root + '/Release/Annotations/mat/' + video[0] + "/{:05d}.mat".format(frame_id)
                frame1_path = os.path.join(self.data_root + '/frames',video[0]) + "/{:05d}.png".format(frame_id)
                frame2_path = os.path.join(self.data_root + '/frames',video[0]) + "/{:05d}.png".format(frame_id+1)
                data.append({'video_id': video[0],
                    'frame1': frame1_path,
                    'frame2': frame2_path})

        return data

    def _read_mask(self, mask_fname):
        anot = mat73.loadmat(mask_fname)
        anot_parsed = anot['reS_id']
        return anot_parsed

    def _get_app_dyn(self):
        actor_actions = list(category_dict.keys())[1:]
        actors = list(dict.fromkeys([action.split('-')[0] for action in actor_actions]))
        actions = list(dict.fromkeys([action.split('-')[1].strip() for action in actor_actions]))
        appearances = ['-']+actors
        dynamics = ['-']+actions

        return appearances, dynamics


    def metadata(self, i):
        '''Returns an object that can be used to create all segmentations.'''
        filename, dynamic, appearance, video_name = self.path_labels[i]['path'], self.path_labels[i]['dynamic'], \
                                                    self.path_labels[i]['appearance'], self.path_labels[i]['video']
        # need to change to list for multi-labeled videos
        dyn_numbers = [self.dynamics_map[dynamic]]
        app_numbers = [self.appearance_map[appearance]]
        return filename, dyn_numbers, app_numbers, video_name

    def filename(self, i):
        '''Returns the filename for the nth dataset image.'''
        return self.path_labels[i]['path']

    def video_size(self):
        '''Returns the number of images in this dataset.'''
        return len(self.data)

    def img_size(self):
        '''Returns the number of images in this dataset.'''
        return len(self.path_labels)

    def all_names(self, category, j):
        if j == 0:
            return []
        if category == 'color':
            return [colorname.color_names[j - 1] + '-c']
        if category == 'flow':
            return [flow_names1[j] + '-f']
        if category == 'dynamic':
            # todo: get pseudo labels for all frames
            print('todo: get pseudo labels for all frames')
            exit()
            return [self.dynamics[j]]
        if category == 'appearance':
            # todo: get pseudo labels for all frames
            print('todo: get pseudo labels for all frames')
            exit()
            return [self.appearances[j]]
        return []

def wants(what, option):
    if option is None:
        return True
    return what in option


