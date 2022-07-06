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
def get_dtdb_data(data_root):
    correspondance_path = data_root + '/app_dyn_correspondance.json'
    if not os.path.exists(correspondance_path):
        dtdb_utils.gen_dtdb_json(data_root)
    with open(correspondance_path) as file:
        data = json.load(file)

    return data

def filter_videos(data_root, data, min_video_frame_length):
    print('Filtering out DTDB videos with less than {} frames...'.format(min_video_frame_length+1))
    # turn data into list for iterators
    list_data = []
    new_data = data.copy()
    # filter out videos of length < min_video_frame_length
    num_removed_videos = 0
    for vid in data:
        info = data[vid]

        # remove all examples without an appearance label
        if 'appearance' in info.keys():
            paths = [os.path.join(data_root, 'frames', info['dynamic'][:-4]) + "/{:06d}.png".format(i) for i in list(range(1,int(info['frame_count'])))]
            length = len(paths)
            dyn_label = info['dynamic'].split('_g')[0]
            app_label = info['appearance'].split('_g')[0]
            if app_label == 'Sliding':
                app_label = 'Sliding_gate'

            if length > min_video_frame_length:
                for path in paths:
                    list_data.append({'path':path, 'dynamic': dyn_label, 'appearance':app_label})
            else:
                num_removed_videos += 1

        # Also remove items from original dictionary if they aren't in the list
                new_data.pop(vid)
        else:
            new_data.pop(vid)

    print('{} videos removed'.format(num_removed_videos))
    return list_data, new_data


# class DTDB(data.Dataset):
class DTDB(AbstractSegmentation):
    def __init__(self, data_root, categories, min_video_frame_length = 63):
        # need to replace this with datalist of all images, with corresponding styles and class labels
        self.data_root = data_root

        # data is a dict of all info, path_labels is list of all frames
        self.data = get_dtdb_data(self.data_root)
        self.path_labels, self.data = filter_videos(self.data_root, self.data, min_video_frame_length)
        self.dynamics = ['-']+ sorted([y.split('/')[-1] for y in glob.glob(self.data_root + '/BY_DYNAMIC_FINAL/TRAIN/*')])
        self.appearances = ['-']+ sorted([y.split('/')[-1] for y in glob.glob(self.data_root + '/BY_APPEARANCE_FINAL/TRAIN/*')])
        self.dynamics_map = dict((t, i) for i, t in enumerate(self.dynamics))
        self.appearance_map = dict((t, i) for i, t in enumerate(self.appearances))
        self.categories = categories

    @classmethod
    def resolve_segmentation(cls, metadata, categories=None):
        filename, dyn_numbers, app_numbers = metadata
        result = {}
        if wants('dynamics', categories):
            result['dynamic'] = dyn_numbers
        if wants('appearance', categories):
            result['appearance'] = app_numbers
        if wants('flow', categories):
            result['flow'] = bin_flow(filename.replace('.png', '_flow.npy'))
        if wants('color', categories):
            result['color'] = colorname.label_major_colors(np.asarray(Image.open(filename))) + 1

        # return a dictionary of segmentation maps (or [1] for video label) and shape of video
        # TODO: retrieve shape from video, maybe build into app_dyn_correspondance.json to save time
        shape = result['color'].shape[-2:] if wants('color', categories) else (1, 1)
        return result, shape

    def metadata(self, i):
        '''Returns an object that can be used to create all segmentations.'''
        filename, dynamic, appearance = self.path_labels[i]['path'], self.path_labels[i]['dynamic'], self.path_labels[i]['appearance']
        # need to change to list for multi-labeled videos
        dyn_numbers = [self.dynamics_map[dynamic]]
        app_numbers = [self.appearance_map[appearance]]
        return filename, dyn_numbers, app_numbers

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
            return [self.dynamics[j]]
        if category == 'appearance':
            return [self.appearances[j]]
        return []

def wants(what, option):
    if option is None:
        return True
    return what in option