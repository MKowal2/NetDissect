import glob
import torch.utils.data as data
import json
import os
import numpy as np
import colorname
import dtdb_utils
from data_processing.bin_flow import *
from loadseg import AbstractSegmentation
from PIL import Image
import os.path
import torch.utils.data as data
import csv
import mat73
from tqdm import tqdm

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
    def __init__(self, data_root, categories, min_video_frame_length=64,choose_ann_idx=0, args=None):
        print('Creating A2D Dataset...')
        self.args = args
        self.data_root = data_root
        self.idx_action_dict = {y: x for x, y in category_dict.items()}
        self.appearances, self.dynamics = self._get_app_dyn()
        self.dynamics_map = dict((t, i) for i, t in enumerate(self.dynamics))
        self.appearance_map = dict((t, i) for i, t in enumerate(self.appearances))
        self.categories = categories
        self.path_labels, self.data = self.get_data(min_video_frame_length,choose_ann_idx)
        self.flow_names= get_flow_names(args.num_flow_mags, args.num_flow_dirs)
        '''
        self.path_labels = [{path (to frames!): x, dynamic: x, appearance: x, video: x}, {}, ...]
        self.data = {OG_vid_id: {duration: x , frame_count: x, fps: x, size: [h, w], dynamic: x, appearance: x, subset: train}}
        '''


    def get_data(self, min_video_frame_length=0,choose_ann_idx=0):
        min_frame_half_count = int(min_video_frame_length/2)

        annotation_path = os.path.join(self.data_root, 'Release/videoset.csv')
        with open(annotation_path, 'r') as f:
            annotations = list(csv.reader(f))
        num_skipped_vids = 0
        list_data = []
        dict_data = {}

        for idx, video in enumerate(tqdm(annotations)):
            num_frames = int(video[6])
            if num_frames < min_video_frame_length:
                num_skipped_vids += 1
                continue

            # need to get labelled frames and then decide on which annotated frame to use as center!
            annotated_frames = glob.glob(self.data_root + '/Release/Annotations/mat/{}/*'.format(video[0]))
            annotated_frames = sorted([int(idx[-9:].strip('.mat')) for idx in annotated_frames])

            valid_ann_frames = []
            for ann in annotated_frames:
                # check if there are 32 frames on either side of any annotated frames
                if (ann) > min_frame_half_count and (num_frames-ann) > min_frame_half_count:
                    valid_ann_frames.append(ann)

            # select the annotated frame if there are some
            if len(valid_ann_frames) == 0:
                num_skipped_vids += 1
                continue
            else:
                ann_idx = valid_ann_frames[choose_ann_idx]
                start_frame = ann_idx - min_frame_half_count
                end_frame = ann_idx + min_frame_half_count

            mat_path = self.data_root + '/Release/Annotations/mat/{}/{:05d}.mat'.format(video[0],ann_idx)
            labels = np.unique(read_mask(mat_path))[1:]
            appearances = [self.idx_action_dict[val].split('-')[0].strip() for val in labels]
            dynamics = [self.idx_action_dict[val].split('-')[1].strip() for val in labels]
            dict_data[video[0]] = {
                'size': [video[4], video[5]],
                'frame_count': video[6],
                'subset': 'train' if video[-1] == '0' else 'test',
                'annotation_frame': ann_idx,
                'dynamic': dynamics,
                'appearance': appearances
            }

            frames = list(range(start_frame,end_frame))
            for frame in frames:
                path = self.data_root + '/frames/{}/{:05d}.png'.format(video[0],frame)
                list_data.append(
                    {
                        'path': path,
                        'dynamic': dynamics,
                        'appearance': appearances,
                        'video': video[0],
                        'label_path': mat_path,
                        'annotated_frame': True if frame == ann_idx else False # indicator whether this is a frame with a label in it
                    }
                )

        print('{} videos removed'.format(num_skipped_vids))
        print('{} total frames'.format(len(list_data)))
        print('{} total videos'.format(len(dict_data.keys())))
        return list_data, dict_data

    def _get_app_dyn(self):
        actor_actions = list(category_dict.keys())[1:]
        actors = list(dict.fromkeys([action.split('-')[0] for action in actor_actions]))
        actions = list(dict.fromkeys([action.split('-')[1].strip() for action in actor_actions]))
        appearances = ['-']+actors
        dynamics = ['-']+actions

        return appearances, dynamics

    @classmethod
    def resolve_segmentation(cls, metadata, categories=None):
        filename, dyn_numbers, app_numbers, labelled, label_path, video_name = metadata
        result = {}
        if wants('dynamics', categories) and labelled:
            result['dynamic']  = dyn_mp(read_mask(filename.replace('frames', 'Release/Annotations/mat').replace('.png', '.mat')))
            # result['dynamic']  = read_mask(filename.replace('frames', 'Release/Annotations/mat').replace('.png', '.mat'))
        if wants('appearance', categories) and labelled:
            result['appearance'] = app_mp(read_mask(filename.replace('frames', 'Release/Annotations/mat').replace('.png', '.mat')))
            # result['appearance'] = read_mask(filename.replace('frames', 'Release/Annotations/mat').replace('.png', '.mat')))
        if wants('flow', categories):
            result['flow'] = bin_flow(filename.replace('.png', '_flow.npy'))
        if wants('color', categories):
            result['color'] = colorname.label_major_colors(np.asarray(Image.open(filename))) + 1

        # return a dictionary of segmentation maps (or [1] for video label) and shape of video
        shape = result['color'].shape[-2:] if wants('color', categories) else (1, 1)
        return result, shape



    def metadata(self, i):
        '''Returns an object that can be used to create all segmentations.'''
        filename, dynamic, appearance, labelled, label_path, video_name = self.path_labels[i]['path'], self.path_labels[i]['dynamic'], \
                                                    self.path_labels[i]['appearance'], self.path_labels[i]['annotated_frame'], self.path_labels[i]['label_path'], self.path_labels[i]['video']
        # need to change to list for multi-labeled videos
        dyn_numbers = [self.dynamics_map[dyn] for dyn in dynamic]
        app_numbers = [self.appearance_map[app] for app in appearance]
        return filename, dyn_numbers, app_numbers, labelled, label_path, video_name

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
            return [self.flow_names[j] + '-f']
        if category == 'dynamic':
            return [self.dynamics[j]]
        if category == 'appearance':
            return [self.appearances[j]]
        return []

def wants(what, option):
    if option is None:
        return True
    return what in option

def read_mask(mask_fname):
    anot = mat73.loadmat(mask_fname)
    anot_parsed = anot['reS_id']
    return anot_parsed


app_dyn_hash_dict = {
    'dynamic': {},
    'appearance': {}
}
for val in category_dict.values():
    if val == 0:
        app_dyn_hash_dict['dynamic'][val] = val
        app_dyn_hash_dict['appearance'][val] = val
    else:
        app_dyn_hash_dict['dynamic'][val] = int(str(val)[1])
        app_dyn_hash_dict['appearance'][val] = int(str(val)[0])

def dyn_mp(entry):
    return app_dyn_hash_dict['dynamic'][entry] if entry in app_dyn_hash_dict['dynamic'] else entry
dyn_mp = np.vectorize(dyn_mp)
def app_mp(entry):
    return app_dyn_hash_dict['appearance'][entry] if entry in app_dyn_hash_dict['appearance'] else entry
app_mp = np.vectorize(app_mp)


