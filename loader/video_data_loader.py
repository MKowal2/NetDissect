from functools import partial
import numpy
import os
import re
import random
import signal
import csv
import video_settings as settings
import numpy as np
from collections import OrderedDict
from imageio import imread
import numpy as np
import os
import random
import torch
from PIL import Image
import torch.utils.data

def load_csv(filename, readfields=None):
    def convert(value):
        if re.match(r'^-?\d+$', value):
            try:
                return int(value)
            except:
                pass
        if re.match(r'^-?[\.\d]+(?:e[+=]\d+)$', value):
            try:
                return float(value)
            except:
                pass
        return value

    with open(filename) as f:
        reader = csv.DictReader(f)
        result = [{k: convert(v) for k, v in row.items()} for row in reader]
        if readfields is not None:
            readfields.extend(reader.fieldnames)
    return result



class VideoSegmentationData(torch.utils.data.Dataset):
    """
    """
    def __init__(self, directory, categories=None, require_all=False, transform=None, seg_return=False):
        directory = os.path.expanduser(directory)
        self.directory = directory
        self.seg_return = seg_return
        self.meta_categories = ['image', 'split', 'ih', 'iw', 'sh', 'sw']

        self.video_input = False if settings.DATASET in settings.IMAGE_DATASETS else True
        self.transform = transform
        with open(os.path.join(directory, settings.INDEX_FILE)) as f:
            self.image = [self.decode_index_dict(r) for r in csv.DictReader(f)]

        self.video = self.generate_video_data()
        with open(os.path.join(directory, 'category.csv')) as f:
            self.category = OrderedDict()
            for row in csv.DictReader(f):
                if categories and row['name'] in categories:
                    self.category[row['name']] = row
        categories = self.category.keys()
        self.categories = list(self.category.keys())
        with open(os.path.join(directory, 'label.csv')) as f:
            label_data = [self.decode_label_dict(r) for r in csv.DictReader(f)]
        self.label = self.build_dense_label_array(label_data)
        # Filter out images with insufficient data
        filter_fn = partial(
                self.index_has_all_data if require_all else self.index_has_any_data,
                categories=categories)
        self.image = [row for row in self.image if filter_fn(row)]
        # Build dense remapping arrays for labels, so that you can
        # get dense ranges of labels for each category.
        self.category_map = {}
        self.category_unmap = {}
        self.category_label = {}
        for cat in self.category:
            with open(os.path.join(directory, 'c_%s.csv' % cat)) as f:
                c_data = [self.decode_label_dict(r) for r in csv.DictReader(f)]
            self.category_unmap[cat], self.category_map[cat] = (
                    self.build_numpy_category_map(c_data))
            self.category_label[cat] = self.build_dense_label_array(
                    c_data, key='code')

        self.labelcat = self.onehot(self.primary_categories_per_index())
        self.root_dir = os.path.join(self.directory, 'images')

    def __getitem__(self, index):
        """
        input: index
        output: result: C x (T) x W x H (T if self.video_input is True)
        """

        '''
        Steps
        1. Get batch = [{'sh': 112, 'sw': 112, 'i': 4, 
        'fn': 'dataset/broden1_224/images/ade20k/ADE_train_00003891.jpg', 
        'image': Array([[[128,  91,  82],...]]])]
        
        2. Normalize image with mean and std
        '''

        if not self.video_input:
            data = self.image[index]
            if not self.seg_return:
                result = Image.open(os.path.join(self.root_dir, data['image']))
                result = self.transform(result)
            else:
                result, shape = self.resolve_segmentation(data, categories=self.categories)
                # data = (j,
                #           self.segmentation.__class__,
                #           self.segmentation.metadata(j),
                #           self.segmentation.filename(j),
                #           self.categories,
                #           self.segmentation_shape)
                # j, typ, m, fn, categories, segmentation_shape = data
                # if segmentation_shape is not None:
                #     for k, v in segs.items():
                #         segs[k] = scale_segmentation(v, segmentation_shape)
                #     shape = segmentation_shape
                # Some additional metadata to provide
                result['sh'], result['sw'] = shape
                result['i'] = index
                result['fn'] = data['image']
                if self.categories is None or 'image' in self.categories:
                    img = Image.open(os.path.join(self.root_dir, data['image']))
                    img = self.transform(img)
                    result['image'] = img
                # batch = {'color': ndarray(1x112x112) , 'scene': list(38), ...
                # 'sh': 112, 'sw': 112, 'i': index, 'fn': 'path_to_img'}
        else:
            video_data = self.video[index]
            result = []
            for frame_data in video_data['frames']:
                frame = Image.open(os.path.join(self.root_dir, frame_data['image']))
                frame = self.transform(frame)
                result.append(frame)

            # stack video, shape of CxTxHxW
            result = torch.stack(result, 1)

        return result


    def onehot(self, arr, minlength=None):
        '''
        Expands an array of integers in one-hot encoding by adding a new last
        dimension, leaving zeros everywhere except for the nth dimension, where
        the original array contained the integer n.  The minlength parameter is
        used to indcate the minimum size of the new dimension.
        '''
        length = np.amax(arr) + 1
        if minlength is not None:
            length = max(minlength, length)
        result = np.zeros(arr.shape + (length,))
        result[list(np.indices(arr.shape)) + [arr]] = 1
        return result

    def generate_video_data(self):
        '''Transform self.image into list-by-video format instead of by list-by-image format'''

        video_dict = {}

        for img in self.image:
            video = img['image'].split('/')[1]
            frame_idx = int(img['image'].split('.')[0][-5:])
            if video not in video_dict.keys():
                video_dict[video] = {'frames': [], 'start_frame': 100000, 'end_frame':0}

            video_dict[video]['frames'].append(img)
            if frame_idx < video_dict[video]['start_frame']:
                video_dict[video]['start_frame'] = frame_idx
            if frame_idx > video_dict[video]['end_frame']:
                video_dict[video]['end_frame'] = frame_idx

        video_data_list = []
        for vid in video_dict:
            # accomodate videos with sparsely labelled frames
            if 'a2d' in video_dict[vid]['frames'][0]['image']:
                labelled_idx = video_dict[vid]['start_frame']+32
            else:
                labelled_idx = None
            data = {'video_id': vid, 'frames': video_dict[vid]['frames'], 'start_frame':video_dict[vid]['start_frame'],
                    'end_frame': video_dict[vid]['end_frame'], 'label_idx': labelled_idx, 'root': video_dict[vid]['frames'][0]['image'][:-10]}
            video_data_list.append(data)

        return video_data_list

    def all_names(self, category, j):
        '''All English synonyms for the given label'''
        if category is not None:
            j = self.category_unmap[category][j]
        return [self.label[j]['name']] + self.label[j]['syns']

    def size(self, split=None):
        '''The number of images in this data set.'''
        if split is None:
            return len(self.image)
        return len([im for im in self.image if im['split'] == split])

    def filename(self, i):
        '''The filename of the ith jpeg (original image).'''
        return os.path.join(self.directory, 'images', self.image[i]['image'])

    def split(self, i):
        '''Which split contains item i.'''
        return self.image[i]['split']

    def metadata(self, i):
        '''Extract metadata for image i, For efficient data loading.'''
        return self.directory, self.image[i]



    # @classmethod
    def resolve_segmentation(self, row, categories=None):
        '''
        Resolves a full segmentation, potentially in a differenct process,
        for efficient multiprocess data loading.
        '''
        result = {}
        for cat, d in row.items():
            if cat in self.meta_categories:
                continue
            if not wants(cat, categories):
                continue
            if all(isinstance(data, int) for data in d):
                result[cat] = d
                continue
            out = numpy.empty((len(d), row['sh'], row['sw']), dtype=numpy.int16)
            for i, channel in enumerate(d):
                if isinstance(channel, int):
                    out[i] = channel
                else:
                    rgb = imread(os.path.join(self.directory, 'images', channel))
                    out[i] = rgb[:,:,0] + rgb[:,:,1] * 256
            result[cat] = out
        return result, (row['sh'], row['sw'])

    def label_size(self, category=None):
        '''
        Returns the number of distinct labels (plus zero), i.e., one
        more than the maximum label number.  If a category is specified,
        returns the number of distinct labels within that category.
        '''
        if category is None:
            return len(self.label)
        else:
            return len(self.category_unmap[category])

    def name(self, category, j):
        '''
        Returns an English name for the jth label.  If a category is
        specified, returns the name for the category-specific nubmer j.
        If category=None, then treats j as a fully unified index number.
        '''
        if category is not None:
            j = self.category_unmap[category][j]
        return self.label[j]['name']

    def frequency(self, category, j):
        '''
        Returns the number of images for which the label appears.
        '''
        if category is not None:
            return self.category_label[category][j]['frequency']
        return self.label[j]['frequency']

    def coverage(self, category, j):
        '''
        Returns the pixel coverage of the label in units of whole-images.
        '''
        if category is not None:
            return self.category_label[category][j]['coverage']
        return self.label[j]['coverage']

    def category_names(self):
        '''
        Returns the set of category names.
        '''
        return list(self.category.keys())

    def category_frequency(self, category):
        '''
        Returns the number of images touched by a category.
        '''
        return float(self.category[category]['frequency'])

    def primary_categories_per_index(self, categories=None):
        '''
        Returns an array of primary category numbers for each label, where
        catagories are indexed according to the list of categories passed,
        or self.category_names() if none.
        '''
        if categories is None:
            categories = self.category_names()
        # Make lists which are nonzero for labels in a category
        catmap = {}
        for cat in categories:
            imap = self.category_index_map(cat)
            if len(imap) < self.label_size(None):
                imap = numpy.concatenate((imap, numpy.zeros(
                    self.label_size(None) - len(imap), dtype=imap.dtype)))
            catmap[cat] = imap
        # For each label, find the category with maximum coverage.
        result = []
        for i in range(self.label_size(None)):
            maxcov, maxcat = max(
                    (self.coverage(cat, catmap[cat][i])
                        if catmap[cat][i] else 0, ic)
                    for ic, cat in enumerate(categories))
            result.append(maxcat)
        # Return the max-coverage cateogry for each label.
        return numpy.array(result)

    def decode_index_dict(self, row):
        result = {}
        for key, val in row.items():
            if key in ['image', 'split']:
                result[key] = val
            elif key in ['sw', 'sh', 'iw', 'ih']:
                result[key] = int(val)
            else:
                item = [s for s in val.split(';') if s]
                for i, v in enumerate(item):
                    if re.match('^\d+$', v):
                        item[i] = int(v)
                result[key] = item
        return result

    def build_dense_label_array(self, label_data, key='number', allow_none=False):
        '''
        Input: set of rows with 'number' fields (or another field name key).
        Output: array such that a[number] = the row with the given number.
        '''
        result = [None] * (max([d[key] for d in label_data]) + 1)
        for d in label_data:
            result[d[key]] = d
        # Fill in none
        if not allow_none:
            example = label_data[0]

            def make_empty(k):
                return dict((c, k if c is key else type(v)())
                            for c, v in example.items())

            for i, d in enumerate(result):
                if d is None:
                    result[i] = dict(make_empty(i))
        return result

    def index_has_any_data(self, row, categories):
        for c in categories:
            for data in row[c]:
                if data: return True
        return False

    def index_has_all_data(self, row, categories):
        for c in categories:
            cat_has = False
            for data in row[c]:
                if data:
                    cat_has = True
                    break
            if not cat_has:
                return False
        return True

    def decode_label_dict(self, row):
        result = {}
        for key, val in row.items():
            if key == 'category':
                result[key] = dict((c, int(n))
                                   for c, n in [re.match('^([^(]*)\(([^)]*)\)$', f).groups()
                                                for f in val.split(';')])
            elif key == 'name':
                result[key] = val
            elif key == 'syns':
                result[key] = val.split(';')
            elif re.match('^\d+$', val):
                result[key] = int(val)
            elif re.match('^\d+\.\d*$', val):
                result[key] = float(val)
            else:
                result[key] = val
        return result

    def build_dense_label_array(self, label_data, key='number', allow_none=False):
        '''
        Input: set of rows with 'number' fields (or another field name key).
        Output: array such that a[number] = the row with the given number.
        '''
        result = [None] * (max([d[key] for d in label_data]) + 1)
        for d in label_data:
            result[d[key]] = d
        # Fill in none
        if not allow_none:
            example = label_data[0]

            def make_empty(k):
                return dict((c, k if c is key else type(v)())
                            for c, v in example.items())

            for i, d in enumerate(result):
                if d is None:
                    result[i] = dict(make_empty(i))
        return result

    def build_numpy_category_map(self, map_data, key1='code', key2='number'):
        '''
        Input: set of rows with 'number' fields (or another field name key).
        Output: array such that a[number] = the row with the given number.
        '''
        results = list(numpy.zeros((max([d[key] for d in map_data]) + 1),
                                   dtype=numpy.int16) for key in (key1, key2))
        for d in map_data:
            results[0][d[key1]] = d[key2]
            results[1][d[key2]] = d[key1]
        return results

    def category_index_map(self, category):
        return numpy.array(self.category_map[category])

    def __len__(self):
        return len(self.image)

def wants(what, option):
    if option is None:
        return True
    return what in option

def seg_map_collate(batch):
    # since we only want to deal with numpy arrays with variable sizes, we just return the list of dictionaries
    return batch



