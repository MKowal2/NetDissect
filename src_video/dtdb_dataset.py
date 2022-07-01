import glob
import torch.utils.data as data
import json
import os
import dtdb_utils


def get_dtdb_data(data_root):
    correspondance_path = data_root + '/app_dyn_correspondance.json'
    if not os.path.exists(correspondance_path):
        dtdb_utils.gen_dtdb_json(data_root)
    with open(correspondance_path) as file:
        data = json.load(file)
    return data

class DTDB(data.Dataset):
    def __init__(self, data_root, categories):
        # need to replace this with datalist of all images, with corresponding styles and class labels
        self.data_root = data_root
        self.data = get_dtdb_data(self.data_root)
        self.dynamics = [y.split('/')[-1] for y in glob.glob(self.data_root + '/BY_DYNAMIC_FINAL/TRAIN/*')]
        self.num_dyns = len(self.dyn_app_list)
        self.categories = categories

    def __getitem__(self, i):
        video = self.data[i]
        result = {}
        if wants('dynamics', self.categories):
            print('dynamcs')
            result['dynamics'] = 1
        if wants('data_processing', self.categories):
            print('data_processing')

        # return a dictionary of segmentation maps (or [1] for video label) and shape of video
        # TODO: retrieve shape from video, maybe build into app_dyn_correspondance.json to save time
        shape = (0,0,0)
        return result, shape

    def __len__(self):
        return len(self.data)

def wants(what, option):
    if option is None:
        return True
    return what in option