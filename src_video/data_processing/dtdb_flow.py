import glob
import torch.utils.data as data
import json


def get_dtdb_data(data_root):
    correspondance_path = data_root + '/app_dyn_correspondance.json'
    with open(correspondance_path) as file:
        data = json.load(file)
    return data

class DTDB(data.Dataset):
    def __init__(self, data_root, categories):
        # need to replace this with datalist of all images, with corresponding styles and class labels
        self.data_root = data_root
        self.data = get_dtdb_data(self.data_root)
        self.dynamics = [y.split('/')[-1] for y in glob.glob(self.data_root + '/BY_DYNAMIC_FINAL/TRAIN/*')]
        self.categories = categories

    def __getitem__(self, i):
        video = self.data[i]
        return video, save_path

    def __len__(self):
        return len(self.data)
