import csv
import os
from tqdm import tqdm

'''
This is a script to select subsets of a larger index.csv file. You can choose which dataset(s)
to be in the smaller subset as well as the number of videos.
'''


dataset_path = 'dataset/A2D_video_broden4_224'
index_file = 'index.csv'
new_file_name = 'index_sm.csv'

def choose_dataset_subset(datasets, num_videos):
    '''
    datasets: list of strings, with each dataset as an element e.g., ['a2d','dtdb']
    num_videos: number of videos per dataset to use e.g., 100
    '''
    with open(os.path.join(dataset_path, index_file), 'r') as f:
        reader = csv.reader(f)
        new_csv_dict = {}
        for dataset in datasets:
            new_csv_dict[dataset] = []

        video_dict = {}
        for dataset in datasets:
            video_dict[dataset] = []

        # # check for duplicates in index.csv
        # check_list = []
        # for x in reader:
        #     if x in check_list:
        #         print('Duplicate in new list:  ',x)
        #     else:
        #         check_list.append(x)

        for idx, row in enumerate(tqdm(reader)):
            if idx == 0:
                new_csv_dict[datasets[0]].append(row)
            if row[0].split('/')[0] in datasets:
                video = row[0].split('/')[1]
                dataset = row[0].split('/')[0]
                # if the video list has less than num_videos videos, add the new video
                if len(video_dict[dataset]) < num_videos:
                    if video not in video_dict[dataset]:
                        video_dict[dataset].append(video)
                    new_csv_dict[dataset].append(row)
                else:
                    if video in video_dict[dataset]:
                        new_csv_dict[dataset].append(row)
    new_csv_list = []
    for dataset in datasets:
        new_csv_list += new_csv_dict[dataset]


    check_list = []
    for x in new_csv_list:
        if x in check_list:
            print('Duplicate in new list:  ',x)
        else:
            check_list.append(x)

    with open(os.path.join(dataset_path, new_file_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_csv_list)


choose_dataset_subset(datasets=['a2d'],num_videos=100)