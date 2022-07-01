from tqdm import tqdm
import os
import json

def convert(args):
    if args.dataset == 'dtdb':
        correspondance_path = args.data_root + '/app_dyn_correspondance.json'
        with open(correspondance_path) as file:
            data = json.load(file)
    if not os.path.exists(os.path.join(args.data_root, args.save_path)):
        os.mkdir(os.path.join(args.data_root, args.save_path))

    data_out_path = os.path.join(args.data_root, args.save_path)
    for vid_id in tqdm(data):
        if len(data[vid_id]) < 8:
            continue

        data_in_path = os.path.join(args.data_root) + '/BY_DYNAMIC_FINAL/' + data[vid_id]['dyn_subset'].upper() + '/' \
                       +data[vid_id]['dynamic'].split('_g')[0] + '/' + data[vid_id]['dynamic']
        save_dir = data_out_path + '/' + data[vid_id]['dynamic'].split('.')[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        command = 'ffmpeg -hide_banner -loglevel error -i "{}" -r 30 -q:v 1 -vf scale=224:224 "{}/%06d.png"'.format(data_in_path, save_dir)
        os.system(command)




if __name__ == '__main__':
    # general imports
    import argparse


    parser = argparse.ArgumentParser(
        description='Generating optical data_processing.')
    parser.add_argument(
            '--save_size',
            type=int, default=224,
            help='pixel size for output videos')
    parser.add_argument(
        '--dataset',
        type=str, default='dtdb',
        help='dataset to generate data_processing for')
    parser.add_argument(
        '--data_root',
        type=str, default='/home/m2kowal/data/DTDB',
        help='dataset to generate data_processing for')
    parser.add_argument(
        '--save_path',
        type=str, default='frames',
        help='save directory name, will save under data_root')
    args = parser.parse_args()
    convert(args)
