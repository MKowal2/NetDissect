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

    new_data = data.copy()
    data_out_path = os.path.join(args.data_root, args.save_path)
    for idx, vid_id in enumerate(tqdm(data)):
        if len(data[vid_id]) < 8:
            continue

        new_data[vid_id]['save_idx'] = idx

        fps = data[vid_id]['fps']
        data_in_path = os.path.join(args.data_root) + '/BY_DYNAMIC_FINAL/' + data[vid_id]['dyn_subset'].upper() + '/' \
                       +data[vid_id]['dynamic'].split('_g')[0] + '/' + data[vid_id]['dynamic']
        # rename files
        save_dir = data_out_path + '/' + str(idx)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        command = 'ffmpeg -hide_banner -loglevel error -i "{}" -r {} -q:v 1 "{}/%06d.png"'.format(data_in_path, fps, save_dir)
        os.system(command)


        # SCALED VERSION
        # command = 'ffmpeg -hide_banner -loglevel error -i "{}" -r 30 -q:v 1 -vf scale=320:-1 "{}/%06d.png"'.format(data_in_path, save_dir)

    # save new json file
    save_json_path = args.data_root + '/app_dyn_correspondance_frames.json'
    with open(save_json_path, 'w') as f:
        json.dump(new_data, f)


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
