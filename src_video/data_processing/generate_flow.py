import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torch.utils.data.dataloader import DataLoader
from dtdb_flow import DTDB
from a2d_flow import A2D

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()

def generateflow(args):

    if args.dataset == 'dtdb':
        categories = ['dynamics']
        data = DTDB(data_root='/home/m2kowal/data/DTDB')
    elif args.dataset == 'a2d':
        data = A2D(data_root='/home/m2kowal/data/a2d_dataset')
    model = raft_large(pretrained=True, progress=False).cuda()
    model = model.eval()
    data_loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers)
    for idx, batch in enumerate(tqdm(data_loader)):
        batch1, batch2, save_paths = batch
        flows = model(batch1.cuda(), batch2.cuda())
        final_flows = F.resize(flows[-1], size=[112, 112])

        ## VISUALIZE
        # flow_imgs = flow_to_image(flows[-1])
        # # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
        # img1_batch = [(batch1[0] + 1) / 2, (batch2[0] + 1) / 2]
        # grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
        # plot(grid)
        ## VISUALIZE

        final_flows = final_flows.detach().cpu().numpy()
        for i, path in enumerate(save_paths):
            final_flow = final_flows[i]
            np.save(path, final_flow)
    print('done!')



if __name__ == '__main__':
    # general imports
    import argparse

    # dataset imports
    import dtdb_flow


    parser = argparse.ArgumentParser(
        description='Generating optical data_processing.')
    parser.add_argument(
            '--save_size',
            type=int, default=112,
            help='pixel size for output videos')
    parser.add_argument(
            '--batch_size',
            type=int, default=6,
            help='batch size for data loader')
    parser.add_argument(
            '--num_workers',
            type=int, default=12,
            help='cpu threads')
    parser.add_argument(
        '--dataset',
        type=str, default='dtdb',
        help='dataset to generate data_processing for')
    args = parser.parse_args()
    generateflow(args)

