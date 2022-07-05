import numpy as np
from torchvision.utils import flow_to_image
import torch
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats


def debug_flow_bin(verbose=True):
    flow_paths = [
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c1/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c2/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c3/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c4/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c5/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c7/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c10/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c11/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c12/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c13/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c14/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c15/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c16/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c17/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Wavy_motion_g1_c1/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Wavy_motion_g1_c2/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Wavy_motion_g1_c3/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Wavy_motion_g1_c4/000050_flow.npy',
                  ]
    for flow_path in flow_paths:
        flow = np.load(flow_path)



        # 1) convert to magnitude and angle
        x = flow[0]
        y = flow[1]

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = np.where(theta < 0, 2 * np.pi + theta, theta)

        # 2) bin based on angle (8 bins) and magnitude (3 bins)
        magnitude_bins = np.array([0.0, 0.5, 1.0, 2])
        angle_bins = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4, 2*np.pi])
        binned_magnitude = np.digitize(r, magnitude_bins)-1
        binned_angle = np.digitize(theta, angle_bins)

        # Compute binned flow by base 10 for each angle, +1 for each magnitude level. Output values: [0,11,12,13,21,22,...,81,82,83]
        binned_flow = np.where(binned_magnitude == 0, 0, binned_angle * 10 + binned_magnitude)
        print(binned_flow.shape)

        if verbose:
            torch_flow = torch.tensor(flow)
            flow_imgs = flow_to_image(torch_flow)

            img_path = flow_path.replace('_flow.npy', '.png')
            img = Image.open(img_path)

            plt.imshow(img)
            plt.show()
            plt.imshow(flow_imgs.permute(1,2,0))
            plt.show()


def bin_flow(path):
    flow = np.load(path)

    # 1) convert to magnitude and angle
    x = flow[0]
    y = flow[1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, 2 * np.pi + theta, theta)

    # 2) bin based on angle (8 bins) and magnitude (3 bins)
    magnitude_bins = np.array([0.0, 0.5, 1.0, 2])
    angle_bins = np.array(
        [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 6 * np.pi / 4, 7 * np.pi / 4, 2 * np.pi])
    binned_magnitude = np.digitize(r, magnitude_bins) - 1
    binned_angle = np.digitize(theta, angle_bins)

    # Compute binned flow by base 10 for each angle, +1 for each magnitude level. Output values: [0,11,12,13,21,22,...,81,82,83]
    binned_flow = np.where(binned_magnitude == 0, 0, binned_angle * 10 + binned_magnitude)
    return binned_flow


# debug_flow_bin()

