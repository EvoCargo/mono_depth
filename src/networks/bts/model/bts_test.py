from __future__ import absolute_import, division, print_function

import argparse
import errno
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from bts import BtsModel
from bts_dataloader import BtsDataLoader
from torch.autograd import Variable
from tqdm import tqdm


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(
    description='BTS PyTorch implementation.', fromfile_prefix_chars='@'
)
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument(
    '--model_name',
    type=str,
    help='model name',
    default='bts_eigen_v2_pytorch_densenet121',
)
parser.add_argument(
    '--encoder',
    type=str,
    help='type of encoder, vgg or desenet121_bts or densenet161_bts',
    default='densenet161_bts',
)
parser.add_argument('--data_path', type=str, help='path to the data')
parser.add_argument('--image_path', type=str, help='path to image')
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file')
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument(
    '--max_depth', type=float, help='maximum depth in estimation', default=80
)
parser.add_argument(
    '--checkpoint_path',
    type=str,
    help='path to a specific checkpoint to load',
    default='',
)
parser.add_argument(
    '--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu'
)
parser.add_argument(
    '--do_kb_crop',
    help='if set, crop input images as kitti benchmark images',
    action='store_true',
)
parser.add_argument(
    '--save_lpg', help='if set, save outputs from lpg layers', action='store_true'
)
parser.add_argument(
    '--bts_size', type=int, help='initial num_filters in bts', default=512
)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args, 'test')

    model = BtsModel(params=args)
    model = nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    # with open(args.filenames_file) as f:
    #     lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []

    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())
            pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')

    save_name = 'result_' + args.model_name

    print('Saving result pngs..')
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for s in tqdm(range(num_test_samples)):
        filename_pred_png = (
            save_name + '/raw/' + args.image_path.split('/')[-1].replace('.jpg', '.png')
        )
        filename_cmap_png = (
            save_name + '/cmap/' + args.image_path.split('/')[-1].replace('.jpg', '.png')
        )
        filename_image_png = save_name + '/rgb/' + args.image_path.split('/')[-1]

        rgb_path = params['image_path']
        image = cv2.imread(rgb_path)

        pred_depth = pred_depths[s]
        pred_8x8 = pred_8x8s[s]
        pred_4x4 = pred_4x4s[s]
        pred_2x2 = pred_2x2s[s]
        pred_1x1 = pred_1x1s[s]

        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth * 1000.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(
            filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )

        if args.save_lpg:
            cv2.imwrite(filename_image_png, image[10 : -1 - 9, 10 : -1 - 9, :])
            plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1), cmap='Greys')

    return


if __name__ == '__main__':
    test(args)
