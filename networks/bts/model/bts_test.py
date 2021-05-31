from __future__ import absolute_import, division, print_function

# import argparse
import errno
import os
import sys
import time

import cv2

# import matplotlib.pyplot as plt
import numpy as np
import torch

# import torch.nn as nn
from bts import BtsModel
from bts_dataloader import BtsDataLoader
from bts_options import BTSOptions
from torch.autograd import Variable
from tqdm import tqdm


options = BTSOptions()
opts = options.parse()

model_dir = os.path.dirname(opts.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(opts.model_name)).items():
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
    dataloader = BtsDataLoader(opts, 'test')

    model = BtsModel(params=opts)
    # model = nn.DataParallel(model)

    checkpoint = torch.load(opts.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    # print('Parameters ', sum(p.numel() for p in model.parameters()))
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(opts.filenames_file)

    with open(params.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, opts.checkpoint_path))

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

    save_name = 'result_' + opts.model_name

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
        scene_name = lines[s].split()[0]
        filename_pred_png = (
            save_name + '/raw/' + scene_name + '_' + lines[s].split()[1] + '_bts.png'
        )

        # rgb_path = params['image_path']
        image = cv2.imread(
            os.path.join(
                params.data_path,
                scene_name,
                'front_rgb_left',
                scene_name + '_' + lines[s].split()[1] + '.jpg',
            )
        )

        pred_depth = pred_depths[s]
        # pred_resized = torch.nn.functional.interpolate(
        #     torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0), (image.shape[:2]), mode="bilinear", align_corners=False
        # ).squeeze().cpu().numpy()
        # pred_8x8 = pred_8x8s[s]
        # pred_4x4 = pred_4x4s[s]
        # pred_2x2 = pred_2x2s[s]
        # pred_1x1 = pred_1x1s[s]

        # vmax = np.percentile(pred_resized, 95)
        # plt.imsave(filename_pred_png, pred_resized, cmap='magma', vmax=vmax)

        # pred_depth_scaled = pred_depth.astype(np.uint16)
        cv2.imwrite(
            filename_pred_png,
            pred_depth.astype(np.uint16),
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        )

    return


if __name__ == '__main__':

    options = BTSOptions()
    opts = options.parse()
    test(opts)
