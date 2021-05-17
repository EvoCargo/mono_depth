# -*- coding: utf-8 -*-
import os

import numpy as np

# import torch
from dp.datasets.utils import EvoDepthLoader
from tqdm import tqdm


root_dir = "/media/data/datasets/bag_depth"
filelist = "/home/penitto/mono_depth/networks/dorn/dp/datasets/lists/evo_trainval.list"
with open(filelist, "r") as f:
    filenames = f.readlines()

alpha = 10000.0
beta = 0.0

for i in tqdm(range(len(filenames))):
    # _, path = filenames[i].split()
    # path = os.path.join(root_dir, path)

    filename = filenames[i].split()

    depth_path = os.path.join(
        root_dir,
        filename[0],
        'front_depth_left',
        '{}_{}.png'.format(filename[0], filename[1]),
    )

    depth = EvoDepthLoader(depth_path)
    mask = depth > 0.0
    valid_depth = depth[mask]
    tmp_max = np.max(valid_depth)
    tmp_min = np.min(valid_depth)
    alpha = min(tmp_min, alpha)
    beta = max(tmp_max, beta)


print("alpha={}, beta={}, gamma={}".format(alpha, beta, 1.0 - alpha))

"""
alpha=1.9765625 beta=90.44140625 gamma=-0.9765625
"""
