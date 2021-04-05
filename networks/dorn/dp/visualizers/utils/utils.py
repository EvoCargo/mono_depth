# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def depth_to_color(depth):
    cmap = plt.cm.jet
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def error_to_color(depth, gt):
    mask = gt <= 0.0
    cmap = plt.cm.Greys
    err = np.abs(depth - gt)
    err[mask] = 0.0
    err_min = np.min(err)
    err_max = np.max(err)
    err_rel = (err - err_min) / (err_max - err_min)
    return 255 * cmap(err_rel)[:, :, :3]
