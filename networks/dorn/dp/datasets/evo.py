#!/usr/bin/python3
# -*- coding: utf-8 -*-
# import math
import os
import random

import cv2
import numpy as np
from dp.datasets.base_dataset import BaseDataset
from dp.datasets.utils import EvoDepthLoader, PILLoader, normalize


# from PIL import Image


class evo(BaseDataset):
    def __init__(
        self,
        config,
        is_train=True,
        image_loader=PILLoader,
        depth_loader=EvoDepthLoader,
    ):
        super().__init__(config, is_train, image_loader, depth_loader)
        file_list = (
            "/home/penitto/mono_depth/networks/dorn/dp/datasets/lists/evo_{}.list".format(
                self.split
            )
        )
        with open(file_list, "r") as f:
            self.filenames = f.readlines()

    def _parse_path(self, index):

        filename = self.filenames[index].split()

        image_path = os.path.join(
            self.root,
            filename[0],
            'front_rgb_left',
            '{}_{}.jpg'.format(filename[0], filename[1]),
        )

        depth_path = os.path.join(
            self.root,
            filename[0],
            'front_depth_left',
            '{}_{}.png'.format(filename[0], filename[1]),
        )

        return image_path, depth_path

    def _tr_preprocess(self, image, depth):
        crop_h, crop_w = self.config["tr_crop_size"]
        # resize
        W, H = image.size
        dH, dW = depth.shape
        # scale = max(crop_h / H, 1.0)      # скорее всего это 1
        # print(scale)

        if dH != H:
            depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)

        # random crop size
        x = random.randint(0, image.size[0] - crop_w)
        y = random.randint(0, image.size[1] - crop_h)
        # dx, dy = math.floor(x / scale), math.floor(y / scale)

        # print('Before preprocess: ', image.size, depth.shape)
        # print('x {} y {} dx {} dy {}'.format(x,y,dx,dy))

        image = image.crop((x, y, x + crop_w, y + crop_h))
        depth = depth[y : y + crop_h, x : x + crop_w]

        # normalize
        image = np.asarray(image).astype(np.float32) / 255.0
        image = normalize(image, type=self.config['norm_type'])
        image = image.transpose(2, 0, 1)

        # print('After preprocess: ', image.shape, depth.shape)

        return image, depth, None

    def _te_preprocess(self, image, depth):
        crop_h, crop_w = self.config["te_crop_size"]
        # resize
        W, H = image.size
        dH, dW = depth.shape
        # scale = max(crop_h / H, 1.0)

        if dH != H:
            depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)

        image_n = image.copy()
        # crop_dh, crop_dw = int(crop_h / scale), int(crop_w / scale)
        # print("corp dh = {}, crop dw = {}".format(crop_dh, crop_dw))
        # depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)

        # center crop
        # x = (W - crop_w) // 2
        # y = (H - crop_h) // 2

        x = random.randint(0, image.size[0] - crop_w)
        y = random.randint(0, image.size[1] - crop_h)

        image = image.crop((x, y, x + crop_w, y + crop_h))
        depth = depth[y : y + crop_h, x : x + crop_w]
        image_n = image_n.crop((x, y, x + crop_w, y + crop_h))

        # normalize
        image_n = np.array(image_n).astype(np.float32)
        image = np.asarray(image).astype(np.float32) / 255.0
        image = normalize(image, type=self.config['norm_type'])
        image = image.transpose(2, 0, 1)

        output_dict = {"image_n": image_n}

        # print('Test:', image.shape)

        return image, depth, output_dict
