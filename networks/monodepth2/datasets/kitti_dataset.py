from __future__ import absolute_import, division, print_function

import os
import random

import numpy as np
import pandas as pd
import PIL.Image as pil
import skimage.transform
import torch
from kitti_utils import generate_depth_map
from torchvision import transforms

from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders"""

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)),
        )

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth"""

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str
        )
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)),
        )

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt,
            self.full_res_shape[::-1],
            order=0,
            preserve_range=True,
            mode='constant',
        )

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing"""

    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str,
        )
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps"""

    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str
        )
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str,
        )

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class EvoDataset(KITTIDataset):
    """Evo dataset which uses the updated ground truth depth maps"""

    def __init__(self, *args, **kwargs):
        super(EvoDataset, self).__init__(*args, **kwargs)

        # self.K = np.array(
        #     [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        #     dtype=np.float32,
        # )

        def mask_first_last(x):
            result = np.ones_like(x)
            result[0] = 0
            result[-1] = 0
            return result

        self.filenames = pd.DataFrame.from_records(
            [i.split() for i in self.filenames], columns=['folder', 'file', 'ind']
        )

        self.filenames.to_csv('filenames.csv', index=False)

        mask = (
            self.filenames.groupby('folder')['folder']
            .transform(mask_first_last)
            .astype(bool)
        )
        self.mod_filenames = self.filenames.loc[mask]
        self.mod_filenames.reset_index(drop=True, inplace=True)

    def check_depth(self):
        return True

    def get_image_path(self, folder, frame_index, side):
        f_str = "{}_{:19d}.jpg".format(folder, frame_index)
        image_path = os.path.join(self.data_path, folder, "front_rgb_left", f_str)

        return image_path

    def __getitem__(self, index):
        # print(index)
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder = self.mod_filenames.iloc[index]['folder']
        frame_index = self.mod_filenames.iloc[index]['file']
        ind = int(self.mod_filenames.iloc[index]['ind'])
        side = None

        for i in self.frame_idxs:
            if (
                len(
                    self.filenames[
                        (self.filenames['folder'] == folder)
                        & (self.filenames['ind'] == str(ind + i))
                    ]
                )
                == 0
            ):
                print(folder, ind, i, frame_index)

            an_file = self.filenames[
                (self.filenames['folder'] == folder)
                & (self.filenames['ind'] == str(ind + i))
            ]['file'].iloc[0]
            inputs[("color", i, -1)] = self.get_color(folder, int(an_file), side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            color_aug = lambda x: x  # noqa

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def __len__(self):
        return len(self.mod_filenames)

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{}_{:19d}.png".format(folder, int(frame_index))

        depth_path = os.path.join(
            self.data_path,
            folder,
            'front_depth_left',
            f_str,
        )

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
