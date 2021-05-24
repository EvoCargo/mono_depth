from __future__ import absolute_import, division, print_function

import datetime
import os
import random

import numpy as np
import pandas as pd
import PIL.Image as pil
import scipy.misc
import torch
from torchvision import transforms

from .kitti_utils import (  # pose_from_oxts_packet,
    generate_depth_map,
    read_calib_file,
    transform_from_rot_trans,
)
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders"""

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

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
        depth_gt = scipy.misc.imresize(depth_gt, self.full_res_shape[::-1], "nearest")

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_pose(self, folder, frame_index, offset):
        oxts_root = os.path.join(self.data_path, folder, 'oxts')
        with open(os.path.join(oxts_root, 'timestamps.txt')) as f:
            timestamps = np.array(
                [
                    datetime.datetime.strptime(
                        ts[:-3], "%Y-%m-%d %H:%M:%S.%f"
                    ).timestamp()
                    for ts in f.read().splitlines()
                ]
            )

        speed0 = np.genfromtxt(
            os.path.join(oxts_root, 'data', '{:010d}.txt'.format(frame_index))
        )[[8, 9, 10]]
        # speed1 = np.genfromtxt(os.path.join(oxts_root, 'data', '{:010d}.txt'.format(frame_index+offset)))[[8, 9, 10]]

        timestamp0 = timestamps[frame_index]
        timestamp1 = timestamps[frame_index + offset]
        # displacement = 0.5 * (speed0 + speed1) * (timestamp1 - timestamp0)
        displacement = speed0 * (timestamp1 - timestamp0)

        imu2velo = read_calib_file(
            os.path.join(self.data_path, os.path.dirname(folder), 'calib_imu_to_velo.txt')
        )
        velo2cam = read_calib_file(
            os.path.join(self.data_path, os.path.dirname(folder), 'calib_velo_to_cam.txt')
        )
        cam2cam = read_calib_file(
            os.path.join(self.data_path, os.path.dirname(folder), 'calib_cam_to_cam.txt')
        )

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

        odo_pose = imu2cam[:3, :3] @ displacement + imu2cam[:3, 3]

        return odo_pose


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing"""

    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        side_map = {"l": 0, "r": 1}
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(side_map[side]),
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

        def mask_first_last(x):
            result = np.ones_like(x)
            result[0] = 0
            result[-1] = 0
            return result

        self.filenames = pd.DataFrame.from_records(
            [i.split() for i in self.filenames],
            columns=['folder', 'file', 'ind', 'x_focal', 'y_focal', 'x_pp', 'y_pp'],
        )

        self.filenames['ind'] = self.filenames['ind'].apply(lambda x: x[:-2])

        if self.is_train:
            self.filenames.to_csv('train_filenames.csv', index=False)
        else:
            self.filenames.to_csv('val_filenames.csv', index=False)

        mask = (
            self.filenames.groupby('folder')['folder']
            .transform(mask_first_last)
            .astype(bool)
        )
        self.mod_filenames = self.filenames.loc[mask]
        self.mod_filenames.reset_index(drop=True, inplace=True)

        # if not self.is_train:
        self.full_res_shape = (1280, 720)

    def check_depth(self):
        return True

    def get_image_path(self, folder, frame_index, side):
        f_str = "{}_{:19d}.jpg".format(folder, frame_index)
        image_path = os.path.join(self.data_path, folder, "front_rgb_left", f_str)

        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color, color.size

    def __getitem__(self, index):
        # print(index)
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder = self.mod_filenames.iloc[index]['folder']
        frame_index = self.mod_filenames.iloc[index]['file']
        ind = int(self.mod_filenames.iloc[index]['ind'])
        side = None
        x_focal = float(self.mod_filenames.iloc[index]['x_focal'])
        y_focal = float(self.mod_filenames.iloc[index]['y_focal'])
        x_pp = float(self.mod_filenames.iloc[index]['x_pp'])
        y_pp = float(self.mod_filenames.iloc[index]['y_pp'])

        for i in self.frame_idxs:

            an_file = self.filenames[
                (self.filenames['folder'] == folder)
                & (self.filenames['ind'] == str(ind + i))
            ]['file'].iloc[0]
            inputs[("color", i, -1)], (real_width, real_height) = self.get_color(
                folder, int(an_file), side, do_flip
            )

        K = np.array(
            [
                [x_focal / real_width, 0, x_pp / real_width, 0],
                [0, y_focal / real_height, y_pp / real_height, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        K[0, :] *= self.width
        K[1, :] *= self.height
        inv_K = np.linalg.pinv(K)

        inputs[("K")] = torch.from_numpy(K)
        inputs[("inv_K")] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            color_aug = lambda x: x  # noqa

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            # del inputs[("color_aug", i, -1)]

        # if not self.is_train and self.gt_depth_path is not None:
        #     gt_depth = self.gt_depths[index]
        #     inputs['gt_depth'] = gt_depth

        # if self.load_depth:
        # depth_gt =
        # print('inside ', depth_gt.shape)
        inputs["gt_depth"] = self.get_depth(folder, frame_index, side, do_flip)
        # inputs["gt_depth"] = np.expand_dims(depth_gt, 0)
        inputs["gt_depth"] = torch.from_numpy(inputs["gt_depth"].astype(np.float32))

        # if "s" in self.frame_idxs:
        #     stereo_T = np.eye(4, dtype=np.float32)
        #     baseline_sign = -1 if do_flip else 1
        #     side_sign = -1 if side == "l" else 1
        #     stereo_T[0, 3] = side_sign * baseline_sign * 0.1
        #     inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def __len__(self):
        return len(self.mod_filenames)

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{}_{}.png".format(folder, frame_index)

        depth_path = os.path.join(
            self.data_path,
            folder,
            'front_depth_left',
            f_str,
        )

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
