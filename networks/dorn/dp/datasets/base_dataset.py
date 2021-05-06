# -*- coding: utf-8 -*-

# import os

import numpy as np
import torch
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, config, is_train=True, image_loader=None, depth_loader=None):
        super(BaseDataset, self).__init__()
        self.config = config
        self.root = self.config['path']
        self.split = self.config["split"]
        self.split = self.split[0] if is_train else self.split[1]
        self.image_loader, self.depth_loader = image_loader, depth_loader
        if is_train:
            self.preprocess = self._tr_preprocess
        else:
            self.preprocess = self._te_preprocess

        # print(type(self.depth_loader), self.depth_loader)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_path, depth_path = self._parse_path(index)
        item_name = image_path.split("/")[-1].split(".")[0]

        image, depth = self._fetch_data(image_path, depth_path)
        image, depth, extra_dict = self.preprocess(image, depth)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        output_dict = dict(
            image=image, fn=str(item_name), image_path=image_path, n=self.get_length()
        )

        if depth is not None:
            output_dict['target'] = torch.from_numpy(np.ascontiguousarray(depth)).float()
            output_dict['target_path'] = depth_path

        if extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, image_path, depth_path=None):
        image = self.image_loader(image_path)
        depth = None
        if depth_path is not None:
            depth = self.depth_loader(depth_path)

        # print('Train:', image.size, depth.shape)

        return image, depth

    def _parse_path(self, index):
        raise NotImplementedError

    def get_length(self):
        return self.__len__()

    def _tr_preprocess(self, image, depth):
        raise NotImplementedError

    def _te_preprocess(self, image, depth):
        raise NotImplementedError
