#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from dp.utils.pyt_ops import interpolate, tensor2numpy
from dp.visualizers.base_visualizer import BaseVisualizer
from dp.visualizers.utils import depth_to_color, error_to_color


class dorn_visualizer(BaseVisualizer):
    def __init__(self, config, writer=None):
        super(dorn_visualizer, self).__init__(config, writer)

    def visualize(self, batch, out, epoch=0):
        """
        :param batch_in: minibatch
        :param pred_out: model output for visualization, dic, {"target": [NxHxW]}
        :param tensorboard: if tensorboard = True, the visualized image should be in [0, 1].
        :return: vis_ims: image for visualization.
        """
        fn = batch["fn"]
        if batch["target"].shape != out["target"][-1].shape:
            h, w = batch["target"].shape[-2:]
            # batch = interpolate(batch, size=(h, w), mode='nearest')
            out = interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        image = batch["image_n"].numpy()

        has_gt = False
        if batch.get("target") is not None:
            depth_gts = tensor2numpy(batch["target"])
            has_gt = True

        for i in range(len(fn)):
            image = image[i].astype(np.float)
            depth = tensor2numpy(out['target'][0][i])
            # print("!! depth shape:", depth.shape)

            if has_gt:
                depth_gt = depth_gts[i]

                err = error_to_color(depth, depth_gt)
                depth_gt = depth_to_color(depth_gt)

            depth = depth_to_color(depth)
            # print("pred:", depth.shape, " target:", depth_gt.shape)
            group = np.concatenate((image, depth), axis=0)

            if has_gt:
                gt_group = np.concatenate((depth_gt, err), axis=0)
                group = np.concatenate((group, gt_group), axis=1)

            if self.writer is not None:
                group = group.transpose((2, 0, 1)) / 255.0
                group = group.astype(np.float32)
                # print("group shape:", group.shape)
                self.writer.add_image(fn[i] + "/image", group, epoch)
