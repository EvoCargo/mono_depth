#!/usr/bin/python3
# -*- coding: utf-8 -*-
from dp.utils.pyt_ops import interpolate


class MaskedL1Loss(object):
    def __init__(self):
        super().__init__()

    def __call__(self, pred, gt):
        assert (
            pred.dim() == gt.dim()
        ), "inconsistent dimensions, pred shape is {}, but gt shape is {}.".format(
            pred.shape, gt.shape
        )

        if pred.shape != gt.shape:
            pred = interpolate(
                pred, size=gt.shape[-2:], mode="bilinear", align_corners=True
            )

        valid_mask = (gt > 0).detach()
        diff = gt - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
