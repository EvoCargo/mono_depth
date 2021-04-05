#!/usr/bin/python3
# -*- coding: utf-8 -*-


class BaseVisualizer(object):
    def __init__(self, config, writer=None):
        self.config = config["vis_config"]
        self.writer = writer

    def visualize(self, batch, out, epoch=0):
        raise NotImplementedError
