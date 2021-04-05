# -*- coding: utf-8 -*-


def build_visualizer(cfg, writer=None):
    mod = __import__('{}.{}'.format(__name__, cfg['vis_config']['name']), fromlist=[''])
    return getattr(mod, cfg["vis_config"]["name"] + "_visualizer")(cfg, writer)
