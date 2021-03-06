#!/usr/bin/python3
# -*- coding: utf-8 -*-


def _get_model(cfg):
    mod = __import__('{}.{}'.format(__name__, cfg['model']['name']), fromlist=[''])
    return getattr(mod, "DepthPredModel")(**cfg["model"]["params"])  # noqa
