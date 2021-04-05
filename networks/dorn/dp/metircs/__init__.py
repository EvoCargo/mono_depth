#!/usr/bin/python3
# -*- coding: utf-8 -*-


def build_metrics(cfg):
    mod = __import__("{}.{}".format(__name__, "metrics"), fromlist=[''])
    return getattr(mod, "Metrics")()  # noqa
    # return mod.Metrics
