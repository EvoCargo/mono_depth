#!/usr/bin/python3
# -*- coding: utf-8 -*-


def _get_dataset(name):
    mod = __import__('{}.{}'.format(__name__, name), fromlist=[''])
    return getattr(mod, name)
