#!/usr/bin/python3
# -*- coding: utf-8 -*-
import collections

from ruamel import yaml


def load_config(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.RoundTripLoader)
    return config


def save_config(path, config):
    with open(path, 'w') as nf:
        yaml.dump(config, nf, Dumper=yaml.RoundTripDumper)


def print_config(config, step=''):
    for k, v in config.items():
        if isinstance(v, collections.OrderedDict):
            new_step = step + '  '
            print(step + k + ':')
            print_config(v, new_step)
        else:
            print(step + k + ':', v)


class Config:
    def __init__(self, defualt_path='./config/default.yaml'):
        with open(defualt_path) as f:
            self.config = yaml.load(f, Loader=yaml.RoundTripLoader)

    def load(self, path):
        with open(path) as f:
            self.config = yaml.load(f, Loader=yaml.RoundTripLoader)

    def save(self, path):
        with open(path, 'w') as nf:
            yaml.dump(self.config, nf, Dumper=yaml.RoundTripDumper)

    def get(self):
        return self.config

    def set(self, config):
        self.config = config
