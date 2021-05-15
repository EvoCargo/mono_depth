# -*- coding: utf-8 -*-

import logging
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_model(model, model_file, device=torch.device('cpu')):  # noqa
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logging.warning(
            'Missing key(s) in state_dict: {}'.format(
                ', '.join('{}'.format(k) for k in missing_keys)
            )
        )

    if len(unexpected_keys) > 0:
        logging.warning(
            'Unexpected key(s) in state_dict: {}'.format(
                ', '.join('{}'.format(k) for k in unexpected_keys)
            )
        )

    del state_dict
    t_end = time.time()
    logging.info(
        "Load model, Time usage: IO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend
        )
    )


def create_summary_writer(logdir=None):
    # assert os.path.exists(logdir), 'Log file dir is not existed.'
    ensure_dir(logdir)

    log_path = os.path.join(logdir, 'tensorboard')
    # if os.path.isdir(log_path):
    #     shutil.rmtree(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = SummaryWriter(log_path)
    return logger
