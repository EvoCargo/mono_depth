from __future__ import division

import argparse

import torch
from mmcv import Config

# from mmcv.runner import load_checkpoint
from mono.apis import get_root_logger, set_random_seed, train_mono
from mono.datasets.get_dataset import get_dataset
from mono.model.registry import MONO


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--config',
        default='./config/cfg_evo.py',
        help='train config file path',
    )
    parser.add_argument(
        '--work_dir',
        default='./log',
        help='the dir to save logs and models',
    )
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True

    cfg.gpus = [0]

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    # logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    # if args.seed is not None:
    # logger.info('Set random seed to {}'.format(args.seed))
    set_random_seed(17)

    model_name = cfg.model['name']
    model = MONO.module_dict[model_name](cfg.model)

    # if cfg.resume_from is not None:
    #     load_checkpoint(model, cfg.resume_from, map_location='cpu')
    # elif cfg.finetune is not None:
    #     print('loading from', cfg.finetune)
    #     checkpoint = torch.load(cfg.finetune, map_location='cpu')
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)

    train_dataset = get_dataset(cfg.data, training=True)
    if cfg.validate:
        val_dataset = get_dataset(cfg.data, training=False)
    else:
        val_dataset = None

    train_mono(
        model,
        train_dataset,
        val_dataset,
        cfg,
        validate=cfg.validate,
        logger=logger,
    )


if __name__ == '__main__':
    main()
