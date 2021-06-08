import inspect
import logging

# import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# import torch.distributed as dist
from dp.core.lr_policys import _get_lr_policy
from dp.core.optimizers import _get_optimizer
from dp.models import _get_model
from dp.utils.comm import synchronize
from dp.utils.pyt_io import load_model
from dp.utils.pyt_ops import tensor2cuda
from dp.version import __version__
from torch.nn.utils import clip_grad_norm_


logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)


class Solver(object):
    def __init__(self):
        """
        :param config: easydict
        """
        self.version = __version__
        self.epoch = 0
        self.iteration = 0
        self.config = None
        self.model, self.optimizer, self.lr_policy = None, None, None
        self.step_decay = 1
        self.filtered_keys = None
        logging.info('[Single GPU mode]')

    def _build_environ(self):
        if self.config['environ']['deterministic']:
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.set_printoptions(precision=10)
        else:
            cudnn.benchmark = True

        # set random seed
        torch.manual_seed(self.config['environ']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['environ']['seed'])
        np.random.seed(self.config['environ']['seed'])
        random.seed(self.config['environ']['seed'])

        # grad clip settings
        self.grad_clip_params = self.config["solver"]["optimizer"].get("grad_clip")
        self.use_grad_clip = True if self.grad_clip_params is not None else False
        if self.use_grad_clip:
            logging.info("Using grad clip and params is {}".format(self.grad_clip_params))
        else:
            logging.info("Not Using grad clip.")

    def init_from_scratch(self, config):
        t_start = time.time()
        self.config = config
        self._build_environ()
        # model and optimizer
        self.model = _get_model(self.config)
        self.filtered_keys = [
            p.name for p in inspect.signature(self.model.forward).parameters.values()
        ]
        # logging.info("filtered keys:{}".format(self.filtered_keys))
        # model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        model_params = []
        for params in self.model.optimizer_params():
            params["lr"] = (
                self.config["solver"]["optimizer"]["params"]["lr"] * params["lr"]
            )
            model_params.append(params)
        self.optimizer = _get_optimizer(
            config['solver']['optimizer'], model_params=model_params
        )

        self.lr_policy = _get_lr_policy(
            config['solver']['lr_policy'], optimizer=self.optimizer
        )
        self.step_decay = config['solver']['step_decay']

        if config['model'].get('pretrained_model') is not None:
            logging.info(
                'loadding pretrained model from {}.'.format(
                    config['model']['pretrained_model']
                )
            )
            load_model(self.model, config['model']['pretrained_model'])

        self.model.cuda(0)
        t_end = time.time()
        logging.info(
            "Init trainer from scratch, Time usage: IO: {}".format(t_end - t_start)
        )

    def init_from_checkpoint(self, continue_state_object):
        t_start = time.time()

        self.config = continue_state_object['config']
        self._build_environ()
        self.model = _get_model(self.config)
        self.filtered_keys = [
            p.name for p in inspect.signature(self.model.forward).parameters.values()
        ]
        # model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        model_params = []
        for params in self.model.optimizer_params():
            params["lr"] = (
                self.config["solver"]["optimizer"]["params"]["lr"] * params["lr"]
            )
            model_params.append(params)
        self.optimizer = _get_optimizer(
            self.config['solver']['optimizer'], model_params=model_params
        )
        self.lr_policy = _get_lr_policy(
            self.config['solver']['lr_policy'], optimizer=self.optimizer
        )

        load_model(self.model, continue_state_object['model'])
        self.model.cuda(0)

        self.optimizer.load_state_dict(continue_state_object['optimizer'])
        self.lr_policy.load_state_dict(continue_state_object['lr_policy'])

        self.step_decay = self.config['solver']['step_decay']
        self.epoch = continue_state_object['epoch']
        self.iteration = continue_state_object["iteration"]

        # del continue_state_object
        t_end = time.time()
        logging.info(
            "Init trainer from checkpoint, Time usage: IO: {}".format(t_end - t_start)
        )

    def parse_kwargs(self, minibatch):
        kwargs = {k: v for k, v in minibatch.items() if k in self.filtered_keys}
        if torch.cuda.is_available():
            kwargs = tensor2cuda(kwargs)
        return kwargs

    def step(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        self.iteration += 1
        loss = self.model(**kwargs)
        loss /= self.step_decay
        loss.backward()

        if self.iteration % self.step_decay == 0:
            if self.use_grad_clip:
                clip_grad_norm_(self.model.parameters(), **self.grad_clip_params)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_policy.step(self.epoch)

        reduced_loss = loss.data
        return reduced_loss

    def step_no_grad(self, **kwargs):
        with torch.no_grad():
            out = self.model(**kwargs)
        return out

    def before_epoch(self, epoch):
        synchronize()
        self.iteration = 0
        self.epoch = epoch
        self.model.train()
        # self.lr_policy.step(epoch)
        torch.cuda.empty_cache()

    def after_epoch(self, epoch=None):
        synchronize()
        self.model.eval()
        # gc.collect()
        torch.cuda.empty_cache()

    def save_checkpoint(self, path):
        t_start = time.time()

        state_dict = {}

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in self.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v

        state_dict['config'] = self.config
        state_dict['model'] = new_state_dict
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['lr_policy'] = self.lr_policy.state_dict()
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration

        t_iobegin = time.time()
        torch.save(state_dict, path)
        del state_dict
        del new_state_dict
        t_end = time.time()
        logging.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare snapshot: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin
            )
        )

    def get_learning_rates(self):
        lrs = []
        for i in range(len(self.optimizer.param_groups)):
            lrs.append(self.optimizer.param_groups[i]['lr'])
        return lrs
