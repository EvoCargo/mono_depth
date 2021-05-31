from __future__ import absolute_import, division, print_function

import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from bts import BtsModel
from bts_dataloader import BtsDataLoader
from bts_options import BTSOptions
from tensorboardX import SummaryWriter
from torch.autograd import Variable


options = BTSOptions()
opts = options.parse()

model_dir = os.path.dirname(opts.checkpoint_path)
sys.path.append(model_dir)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))

    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    global gt_depths, is_missing, missing_ids
    gt_depths = []
    is_missing = []
    missing_ids = set()
    write_summary = False
    steps = set()

    if os.path.isdir(opts.checkpoint_path):
        import glob

        models = [f for f in glob.glob(opts.checkpoint_path + "/model*")]

        for model in models:
            # print(model)
            step = model.split('-')[-1]
            steps.add('{:06d}'.format(int(step)))

        lines = []
        if os.path.exists(opts.checkpoint_path + '/evaluated_checkpoints'):
            with open(opts.checkpoint_path + '/evaluated_checkpoints') as file:
                lines = file.readlines()

        for line in lines:
            if line.rstrip() in steps:
                steps.remove(line.rstrip())

        steps = sorted(steps)
        if opts.log_directory != '':
            summary_path = os.path.join(opts.log_directory, opts.model_name)
        else:
            summary_path = os.path.join(opts.checkpoint_path, 'eval')

        write_summary = True
    else:
        steps.add('{:06d}'.format(int(opts.checkpoint_path.split('-')[-1])))

    if len(steps) == 0:
        print('No new model to evaluate. Abort.')
        return

    opts.mode = 'test'
    dataloader = BtsDataLoader(opts, 'test')

    model = BtsModel(params=params)

    cudnn.benchmark = True

    if write_summary:
        summary_writer = SummaryWriter(summary_path, flush_secs=30)

    for step in steps:
        if os.path.isdir(opts.checkpoint_path):
            checkpoint = torch.load(
                os.path.join(opts.checkpoint_path, 'model-' + str(int(step)))
            )
            model.load_state_dict(checkpoint['model'])
        else:
            checkpoint = torch.load(opts.checkpoint_path)
            model.load_state_dict(checkpoint['model'])

        model.eval()
        model.cuda()

        num_test_samples = get_num_lines(opts.filenames_file)

        with open(opts.filenames_file) as f:
            lines = f.readlines()

        print('now testing {} files for step {}'.format(num_test_samples, step))

        pred_depths = []

        start_time = time.time()
        with torch.no_grad():
            for _, sample in enumerate(dataloader.data):
                image = Variable(sample['image'].cuda())
                focal = Variable(sample['focal'].cuda())
                # Predict
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
                pred_depths.append(depth_est.cpu().numpy().squeeze())

        elapsed_time = time.time() - start_time
        print('Elapesed time: %s' % str(elapsed_time))
        print('Done.')

        if len(gt_depths) == 0:
            for t_id in range(num_test_samples):

                splitted = lines[t_id].split()

                gt_depth_path = os.path.join(
                    opts.data_path,
                    splitted[0],
                    'front_depth_left',
                    splitted[0] + '_' + splitted[1] + '.png',
                )
                depth = cv2.imread(gt_depth_path, -1)
                gt_depths.append(depth)

        print('Computing errors')
        silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3 = eval(
            pred_depths, int(step)
        )

        if write_summary:
            summary_writer.add_scalar('silog', silog.mean(), int(step))
            summary_writer.add_scalar('abs_rel', abs_rel.mean(), int(step))
            summary_writer.add_scalar('log10', log10.mean(), int(step))
            summary_writer.add_scalar('sq_rel', sq_rel.mean(), int(step))
            summary_writer.add_scalar('rms', rms.mean(), int(step))
            summary_writer.add_scalar('log_rms', log_rms.mean(), int(step))
            summary_writer.add_scalar('d1', d1.mean(), int(step))
            summary_writer.add_scalar('d2', d2.mean(), int(step))
            summary_writer.add_scalar('d3', d3.mean(), int(step))
            summary_writer.flush()

            with open(
                os.path.dirname(opts.checkpoint_path) + '/evaluated_checkpoints', 'a'
            ) as file:
                file.write(step + '\n')

        print('Evaluation done')


def eval(pred_depths, step):
    num_samples = get_num_lines(opts.filenames_file)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):

        gt_depth = gt_depths[i]
        cv2.imwrite(f'/home/penitto/mono_depth/networks/bts/log/tstP{i}.png', gt_depth)
        pred_depth = cv2.resize(
            pred_depths[i],
            (gt_depths[i].shape[1], gt_depths[i].shape[0]),
            cv2.INTER_LINEAR,
        )

        # alt_gt_depth = gt_depths[i]
        # alt_pred_depth = F.interpolate(
        #     pred_depths[i],
        #     size=gt_depths[i].shape,
        #     mode="bilinear",
        #     align_corners=True)

        pred_depth[pred_depth < opts.min_depth_eval] = opts.min_depth_eval
        pred_depth[pred_depth > opts.max_depth_eval] = opts.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = opts.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = opts.min_depth_eval

        valid_mask = np.logical_and(
            gt_depth > opts.min_depth_eval, gt_depth < opts.max_depth_eval
        )

        (
            silog[i],
            log10[i],
            abs_rel[i],
            sq_rel[i],
            rms[i],
            log_rms[i],
            d1[i],
            d2[i],
            d3[i],
        ) = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

    print(
        "{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
            'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'
        )
    )
    print(
        "{:7.4f}, {:7.4f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
            silog.mean(),
            abs_rel.mean(),
            log10.mean(),
            rms.mean(),
            sq_rel.mean(),
            log_rms.mean(),
            d1.mean(),
            d2.mean(),
            d3.mean(),
        )
    )

    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3


if __name__ == '__main__':
    test(opts)
