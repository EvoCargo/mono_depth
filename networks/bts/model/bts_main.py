# import argparse

# import datetime
import os
import sys

# import threading
import time

import matplotlib
import matplotlib.cm
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# import torch.nn.utils as utils
from bts import BtsModel, bn_init_as_tf, silog_loss, weights_init_xavier
from bts_dataloader import BtsDataLoader
from bts_options import BTSOptions
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm


options = BTSOptions()
opts = options.parse()

if (opts.mode == 'train') and (opts.checkpoint_path):
    model_dir = os.path.dirname(opts.checkpoint_path)
    model_name = os.path.basename(model_dir)

    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val


# А что это такое?
inv_normalize = transforms.Normalize(
    mean=torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], device='cuda:0'),
    std=torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225], device='cuda:0'),
)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.0

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.0

    return np.expand_dims(value, 0)


def set_misc(model):
    if opts.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if opts.fix_first_conv_blocks:
        if 'resne' in opts.encoder:
            fixing_layers = [
                'base_model.conv1',
                'base_model.layer1.0',
                'base_model.layer1.1',
                '.bn',
            ]
        else:
            fixing_layers = [
                'conv0',
                'denseblock1.denselayer1',
                'denseblock1.denselayer2',
                'norm',
            ]
        print("Fixing first two conv blocks")
    elif opts.fix_first_conv_block:
        if 'resne' in opts.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in opts.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if 'encoder' not in name:
            continue
        for name2, parameters in child.named_parameters():
            # print(name, name2)
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False


def online_eval(model, dataloader_eval):
    eval_measures = torch.zeros(10).cuda()
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(
                eval_sample_batched['image'].cuda(non_blocking=True)
            )
            focal = torch.autograd.Variable(
                eval_sample_batched['focal'].cuda(non_blocking=True)
            )
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            # print(image.shape, focal.shape, gt_depth.shape)
            _, _, _, _, pred_depth = model(image, focal)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        pred_depth[pred_depth < opts.min_depth_eval] = opts.min_depth_eval
        pred_depth[pred_depth > opts.max_depth_eval] = opts.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = opts.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = opts.min_depth_eval

        valid_mask = np.logical_and(
            gt_depth > opts.min_depth_eval, gt_depth < opts.max_depth_eval
        )

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        # print('Measures', measures, '\n')

        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    # print('Eval_measures\n')
    # print(type(eval_measures))
    # print(eval_measures)
    # print(eval_measures.device, '\n')

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)))
    print(
        "{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
            'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'
        )
    )
    for i in range(8):
        print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.3f}'.format(eval_measures_cpu[8]))
    return eval_measures_cpu


def main_worker(opts):

    # Create model
    model = BtsModel(opts)
    model.train()
    model.decoder.apply(weights_init_xavier)
    set_misc(model)

    # model = nn.DataParallel(model)
    model.cuda()

    print("Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.AdamW(
        [
            {
                'params': model.encoder.parameters(),
                'weight_decay': opts.weight_decay,
            },
            {'params': model.decoder.parameters(), 'weight_decay': 0},
        ],
        lr=opts.learning_rate,
        eps=opts.adam_eps,
    )

    # if opts.checkpoint_path != '':
    #     if os.path.isfile(opts.checkpoint_path):
    #         print("Loading checkpoint '{}'".format(opts.checkpoint_path))
    #         checkpoint = torch.load(opts.checkpoint_path)
    #         global_step = checkpoint['global_step']
    #         model.load_state_dict(checkpoint['model'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         try:
    #             best_eval_measures_higher_better = checkpoint[
    #                 'best_eval_measures_higher_better'
    #             ].cpu()
    #             best_eval_measures_lower_better = checkpoint[
    #                 'best_eval_measures_lower_better'
    #             ].cpu()
    #             best_eval_steps = checkpoint['best_eval_steps']
    #         except KeyError:
    #             print("Could not load values for online evaluation")

    #         print(
    #             "Loaded checkpoint '{}' (global_step {})".format(
    #                 opts.checkpoint_path, checkpoint['global_step']
    #             )
    #         )
    #     else:
    #         print("No checkpoint found at '{}'".format(opts.checkpoint_path))

    # if opts.retrain:
    #     global_step = 0

    cudnn.benchmark = True

    dataloader = BtsDataLoader(opts, 'train')
    dataloader_eval = BtsDataLoader(opts, 'online_eval')

    # Logging
    writer = SummaryWriter(
        os.path.join(opts.log_directory, opts.model_name, 'summaries'), flush_secs=30
    )

    if opts.do_online_eval:
        eval_summary_writer = SummaryWriter(
            os.path.join(opts.log_directory, opts.model_name, 'eval'), flush_secs=30
        )

    silog_criterion = silog_loss(variance_focus=opts.variance_focus)

    start_time = time.time()

    num_log_images = opts.batch_size
    end_learning_rate = (
        opts.end_learning_rate
        if opts.end_learning_rate != -1
        else 0.1 * opts.learning_rate
    )

    steps_per_epoch = len(dataloader.data)
    num_total_steps = opts.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < opts.num_epochs:

        for _, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            # before_op_time = time.time()

            image = torch.autograd.Variable(
                sample_batched['image'].cuda(non_blocking=True)
            )
            focal = torch.autograd.Variable(
                sample_batched['focal'].cuda(non_blocking=True)
            )
            depth_gt = torch.autograd.Variable(
                sample_batched['depth'].cuda(non_blocking=True)
            )

            # print('Model.input: ', image.size(), depth_gt.size())
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)

            # print('Model output: ', depth_est.size())

            mask = depth_gt > 1.0

            loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (opts.learning_rate - end_learning_rate) * (
                    1 - global_step / num_total_steps
                ) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            # print(
            #     '[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(
            #         epoch, step, steps_per_epoch, global_step, current_lr, loss
            #     )
            # )
            # if np.isnan(loss.cpu().item()):
            #     print('NaN in loss occurred. Aborting training.')
            #     return -1

            # Какой-то лог
            if global_step and global_step % opts.log_freq == 0:
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar

                # print("{}".format(opts.model_name))
                print_string = 'loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(
                    print_string.format(
                        loss,
                        time_sofar,
                        training_time_left,
                    )
                )

                writer.add_scalar('silog_loss', loss, global_step)
                writer.add_scalar('learning_rate', current_lr, global_step)
                depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                for i in range(num_log_images):
                    writer.add_image(
                        'depth_gt/image/{}'.format(i),
                        normalize_result(1 / depth_gt[i, :, :, :].data),
                        global_step,
                    )
                    writer.add_image(
                        'depth_est/image/{}'.format(i),
                        normalize_result(1 / depth_est[i, :, :, :].data),
                        global_step,
                    )
                    writer.add_image(
                        'reduc1x1/image/{}'.format(i),
                        normalize_result(1 / reduc1x1[i, :, :, :].data),
                        global_step,
                    )
                    writer.add_image(
                        'lpg2x2/image/{}'.format(i),
                        normalize_result(1 / lpg2x2[i, :, :, :].data),
                        global_step,
                    )
                    writer.add_image(
                        'lpg4x4/image/{}'.format(i),
                        normalize_result(1 / lpg4x4[i, :, :, :].data),
                        global_step,
                    )
                    writer.add_image(
                        'lpg8x8/image/{}'.format(i),
                        normalize_result(1 / lpg8x8[i, :, :, :].data),
                        global_step,
                    )
                    writer.add_image(
                        'image/image/{}'.format(i),
                        inv_normalize(image[i, :, :, :]).data,
                        global_step,
                    )
                writer.flush()
            global_step += 1

        if opts.do_online_eval:
            time.sleep(0.1)
            model.eval()
            eval_measures = online_eval(model, dataloader_eval)
            if eval_measures is not None:
                for i in range(9):
                    eval_summary_writer.add_scalar(
                        eval_metrics[i], eval_measures[i].cpu(), int(epoch)
                    )
                    measure = eval_measures[i]
                    is_best = False
                    if i < 6 and measure < best_eval_measures_lower_better[i]:
                        old_best = best_eval_measures_lower_better[i].item()
                        best_eval_measures_lower_better[i] = measure.item()
                        is_best = True
                    elif i >= 6 and measure > best_eval_measures_higher_better[i - 6]:
                        old_best = best_eval_measures_higher_better[i - 6].item()
                        best_eval_measures_higher_better[i - 6] = measure.item()
                        is_best = True
                    if is_best:
                        old_best_step = best_eval_steps[i]
                        old_best_name = '/model-{}-best_{}_{:.5f}'.format(
                            old_best_step, eval_metrics[i], old_best
                        )
                        model_path = (
                            opts.log_directory + '/' + opts.model_name + old_best_name
                        )
                        if os.path.exists(model_path):
                            command = 'rm {}'.format(model_path)
                            os.system(command)
                        best_eval_steps[i] = epoch
                        model_save_name = '/model-{}-best_{}_{:.5f}'.format(
                            epoch, eval_metrics[i], measure
                        )
                        print(
                            'New best for {}. Saving model: {}'.format(
                                eval_metrics[i], model_save_name
                            )
                        )
                        checkpoint = {
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_eval_measures_higher_better': best_eval_measures_higher_better,
                            'best_eval_measures_lower_better': best_eval_measures_lower_better,
                            'best_eval_steps': best_eval_steps,
                        }
                        torch.save(
                            checkpoint,
                            opts.log_directory + '/' + opts.model_name + model_save_name,
                        )
                eval_summary_writer.flush()
            model.train()
            block_print()
            set_misc(model)
            enable_print()

        epoch += 1


def main():

    model_filename = opts.model_name + '.py'
    command = 'mkdir ' + opts.log_directory + '/' + opts.model_name
    os.system(command)

    opts_out_path = opts.log_directory + '/' + opts.model_name + '/' + sys.argv[1]
    command = 'cp ' + sys.argv[1] + ' ' + opts_out_path
    os.system(command)

    if opts.checkpoint_path == '':
        model_out_path = opts.log_directory + '/' + opts.model_name + '/' + model_filename
        command = 'cp bts.py ' + model_out_path
        os.system(command)
        aux_out_path = opts.log_directory + '/' + opts.model_name + '/.'
        command = 'cp bts_main.py ' + aux_out_path
        os.system(command)
        command = 'cp bts_dataloader.py ' + aux_out_path
        os.system(command)
    else:
        loaded_model_dir = os.path.dirname(opts.checkpoint_path)
        loaded_model_name = os.path.basename(loaded_model_dir)
        loaded_model_filename = loaded_model_name + '.py'

        model_out_path = opts.log_directory + '/' + opts.model_name + '/' + model_filename
        command = (
            'cp ' + loaded_model_dir + '/' + loaded_model_filename + ' ' + model_out_path
        )
        os.system(command)

    torch.cuda.empty_cache()

    if opts.do_online_eval:
        print("You have specified --do_online_eval.")
        print(
            "This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics.".format(
                opts.eval_freq
            )
        )

    main_worker(opts)


if __name__ == '__main__':
    main()
