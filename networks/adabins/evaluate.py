import os

import matplotlib.pyplot as plt
import model_io
import numpy as np
import torch
import torch.nn as nn
from dataloader import DepthDataLoader
from models import UnetAdaptiveBins
from options import AdabinsOptions
from tqdm import tqdm
from utils import RunningAverageDict


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(
        a1=a1,
        a2=a2,
        a3=a3,
        abs_rel=abs_rel,
        rmse=rmse,
        log_10=log_10,
        rmse_log=rmse_log,
        silog=silog,
        sq_rel=sq_rel,
    )


def predict_tta(model, image, args):
    pred = model(image)[-1]
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)

    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)

    pred_lr = model(image)[-1]
    pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], args.min_depth, args.max_depth)
    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(
        torch.Tensor(final), image.shape[-2:], mode='bilinear', align_corners=True
    )
    return torch.Tensor(final)


def eval(
    model,
    test_loader,
    args,
    gpus=None,
):
    if gpus is None:
        device = (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
    else:
        device = gpus[0]

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    metrics = RunningAverageDict()
    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):

            image = batch['image'].to(device)
            gt = batch['depth'].to(device)
            final = predict_tta(model, image, args)
            final = final.squeeze().cpu().numpy()

            final[np.isinf(final)] = args.max_depth
            final[np.isnan(final)] = args.min_depth

            if args.save_dir is not None:
                dpath = batch['image_path'][0].split('/')
                impath = dpath[1] + "_" + dpath[-1]
                impath = impath.split('.')[0]

                pred_path = os.path.join(args.save_dir, f"{impath}.png")

                final_resized = (
                    nn.functional.interpolate(
                        torch.from_numpy(final).unsqueeze(0).unsqueeze(0),
                        (batch['original_size'][1], batch['original_size'][0]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                pred = (final_resized).astype('uint16')
                vmax = np.percentile(pred, 95)
                plt.imsave(pred_path, pred, cmap='magma', vmax=vmax)

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    total_invalid += 1
                    continue

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)

            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))

    print(f"Total invalid: {total_invalid}")
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")


if __name__ == '__main__':
    options = AdabinsOptions()
    opts = options.parse()
    opts.gpu = int(opts.gpu) if opts.gpu is not None else 0
    opts.distributed = False
    device = torch.device('cuda:{}'.format(opts.gpu))
    test = DepthDataLoader(opts, 'online_eval').data
    model = UnetAdaptiveBins.build(
        n_bins=opts.n_bins, min_val=opts.min_depth, max_val=opts.max_depth, norm='linear'
    ).to(device)
    print('Parameters ', sum(p.numel() for p in model.parameters()))
    model = model_io.load_checkpoint(opts.checkpoint_path, model)[0]
    model = model.eval()

    eval(model, test, opts, gpus=[device])
