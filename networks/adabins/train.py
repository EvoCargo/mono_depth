import os
import uuid
from datetime import datetime as dt

# import matplotlib
import model_io
import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import wandb
from dataloader import DepthDataLoader
from loss import BinsChamferLoss, SILogLoss
from options import AdabinsOptions
from tqdm import tqdm
from utils import RunningAverage, colorize


# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "monodepth_Adabins"
logging = True


def is_rank_zero(args):
    return args.rank == 0


def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)],
        },
        step=step,
    )


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # Load model

    model = models.UnetAdaptiveBins.build(
        n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm
    )

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False

    args.epoch = 0
    args.last_epoch = -1
    train(
        model,
        args,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=args.gpu,
        root=args.log_dir,
        experiment_name=args.name,
        optimizer_state_dict=None,
    )


def train(
    model,
    args,
    epochs=10,
    experiment_name="DeepLab",
    lr=0.0001,
    root=".",
    device=None,
    optimizer_state_dict=None,
):
    global PROJECT
    if device is None:
        device = (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

    # Logging setup
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.batch_size}-tep{epochs}-lr{lr}-wd{args.weight_decay}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"

    tags = args.tags.split(',') if args.tags != '' else None
    if args.dataset != 'nyu':
        PROJECT = PROJECT + f"-{args.dataset}"
    wandb.init(
        project=PROJECT,
        name=name,
        config=args,
        dir=args.log_dir,
        tags=tags,
        notes=args.notes,
    )
    # wandb.watch(model)

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    # losses
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None

    model.train()

    # Optimizer
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [
            {"params": m.get_1x_lr_params(), "lr": lr / 10},
            {"params": m.get_10x_lr_params(), "lr": lr},
        ]

    optimizer = optim.AdamW(params, weight_decay=args.weight_decay, lr=args.learning_rate)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        last_epoch=args.last_epoch,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
    )
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)

    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        # Train loop
        wandb.log({"Epoch": epoch}, step=step)
        for _, batch in tqdm(
            enumerate(train_loader),
            desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
            total=len(train_loader),
        ):

            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            bin_edges, pred = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(
                pred, depth, mask=mask.to(torch.bool), interpolate=True
            )

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)

            loss = l_dense + args.w_chamfer * l_chamfer
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            if step % 5 == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)

            step += 1
            scheduler.step()

        # Validation loop
        model.eval()
        metrics, val_si = validate(
            args, model, test_loader, criterion_ueff, epoch, epochs, device
        )

        # print("Validated: {}".format(metrics))
        wandb.log(
            {
                f"Test/{criterion_ueff.name}": val_si.get_value(),
                # f"Test/{criterion_bins.name}": val_bins.get_value()
            },
            step=step,
        )

        wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
        model_io.save_checkpoint(
            model,
            optimizer,
            epoch,
            f"{experiment_name}_{run_id}_latest.pt",
            root=os.path.join(root, "checkpoints"),
        )

        if metrics['abs_rel'] < best_loss:
            model_io.save_checkpoint(
                model,
                optimizer,
                epoch,
                f"{experiment_name}_{run_id}_best.pt",
                root=os.path.join(root, "checkpoints"),
            )
            best_loss = metrics['abs_rel']
        model.train()

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(
            test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation"
        ):
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            bins, pred = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(
                pred, depth, mask=mask.to(torch.bool), interpolate=True
            )
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(
                pred, depth.shape[-2:], mode='bilinear', align_corners=True
            )

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(
                gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval
            )

            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


if __name__ == '__main__':

    # Arguments
    options = AdabinsOptions()
    opts = options.parse()

    opts.chamfer = opts.w_chamfer > 0
    if opts.log_dir != "." and not os.path.isdir(opts.log_dir):
        os.makedirs(opts.log_dir)

    # try:
    #     node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
    #     nodes = node_str.split(',')

    #     args.world_size = len(nodes)
    #     args.rank = int(os.environ['SLURM_PROCID'])

    # except KeyError:
    #     # We are NOT using SLURM
    #     args.world_size = 1
    #     args.rank = 0
    #     nodes = ["127.0.0.1"]

    ngpus_per_node = torch.cuda.device_count()
    opts.num_workers = opts.workers
    opts.ngpus_per_node = ngpus_per_node
    opts.gpu = 0

    main_worker(opts.gpu, ngpus_per_node, opts)
