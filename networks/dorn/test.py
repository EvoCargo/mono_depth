# import argparse
# import logging
# import sys
# import time
# import warnings

# import torch
# from dp.core.solver import Solver
# from dp.datasets.loader import build_loader
# from dp.utils.config import load_config, print_config
# from dp.visualizers import build_visualizer
# from tqdm import tqdm


# warnings.filterwarnings("ignore")
# logging.basicConfig(
#     format='[%(asctime)s %(levelname)s] %(message)s',
#     datefmt='%m/%d/%Y %H:%M:%S',
#     level=logging.INFO,
# )

# parser = argparse.ArgumentParser(description='Training script')
# parser.add_argument('-c', '--config', type=str)
# parser.add_argument('-r', '--resumed', type=str, default=None, required=False)

# parser.add_argument("--vpath", type=str, default="vis")
# parser.add_argument(
#     "--vdepth", action="store_true", help="visualize depth gt and predictions"
# )

# args = parser.parse_args()

# solver = Solver()

# continue_state_object = torch.load(args.resumed, map_location=torch.device("cpu"))
# config = continue_state_object['config']

# solver.init_from_checkpoint(continue_state_object=continue_state_object)

# te_loader, _, niter_test = build_loader(config, False)

# dataset_name = config["data"]["name"][1]
# epoch = config['solver']['epochs']
# solver.after_epoch()

# # validation

# bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
# pbar = tqdm(range(niter_test), file=sys.stdout, bar_format=bar_format)

# test_iter_l = iter(te_loader_l)
# test_iter_r = iter(te_loader_r)
# for idx in pbar:
#     t_start = time.time()
#     minibatch_l = next(test_iter_l)
#     filtered_kwargs_l = solver.parse_kwargs(minibatch_l)
#     minibatch_r = next(test_iter_r)
#     filtered_kwargs_r = solver.parse_kwargs(minibatch_r)

#     # print(filtered_kwargs)
#     t_end = time.time()
#     io_time = t_end - t_start
#     t_start = time.time()
#     pred_l = solver.step_no_grad(**filtered_kwargs_l)
#     pred_r = solver.step_no_grad(**filtered_kwargs_r)
#     d_pred_l = pred_l["target"][-1]  # B*H*W
#     d_pred_r = pred_r["target"][-1]  # B*H*W

#     full_width = minibatch_l["depth_full"].shape[
#         -1
#     ]  # minibatch_l["depth_full"].shape=(B*H*W)
#     full_height = minibatch_l["depth_full"].shape[-2]
#     pred_full = compose_preds(d_pred_l, d_pred_r, full_width, full_height)
#     pred_kb_crop = kb_crop_preds(pred_full)  # B*H*W

#     t_end = time.time()
#     inf_time = t_end - t_start
#     t_start = time.time()

#     # remove batch dim
#     pred_crop_hw = pred_kb_crop[0]
#     pred_full_hw = pred_full[0]
#     gt_full_hw = minibatch_l["depth_full"][0]

#     t_end = time.time()
#     cmp_time = t_end - t_start

#     # visualize predictions
#     vis_pred = vis_depth(pred_full)
#     vis_pred = uint8_np_from_img_tensor(vis_pred)
#     vis_gt = vis_depth(minibatch_l["depth_full"])
#     vis_gt = uint8_np_from_img_tensor(vis_gt)
#     save_np_to_img(vis_pred, "{}/{}_pred".format(args.vpath, idx))

#     pbar.set_description(print_str, refresh=False)


# print("Mean pred/gt ratio:", mean_tracker.mean())
