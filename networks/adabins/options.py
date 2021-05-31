import argparse
import sys


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


class AdabinsOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Training script. Default values of all arguments are recommended for reproducibility',
            fromfile_prefix_chars='@',
            conflict_handler='resolve',
        )
        self.parser.convert_arg_line_to_args = convert_arg_line_to_args

        self.parser.add_argument(
            '--mode', type=str, help='train or test', default='train'
        )
        self.parser.add_argument(
            '--epochs', default=25, type=int, help='number of total epochs to run'
        )
        self.parser.add_argument(
            '--n_bins',
            default=80,
            type=int,
            help='number of bins/buckets to divide depth range into',
        )
        self.parser.add_argument(
            '--learning_rate', default=0.000357, type=float, help='max learning rate'
        )
        self.parser.add_argument(
            '--weight_decay', default=0.1, type=float, help='weight decay'
        )
        self.parser.add_argument(
            '--w_chamfer',
            default=0.1,
            type=float,
            help="weight value for chamfer loss",
        )
        self.parser.add_argument(
            '--div_factor',
            default=25,
            type=float,
            help="Initial div factor for lr",
        )
        self.parser.add_argument(
            '--final_div_factor',
            default=100,
            type=float,
            help="final div factor for lr",
        )

        self.parser.add_argument('--batch_size', default=16, type=int, help='batch size')
        self.parser.add_argument(
            '--validate_every',
            default=100,
            type=int,
            help='validation period',
        )
        self.parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
        self.parser.add_argument("--name", default="UnetAdaptiveBins")
        self.parser.add_argument(
            "--norm",
            default="linear",
            type=str,
            help="Type of norm/competition for bin-widths",
            choices=['linear', 'softmax', 'sigmoid'],
        )
        self.parser.add_argument(
            '--same_lr',
            default=False,
            action="store_true",
            help="Use same LR for all param groups",
        )
        self.parser.add_argument(
            "--log_dir", default=".", type=str, help="Root folder to save data in"
        )
        self.parser.add_argument(
            "--resume", default='', type=str, help="Resume from checkpoint"
        )

        self.parser.add_argument("--notes", default='', type=str, help="Wandb notes")
        self.parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")
        self.parser.add_argument(
            "--workers", default=11, type=int, help="Number of workers for data loading"
        )
        self.parser.add_argument(
            "--dataset", default='evo', type=str, help="Dataset to train on"
        )
        self.parser.add_argument(
            "--data_path",
            default='../dataset/nyu/sync/',
            type=str,
            help="path to dataset",
        )
        self.parser.add_argument(
            "--gt_path", default='../dataset/nyu/sync/', type=str, help="path to dataset"
        )
        self.parser.add_argument(
            '--filenames_file',
            default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
            type=str,
            help='path to the filenames text file',
        )

        self.parser.add_argument(
            '--input_height', type=int, help='input height', default=416
        )
        self.parser.add_argument(
            '--input_width', type=int, help='input width', default=544
        )
        self.parser.add_argument(
            '--max_depth', type=float, help='maximum depth in estimation', default=10
        )
        self.parser.add_argument(
            '--min_depth', type=float, help='minimum depth in estimation', default=1e-3
        )
        self.parser.add_argument(
            '--do_random_rotate',
            default=True,
            help='if set, will perform random rotation for augmentation',
            action='store_true',
        )
        self.parser.add_argument(
            '--degree', type=float, help='random rotation maximum degree', default=2.5
        )
        self.parser.add_argument(
            '--use_right',
            help='if set, will randomly use right images when train on KITTI',
            action='store_true',
        )

        self.parser.add_argument(
            '--data_path_eval',
            default="../dataset/nyu/official_splits/test/",
            type=str,
            help='path to the data for online evaluation',
        )
        self.parser.add_argument(
            '--gt_path_eval',
            default="../dataset/nyu/official_splits/test/",
            type=str,
            help='path to the groundtruth data for online evaluation',
        )
        self.parser.add_argument(
            '--filenames_file_eval',
            default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
            type=str,
            help='path to the filenames text file for online evaluation',
        )

        self.parser.add_argument(
            '--min_depth_eval',
            type=float,
            help='minimum depth for evaluation',
            default=1e-3,
        )
        self.parser.add_argument(
            '--max_depth_eval',
            type=float,
            help='maximum depth for evaluation',
            default=10,
        )
        self.parser.add_argument(
            '--save_dir',
            default=None,
            type=str,
            help='Store predictions in folder',
        )
        self.parser.add_argument(
            '--checkpoint_path',
            type=str,
            required=True,
            help="checkpoint file to use for prediction",
        )

    def parse(self):
        arg_filename_with_prefix = '@' + sys.argv[1]
        self.opts = self.parser.parse_args([arg_filename_with_prefix])
        # self.options = self.parser.parse_args()
        return self.opts
