import argparse
import sys


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


class BTSOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="BTS options", fromfile_prefix_chars='@'
        )
        self.parser.convert_arg_line_to_args = convert_arg_line_to_args
        self.parser.add_argument(
            '--mode', type=str, help='train or test', default='train'
        )
        self.parser.add_argument(
            '--model_name', type=str, help='model name', default='bts_eigen_v2'
        )
        self.parser.add_argument(
            '--encoder',
            type=str,
            help='type of encoder, desenet121_bts, densenet161_bts, '
            'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
            default='densenet161_bts',
        )
        # Dataset
        self.parser.add_argument(
            '--dataset',
            type=str,
            help='dataset to train on, kitti or evo',
            default='kitti',
        )
        self.parser.add_argument(
            '--data_path', type=str, help='path to the data', required=True
        )
        self.parser.add_argument(
            '--filenames_file',
            type=str,
            help='path to the filenames text file',
            required=True,
        )
        self.parser.add_argument(
            '--input_height', type=int, help='input height', default=480
        )
        self.parser.add_argument(
            '--input_width', type=int, help='input width', default=640
        )
        self.parser.add_argument(
            '--max_depth', type=float, help='maximum depth in estimation', default=10
        )

        # Log and save
        self.parser.add_argument(
            '--log_directory',
            type=str,
            help='directory to save checkpoints and summaries',
            default='',
        )
        self.parser.add_argument(
            '--checkpoint_path', type=str, help='path to a checkpoint to load', default=''
        )
        self.parser.add_argument(
            '--log_freq', type=int, help='Logging frequency in global steps', default=100
        )
        self.parser.add_argument(
            '--save_freq',
            type=int,
            help='Checkpoint saving frequency in global steps',
            default=500,
        )

        # Training
        self.parser.add_argument(
            '--fix_first_conv_blocks',
            help='if set, will fix the first two conv blocks',
            action='store_true',
        )
        self.parser.add_argument(
            '--fix_first_conv_block',
            help='if set, will fix the first conv block',
            action='store_true',
        )
        self.parser.add_argument(
            '--bn_no_track_stats',
            help='if set, will not track running stats in batch norm layers',
            action='store_true',
        )
        self.parser.add_argument(
            '--weight_decay',
            type=float,
            help='weight decay factor for optimization',
            default=1e-2,
        )
        self.parser.add_argument(
            '--bts_size', type=int, help='initial num_filters in bts', default=512
        )
        self.parser.add_argument(
            '--retrain',
            help='if used with checkpoint_path, will restart training from step zero',
            action='store_true',
        )
        self.parser.add_argument(
            '--adam_eps', type=float, help='epsilon in Adam optimizer', default=1e-6
        )
        self.parser.add_argument('--batch_size', type=int, help='batch size', default=4)
        self.parser.add_argument(
            '--num_epochs', type=int, help='number of epochs', default=50
        )
        self.parser.add_argument(
            '--learning_rate', type=float, help='initial learning rate', default=1e-4
        )
        self.parser.add_argument(
            '--end_learning_rate', type=float, help='end learning rate', default=-1
        )
        self.parser.add_argument(
            '--variance_focus',
            type=float,
            help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error',
            default=0.85,
        )

        # Preprocessing
        self.parser.add_argument(
            '--do_random_rotate',
            help='if set, will perform random rotation for augmentation',
            action='store_true',
        )
        self.parser.add_argument(
            '--degree', type=float, help='random rotation maximum degree', default=2.5
        )

        # Multi-gpu training
        self.parser.add_argument(
            '--num_threads',
            type=int,
            help='number of threads to use for data loading',
            default=1,
        )
        # Online eval
        self.parser.add_argument(
            '--do_online_eval',
            help='if set, perform online eval in every eval_freq steps',
            action='store_true',
        )
        self.parser.add_argument(
            '--filenames_file_eval',
            type=str,
            help='path to the filenames text file for online evaluation',
            required=False,
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
            default=80,
        )

        self.parser.add_argument(
            '--eval_freq',
            type=int,
            help='Online evaluation frequency in global steps',
            default=500,
        )
        self.parser.add_argument(
            '--eval_summary_directory',
            type=str,
            help='output directory for eval summary,'
            'if empty outputs to checkpoint folder',
            default='',
        )

    def parse(self):
        arg_filename_with_prefix = '@' + sys.argv[1]
        self.opts = self.parser.parse_args([arg_filename_with_prefix])
        # self.options = self.parser.parse_args()
        return self.opts
