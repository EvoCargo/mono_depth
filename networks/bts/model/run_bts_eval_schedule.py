import datetime
import os

from apscheduler.schedulers.blocking import BlockingScheduler


scheduler = BlockingScheduler()


@scheduler.scheduled_job(
    'interval', minutes=1, start_date=datetime.datetime.now() + datetime.timedelta(0, 3)
)
def run_eval():
    command = (
        'export CUDA_VISIBLE_DEVICES=0; '
        '/usr/bin/python '
        'bts_eval.py '
        '--encoder densenet161_bts '
        '--dataset kitti '
        '--data_path ../../dataset/kitti_dataset/ '
        '--gt_path ../../dataset/kitti_dataset/data_depth_annotated/ '
        '--filenames_file ../train_test_inputs/eigen_test_files_with_gt.txt '
        '--input_height 352 '
        '--input_width 1216 '
        '--garg_crop '
        '--max_depth 80 '
        '--max_depth_eval 80 '
        '--output_directory ./models/eval-eigen/ '
        '--model_name bts_eigen_v0_0_1 '
        '--checkpoint_path ./models/bts_eigen_v0_0_1/ '
        '--do_kb_crop '
    )

    print('Executing: %s' % command)
    os.system(command)
    print('Finished: %s' % datetime.datetime.now())


scheduler.configure()
scheduler.start()
