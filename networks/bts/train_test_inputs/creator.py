import csv

# import os
from pathlib import Path

import numpy as np
import pandas as pd


np.random.seed(17)

global_path = Path('/media/data/datasets/bag_depth')
train_list = [
    'kalibr_002_kalibr-2_n1-002_2021-03-02-12-35-04Z_.evo1a_record_default',
    'kalibr_002_2021-03-10_gnss_tests_kalibr-2_n1-002_2021-03-10-10-59-42Z_.evo1a_record_default',
    'kalibr_002_kalibr-2_n1-002_2021-03-02-12-48-36Z_.evo1a_record_default',
    'kalibr_04_2021-01-18_snow_2021-01-18-15-29-39_0',
    'hospital_01_2020-07-14-14-03-23_0',
    'ipcp_03_2020-06-23_ipcp_ipcp_n1-03_2020-06-23-16-54-40Z_converted.evo1a_old_default',
    'ipcp_03_2020-08-19_ipcp_ipcp_n1-03_2020-08-18-10-14-48Z_converted.evo1a_old_default',
    'ckad_01_ckad_2020-10-29-16-53-38_0',
]

val_list = [
    'kalibr_04_2021-01-18_snow_2021-01-18-15-37-39_0',
    'kalibr_04_2020-12-03_2020-12-03-13-26-10_0_ZED',
    'kalibr_03_2020-06-01_kalibr_2020-06-01-16-38-16_0',
    'kalibr_03_2020-05-22_kalibr_2020-05-22-18-40-28_full_route_high_speed',
    'hospital_01_2020-07-14-16-59-35_0',
    'ipcp_03_2020-08-19_ipcp_2020-08-19-20-23-09_0',
    'ipcp_03_2020-06-23_ipcp_2020-06-23-16-48-18_0',
    'ckad_01_ckad_2020-10-29-17-01-56_0',
]

# Intrinsics matrices: (x_focal, y_focal, x_pp, y_pp)
K_maxtrces = {
    'ckad_01_ckad_2020-10-29-16-53-38_0': (
        1052.69873046875,
        1052.69873046875,
        969.119079589844,
        555.560913085938,
    ),
    'hospital_01_2020-07-14-14-03-23_0': (
        1412.705078125,
        1412.705078125,
        967.168579101562,
        543.377807617188,
    ),
    'ipcp_03_2020-06-23_ipcp_ipcp_n1-03_2020-06-23-16-54-40Z_converted.evo1a_old_default': (
        1411.85119628906,
        1411.85119628906,
        991.805480957031,
        505.537628173828,
    ),
    'ipcp_03_2020-08-19_ipcp_ipcp_n1-03_2020-08-18-10-14-48Z_converted.evo1a_old_default': (
        1418.53186035156,
        1418.53186035156,
        963.149108886719,
        525.431640625,
    ),
    'kalibr_002_2021-03-10_gnss_tests_kalibr-2_n1-002_2021-03-10-10-59-42Z_.evo1a_record_default': (
        521.493103027344,
        521.493103027344,
        630.518432617188,
        362.928863525391,
    ),
    'kalibr_002_kalibr-2_n1-002_2021-03-02-12-35-04Z_.evo1a_record_default': (
        521.493103027344,
        521.493103027344,
        630.518432617188,
        362.928863525391,
    ),
    'kalibr_002_kalibr-2_n1-002_2021-03-02-12-48-36Z_.evo1a_record_default': (
        521.493103027344,
        521.493103027344,
        630.518432617188,
        362.928863525391,
    ),
    'kalibr_03_2020-05-22_kalibr_2020-05-22-18-40-28_full_route_high_speed': (
        1410.72094726563,
        1410.72094726563,
        991.791381835938,
        505.526275634766,
    ),
    'kalibr_03_2020-06-01_kalibr_2020-06-01-16-38-16_0': (
        1411.34484863281,
        1411.34484863281,
        991.828979492188,
        505.558074951172,
    ),
    'kalibr_04_2020-12-03_2020-12-03-13-26-10_0_ZED': (
        523.876708984375,
        523.876708984375,
        647.893188476563,
        369.270965576172,
    ),
    'ckad_01_ckad_2020-10-29-17-01-56_0': (
        1052.69873046875,
        1052.69873046875,
        969.119079589844,
        555.560913085938,
    ),
    'hospital_01_2020-07-14-16-59-35_0': (
        1412.67077636719,
        1412.67077636719,
        967.1748046875,
        543.375671386719,
    ),
    'ipcp_03_2020-06-23_ipcp_2020-06-23-16-48-18_0': (
        1418.07263183594,
        1418.07263183594,
        991.793395996094,
        505.523162841797,
    ),
    'ipcp_03_2020-08-19_ipcp_2020-08-19-20-23-09_0': (
        1418.53186035156,
        1418.53186035156,
        963.149108886719,
        525.431640625,
    ),
    'kalibr_04_2021-01-18_snow_2021-01-18-15-29-39_0': (
        523.876708984375,
        523.876708984375,
        647.893188476563,
        369.270965576172,
    ),
    'kalibr_04_2021-01-18_snow_2021-01-18-15-37-39_0': (
        523.876708984375,
        523.876708984375,
        647.893188476563,
        369.270965576172,
    ),
    'kapotnya_02_2020-02-26_kapotnya_2020-02-26-15-27-06_0': (
        1409.71423339844,
        1409.71423339844,
        991.831970214844,
        505.561462402344,
    ),
    'kapotnya_02_2020-02-26_kapotnya_2020-02-26-15-09-42_0': (
        1418.07263183594,
        1418.07263183594,
        991.793395996094,
        505.523162841797,
    ),
    'kapotnya_02_2020-02-26_kapotnya_2020-02-26-15-34-18_0': (
        1409.71423339844,
        1409.71423339844,
        991.831970214844,
        505.561462402344,
    ),
    'kapotnya_02_2020-02-26_kapotnya_2020-02-26-15-42-08_0': (
        1409.71423339844,
        1409.71423339844,
        991.831970214844,
        505.561462402344,
    ),
}


def get_images_paths(path: Path):

    assert isinstance(path, Path)

    folders_paths = [folder for folder in path.iterdir() if folder.is_dir()]

    res = pd.DataFrame({'folder': [], 'filenum': [], 'ind': []})
    for folder in folders_paths:

        images = [
            image for image in (folder / 'front_rgb_left').iterdir() if image.is_file()
        ]
        depths = [
            depth for depth in (folder / 'front_depth_left').iterdir() if depth.is_file()
        ]

        images_st = np.array([image.stem for image in images])
        depths_st = np.array([depth.stem for depth in depths])

        _, images_mask, _ = np.intersect1d(images_st, depths_st, return_indices=True)

        images_masked = np.array(images_st)[images_mask]

        df = pd.DataFrame({'folder': [], 'filenum': [], 'ind': []})
        df['folder'] = [i[:-20] for i in images_masked]
        df['filenum'] = [i[-19:] for i in images_masked]

        df = df.sort_values('filenum').reset_index(drop=True)
        df['ind'] = df.index

        res = res.append(df[['folder', 'filenum', 'ind']])
    res.reset_index(inplace=True, drop=True)

    res['x_focal'] = res['folder'].apply(lambda x: K_maxtrces[x][0])
    res['y_focal'] = res['folder'].apply(lambda x: K_maxtrces[x][1])
    res['x_pp'] = res['folder'].apply(lambda x: K_maxtrces[x][2])
    res['y_pp'] = res['folder'].apply(lambda x: K_maxtrces[x][3])

    return res


global_df = get_images_paths(global_path)
train = global_df[global_df['folder'].isin(train_list)]
val = global_df[global_df['folder'].isin(val_list)]
test = val.iloc[np.random.choice(np.arange(len(val)), size=len(val) // 10, replace=False)]
train_val = global_df[
    global_df['folder'].isin(train_list) | global_df['folder'].isin(val_list)
]

to_write = ['folder', 'filenum', 'x_focal', 'y_focal', 'x_pp', 'y_pp']

train[to_write].to_csv(
    'train_files.txt',
    header=False,
    index=False,
    sep=' ',
    quoting=csv.QUOTE_NONE,
    escapechar=' ',
)
val[to_write].to_csv(
    'val_files.txt',
    header=False,
    index=False,
    sep=' ',
    quoting=csv.QUOTE_NONE,
    escapechar=' ',
)

test[to_write].to_csv(
    'test_files.txt',
    header=False,
    index=False,
    sep=' ',
    quoting=csv.QUOTE_NONE,
    escapechar=' ',
)

train_val[to_write].to_csv(
    'trainval_files.txt',
    header=False,
    index=False,
    sep=' ',
    quoting=csv.QUOTE_NONE,
    escapechar=' ',
)
# with open('train_files.txt', 'w') as f:
#     for it in train:
#         f.write(path.parents[1].as_posix() + ' ' + path.stem.split('_')[-1] + ' \n')

# with open('val_files.txt', 'w') as f:
#     for path in test:
#         f.write(path.parents[1].as_posix() + ' ' + path.stem.split('_')[-1] + ' \n')
