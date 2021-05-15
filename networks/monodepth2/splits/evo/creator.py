import csv
from pathlib import Path

import numpy as np
import pandas as pd


global_path = Path('/media/data/datasets/bag_depth')
train_list = [
    'ckad_01_ckad_2020-10-29-16-53-38_0',
    'hospital_01_2020-07-14-14-03-23_0',
    'ipcp_03_2020-06-23_ipcp_ipcp_n1-03_2020-06-23-16-54-40Z_converted.evo1a_old_default',
    'ipcp_03_2020-08-19_ipcp_ipcp_n1-03_2020-08-18-10-14-48Z_converted.evo1a_old_default',
    'kalibr_002_2021-03-10_gnss_tests_kalibr-2_n1-002_2021-03-10-10-59-42Z_.evo1a_record_default',
    'kalibr_002_kalibr-2_n1-002_2021-03-02-12-35-04Z_.evo1a_record_default',
    'kalibr_002_kalibr-2_n1-002_2021-03-02-12-48-36Z_.evo1a_record_default',
    'kalibr_03_2020-05-22_kalibr_2020-05-22-18-40-28_full_route_high_speed',
    'kalibr_03_2020-06-01_kalibr_2020-06-01-16-38-16_0',
    'kalibr_04_2020-12-03_2020-12-03-13-26-10_0_ZED',
]

val_list = [
    'ckad_01_ckad_2020-10-29-17-01-56_0',
    'hospital_01_2020-07-14-16-59-35_0',
    'ipcp_03_2020-06-23_ipcp_2020-06-23-16-48-18_0',
    'ipcp_03_2020-08-19_ipcp_2020-08-19-20-23-09_0',
    'kalibr_04_2021-01-18_snow_2021-01-18-15-29-39_0',
    'kalibr_04_2021-01-18_snow_2021-01-18-15-37-39_0',
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
    xlsx_paths = [
        [file for file in folder.iterdir() if file.is_file() and file.suffix == '.xlsx']
        for folder in folders_paths
    ]
    flat_xlsx_paths = sorted([item for sublist in xlsx_paths for item in sublist])

    res = pd.DataFrame({'image': [], 'target': [], 'ind': []})
    for xlsx in flat_xlsx_paths:
        df = pd.read_excel(xlsx.as_posix())
        df = df.dropna()
        df['image'] = xlsx.parent / df['front_rgb_left']
        df['image'] = df['image'].apply(lambda x: '/'.join(x.as_posix().split('/')[-3:]))

        df['target'] = df['front_dp_classic']
        df['target'] = df['target'].apply(lambda x: '/'.join(x.split('/')[1:]))

        df = df.sort_values('image').reset_index(drop=True)
        df['ind'] = df.index

        res = res.append(df[['image', 'target', 'ind']])
    res.reset_index(inplace=True, drop=True)

    res['image'] = res['image'].apply(lambda x: Path(x))
    res['target'] = res['target'].apply(lambda x: Path(x))
    res['ind'] = res['ind'].astype(int)
    res['image_mod'] = res['image'].apply(
        lambda x: x.parents[1].as_posix() + ' ' + x.stem.split('_')[-1]
    )
    res['parent_folder'] = res['image'].apply(lambda x: x.parents[1].as_posix())

    res['x_focal'] = res['parent_folder'].apply(lambda x: K_maxtrces[x][0])
    res['y_focal'] = res['parent_folder'].apply(lambda x: K_maxtrces[x][1])
    res['x_pp'] = res['parent_folder'].apply(lambda x: K_maxtrces[x][2])
    res['y_pp'] = res['parent_folder'].apply(lambda x: K_maxtrces[x][3])

    return res


global_df = get_images_paths(global_path)
train = global_df[global_df['parent_folder'].isin(train_list)]
val = global_df[global_df['parent_folder'].isin(val_list)]

to_write = ['image_mod', 'ind', 'x_focal', 'y_focal', 'x_pp', 'y_pp']

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

test = val.iloc[np.random.choice(np.arange(len(val)), size=120, replace=False)]

test[to_write].to_csv(
    'test_files.txt',
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
