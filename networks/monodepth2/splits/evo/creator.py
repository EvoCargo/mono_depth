import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


global_path = Path('/media/data/datasets/bag_depth')


def get_images_paths(path: Path):

    assert isinstance(path, Path)

    folders_paths = [folder for folder in path.iterdir() if folder.is_dir()]
    # print(folders_paths)
    xlsx_paths = [
        [file for file in folder.iterdir() if file.is_file() and file.suffix == '.xlsx']
        for folder in folders_paths
    ]
    flat_xlsx_paths = [item for sublist in xlsx_paths for item in sublist]

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

    return res


train, val = train_test_split(get_images_paths(global_path), random_state=17)

train[['image_mod', 'ind']].to_csv(
    'train_files.txt',
    header=False,
    index=False,
    sep=' ',
    quoting=csv.QUOTE_NONE,
    escapechar=' ',
)
val[['image_mod', 'ind']].to_csv(
    'val_files.txt',
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
