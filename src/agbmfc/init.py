import json
import os
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

from src.agbmfc.loading import DATA_ROOT


def generate_train_dev_chip_split():
    features_meta = pd.read_csv(os.path.join(DATA_ROOT, 'features_metadata.csv'))
    train_chip_ids = features_meta[features_meta.split == 'train'].chip_id.unique()
    train, dev = train_test_split(train_chip_ids, test_size=0.20, shuffle=True)

    chips_from_partition = {
        'train': list(train),
        'dev': list(dev),
    }

    path = pathlib.Path(__file__)
    folder_path = path.parent.parent.parent / 'data' / 'md'
    path_to_file = folder_path / 'chips_from_partition.json'

    if os.path.exists(path_to_file):
        raise ValueError(f'file {path_to_file} already exists')

    with open(path_to_file, 'w') as fd:
        json.dump(chips_from_partition, fd)

    return chips_from_partition


if __name__ == '__main__':

    generate_train_dev_chip_split()
