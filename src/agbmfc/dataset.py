import glob
import os
from typing import Union

import torch


class PSSBatchDataset(torch.utils.data.Dataset):
    """ProcessedShuffledSampledDataset"""

    def __init__(self, data_dir, _limit: Union[int, None] = None):
        self.data_dir = data_dir
        self.features_files = sorted(glob.glob(os.path.join(self.data_dir, rf'batch-*-features.pt')))
        self.target_files = sorted(glob.glob(os.path.join(self.data_dir, rf'batch-*-target.pt')))
        self._limit = _limit

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        target_file = self.target_files[idx]

        target = torch.load(target_file)
        features = torch.load(feature_file)

        if self._limit is not None:
            target = target[:self._limit]
            features = features[:self._limit]

        return features, target

    def __len__(self):
        return len(self.features_files)
