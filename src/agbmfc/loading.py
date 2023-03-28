import glob
import os

from typing import List, Literal

import tqdm
import rasterio
import torch
import numpy as np
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


DATA_ROOT = r'/data/driven_data_bio_massters'
DATA_TARGET_ROOT = os.path.join(DATA_ROOT, 'train_agbm')
DATA_FEATURES_ROOT = os.path.join(DATA_ROOT, 'train_features')

MISSED_S2_CHIP_TENSOR = torch.zeros([11, 256, 256])
MISSED_S2_CHIP_TENSOR[10] = 255  # broadcasts! broadcasts are everywhere!

S1_EXPECTED_SHAPE = torch.Size([12, 4, 256, 256])
S2_EXPECTED_SHAPE = torch.Size([12, 11, 256, 256])


def read_image_tensor(file_path: str, precision: int = 32, device: str = 'cpu'):

    if precision == 32:
        numpy_type = np.float32
        torch_type = torch.float32
    elif precision == 16:
        numpy_type = np.float16
        torch_type = torch.float16
    else:
        raise ValueError(f'got unknown precision: {precision}')

    with rasterio.open(file_path) as fd:
        # ds: is it ok to keep it as float16? usually it reads as float32
        image_tensor = torch.tensor(fd.read().astype(numpy_type), dtype=torch_type,
                                    requires_grad=False, device=device)

    return image_tensor


def get_chip_tensor(chip_files: List[str], precision: int = 32, device: str = 'cpu', restore_missed=True):
    chip_files = sorted(chip_files)
    # we use sorted to produce order of our images, because we checked that name defines position
    chip_tensors = [read_image_tensor(path, precision=precision, device=device) for path in chip_files]

    if restore_missed:
        # missed_id_files = []

        for season_idx in range(0, 12):
            season_mask = f'_{season_idx:02}.'
            #     print(season_mask)
            chip_file = chip_files[season_idx]
            if season_mask not in chip_file:
                # print(season_mask, 'was missed')
                missed_chip_file = chip_files[0][:-7] + season_mask + chip_files[0][-3:]
                chip_files.insert(season_idx, missed_chip_file)
                # missed_id_files.append((season_idx, missed_chip_file))

                chip_tensors.insert(season_idx, MISSED_S2_CHIP_TENSOR)

    return torch.stack(chip_tensors)


def get_chip_files(chip: str, s1: bool = True, s2: bool = True):
    if not (s1 or s2):
        raise ValueError('s1 and s2 are False. At lease one of them must be True')

    chip_files = sorted(glob.glob(DATA_FEATURES_ROOT + f'/*{chip}*'))
    if s1 and s2:
        return chip_files

    if s1:
        return [f for f in chip_files if 'S1' in f]
    if s2:
        return [f for f in chip_files if 'S2' in f]


def chip_tensor_to_pixel_tensor(chip: torch.tensor):

    if not(chip.shape == S1_EXPECTED_SHAPE or chip.shape == S2_EXPECTED_SHAPE):
        raise ValueError(f'expected shape is {S1_EXPECTED_SHAPE} or {S2_EXPECTED_SHAPE}, but got {chip.shape}')

    chip = torch.permute(chip, (2, 3, 0, 1))
    chip = chip.flatten(start_dim=0, end_dim=1)
    return chip


# processing.target
def get_pixel_target_tensor(chip: str):
    file = os.path.join(DATA_TARGET_ROOT, f'{chip}_agbm.tif')
    image = read_image_tensor(file)

    # pixel_target_tensor
    pixel_target_tensor = torch.permute(image, (1, 2, 0))
    pixel_target_tensor = pixel_target_tensor.flatten(start_dim=0, end_dim=1)

    return pixel_target_tensor


def select_pixel_indexes(pixel_tensor, n_samples: int = 1_000, stratify=True, step: int = 100, step_count: int = 10):

    if not stratify:
        return torch.randperm(len(pixel_tensor))[:n_samples]

    fl = pixel_tensor.flatten()

    bins = [i * step for i in range(step_count + 1)]
    bins.append(np.inf)

    boundaries = []
    for i in range(len(bins) - 1):
        boundaries.append((bins[i], bins[i + 1]))

    counts, _ = np.histogram(fl, bins)
    n_samples_per_bin = (counts * n_samples / len(fl)).astype(int)

    # TODO: we need to add missed indexes to support n_samples per batch

    index_buffer = []
    for bin_idx, (left_boundary, right_boundary) in enumerate(boundaries):
        n_samples_in_bin = n_samples_per_bin[bin_idx]
        if n_samples_in_bin == 0:
            continue
        bin_mask = (left_boundary <= fl) * (fl < right_boundary)
        bin_indexes = torch.nonzero(bin_mask).flatten()

        selected_idxs = torch.randperm(len(bin_indexes))[:n_samples_in_bin]
        index_buffer.extend(selected_idxs)

    return torch.stack(index_buffer)


def get_batch(batch_chips, samples_from_chip: int = 1_000):
    batch_data = []
    batch_data_target = []
    # ds: its slow enough : 20 secs for 100 chips : Could read and build pixel_tensors as multiprocessing task
    for chip in batch_chips:
        chip_files = get_chip_files(chip, s1=False)
        chip_tensor = get_chip_tensor(chip_files)

        pixel_tensor = chip_tensor_to_pixel_tensor(chip_tensor)
        target_pixel_tensor = get_pixel_target_tensor(chip)
        indexes = select_pixel_indexes(target_pixel_tensor, samples_from_chip)
        batch_data.append(pixel_tensor[indexes])

        batch_data_target.append(target_pixel_tensor[indexes])

    batch = torch.stack(batch_data)
    batch = batch.flatten(start_dim=0, end_dim=1)

    batch_target = torch.stack(batch_data_target)
    batch_target = batch_target.flatten(start_dim=0, end_dim=1)

    # TODO: ds: lock random state!
    return batch, batch_target


def generate_processed_files(chips: List[str], split: Literal['train', 'val'],
                             chip_batch_size: int = 100, samples_from_chip: int = 1_000):

    batches = []

    length = len(chips)
    for i in range(0, length, chip_batch_size):
        _batch = chips[i: min(i + chip_batch_size, length)]
        batches.append(_batch)

    def gen_batch_file(batch_chips, idx):

        batch, batch_target = get_batch(batch_chips, samples_from_chip=1_000)

        torch.save(batch, rf'../data/processed/{split}/batch-{idx:06}-features.pt')
        torch.save(batch_target, rf'../data/processed/{split}/batch-{idx:06}-target.pt')

    task_gen = ((_chips, _idx) for _chips, _idx in zip(batches, range(len(batches))))
    _ = Parallel(backend='loky', n_jobs=-1)(
        delayed(gen_batch_file)(_chips, _idx) for (_chips, _idx) in tqdm.auto.tqdm(task_gen, total=len(batches)))
