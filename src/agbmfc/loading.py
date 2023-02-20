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


def get_chip_tensor(chip_files: List[str], precision: int = 32, device: str = 'cpu'):
    if len(chip_files) != 12:
        # ds: for s2 it will be big problem: s2 contain a lot of missed data
        raise ValueError(f'found exactly 12 files, but got: {len(chip_files)}')

    # we use sorted to produce order of our images, because we checked that name defines position
    chip_tensors = [read_image_tensor(path, precision=precision, device=device) for path in sorted(chip_files)]
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
    expected_shape = torch.Size([12, 4, 256, 256])
    if chip.shape != expected_shape:
        raise ValueError(f'expected shape is {expected_shape}, but got {chip.shape}')

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


def get_batch(batch_chips, samples_from_chip: int = 1_000, target: bool = True):
    batch_data = []
    batch_data_target = []
    # ds: its slow enough : 20 secs for 100 chips : Could read and build pixel_tensors as multiprocessing task
    for chip in batch_chips:
        chip_files = get_chip_files(chip, s2=False)
        chip_tensor = get_chip_tensor(chip_files)

        pixel_tensor = chip_tensor_to_pixel_tensor(chip_tensor)
        shufled_indexes = torch.randperm(len(pixel_tensor))
        batch_data.append(pixel_tensor[shufled_indexes[:samples_from_chip]])

        if target:
            target_pixel_tensor = get_pixel_target_tensor(chip)
            batch_data_target.append(target_pixel_tensor[shufled_indexes[:samples_from_chip]])

    batch = torch.stack(batch_data)
    batch = batch.flatten(start_dim=0, end_dim=1)

    if not target:
        return batch, None

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
