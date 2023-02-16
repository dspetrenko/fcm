import glob
import os

from typing import List

import rasterio
import torch
import numpy as np

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
