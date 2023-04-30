import os.path
from pathlib import Path
import gc

import numpy as np
import onnxruntime
import rasterio

MISSED_S2_CHIP_ARRAY = np.zeros((11, 256, 256), dtype=np.float32)
MISSED_S2_CHIP_ARRAY[10] = 255

_file_path = Path(os.path.abspath(__file__))
_project_root = _file_path.parent.parent.parent
_model_path = _project_root / 'models' / 'baseline-model-05.onnx'


def onnx_inference(chip_tensor: np.ndarray, expected_batch_size: int = 10_000) -> np.ndarray:
    # agbmfc.loading.chip_tensor_to_pixel_tensor
    pixel_tensor = np.transpose(chip_tensor, (2, 3, 0, 1)).reshape([65536, 12, 11])

    result = []
    first = 0
    last = expected_batch_size

    while first <= len(pixel_tensor):

        # there is a reason why we create _ORT_SESSION for every batch:
        # w/o this hack worker failed and inference impossible
        _ORT_SESSION = onnxruntime.InferenceSession(_model_path.as_posix())
        batch = pixel_tensor[first:last]

        real_batch_size = len(batch)
        padding_size = expected_batch_size - real_batch_size
        padding = np.zeros((padding_size, 12, 11), dtype=np.float32)
        batch = np.concatenate((batch, padding))
        ort_inputs = {'pixel_tensor': batch}

        batch_result = _ORT_SESSION.run(None, ort_inputs)[0]
        batch_result = batch_result[:real_batch_size] if padding_size else batch_result
        result.append(batch_result)

        first = last
        last += expected_batch_size

        del _ORT_SESSION
        del batch
        del ort_inputs
        _ = gc.collect()

    result = np.concatenate(result, axis=None)

    return result.reshape(256, 256)


def read_image_as_array(file_path: str):

    numpy_type = np.float32

    with rasterio.open(file_path) as fd:
        image_array = fd.read().astype(numpy_type)
    return image_array
