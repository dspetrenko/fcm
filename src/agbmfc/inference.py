import numpy as np
import onnxruntime

MISSED_S2_CHIP_ARRAY = np.zeros((11, 256, 256), dtype=np.float32)
MISSED_S2_CHIP_ARRAY[10] = 255

_ORT_SESSION = onnxruntime.InferenceSession(r"../models/baseline-model-05.onnx")


def onnx_inference(chip_tensor: np.ndarray, expected_batch_size: int = 10_000):
    # agbmfc.loading.chip_tensor_to_pixel_tensor
    pixel_tensor = np.transpose(chip_tensor, (2, 3, 0, 1)).reshape([65536, 12, 11])

    result = []
    first = 0
    last = expected_batch_size

    while first <= len(pixel_tensor):
        batch = pixel_tensor[first:last]

        real_batch_size = len(batch)
        padding_size = expected_batch_size - real_batch_size
        padding = np.zeros((padding_size, 12, 11), dtype=np.float32)
        batch = np.concatenate((batch, padding))
        ort_inputs = {'pixel_tensor': batch}
        batch_result = _ORT_SESSION.run(None, ort_inputs)[0]
        #         batch_result = np.random.randint(low=0, high=255, size=expected_batch_size)
        batch_result = batch_result[:real_batch_size] if padding_size else batch_result

        #         debug
        #         print(first, real_batch_size, len(batch_result))

        result.append(batch_result)

        first = last
        last += expected_batch_size

    result = np.concatenate(result, axis=None)

    return result.reshape(256, 256)
