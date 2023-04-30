import os
from typing import Literal

import numpy as np
from celery import Celery

from src.agbmfc.inference import onnx_inference

celery_worker = Celery(__name__)
celery_worker.conf.broker_url = os.environ.get("CELERY_BROKER_URL", 'redis://localhost:6379')
celery_worker.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", 'redis://localhost:6379')


@celery_worker.task(name='inference')
def create_inference_task(image_tensor: list, model_type: Literal['trivial', 'baseline-pixel'] = 'trivial') -> list:

    image_tensor = np.array(image_tensor, dtype=np.float32)
    prediction = onnx_inference(image_tensor)

    return prediction.tolist()

