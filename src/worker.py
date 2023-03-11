import io
import os
from typing import Literal

import torch
from celery import Celery

from src.agbmfc.model import pickup_model
from src.agbmfc.model import inference

celery_worker = Celery(__name__)
celery_worker.conf.broker_url = os.environ.get("CELERY_BROKER_URL", 'redis://localhost:6379')
celery_worker.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", 'redis://localhost:6379')


@celery_worker.task(name='inference')
def create_inference_task(image_tensor: list, model_type: Literal['trivial', 'baseline-pixel'] = 'trivial') -> list:

    # image_tensor: torch.Tensor = torch.load(o.BytesIO(image_bytes))
    image_tensor = torch.Tensor(image_tensor)
    model = pickup_model(model_type)
    prediction = inference(model, image_tensor)

    # buffer = io.BytesIO()
    # torch.save(prediction, buffer)
    # prediction_bytes = buffer.getvalue()

    return prediction.numpy().tolist()

