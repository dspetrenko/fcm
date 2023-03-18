import io
import os.path
from typing import List, Literal

import numpy as np
from dotenv import load_dotenv
from starlette.responses import JSONResponse

from src import monitoring

load_dotenv('.env')

from fastapi import FastAPI, Request, Response, Depends, UploadFile
from sqlalchemy.orm import Session

import prometheus_client

import torch
from rasterio.io import MemoryFile
from PIL import Image

from src.service import crud, schemas
from src.service.db import Base, engine, SessionLocal

from src.agbmfc.model import inference as inference_chip
from src.agbmfc.model import pickup_model
from src.agbmfc.loading import read_image_tensor
from src.worker import celery_worker, create_inference_task

DATA_PATH = os.path.join('..', 'data')

PROJECT_TITLE = "Project to solve problem of AGB estimation on satellite images"
PROJECT_DESC = "This project is a diploma work. "


Base.metadata.create_all(bind=engine)

app = FastAPI(title=PROJECT_TITLE, description=PROJECT_DESC)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@monitoring.METRIC_STORAGE['echo_request_duration'].time()
@app.route('/echo')
async def hello(request: Request):
    body = await request.body()
    return Response(content=body, status_code=200)


@app.get('/users/', response_model=List[schemas.User])
async def get_users(db: Session = Depends(get_db)):
    return crud.get_users(db)


@app.post(r'/agbmfc/inference')
async def inference(chip_files: list[UploadFile], model_type: Literal['trivial', 'baseline-pixel'] = 'trivial'):

    mem_files = [MemoryFile(await file_data.read()) for file_data in sorted(chip_files, key=lambda x: x.filename)]
    chip_tensors = [read_image_tensor(mf) for mf in mem_files]
    image_tensor = torch.stack(chip_tensors)

    model = pickup_model(model_type)
    prediction = inference_chip(model, image_tensor)

    img = Image.fromarray(prediction.numpy())
    with io.BytesIO() as buffer:
        img.save(buffer, format='tiff')
        img_bytes = buffer.getvalue()

    response = Response(img_bytes, media_type='image/tiff')
    return response


@app.post('/agbmfc/inference/task', status_code=201)
async def inference_task(chip_files: list[UploadFile], model_type: Literal['trivial', 'baseline-pixel'] = 'trivial'):

    mem_files = [MemoryFile(await file_data.read()) for file_data in sorted(chip_files, key=lambda x: x.filename)]
    chip_tensors = [read_image_tensor(mf) for mf in mem_files]
    image_tensor = torch.stack(chip_tensors)

    # buffer = io.BytesIO()
    # torch.save(image_tensor, buffer)
    # image_bytes = buffer.getvalue()

    task = create_inference_task.delay(image_tensor.numpy().tolist(), model_type)

    monitoring.METRIC_STORAGE['created_inference_tasks'].inc()
    return JSONResponse({'task_id': task.id})


@app.get('/agbmfc/inference/result')
async def inference_result(task_id: str):

    monitoring.METRIC_STORAGE['requests_to_fetch_inference_result'].inc()

    result = celery_worker.AsyncResult(task_id)
    prediction = result.get()

    img = Image.fromarray(np.array(prediction))
    with io.BytesIO() as buffer:
        img.save(buffer, format='tiff')
        img_bytes = buffer.getvalue()

    response = Response(img_bytes, media_type='image/tiff')
    return response


@app.route('/metrics/')
def metrics(request: Request):
    res = prometheus_client.generate_latest()

    # app_res = []
    # for metric, value in monitoring.METRIC_STORAGE.items():
    #     app_res.append(prometheus_client.generate_latest(value))

    # return Response(res + b''.join(app_res))
    return Response(res)
