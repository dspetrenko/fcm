import io
import os.path
from typing import List

from dotenv import load_dotenv
load_dotenv('.env')

from fastapi import FastAPI, Request, Response, Depends, UploadFile
from sqlalchemy.orm import Session

import torch
from rasterio.io import MemoryFile
from PIL import Image

from src.service import crud, schemas
from src.service.db import Base, engine, SessionLocal

from src.agbmfc.model import inference as inference_chip
from src.agbmfc.model import pickup_model
from src.agbmfc.loading import read_image_tensor

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


@app.route('/echo')
async def hello(request: Request):
    body = await request.body()
    return Response(content=body, status_code=200)


@app.get('/users/', response_model=List[schemas.User])
async def get_users(db: Session = Depends(get_db)):
    return crud.get_users(db)


@app.post(r'/agbmfc/inference')
async def inference(chip_files: list[UploadFile]):
    model = pickup_model('baseline-pixel')

    mem_files = [MemoryFile(await file_data.read()) for file_data in sorted(chip_files, key=lambda x: x.filename)]
    chip_tensors = [read_image_tensor(mf) for mf in mem_files]
    image_tensor = torch.stack(chip_tensors)

    prediction = inference_chip(model, image_tensor)

    img = Image.fromarray(prediction.numpy())
    with io.BytesIO() as buffer:
        img.save(buffer, format='tiff')
        img_bytes = buffer.getvalue()

    response = Response(img_bytes, media_type='image/tiff')
    return response
