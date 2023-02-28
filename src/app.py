import os.path
from typing import List

from dotenv import load_dotenv
load_dotenv('.env')

from fastapi import FastAPI, Request, Response, Depends, UploadFile
from sqlalchemy.orm import Session

from src.service import crud, schemas
# from src.service.db import Base, engine, SessionLocal

from src.agbmfc.model import inference as inference_chip
from src.agbmfc.model import pickup_model

DATA_PATH = os.path.join('..', 'data')

PROJECT_TITLE = "Project to solve problem of AGB estimation on satellite images"
PROJECT_DESC = "This project is a diploma work. "

# Base.metadata.create_all(bind=engine)

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
    model = pickup_model()

    print({'filenames': [file.filename for file in chip_files]})

    prediction = inference_chip(model, None)
    return prediction
