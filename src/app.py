import os.path

from fastapi import FastAPI, Request, Response

DATA_PATH = os.path.join('..', 'data')

PROJECT_TITLE = "Project to solve problem of AGB estimation on satellite images"
PROJECT_DESC = "This project is a diploma work. "


app = FastAPI(title=PROJECT_TITLE, description=PROJECT_DESC)


@app.route('/echo')
async def hello(request: Request):
    body = await request.body()
    return Response(content=body, status_code=200)
