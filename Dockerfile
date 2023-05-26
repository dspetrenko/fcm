FROM python:3.10-slim

WORKDIR /app

COPY ./requirenments.txt /app/requirenments.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade -r /app/requirenments.txt

COPY ./ /app

RUN python startup.py

CMD ["uvicorn", "src.app:app", \
     "--host", "0.0.0.0", \
     "--port", "80"]