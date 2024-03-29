# fcm
This project is implementation of the diploma HSE Master program

---


Тема: Применение методов машинного обучения для оценки количества биомассы по данным спутниковых снимков Sentinel-1 и Sentinel-2

Описание: Данные конкурса [The BioMassters - Competition](https://www.drivendata.org/competitions/99/biomass-estimation/page/536/)

---

# How to run it locally: 

```commandline
    uvicorn src.app:app
```

You should receive message that uvicorn started up on localhost:8000

```commandline
INFO:     Started server process [8084]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---
# How to run it in docker

```commandline
    docker build --tag agbm:latest .
    docker run -p 8000:80 -d --rm --name agbm agbm:latest
```


## How to check:
* visit localhost:8000/echo or localhost:8000/docs# 
* send get request via curl ```curl  {"Hello, diploma!"} localhost:8000/echo```

---
# How to run using docker compose

1. create `.env` file. See `.env_example` for a reference
2. build and run services:
```commandline
    docker compose build
    docker compose up
``` 
## How to check:
Simplest way is to use `curl` (Or you can visit a docs page `localhost:8000/docs#`)
```commandline
    curl localhost:8000/users/ -H 'accept: application/json'
``` 

**Note**: keep in mind that if you run it first time user table will be empty

# How to check if creation inference task works properly

1. init `/data_sample` folder (To avoid monkey job we will use the simplest scripts)
```commandline
$ python -m src.utils.init_data_samples
```
2. create a task for inference
```commandline
$ python -m src.utils.client_inference_task
files:
         fed6bb57_S1_00.tif
         fed6bb57_S1_01.tif
         fed6bb57_S1_02.tif
         fed6bb57_S1_03.tif
         fed6bb57_S1_04.tif
         fed6bb57_S1_05.tif
         fed6bb57_S1_06.tif
         fed6bb57_S1_07.tif
         fed6bb57_S1_08.tif
         fed6bb57_S1_09.tif
         fed6bb57_S1_10.tif
         fed6bb57_S1_11.tif
--------------------------------------------------
fb38b477-1add-481e-9bc4-243f4c7f28a1
```
3. get result from task (keep attention: we will use task id printed in the last line)
```commandline
$ python -m src.utils.client_inference_result -i fb38b477-1add-481e-9bc4-243f4c7f28a1
response code: 200 - response content saved to response.tif
```
You should find a `response.tif` file in your folder. (Keep calm. I's ok if you see nothing in `response.tif`. I just switched off model to save some resources)

4. bonus: There is a flower in the docker-compose.yml. So, you could visit [localhost:5555](http://localhost:5555)

# Monitoring
There are all needed stuff in the `docker-compose.yaml` to set up monitoring of the service: Prometheus, exporters, Grafana. 
Followed objects in the service were added for monitoring:
1. application 
2. celery (via flower)
3. celery via celery-exporter
4. postgresql via postgresql-exporter
5. host via host-exporter (Be carefully, if you are using windows on your host you will face fail. So the best way is to ~~migrate to linux~~ drop `node-exporter` from `docker-compose.yml` )

## How to check monitoring

1. run service in the docker compose manner:
```commandline
docker compose up --build
```
2. check if every exporter ready. Every of the listed uri should return you current status:
   1. http://localhost:8000/metrics/
   2. http://localhost:5555/metrics
   3. http://localhost:9808/metrics
   4. http://localhost:9187/metrics

3. check if Prometheus scrape every of exports above. Visit http://localhost:9090/targets. Every target should be healthy.
4. check if Grafana successfully connect to Prometheus. Visit http http://localhost:3000/dashboards. A few dashboards should be found: 
   1. Application status - just one chart to proof that we successfully pick up metrics from our app
   2. Celery / Tasks / By Task - show details about Celery. It could be empty if you haven't created inference task yet. (If it happened just pass `How to check if creation inference task works properly` section)
   3. Docker Prometheus Monitoring - in progress now. It should show docker container metrics, but now just count how many exporters do we have
   4. PostgresSQL Database - monitor Postgres. It works fine, but need a few minutes to load metrics.
   