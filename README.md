# fcm
This project is implementation of the diploma HSE Master program

---


Тема: Применение методов машинного обучения для оценки количества биомассы по данным спутниковых снимков Sentinel-1 и Sentinel-2

Описание: Данные конкурса [The BioMassters - Competition](https://www.drivendata.org/competitions/99/biomass-estimation/page/536/)

---

# How to run it locally: 

```commandline
    cd src
    uvicorn app:app
```

You should receive message that uvicorn started up on localhost:8000

```commandline
INFO:     Started server process [8084]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

How to check:
    * visit localhost:8000/echo
    * send get request via curl ```curl  {"Hello, diploma!"} localhost:8000/echo```