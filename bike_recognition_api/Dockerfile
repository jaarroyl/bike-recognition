

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /bike_app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./model ./model

COPY ./app ./app

EXPOSE 8080

CMD ["python", "./app/main.py"]