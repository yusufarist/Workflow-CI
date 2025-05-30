FROM python:3.12.7-slim

WORKDIR /app

COPY MLProject/requirements.txt .
RUN pip install -r requirements.txt

COPY MLProject/ .

ENV MLFLOW_TRACKING_URI="http://localhost:5000"

CMD ["python", "modelling.py"]
