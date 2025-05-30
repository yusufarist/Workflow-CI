FROM python:3.12.7-slim

WORKDIR /app

COPY MLProject/requirements.txt .
RUN pip install -r requirements.txt

COPY MLProject/ .

CMD ["python", "modelling.py"]