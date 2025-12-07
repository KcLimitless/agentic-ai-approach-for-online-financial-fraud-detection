FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

CMD ["python", "main.py"]