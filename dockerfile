FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY saved_model ./saved_model
COPY App.py .

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

EXPOSE 5000

CMD ["python", "App.py"]
