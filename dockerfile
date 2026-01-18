FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY src/data/ ./data/

ENV PYTHONPATH=/app

CMD ["python", "-u", "src/models/train_rnn.py"]
