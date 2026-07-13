FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# CPU version of Paddle
RUN pip install --no-cache-dir paddlepaddle

# PaddleOCR
RUN pip install --no-cache-dir paddleocr

# FastAPI
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--root-path", "/paddleocr"]
