FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install --no-cache-dir -e '.[models,media]'
ENTRYPOINT ["websleuth"]
