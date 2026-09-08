FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY main.py model.py advice.py ./
# int8-quantized CustomCNN (~26 MB). Regenerate with export_onnx.py.
COPY cassava_customcnn.int8.onnx ./

ENV CASSAVA_MODEL_PATH=/app/cassava_customcnn.int8.onnx
ENV CASSAVA_ORT_THREADS=1

EXPOSE 8000

# Render and other hosts set PORT; default 8000 for local docker run.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
