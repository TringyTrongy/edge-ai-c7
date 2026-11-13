# M3 container for the detection + decision service.
# Written for a cold start on a Raspberry Pi (linux/arm64) or a PC. NOT built or run in this repo's session
# (no Docker daemon was available), so it is untested.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY m3/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY edge/ edge/
COPY m3/ m3/
COPY runs/synthetic_v1/weights/best.onnx runs/synthetic_v1/weights/best.onnx
# Start the two-stream pipeline; on a real device replace the image folders with /dev/video0.
CMD ["python", "m3/pipeline.py"]
