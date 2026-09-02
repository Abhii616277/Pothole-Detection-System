"""
Pothole Detection API
----------------------
FastAPI backend that serves a YOLOv8 object-detection model fine-tuned
to detect potholes in road images.

Endpoints:
  GET  /health              -> service + model status
  POST /predict              -> upload an image, get back JSON detections
  POST /predict/annotated    -> upload an image, get back the image with
                                 bounding boxes drawn on it

Run locally:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = FastAPI(
    title="Pothole Detection API",
    description="Detects potholes in road images using a fine-tuned YOLOv8 model.",
    version="1.0.0",
)

_model: YOLO | None = None


def get_model() -> YOLO:
    """Lazy-load the model once and cache it in memory."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model weights not found at {MODEL_PATH}. "
                "Train the model first (see train.py) or point MODEL_PATH "
                "at an existing .pt checkpoint."
            )
        _model = YOLO(str(MODEL_PATH))
    return _model


def read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def validate_upload(file: UploadFile):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. "
            f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists()
    return {
        "status": "ok",
        "model_loaded": model_ready,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Return JSON detections: bounding boxes, confidence, class."""
    validate_upload(file)
    model = get_model()

    file_bytes = await file.read()
    img = read_image(file_bytes)

    t0 = time.time()
    results = model.predict(
        img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False
    )[0]
    latency_ms = round((time.time() - t0) * 1000, 1)

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "class": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )

    return JSONResponse(
        {
            "filename": file.filename,
            "pothole_count": len(detections),
            "detections": detections,
            "inference_ms": latency_ms,
        }
    )


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    """Return the same image with bounding boxes drawn, as a JPEG stream."""
    validate_upload(file)
    model = get_model()

    file_bytes = await file.read()
    img = read_image(file_bytes)

    results = model.predict(
        img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False
    )[0]
    annotated = results.plot()  # BGR numpy array with boxes drawn

    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
