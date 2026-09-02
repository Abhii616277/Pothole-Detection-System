# Pothole Detection System

An end-to-end backend ML project: a YOLOv8 object-detection model fine-tuned
to spot potholes in road images, served through a FastAPI backend with JSON
and annotated-image endpoints.

## Architecture

```
Image upload → FastAPI (/predict, /predict/annotated)
                    ↓
            YOLOv8n (fine-tuned) inference
                    ↓
       JSON detections  |  Annotated JPEG output
```

## Results

Trained on a 665-image labeled pothole dataset (YOLO format, 70/20/10
train/val/test split), fine-tuning YOLOv8n (pretrained on COCO) for 20
epochs at 384px on CPU.

| Split | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|----------|-----------|--------|
| Validation (final epoch) | 0.76 | 0.44 | 0.80 | 0.65 |
| Held-out test set        | 0.70 | 0.44 | 0.70 | 0.64 |

Inference latency: ~20-35ms/image on CPU (single core).

**Honest caveats:** this was trained for 20 epochs on a single CPU core in a
sandboxed environment as a portfolio/demo run, not a production run. The
`train.py` script defaults to 60 epochs at 640px, which is what you'd run
with a GPU for a stronger model — I've included the shorter run's actual
results above rather than projected numbers.

## Dataset

665 images of road potholes, originally annotated by Atikur Rahman Chitholian
as part of academic work, widely used as a benchmark pothole-detection set.
YOLO-format bounding box labels, one class (`pothole`).

- `dataset/images/{train,valid,test}` — 465 / 133 / 67 images
- `dataset/labels/{train,valid,test}` — matching YOLO `.txt` labels
- `dataset/data.yaml` — class + path config for Ultralytics

## Project structure

```
pothole-project/
├── app/
│   └── main.py               # FastAPI backend (health, predict, predict/annotated)
├── dataset/                  # YOLO-format images + labels + data.yaml
├── models/
│   └── best.pt                # Fine-tuned model checkpoint (this run's best weights)
├── samples/                    # Sample road images + one annotated example
├── training_results/            # Loss curves, confusion matrix, prediction grids
├── pothole_detection.ipynb       # Notebook: dataset EDA, training, evaluation, inference
├── train.py                      # Standalone training script (configurable epochs/imgsz)
├── requirements.txt
└── README.md
```

`pothole_detection.ipynb` is the data-science companion to the API: it walks
through dataset stats, ground-truth visualization, the training run, the
loss/mAP curves, test-set evaluation, and inference on sample images — all
with real output already saved in the notebook, so it renders fully on
GitHub without needing to re-run anything.

## Setup

```bash
pip install -r requirements.txt
```

## Train (optional — a trained checkpoint is already included)

```bash
python train.py --epochs 60 --imgsz 640 --batch 16 --device 0   # GPU
python train.py --epochs 20 --imgsz 384 --batch 8  --device cpu # CPU (slower)
```

This saves the best checkpoint to `models/best.pt`, which the API loads.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

**`GET /health`** — service + model status.

**`POST /predict`** — upload an image, get JSON back:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/pothole-in-road.jpg;type=image/jpeg"
```

```json
{
  "filename": "pothole-in-road.jpg",
  "pothole_count": 1,
  "detections": [
    {"class": "pothole", "confidence": 0.8817, "bbox_xyxy": [200.71, 204.44, 334.70, 293.50]}
  ],
  "inference_ms": 22.4
}
```

**`POST /predict/annotated`** — same input, returns the image with boxes drawn
(JPEG stream):

```bash
curl -X POST http://localhost:8000/predict/annotated \
  -F "file=@samples/pothole-in-road.jpg;type=image/jpeg" \
  --output annotated.jpg
```

## Tech stack

Python, PyTorch, Ultralytics YOLOv8, FastAPI, Uvicorn, OpenCV.

## Possible extensions (not implemented here)

- Swap the in-memory model cache for a model-serving layer (TorchServe /
  Triton) if scaling beyond one process.
- Add a `/predict/batch` endpoint for multiple images per request.
- Persist detections to a database (e.g. Postgres + PostGIS) with GPS
  coordinates for a real road-maintenance use case.
- Containerize with Docker for deployment.
