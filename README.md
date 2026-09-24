# RoadSentinel — AI-Powered Pothole Intelligence Platform

> **A production-grade smart-city tool, not a toy detector.**
> RoadSentinel goes beyond bounding boxes: it grades damage severity, powers a citizen reporting portal, manages a municipality triage workflow, and surfaces live analytics — all served from a single FastAPI backend with a polished dark SPA frontend.

---

## Live Features

| Feature | Detail |
|---|---|
| 🔍 **AI Detection** | YOLOv8n fine-tuned on 665 annotated road images |
| 🎯 **Severity Grading** | Each pothole graded Low / Medium / High / Critical by bounding-box area |
| 📋 **Citizen Reporting Portal** | Submit GPS-tagged reports with photo; receive a tracking ID |
| 🏛️ **Municipality Triage** | Reports flow Open → In Review → Resolved via PATCH API |
| 📊 **Analytics Dashboard** | Live severity distribution chart, status counts, recent activity |
| ⬇️ **CSV Export** | Download filtered reports for offline GIS / civic use |
| 💅 **Glassmorphism UI** | Dark-themed SPA, Tailwind CSS, Canvas bar chart — zero JS frameworks |

---

## Architecture

```
Browser  ──── GET /  ────────────────────────►  FastAPI static SPA (index.html)
              │
              ├─ POST /predict                  YOLOv8 inference + severity scoring
              ├─ POST /predict/annotated         Annotated JPEG stream
              │
              ├─ POST /reports                  Submit report → DB + AI analysis
              ├─ GET  /reports                  List with status/severity filters
              ├─ GET  /reports/{id}             Single report + detections JSON
              ├─ PATCH /reports/{id}            Update status / municipality notes
              │
              ├─ GET  /analytics/summary        Dashboard aggregates
              └─ GET  /export/csv               CSV download (filterable)
                            │
                     SQLAlchemy ORM
                            │
                      SQLite  (roadsentinel.db)
                      Table: pothole_reports
                      Columns: id, reporter, location, lat/lng,
                               severity, pothole_count, status,
                               priority_score, notes, timestamps
```

> **Production upgrade path:** swap `roadsentinel.db` for PostgreSQL + PostGIS for spatial queries (`ST_Within`, heatmaps by ward). No other code changes needed — SQLAlchemy handles the difference.

---

## Model Performance

Trained on 665 annotated road images (YOLO format, 70/20/10 split), fine-tuning YOLOv8n from COCO weights.

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Validation | 0.76 | 0.44 | 0.80 | 0.65 |
| Test set | 0.70 | 0.44 | 0.70 | 0.64 |

Inference latency: ~20–35 ms/image on CPU.

---

## Severity Engine

The custom `severity.py` module grades each detected pothole by its **bounding-box area as a percentage of the total image**:

| Tier | Area % | Color | Meaning |
|---|---|---|---|
| 🟢 Low | < 2% | Green | Minor surface deformation — monitor |
| 🟡 Medium | 2–5% | Amber | Noticeable damage — schedule repair |
| 🔴 High | 5–10% | Red | Significant hazard — prioritise |
| 🟣 Critical | > 10% | Purple | Immediate danger — emergency repair |

A **priority score** (0–100) is also computed from `confidence × severity_weight` across all detections, enabling sorted municipality triage queues.

---

## Project Structure

```
pothole-project/
├── app/
│   ├── main.py            # FastAPI — 9 endpoints, CORS, static serving
│   ├── database.py        # SQLAlchemy ORM — PotholeReport model
│   ├── severity.py        # Severity scoring engine
│   └── static/
│       ├── index.html     # Full SPA frontend (4 panels)
│       └── annotated/     # Auto-saved annotated images (created at runtime)
├── dataset/               # YOLO-format images + labels + data.yaml
├── models/
│   └── best.pt            # Fine-tuned YOLOv8n checkpoint
├── samples/               # Sample road images
├── training_results/      # Loss curves, confusion matrix, sample predictions
├── pothole_detection.ipynb  # Dataset EDA, training, evaluation, inference
├── train.py               # Training script (configurable)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Train (optional — checkpoint already included)

```bash
python train.py --epochs 60 --imgsz 640 --batch 16 --device 0   # GPU
python train.py --epochs 20 --imgsz 384 --batch 8  --device cpu # CPU
```

---

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "version": "2.0.0", "model_loaded": true, "database_ok": true }
```

### `POST /predict`
Upload an image, get severity-graded detections:
```bash
curl -X POST http://localhost:8000/predict -F "file=@samples/pothole-in-road.jpg"
```
```json
{
  "filename": "pothole-in-road.jpg",
  "pothole_count": 1,
  "overall_severity": "High",
  "priority_score": 44.1,
  "detections": [
    {
      "class": "pothole", "confidence": 0.8817,
      "bbox_xyxy": [200.71, 204.44, 334.70, 293.50],
      "area_pct": 6.21, "severity": "High", "severity_color": "#ef4444"
    }
  ],
  "annotated_image_url": "/static/annotated/quick_a1b2c3d4.jpg",
  "inference_ms": 22.4
}
```

### `POST /reports`
Submit a citizen report with metadata:
```bash
curl -X POST "http://localhost:8000/reports?reporter_name=Rajan&location_description=MG+Road+Bengaluru&latitude=12.9716&longitude=77.5946" \
  -F "file=@pothole.jpg"
```

### `GET /reports`
```bash
curl "http://localhost:8000/reports?status=Open&severity=High&limit=20"
```

### `PATCH /reports/{id}`
Municipality triage update:
```bash
curl -X PATCH http://localhost:8000/reports/1 \
  -H "Content-Type: application/json" \
  -d '{"status": "In Review", "notes": "Repair crew dispatched for 24 Oct."}'
```

### `GET /analytics/summary`
Dashboard aggregates — total, by status, by severity, recent reports.

### `GET /export/csv`
```bash
curl "http://localhost:8000/export/csv?status=Open" -o open_reports.csv
```

---

## Tech Stack

`Python 3.10+` · `FastAPI` · `SQLAlchemy 2.0` · `SQLite` · `Ultralytics YOLOv8` · `OpenCV` · `PyTorch` · `TailwindCSS` · `Canvas API`

---

## CV Highlights

- End-to-end ML system with **custom feature engineering** (severity scoring, priority ranking)
- **Full REST API** with filtering, pagination, lifecycle management, and CSV export
- **Database-backed** citizen reporting with municipality workflow (Open → Resolved)
- **Production-ready patterns**: lazy model loading, ORM, CORS, static file serving, error handling
- **Zero-framework frontend** using only Tailwind CDN + vanilla JS + Canvas API
