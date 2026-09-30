"""
RoadGuard — API Backend
------------------------
FastAPI backend powering the RoadGuard pothole intelligence & reporting platform.

Endpoints:
  GET  /health                  → service + model + DB status
  POST /auth/login              → authenticate user
  POST /auth/signup             → register new citizen user
  POST /auth/google             → reserved until Google ID token verification is configured
  POST /predict                 → detect potholes, return JSON with severity & dimensions
  POST /predict/annotated       → detect + return annotated JPEG stream
  POST /reports                 → submit citizen report (image + metadata + GPS)
  GET  /reports                 → list reports (filter by status/severity/search)
  GET  /reports/{id}            → single report with full detection JSON
  PATCH /reports/{id}           → update status/notes (municipality triage)
  GET  /analytics/summary       → live dashboard stats & distribution
  GET  /export/csv              → download reports as CSV

Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import logging
from dotenv import load_dotenv
import csv
import io
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
import onnxruntime as ort
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from app.severity import score_detections

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
from app.database import PotholeReport, User, get_db, init_db

MODEL_PATH = BASE_DIR / "models" / "best.onnx"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ANNOTATED_DIR = STATIC_DIR / "annotated"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 50
security = HTTPBearer()
logger = logging.getLogger(__name__)
DB_INIT_ERROR: Exception | None = None

# ---------------------------------------------------------------------------
#JWT CODE 
def create_access_token(user:User):
    expire = datetime.now(timezone.utc)+timedelta(
        minutes=JWT_EXPIRE_MINUTES
    )
    payload_id = {
        "Id" : str(user.id),
        "Email" : user.email,
        "Role" : user.role,
        "exp" : expire,
    }
    return jwt.encode(
        payload_id,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )
# Ensure directories exist
STATIC_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)
#-----------------------------------------------------------------------------------------
#JWT VALIDATION - VERIFY ACCESS TOKEN 
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )

        user_id = payload.get("Id")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token.",
            )

    except (JWTError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token.",
        )

    user = db.query(User).filter(User.id == int(user_id)).first()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User no longer exists.",
        )

    return user
# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RoadGuard API",
    description=(
        "AI-powered pothole intelligence & civic reporting platform. "
        "Detects, grades, maps, and tracks road hazards using YOLOv8."
    ),
    version="2.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend & images
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------
_model: ort.InferenceSession | None = None
_password_hasher = PasswordHasher()


def get_model() -> ort.InferenceSession:
    """Lazy-load and cache the CPU ONNX model."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"ONNX model not found at {MODEL_PATH}. "
                "Export models/best.pt to ONNX before deploying."
            )
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        _model = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    return _model


# ---------------------------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------------------------
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
            detail=f"Unsupported content type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )


async def read_upload(file: UploadFile) -> bytes:
    """Read an image while enforcing a payload limit below Vercel's body cap."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 4 MB or smaller.")
    return data


def save_annotated(img_bgr: np.ndarray, prefix: str = "") -> str:
    """Save annotated image to static/annotated/ and return relative URL."""
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    path = ANNOTATED_DIR / name
    cv2.imwrite(str(path), img_bgr)
    return f"/static/annotated/{name}"


def run_detection(img: np.ndarray) -> tuple[list[dict], np.ndarray, float]:
    """Run YOLO ONNX inference and NMS. Returns detections, image, latency."""
    model = get_model()
    img_h, img_w = img.shape[:2]
    input_size = 640
    scale = min(input_size / img_w, input_size / img_h)
    resized_w, resized_h = round(img_w * scale), round(img_h * scale)
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = input_size - resized_w, input_size - resized_h
    left, top = round(pad_w / 2 - 0.1), round(pad_h / 2 - 0.1)
    right, bottom = pad_w - left, pad_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    input_tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    t0 = time.time()
    output = model.run(None, {model.get_inputs()[0].name: input_tensor})[0]
    latency_ms = round((time.time() - t0) * 1000, 1)

    predictions = output[0].T
    scores = predictions[:, 4]
    candidate_indices = np.flatnonzero(scores >= CONF_THRESHOLD)
    boxes = []
    confidences = []
    coordinates = []
    for index in candidate_indices:
        cx, cy, width, height = predictions[index, :4]
        x1 = (cx - width / 2 - left) / scale
        y1 = (cy - height / 2 - top) / scale
        x2 = (cx + width / 2 - left) / scale
        y2 = (cy + height / 2 - top) / scale
        x1, x2 = np.clip([x1, x2], 0, img_w)
        y1, y2 = np.clip([y1, y2], 0, img_h)
        coordinates.append([float(x1), float(y1), float(x2), float(y2)])
        boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        confidences.append(float(scores[index]))

    kept = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)
    raw = []
    annotated = img.copy()
    for kept_index in np.asarray(kept).reshape(-1):
        x1, y1, x2, y2 = [round(value, 2) for value in coordinates[int(kept_index)]]
        confidence = round(confidences[int(kept_index)], 4)
        raw.append({"class": "pothole", "confidence": confidence, "bbox_xyxy": [x1, y1, x2, y2]})
        point1, point2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(annotated, point1, point2, (16, 185, 129), 2)
        cv2.putText(
            annotated, f"pothole {confidence:.2f}", (point1[0], max(point1[1] - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 185, 129), 2,
        )
    return raw, annotated, latency_ms


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ReportUpdateRequest(BaseModel):
    status: Optional[str] = None   # Open | In Review | Resolved
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    global DB_INIT_ERROR
    try:
        init_db()
        DB_INIT_ERROR = None
    except Exception as exc:
        # Keep non-database routes (including the frontend) available. The
        # exception and traceback remain available in Vercel function logs.
        DB_INIT_ERROR = exc
        logger.exception("Database initialization failed")


# ---------------------------------------------------------------------------
# Routes — System & Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health():
    from app.database import engine
    db_ok = DB_INIT_ERROR is None
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "app": "RoadGuard",
        "version": "2.5.0",
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "database_ok": db_ok,
        "database_init_error": type(DB_INIT_ERROR).__name__ if DB_INIT_ERROR else None,
    }


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class GoogleAuthRequest(BaseModel):
    credential: str


@app.post("/auth/login", tags=["Auth"])
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email = req.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()

    if not user or not user.password_hash:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    try:
        _password_hasher.verify(user.password_hash, req.password)
    except VerificationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    access_token = create_access_token(user)
    return {
    "status": "success",
    "message": "Logged in successfully",
    "access_token": access_token,
    "token_type": "bearer",
    "user": user.to_dict(),
}


@app.post("/auth/signup", tags=["Auth"])
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    name = req.name.strip()
    email = req.email.lower().strip()

    if not name:
        raise HTTPException(
            status_code=422,
            detail="Name is required."
        )

    if len(req.password) < 8:
        raise HTTPException(
            status_code=422,
            detail="Password must be at least 8 characters."
        )

    existing = db.query(User).filter(User.email == email).first()

    if existing:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    user = User(
        name=name,
        email=email,
        password_hash=_password_hasher.hash(req.password),
        provider="local",
        role="Citizen",
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    access_token = create_access_token(user)
    return {
    "status": "success",
    "message": "Account created successfully",
    "access_token": access_token,
    "token_type": "bearer",
    "user": user.to_dict(),
}


# ---------------------------------------------------------------------------
# Google authentication
# ---------------------------------------------------------------------------

@app.post("/auth/google", tags=["Auth"])
def google_auth(
    req: GoogleAuthRequest,
    db: Session = Depends(get_db)
):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="Google authentication is not configured on the server."
        )

    try:
        google_user = id_token.verify_oauth2_token(
            req.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired Google credential."
        )

    if google_user.get("iss") not in {
        "accounts.google.com",
        "https://accounts.google.com",
    }:
        raise HTTPException(
            status_code=401,
            detail="Invalid Google token issuer."
        )

    google_sub = google_user.get("sub")
    email = google_user.get("email")
    email_verified = google_user.get("email_verified", False)

    if not google_sub or not email:
        raise HTTPException(
            status_code=401,
            detail="Google account information is incomplete."
        )

    if not email_verified:
        raise HTTPException(
            status_code=401,
            detail="Google email address is not verified."
        )

    name = google_user.get("name") or email.split("@")[0]
    picture = google_user.get("picture")
    email = email.lower().strip()

    user = db.query(User).filter(User.email == email).first()

    if user:
        user.name = name
        user.avatar_url = picture
        user.provider = "google"
    else:
        user = User(
            name=name,
            email=email,
            avatar_url=picture,
            provider="google",
            password_hash=None,
            role="Citizen",
        )
        db.add(user)

    db.commit()
    db.refresh(user)

    return {
        "status": "success",
        "message": "Google sign-in successful",
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "user": user.to_dict(),
    }
# ---------------------------------------------------------------------------
# Routes — AI Detection
# ---------------------------------------------------------------------------
@app.post("/predict", tags=["Detection"])
async def predict(file: UploadFile = File(...)):
    """
    Detect potholes in an uploaded image.
    Returns JSON with detections, confidence %, estimated physical size, and severity.
    """
    validate_upload(file)
    file_bytes = await read_upload(file)
    img = read_image(file_bytes)
    h, w = img.shape[:2]

    raw, annotated, latency_ms = run_detection(img)
    severity_result = score_detections(raw, h, w)

    # Estimate physical dimensions based on perspective width ratio
    if severity_result.detections:
        max_det = max(severity_result.detections, key=lambda d: d.area_pct)
        x1, y1, x2, y2 = max_det.bbox_xyxy
        w_ratio = (x2 - x1) / max(w, 1)
        est_width = round(max(0.3, w_ratio * 3.2), 1)
        estimated_size = f"~ {est_width} m (width)"
        top_conf = max_det.confidence
    else:
        estimated_size = "~ 0.0 m"
        top_conf = 0.0

    annotated_url = save_annotated(annotated, prefix="quick")

    return JSONResponse(
        {
            "filename": file.filename,
            "image_dims": {"width": w, "height": h},
            "pothole_count": len(severity_result.detections),
            "overall_severity": severity_result.overall_severity,
            "confidence": round(top_conf * 100, 1),
            "estimated_size": estimated_size,
            "priority_score": severity_result.priority_score,
            "detections": [d.to_dict() for d in severity_result.detections],
            "annotated_image_url": annotated_url,
            "inference_ms": latency_ms,
        }
    )


@app.post("/predict/annotated", tags=["Detection"])
async def predict_annotated(file: UploadFile = File(...)):
    """Return the uploaded image with bounding boxes drawn as JPEG stream."""
    validate_upload(file)
    file_bytes = await read_upload(file)
    img = read_image(file_bytes)

    _, annotated, _ = run_detection(img)

    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Routes — Reports & Portal
# ---------------------------------------------------------------------------
@app.post("/reports", tags=["Reports"])
async def submit_report(
    file: UploadFile = File(...),
    reporter_name: Optional[str] = None,
    reporter_email: Optional[str] = None,
    location_description: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submit a citizen pothole report.
    Automatically executes YOLOv8 AI detection, grades severity, and stores GPS data.
    """
    validate_upload(file)
    file_bytes = await read_upload(file)
    img = read_image(file_bytes)
    h, w = img.shape[:2]

    # AI Detection
    raw, annotated, _ = run_detection(img)
    severity_result = score_detections(raw, h, w)

    # Estimate dimensions
    if severity_result.detections:
        max_det = max(severity_result.detections, key=lambda d: d.area_pct)
        x1, y1, x2, y2 = max_det.bbox_xyxy
        w_ratio = (x2 - x1) / max(w, 1)
        est_width = round(max(0.3, w_ratio * 3.2), 1)
        estimated_size = f"~ {est_width} m (width)"
        top_conf = max_det.confidence
    else:
        estimated_size = "~ 0.0 m"
        top_conf = 0.0

    annotated_url = save_annotated(annotated, prefix="report")

    orig_name = f"orig_{uuid.uuid4().hex[:8]}_{file.filename or 'pothole.jpg'}"
    orig_path = ANNOTATED_DIR / orig_name
    orig_path.write_bytes(file_bytes)

    report = PotholeReport(
        reporter_name=current_user.name,
        reporter_email=current_user.email,
        location_description=location_description,
        latitude=latitude,
        longitude=longitude,
        image_filename=orig_name,
        annotated_image_url=annotated_url,
        pothole_count=len(severity_result.detections),
        overall_severity=severity_result.overall_severity,
        estimated_size_m=estimated_size,
        confidence=top_conf,
        priority_score=severity_result.priority_score,
        detection_json=json.dumps([d.to_dict() for d in severity_result.detections]),
        status="Open",
        submitted_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return JSONResponse(
        {
            "message": "Report submitted successfully.",
            "report_id": report.id,
            "pothole_count": report.pothole_count,
            "overall_severity": report.overall_severity,
            "estimated_size": report.estimated_size_m,
            "confidence": round(report.confidence * 100, 1),
            "priority_score": report.priority_score,
            "annotated_image_url": report.annotated_image_url,
            "status": report.status,
        },
        status_code=201,
    )


@app.get("/reports", tags=["Reports"])
def list_reports(
    status: Optional[str] = Query(None, description="Filter: Open | In Review | Resolved"),
    severity: Optional[str] = Query(None, description="Filter: Low | Medium | High | Critical"),
    search: Optional[str] = Query(None, description="Search by location or road"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List reports with optional filters."""
    q = db.query(PotholeReport)
    if status:
        q = q.filter(PotholeReport.status == status)
    if severity:
        q = q.filter(PotholeReport.overall_severity == severity)
    if search:
        pattern = f"%{search.strip()}%"
        q = q.filter(PotholeReport.location_description.ilike(pattern))

    total = q.count()
    reports = q.order_by(PotholeReport.submitted_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "reports": [r.to_dict() for r in reports],
    }


@app.get("/reports/{report_id}", tags=["Reports"])
def get_report(report_id: int, db: Session = Depends(get_db)):
    """Get single report by ID."""
    report = db.query(PotholeReport).filter(PotholeReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail=f"Report #{report_id} not found.")
    data = report.to_dict()
    data["detections"] = json.loads(report.detection_json or "[]")
    return data


@app.patch("/reports/{report_id}", tags=["Reports"])
def update_report(
    report_id: int,
    body: ReportUpdateRequest,
    db: Session = Depends(get_db),
):
    """Update report status or municipality notes."""
    report = db.query(PotholeReport).filter(PotholeReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail=f"Report #{report_id} not found.")

    if body.status:
        report.status = body.status
    if body.notes is not None:
        report.notes = body.notes
    report.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(report)
    return report.to_dict()


# ---------------------------------------------------------------------------
# Routes — Analytics
# ---------------------------------------------------------------------------
@app.get("/analytics/summary", tags=["Analytics"])
def analytics_summary(db: Session = Depends(get_db)):
    """Dashboard statistics and metrics matching RoadGuard reference."""
    total = db.query(PotholeReport).count()
    open_count = db.query(PotholeReport).filter(PotholeReport.status == "Open").count()
    in_review = db.query(PotholeReport).filter(PotholeReport.status == "In Review").count()
    resolved = db.query(PotholeReport).filter(PotholeReport.status == "Resolved").count()

    by_severity = {
        sev: db.query(PotholeReport).filter(PotholeReport.overall_severity == sev).count()
        for sev in ["Low", "Medium", "High", "Critical"]
    }

    recent = (
        db.query(PotholeReport)
        .order_by(PotholeReport.submitted_at.desc())
        .limit(10)
        .all()
    )

    all_reports = db.query(PotholeReport).all()
    total_potholes = sum(r.pothole_count for r in all_reports) if all_reports else 0
    user_count = db.query(User).count()

    return {
        "total_potholes_detected": total_potholes,
        "reports_resolved": resolved,
        "under_review": in_review,
        "active_contributors": user_count,
        "by_status": {
            "Open": open_count,
            "In Review": in_review,
            "Resolved": resolved,
        },
        "by_severity": by_severity,
        "recent_reports": [r.to_dict() for r in recent],
    }


# ---------------------------------------------------------------------------
# Routes — Export
# ---------------------------------------------------------------------------
@app.get("/export/csv", tags=["Export"])
def export_csv(
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Download reports as CSV."""
    q = db.query(PotholeReport)
    if status:
        q = q.filter(PotholeReport.status == status)
    if severity:
        q = q.filter(PotholeReport.overall_severity == severity)
    reports = q.order_by(PotholeReport.submitted_at.desc()).all()

    output = io.StringIO()
    fields = [
        "id", "reporter_name", "reporter_email", "location_description",
        "latitude", "longitude", "pothole_count", "overall_severity",
        "estimated_size_m", "confidence", "status", "notes", "submitted_at"
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in reports:
        writer.writerow(r.to_dict())

    output.seek(0)
    filename = f"roadguard_reports_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.StringIO(output.getvalue()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Frontend Root
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "RoadGuard API is running. Visit /docs"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


#TESTING 
@app.get("/auth/me", tags=["Auth"])
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "status": "success",
        "user": current_user.to_dict(),
    }
