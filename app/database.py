"""
RoadGuard — Database Layer
---------------------------
SQLAlchemy ORM setup with PotholeReport and User models.
Clean, production schema without sample or mock seed data.
"""

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'roadsentinel.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite-specific
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class User(Base):
    """
    User account supporting local and Google OAuth authentication.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(150), nullable=False)
    avatar_url = Column(String(500), nullable=True)
    provider = Column(String(50), default="local")  # local | google
    password_hash = Column(String(255), nullable=True)  # optional for oauth
    role = Column(String(50), default="Citizen")  # Citizen | Municipality Admin
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
            "provider": self.provider,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PotholeReport(Base):
    """
    Represents a citizen-submitted pothole report.

    Lifecycle: Open → In Review → Resolved
    Severity:  Low | Medium | High | Critical
    """

    __tablename__ = "pothole_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Reporter info
    reporter_name = Column(String(120), nullable=True)
    reporter_email = Column(String(200), nullable=True)

    # Location
    location_description = Column(String(500), nullable=True)
    latitude = Column(Float, nullable=True)   # decimal degrees, WGS-84
    longitude = Column(Float, nullable=True)  # decimal degrees, WGS-84

    # Detection results
    image_filename = Column(String(300), nullable=False)
    annotated_image_url = Column(String(500), nullable=True)
    pothole_count = Column(Integer, default=0)
    overall_severity = Column(String(20), default="None")  # Low/Medium/High/Critical/None
    estimated_size_m = Column(String(50), default="~0.0 m")
    confidence = Column(Float, default=0.0)
    detection_json = Column(Text, nullable=True)  # full JSON from model

    # Triage / workflow
    status = Column(String(30), default="Open")      # Open | In Review | Resolved
    priority_score = Column(Float, default=0.0)       # computed 0-100
    notes = Column(Text, nullable=True)               # municipality notes

    # Timestamps
    submitted_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=True,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "reporter_name": self.reporter_name,
            "reporter_email": self.reporter_email,
            "location_description": self.location_description,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "image_filename": self.image_filename,
            "annotated_image_url": self.annotated_image_url,
            "pothole_count": self.pothole_count,
            "overall_severity": self.overall_severity,
            "estimated_size_m": self.estimated_size_m,
            "confidence": self.confidence,
            "status": self.status,
            "priority_score": self.priority_score,
            "notes": self.notes,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def init_db():
    """Create all tables cleanly without seeding any mock data."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session, closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
