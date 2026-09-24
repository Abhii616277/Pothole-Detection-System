"""
RoadSentinel — Severity Scoring Engine
----------------------------------------
Grades each detected pothole and the overall frame from Low → Critical
based on bounding-box area as a percentage of the image + confidence.

Severity tiers (by area percentage of total image):
    Low      : < 2 %   → minor surface deformation, monitor
    Medium   : 2–5 %   → noticeable damage, schedule repair
    High     : 5–10%   → significant hazard, prioritise
    Critical : > 10%   → immediate danger, emergency repair

A priority score (0–100) is also computed for ranking reports in the
municipality triage queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]

# (area_pct_threshold, label, hex_color, priority_weight)
SEVERITY_CONFIG = [
    (2.0,  "Low",      "#22c55e", 1.0),   # green
    (5.0,  "Medium",   "#f59e0b", 2.5),   # amber
    (10.0, "High",     "#ef4444", 5.0),   # red
    (100.0,"Critical", "#7c3aed", 10.0),  # purple
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ScoredDetection:
    cls: str
    confidence: float
    bbox_xyxy: List[float]
    area_pct: float          # bbox area / image area × 100
    severity: str            # Low | Medium | High | Critical
    severity_color: str      # hex colour for UI
    priority_weight: float   # multiplier used in priority_score

    def to_dict(self) -> dict:
        return {
            "class": self.cls,
            "confidence": self.confidence,
            "bbox_xyxy": self.bbox_xyxy,
            "area_pct": round(self.area_pct, 3),
            "severity": self.severity,
            "severity_color": self.severity_color,
        }


@dataclass
class SeverityResult:
    detections: List[ScoredDetection]
    overall_severity: str    # max severity across all detections
    priority_score: float    # 0–100, for triage ordering


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def _classify_area(area_pct: float) -> tuple[str, str, float]:
    """Return (severity_label, hex_color, priority_weight) for a given area %."""
    for threshold, label, color, weight in SEVERITY_CONFIG:
        if area_pct < threshold:
            return label, color, weight
    # Fallback (area_pct >= 100 is physically impossible but guard anyway)
    return "Critical", "#7c3aed", 10.0


def score_detections(
    raw_detections: list[dict],
    img_h: int,
    img_w: int,
) -> SeverityResult:
    """
    Args:
        raw_detections: list of dicts with keys
            {"class", "confidence", "bbox_xyxy": [x1,y1,x2,y2]}
        img_h: image height in pixels
        img_w: image width in pixels

    Returns:
        SeverityResult with per-detection scores and overall metrics.
    """
    img_area = img_h * img_w if (img_h * img_w) > 0 else 1

    scored: List[ScoredDetection] = []
    for d in raw_detections:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        bbox_area = max(0.0, (x2 - x1) * (y2 - y1))
        area_pct = (bbox_area / img_area) * 100.0

        label, color, weight = _classify_area(area_pct)

        scored.append(
            ScoredDetection(
                cls=d["class"],
                confidence=d["confidence"],
                bbox_xyxy=d["bbox_xyxy"],
                area_pct=area_pct,
                severity=label,
                severity_color=color,
                priority_weight=weight,
            )
        )

    # Overall severity = highest tier present
    if not scored:
        overall = "None"
        priority = 0.0
    else:
        tier_order = {lvl: i for i, lvl in enumerate(["None"] + SEVERITY_LEVELS)}
        overall = max(
            (s.severity for s in scored),
            key=lambda lv: tier_order.get(lv, 0),
        )

        # Priority score: sum of (confidence × weight) normalised to 0–100
        raw_priority = sum(s.confidence * s.priority_weight for s in scored)
        # Cap at 100 — a single Critical detection at conf=1.0 gives weight=10
        priority = min(round(raw_priority * 10, 1), 100.0)

    return SeverityResult(
        detections=scored,
        overall_severity=overall,
        priority_score=priority,
    )


def severity_rank(label: str) -> int:
    """Numeric rank for sorting (higher = worse). Returns 0 for 'None'."""
    return (["None"] + SEVERITY_LEVELS).index(label) if label in (["None"] + SEVERITY_LEVELS) else 0
