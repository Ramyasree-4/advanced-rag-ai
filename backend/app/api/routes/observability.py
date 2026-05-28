from __future__ import annotations

from fastapi import APIRouter

from app.services.metrics import metrics

router = APIRouter()


@router.get("/summary")
def observability_summary():
    return metrics.snapshot()
