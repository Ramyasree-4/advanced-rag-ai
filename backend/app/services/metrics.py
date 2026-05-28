from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import Lock
from typing import Any


class ObservabilityMetrics:
    def __init__(self, max_events: int = 100) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = Lock()

    def record(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.appendleft({"created_at": datetime.utcnow().isoformat(), **event})

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
        scores = [event.get("quality_score", 0) for event in events if event.get("quality_score") is not None]
        return {
            "total_queries": len(events),
            "average_quality_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "latest_events": events[:20],
        }


metrics = ObservabilityMetrics()
