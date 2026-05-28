from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock

from app.models.schemas import ChatMessage


class ConversationMemory:
    def __init__(self, max_messages: int = 12) -> None:
        self.max_messages = max_messages
        self._messages: dict[str, deque[ChatMessage]] = defaultdict(lambda: deque(maxlen=max_messages))
        self._lock = Lock()

    def add(self, session_id: str, message: ChatMessage) -> None:
        with self._lock:
            self._messages[session_id].append(message)

    def get(self, session_id: str) -> list[ChatMessage]:
        with self._lock:
            return list(self._messages[session_id])

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._messages.pop(session_id, None)


memory = ConversationMemory()
