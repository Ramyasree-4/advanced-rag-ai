from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.errors import GenerationError
from app.models.schemas import ChatRequest, ChatResponse
from app.services.generator import ResponseGenerator
from app.services.memory import memory

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    generator = ResponseGenerator()
    try:
        if request.stream:
            return StreamingResponse(
                generator.stream_answer(
                    request.session_id,
                    request.message,
                    request.document_ids,
                    request.filters,
                    request.debug,
                ),
                media_type="text/event-stream",
            )
        return await generator.answer(
            request.session_id,
            request.message,
            request.document_ids,
            request.filters,
            request.debug,
        )
    except GenerationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{session_id}/history")
def history(session_id: str):
    return memory.get(session_id)


@router.delete("/{session_id}/history")
def clear_history(session_id: str) -> dict[str, str]:
    memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
