from __future__ import annotations

import logging
import json
import re
from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

from app.core.config import get_settings
from app.core.errors import GenerationError
from app.models.schemas import ChatMessage, ChatResponse
from app.services.memory import memory
from app.services.prompts import (
    ANSWER_EVALUATION_PROMPT,
    CONTEXT_COMPRESSION_PROMPT,
    QUERY_REWRITE_PROMPT,
    RAG_PROMPT,
)
from app.services.retriever import AdvancedRetriever
from app.services.metrics import metrics
from app.services.query_normalizer import query_normalizer

logger = logging.getLogger(__name__)


class ResponseGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = AdvancedRetriever()
        self.llm = ChatMistralAI(
            api_key=self.settings.mistral_api_key,
            model=self.settings.mistral_model,
            temperature=self.settings.mistral_temperature,
            max_tokens=self.settings.mistral_max_tokens,
            max_retries=3,
        )

    async def answer(self, session_id: str, question: str, document_ids: list[str] | None, filters: dict, debug: bool) -> ChatResponse:
        try:
            history = self._format_history(memory.get(session_id))
            rewritten_query = await self._rewrite_query(history, question)
            docs, citations, retrieval_debug = self.retriever.retrieve(rewritten_query, document_ids, filters)
            context = await self._compress_context(rewritten_query, docs)
            prompt = RAG_PROMPT.format_messages(history=history, context=context, question=question)
            result = await self.llm.ainvoke(prompt)
            answer_text = str(result.content)
            quality = await self._evaluate_answer(question, context, answer_text)
            self._record_metrics(session_id, rewritten_query, retrieval_debug, quality, len(answer_text))
            memory.add(session_id, ChatMessage(role="user", content=question))
            memory.add(session_id, ChatMessage(role="assistant", content=answer_text, citations=citations))
            return ChatResponse(
                session_id=session_id,
                answer=answer_text,
                citations=citations,
                rewritten_query=rewritten_query,
                retrieval_debug=retrieval_debug,
                quality_evaluation=quality,
                token_usage=getattr(result, "usage_metadata", {}) or {},
            )
        except Exception as exc:
            logger.exception("Generation failed")
            raise GenerationError(str(exc)) from exc

    async def stream_answer(
        self,
        session_id: str,
        question: str,
        document_ids: list[str] | None,
        filters: dict,
        debug: bool,
    ) -> AsyncIterator[str]:
        history = self._format_history(memory.get(session_id))
        rewritten_query = await self._rewrite_query(history, question)
        docs, citations, retrieval_debug = self.retriever.retrieve(rewritten_query, document_ids, filters)
        context = await self._compress_context(rewritten_query, docs)
        metadata = ChatResponse(
            session_id=session_id,
            answer="",
            citations=citations,
            rewritten_query=rewritten_query,
            retrieval_debug=retrieval_debug,
        )
        yield f"event: metadata\ndata: {metadata.model_dump_json()}\n\n"

        prompt = RAG_PROMPT.format_messages(history=history, context=context, question=question)
        answer_parts: list[str] = []
        async for chunk in self.llm.astream(prompt):
            token = str(chunk.content)
            answer_parts.append(token)
            yield f"event: token\ndata: {json.dumps(token)}\n\n"
        answer_text = "".join(answer_parts)
        quality = await self._evaluate_answer(question, context, answer_text)
        self._record_metrics(session_id, rewritten_query, retrieval_debug, quality, len(answer_text))
        yield f"event: quality\ndata: {json.dumps(quality)}\n\n"
        memory.add(session_id, ChatMessage(role="user", content=question))
        memory.add(session_id, ChatMessage(role="assistant", content=answer_text, citations=citations))
        yield "event: done\ndata: [DONE]\n\n"

    async def _rewrite_query(self, history: str, question: str) -> str:
        result = await self.llm.ainvoke(QUERY_REWRITE_PROMPT.format_messages(history=history, question=question))
        return query_normalizer.normalize(str(result.content), question)

    async def _compress_context(self, query: str, docs: list[Document]) -> str:
        raw_context = self._build_context(docs)
        if not raw_context:
            return ""
        result = await self.llm.ainvoke(CONTEXT_COMPRESSION_PROMPT.format_messages(query=query, chunks=raw_context))
        compressed = str(result.content).strip()
        return compressed or raw_context

    async def _evaluate_answer(self, question: str, context: str, answer: str) -> dict:
        try:
            result = await self.llm.ainvoke(
                ANSWER_EVALUATION_PROMPT.format_messages(question=question, context=context, answer=answer)
            )
            raw = str(result.content).strip()
            parsed = self._parse_quality_evaluation(raw)
            if parsed:
                return parsed
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                json_payload = json.loads(match.group(0))
                return {
                    "score": self._clamp_score(json_payload.get("score", 0)),
                    "reason": str(json_payload.get("reason", "")),
                }
            return self._heuristic_quality_score(question, context, answer, raw[:200] or "Evaluator returned an unparseable response")
        except Exception:
            logger.exception("Answer quality evaluation failed")
            return self._heuristic_quality_score(question, context, answer, "Evaluator unavailable; heuristic fallback used")

    def _build_context(self, docs) -> str:
        blocks: list[str] = []
        used_chars = 0
        for doc in docs:
            text = doc.page_content
            if used_chars + len(text) > self.settings.context_token_budget * 4:
                break
            blocks.append(
                f"Source: {doc.metadata.get('filename')} p.{doc.metadata.get('page')} "
                f"chunk {doc.metadata.get('chunk_id')}\n{text}"
            )
            used_chars += len(text)
        return "\n\n---\n\n".join(blocks)

    def _format_chunks_for_prompt(self, docs: list[Document]) -> str:
        blocks = []
        for doc in docs:
            blocks.append(
                f"chunk_id: {doc.metadata.get('chunk_id')}\n"
                f"source: {doc.metadata.get('filename')} p.{doc.metadata.get('page')}\n"
                f"text: {doc.page_content[:1200]}"
            )
        return "\n\n---\n\n".join(blocks)

    def _parse_quality_evaluation(self, text: str) -> dict | None:
        score_match = re.search(r"score\s*:\s*(10|[0-9])\s*/\s*10", text, re.IGNORECASE)
        reason_match = re.search(r"reason\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if not score_match:
            return None
        return {
            "score": self._clamp_score(score_match.group(1)),
            "grounded": self._extract_yes_no(text, "grounded"),
            "faithful": self._extract_yes_no(text, "faithful"),
            "complete": self._extract_yes_no(text, "complete"),
            "reason": reason_match.group(1).strip()[:240] if reason_match else "Evaluation completed",
        }

    def _extract_yes_no(self, text: str, label: str) -> str:
        match = re.search(rf"{label}\s*:\s*(yes|no)", text, re.IGNORECASE)
        return match.group(1).lower() if match else "unknown"

    def _clamp_score(self, value) -> int:
        try:
            score = int(float(value))
        except (TypeError, ValueError):
            score = 0
        return max(0, min(10, score))

    def _heuristic_quality_score(self, question: str, context: str, answer: str, reason: str) -> dict:
        if not context.strip() or not answer.strip():
            return {"score": 0, "reason": reason}
        if "not found in documents" in answer.lower():
            return {"score": 6, "grounded": "yes", "faithful": "yes", "complete": "partial", "reason": "Conservative not-found answer; " + reason}
        answer_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", answer) if len(term) > 3}
        context_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", context) if len(term) > 3}
        overlap = len(answer_terms & context_terms) / max(1, len(answer_terms))
        has_source = bool(re.search(r"\b(p\.|page|source|chunk)\b", answer, re.IGNORECASE))
        score = 5 + round(overlap * 3) + (1 if has_source else 0)
        return {
            "score": max(1, min(9, score)),
            "grounded": "yes" if overlap > 0.55 else "partial",
            "faithful": "unknown",
            "complete": "unknown",
            "reason": f"Heuristic fallback: answer overlap with context is {overlap:.0%}. {reason}",
        }

    def _record_metrics(
        self,
        session_id: str,
        rewritten_query: str,
        retrieval_debug: list[dict],
        quality: dict,
        answer_chars: int,
    ) -> None:
        metrics.record(
            {
                "session_id": session_id,
                "rewritten_query": rewritten_query,
                "quality_score": quality.get("score"),
                "quality_reason": quality.get("reason"),
                "answer_chars": answer_chars,
                "retrieval_stages": sorted({str(row.get("stage", "unknown")) for row in retrieval_debug}),
                "retrieval_coverage": len({str(row.get("chunk_id")) for row in retrieval_debug if row.get("chunk_id")}),
                "top_chunks": retrieval_debug[:6],
            }
        )

    def _format_history(self, messages: list[ChatMessage]) -> str:
        return "\n".join(f"{message.role}: {message.content}" for message in messages[-6:])
