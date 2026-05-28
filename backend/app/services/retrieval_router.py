from __future__ import annotations

import re


class RetrievalRouter:
    def route(self, query: str) -> dict[str, float | str]:
        lowered = query.lower()
        has_numbers = bool(re.search(r"\b(revenue|profit|cost|margin|growth|percent|year|202\d|q[1-4])\b", lowered))
        asks_summary = bool(re.search(r"\b(summary|summarize|overview|about|themes|purpose)\b", lowered))
        asks_fact = bool(re.search(r"\b(what|when|where|who|how much|list|name)\b", lowered))

        if has_numbers or asks_fact:
            return {"strategy": "factual", "vector_weight": 0.5, "lexical_weight": 0.5}
        if asks_summary:
            return {"strategy": "semantic", "vector_weight": 0.82, "lexical_weight": 0.18}
        return {"strategy": "complex_hybrid", "vector_weight": 0.68, "lexical_weight": 0.32}


retrieval_router = RetrievalRouter()
