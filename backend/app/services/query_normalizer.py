from __future__ import annotations

import re


class QueryNormalizer:
    GENERIC_TERMS = {
        "comprehensive",
        "detailed",
        "analysis",
        "include",
        "provide",
        "information",
        "specific",
        "document",
        "documents",
        "contents",
        "covered",
        "intended",
        "audience",
        "field",
        "study",
    }

    def normalize(self, query: str, fallback: str) -> str:
        cleaned = re.sub(r"[\n\r\t]+", " ", query).strip().strip("\"'")
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", cleaned)
        useful = [token for token in tokens if token.lower() not in self.GENERIC_TERMS]
        normalized = " ".join(useful[:18]).strip()
        if len(normalized.split()) < 3:
            normalized = fallback.strip()
        return normalized[:240]


query_normalizer = QueryNormalizer()
