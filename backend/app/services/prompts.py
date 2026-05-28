from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise query optimization system for a RAG pipeline.

Rewrite the user query into a concise, information-rich search query that improves document retrieval.

Rules:
- Preserve intent
- Expand vague terms only when needed
- Keep the rewrite short, normally under 25 words
- Add missing context from conversation history when available
- If the user asks for a summary, overview, or "what is this document about", preserve that simple intent
- Make it suitable for semantic vector search
- Do NOT answer the question
- Do NOT invent document sections, chapters, claims, fields, or audiences

Return only the rewritten query.""",
        ),
        ("human", "History:\n{history}\n\nUser Query:\n{question}"),
    ]
)

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a relevance scoring system.

Given a user query and multiple retrieved document chunks, rank them by relevance.
Return the most relevant chunks with scores from 0 to 1.

Output format:
- chunk_id: score""",
        ),
        ("human", "User Query:\n{query}\n\nChunks:\n{chunks}\n\nReturn top {top_n} most relevant chunks."),
    ]
)

CONTEXT_COMPRESSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a context compression engine.

Compress document chunks into concise, information-dense notes.

Rules:
- Keep only facts relevant to the query
- Remove repetition
- Preserve numbers, names, dates, and metrics
- Keep source filename, page, and chunk_id citations intact
- Do not introduce facts that are not present""",
        ),
        ("human", "Query:\n{query}\n\nChunks:\n{chunks}"),
    ]
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a STRICT Retrieval-Augmented Generation (RAG) assistant.

You must answer the question using ONLY the provided context AND conversation history.

RULES (VERY IMPORTANT):
- Use ONLY the context and chat history below
- Do NOT use external knowledge
- Do NOT guess, assume, or hallucinate
- If the answer is not explicitly present in context or history, respond exactly:
  "Not found in documents"
- Ignore irrelevant or low-similarity chunks
- Do NOT mention embeddings, vectors, retrieval scores, or system behavior
- Be concise, factual, and grounded""",
        ),
        (
            "human",
            """========================
CONVERSATION HISTORY
========================
{history}

========================
CONTEXT
========================
{context}

========================
QUESTION
========================
{question}

========================
FINAL ANSWER
========================""",
        ),
    ]
)

ANSWER_EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Evaluate this RAG answer quality using only the supplied context.

Check:
- Groundedness: answer claims are supported by context
- Faithfulness: no unsupported additions
- Completeness: answers the question as fully as context allows
- Citation quality: cites filename/page when available

Return EXACTLY this format:
Score: X/10
Grounded: yes/no
Faithful: yes/no
Complete: yes/no
Reason: short reason

Rules:
- X must be an integer from 0 to 10
- Do not return JSON
- Do not add extra sections""",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
    ]
)
