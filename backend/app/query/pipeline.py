"""
CodeLens — Query Pipeline (app/query/pipeline.py)

Orchestrates the full RAG query flow:
  Question → Retrieve (vector search) → Rerank (cross-encoder) → Generate (LLM) → Response.

Replaces: app/core/pipeline.py
"""

import asyncio
import time
from typing import Dict, Any

from app.retrieval.retriever import retrieve
from app.retrieval.reranker import rerank_chunks
from app.llm.generator import generate
from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.metrics import query_latency_seconds, queries_total

logger = get_logger(__name__)
settings = get_settings()


async def query_pipeline(
    question: str,
    repo_id: str,
    top_k: int = 5,
    where: dict | None = None,
    path_filter: str | None = None,
) -> Dict[str, Any]:
    """
    Execute the full RAG pipeline for a codebase question.

    Flow:
        1. Retrieve top-k chunks from ChromaDB via vector search
        2. Rerank with cross-encoder for precision relevance
        3. Generate answer with LLM (Groq/LLaMA3)
        4. Return answer with cited sources

    Args:
        question: Natural language question about the codebase.
        repo_id: Repository identifier.
        top_k: Number of final chunks to send to the LLM.

    Returns:
        Complete response with answer, citations, confidence, and metrics.
    """
    start_time = time.time()

    logger.info(
        "query_pipeline_start",
        repo_id=repo_id,
        question=question[:100],
        top_k=top_k,
        retrieve_k=settings.retriever_top_k,
    )

    try:
        # ── Step 1: Retrieve (async fanout — 4 searches in parallel) ────────
        retrieved_chunks = await retrieve(
            question=question,
            repo_id=repo_id,
            top_k=settings.retriever_top_k,
            where=where,
        )

        # ── Step 1b: Post-filter by path substring (if given) ─────────
        if path_filter and retrieved_chunks:
            filtered = [
                c for c in retrieved_chunks
                if path_filter.lower() in c.get("metadata", {}).get("file_path", "").lower()
            ]
            # Only apply filter if it leaves at least 1 result; otherwise skip
            if filtered:
                logger.info(
                    "path_filter_applied",
                    filter=path_filter,
                    before=len(retrieved_chunks),
                    after=len(filtered),
                )
                retrieved_chunks = filtered

        if not retrieved_chunks:
            queries_total.labels(repo_id=repo_id, status="no_results").inc()
            logger.warning("query_no_results", repo_id=repo_id, question=question[:80])
            return {
                "answer": (
                    "I couldn't find any relevant code in this repository to answer "
                    "your question. The repository might not be indexed yet, or the "
                    "question might not relate to the codebase content."
                ),
                "citations": [],
                "confidence_score": 0.0,
                "query": question,
                "repo_id": repo_id,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }

        # ── Step 2: Rerank ────────────────────────────────────────────
        reranked_chunks = rerank_chunks(
            question=question,
            chunks=retrieved_chunks,
            top_k=top_k,
        )

        # ── Step 3: Generate ──────────────────────────────────────────
        result = generate(
            question=question,
            chunks=reranked_chunks,
            repo_id=repo_id,
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)
        query_latency_seconds.observe(latency_ms / 1000)
        queries_total.labels(repo_id=repo_id, status="success").inc()

        # Confidence = average rerank score
        if reranked_chunks:
            avg_score = sum(c.get("rerank_score", 0) for c in reranked_chunks) / len(reranked_chunks)
            confidence = round(min(max(avg_score, 0.0), 1.0), 4)
        else:
            confidence = 0.0

        response = {
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence_score": confidence,
            "query": question,
            "repo_id": repo_id,
            "latency_ms": latency_ms,
        }

        logger.info(
            "query_pipeline_complete",
            repo_id=repo_id,
            latency_ms=latency_ms,
            citations=len(result["citations"]),
            confidence=confidence,
        )

        return response

    except Exception as e:
        queries_total.labels(repo_id=repo_id, status="error").inc()
        logger.error("query_pipeline_failed", repo_id=repo_id, error=str(e))
        raise
