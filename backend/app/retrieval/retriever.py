"""
CodeLens — Retriever (app/retrieval/retriever.py)

Implements Retrieval Fanout with Hybrid Search + Async Parallel Execution.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  User Query                                                  │
  │       │                                                      │
  │  rewrite_query()  ──────────────────────────┐               │
  │       │                                     │               │
  │  ORIGINAL QUERY                     REWRITTEN QUERY         │
  │       │                                     │               │
  │  ┌────┴─────┐                        ┌──────┴────┐          │
  │  │  Vector  │  asyncio.gather()      │  Vector   │          │
  │  │  Search  │  ← ALL 4 run in ────── │  Search   │          │
  │  │  (dense) │    PARALLEL            │  (dense)  │          │
  │  └────┬─────┘                        └──────┬────┘          │
  │  ┌────┴─────┐                        ┌──────┴────┐          │
  │  │  BM25    │                        │  BM25     │          │
  │  │  Search  │                        │  Search   │          │
  │  │ (sparse) │                        │  (sparse) │          │
  │  └────┬─────┘                        └──────┬────┘          │
  │       └──────────────┬──────────────────────┘               │
  │                      │                                      │
  │         RRF Fusion + Deduplication                          │
  │    (chunks found by multiple searches rank higher)          │
  │                      │                                      │
  │              Final ranked results                           │
  └──────────────────────────────────────────────────────────────┘

Why asyncio.gather?  All 4 searches start at the same time.
  Sequential: 4 × 100ms = 400ms total
  Parallel:   max(100ms, 100ms, 100ms, 100ms) = 100ms total  ← 4x faster

Why RRF over simple dedup?  If a chunk appears in BOTH vector and BM25 results,
  it has stronger evidence of relevance and is ranked higher accordingly.

Replaces: app/core/retriever.py
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from app.embeddings.embedder import embed_query
from app.vectordb.vector_store import search_chunks
from app.vectordb.bm25_store import search_bm25
from app.query.rewriter import rewrite_query
from app.utils.logger import get_logger
from app.utils.metrics import retrieval_latency_seconds, chunks_retrieved

logger = get_logger(__name__)

# RRF constant — standard value; higher = gentler fusion
_RRF_K = 60

# Thread pool for running sync I/O (ChromaDB + BM25) without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=8)


# ── Async wrappers for sync search functions ─────────────────────────────────

async def _async_vector_search(
    repo_id: str,
    query: str,
    top_k: int,
    where: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Async wrapper for vector search.

    embed_query + search_chunks are CPU/IO bound sync functions.
    We run them in a thread pool so they don't block the event loop.
    """
    loop = asyncio.get_event_loop()
    # Embed in thread (CPU-bound)
    embedding = await loop.run_in_executor(_executor, embed_query, query)
    # ChromaDB search in thread (I/O-bound)
    results = await loop.run_in_executor(
        _executor, lambda: search_chunks(repo_id, embedding, top_k=top_k, where=where)
    )
    return results


async def _async_bm25_search(
    repo_id: str,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Async wrapper for BM25 search.
    Runs in thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        _executor, lambda: search_bm25(repo_id, query, top_k=top_k)
    )
    return results


# ── RRF Fusion ───────────────────────────────────────────────────────────────

def _rrf_merge(
    result_lists: List[List[Dict[str, Any]]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion across multiple result lists.

    Score formula per chunk per list: 1 / (RRF_K + rank + 1)
    A chunk found in TWO lists gets DOUBLE the contribution — this is
    the key insight: independent agreement = stronger evidence of relevance.

    Args:
        result_lists: List of chunk lists (each from a different search signal).
        top_k:        Final number of chunks to return.

    Returns:
        Deduplicated, RRF-fused chunks sorted by score descending.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            cid = chunk["id"]
            # Accumulate RRF score — chunks from multiple lists score higher
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (_RRF_K + rank + 1))
            if cid not in chunk_map:
                # Ensure BM25-only chunks have a baseline relevance score
                if "relevance_score" not in chunk:
                    chunk["relevance_score"] = 0.5
                chunk_map[cid] = chunk

    fused_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in fused_ids[:top_k]]


# ── Main Retriever ────────────────────────────────────────────────────────────

async def retrieve(
    question: str,
    repo_id: str,
    top_k: int = 20,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant code chunks using Retrieval Fanout + Async Parallel Search.

    Fanout strategy:
      1. Rewrite query for semantic coverage (LLM call, async)
      2. Launch 4 searches in PARALLEL via asyncio.gather:
           (a) Vector search  with ORIGINAL  query — preserves exact identifiers
           (b) BM25 search    with ORIGINAL  query — exact token matching
           (c) Vector search  with REWRITTEN query — semantic / synonym coverage
           (d) BM25 search    with REWRITTEN query — code-keyword expansion
      3. Merge all 4 result sets with RRF (deduplicates + rewards multi-signal chunks)

    Args:
        question: User's raw natural language question.
        repo_id:  Repository identifier.
        top_k:    Max chunks to return after fusion.
        where:    Optional ChromaDB metadata filter.

    Returns:
        Deduplicated, RRF-ranked list of relevant code chunks.
    """
    start_time = time.time()

    logger.info(
        "retrieve_fanout_start",
        repo_id=repo_id,
        question=question[:100],
        top_k=top_k,
    )

    # ── Step 1: Rewrite query (async, does not block) ────────────────────────
    loop = asyncio.get_event_loop()
    rewritten_query = await loop.run_in_executor(_executor, rewrite_query, question)
    original_query = question

    logger.info(
        "retrieve_fanout_queries",
        original=original_query[:80],
        rewritten=rewritten_query[:80],
    )

    # ── Step 2: Fan out — 4 searches running in parallel ─────────────────────
    #
    #  asyncio.gather() launches all 4 coroutines simultaneously.
    #  Total latency = max(individual latencies) instead of their sum.
    #
    #   (a) vec_original  — semantic search on user's raw phrasing
    #   (b) bm25_original — exact token match on user's raw phrasing
    #   (c) vec_rewritten — semantic search on LLM-expanded code query
    #   (d) bm25_rewritten— keyword match on LLM-expanded code query

    (
        vec_original,   # (a)
        bm25_original,  # (b)
        vec_rewritten,  # (c)
        bm25_rewritten, # (d)
    ) = await asyncio.gather(
        _async_vector_search(repo_id, original_query,  top_k, where),  # (a)
        _async_bm25_search  (repo_id, original_query,  top_k),         # (b)
        _async_vector_search(repo_id, rewritten_query, top_k, where),  # (c)
        _async_bm25_search  (repo_id, rewritten_query, top_k),         # (d)
    )

    logger.info(
        "retrieve_fanout_results",
        vec_original=len(vec_original),
        bm25_original=len(bm25_original),
        vec_rewritten=len(vec_rewritten),
        bm25_rewritten=len(bm25_rewritten),
    )

    # ── Step 3: RRF Merge + Deduplicate ──────────────────────────────────────
    #
    #  Chunks appearing in multiple result lists accumulate higher RRF scores.
    #  Example: a chunk found by vec_original AND bm25_original ranks higher
    #  than one found only by vec_rewritten — independent signals agree.

    results = _rrf_merge(
        result_lists=[vec_original, bm25_original, vec_rewritten, bm25_rewritten],
        top_k=top_k,
    )

    # ── Step 4: Metrics + Logging ─────────────────────────────────────────────
    duration = round(time.time() - start_time, 3)
    retrieval_latency_seconds.observe(duration)
    chunks_retrieved.observe(len(results))

    logger.info(
        "retrieve_fanout_complete",
        repo_id=repo_id,
        fused_unique=len(results),
        duration_seconds=duration,
    )

    if results:
        top = results[0]
        logger.info(
            "retrieve_top_result",
            file=top["metadata"].get("file_path", "?"),
            relevance=top.get("relevance_score", 0),
        )
    else:
        logger.warning("retrieve_no_results", repo_id=repo_id)

    return results
