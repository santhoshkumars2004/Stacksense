"""
CodeLens Query Endpoint.

POST /api/query — ask a question about an indexed repository.
"""

from fastapi import APIRouter, HTTPException

from app.models.request_models import QueryRequest
from app.models.response_models import QueryResponse
from app.query.pipeline import query_pipeline
from app.utils.logger import get_logger

router = APIRouter(prefix="/api", tags=["Query"])
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_repo(request: QueryRequest):
    """
    Ask a natural language question about an indexed codebase.

    Returns an AI-generated answer with exact file:line citations.
    """
    try:
        logger.info(
            "query_request",
            repo_id=request.repo_id,
            question=request.question[:100],
        )

        # Build metadata filter for ChromaDB (where clause)
        where_filter = None
        if request.language_filter:
            where_filter = {"language": {"$eq": request.language_filter.lower()}}

        result = await query_pipeline(
            question=request.question,
            repo_id=request.repo_id,
            top_k=request.top_k or 5,
            where=where_filter,
            path_filter=request.path_filter,
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(
            "query_error",
            repo_id=request.repo_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
