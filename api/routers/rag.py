from fastapi import APIRouter, Body, HTTPException, Query, status
from typing import Optional

from config import settings
from models import RagMatchModel, RagSearchResponse
from services import RAGService

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

_rag_service: Optional[RAGService] = None


def get_rag_service() -> Optional[RAGService]:
    """Lazy getter for the shared RAG service instance."""
    global _rag_service
    if not settings.rag_enabled:
        return None
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@router.get(
    "/search",
    response_model=RagSearchResponse,
    summary="Search vegetarian knowledge base for supporting evidence",
)
async def rag_search(
    query: str = Query(..., min_length=2, description="Dish name or description to search for."),
    top_k: int = Query(3, ge=1, le=10, description="Number of similar dishes to retrieve."),
) -> RagSearchResponse:
    """Return retrieval matches for the supplied query text."""
    service = get_rag_service()
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is disabled or unavailable.",
        )

    matches = service.search(query, top_k=top_k)
    return RagSearchResponse(
        query=query,
        top_k=top_k,
        matches=[
            RagMatchModel(
                name=match.name,
                category=match.category,
                description=match.description,
                score=match.score,
                chunk_index=match.chunk_index,
            )
            for match in matches
        ],
    )


@router.post(
    "/reseed",
    summary="Rebuild the RAG vector store from the vegetarian dataset",
    status_code=status.HTTP_202_ACCEPTED,
)
async def rag_reseed(force: bool = Body(False, embed=True)) -> dict:
    """Clear and repopulate the RAG database."""
    service = get_rag_service()
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is disabled or unavailable.",
        )

    count = service.reseed(force=force)
    return {"status": "ok", "documents": count}
