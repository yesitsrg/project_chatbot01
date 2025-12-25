"""Generic RAG tool - your existing hybrid retrieval."""
from typing import Dict, Any, List
from core.retrieval import get_hybrid_retriever
from services.rag_pipeline import RAGPipeline  # Your existing class
from core import get_logger
from models.schemas import ToolResult


logger = get_logger(__name__)


def answer(question: str, params: Dict[str, Any] = None) -> ToolResult:
    """Generic hybrid RAG over all documents."""
    retriever = get_hybrid_retriever()
    rag_pipeline = RAGPipeline()  # Your existing pipeline

    # Make params safe
    params = params or {}

    # Retrieve with optional domain filter
    domain_filter = params.get("domain_filter")
    filters = None

    logger.info(f"[GenericRagTool] question={question!r} filters={filters}")

    results = retriever.retrieve(question, top_k=5, filters=filters)

    # Log retrieved filenames + short snippets
    logger.info(
        "[GenericRagTool] retrieved %d chunks: %s",
        len(results),
        [r.metadata.get("filename", "unknown") for r in results],
    )

    if not results:
        # No hits at all
        return ToolResult(
            data={"retrieved_chunks": 0},
            explanation="The documents do not contain information about this question.",
            confidence=0.1,
            format_hint="text",
            citations=[],
            sources=[],
        )

    # Similarity threshold for "not found"
    avg_similarity = sum(r.score for r in results) / len(results)
    logger.info("[GenericRagTool] avg_similarity=%.3f", avg_similarity)

    if avg_similarity < 0.3:
        return ToolResult(
            data={"retrieved_chunks": len(results), "avg_similarity": avg_similarity},
            explanation="The documents do not contain information about this question.",
            confidence=0.1,
            format_hint="text",
            citations=[r.metadata.get("filename", "unknown") for r in results],
            sources=[
                {
                    "content": r.content[:200],
                    "filename": r.metadata.get("filename", "unknown"),
                    "score": r.score,
                }
                for r in results
            ],
        )

    context = "\n\n".join(r.content for r in results)
    answer_text = rag_pipeline._generate_answer(context, question)  # Your existing method

    sources = [
        {
            "content": r.content[:200],
            "filename": r.metadata.get("filename", "unknown"),
            "score": r.score,
        }
        for r in results
    ]

    return ToolResult(
        data={"retrieved_chunks": len(results), "avg_similarity": avg_similarity},
        explanation=answer_text,
        confidence=min(avg_similarity, 1.0),
        format_hint="text",
        citations=[r.metadata.get("filename", "unknown") for r in results],
        sources=sources,
    )
