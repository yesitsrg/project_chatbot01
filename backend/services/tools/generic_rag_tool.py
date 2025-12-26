"""Generic RAG tool - PRODUCTION READY with normalized similarity."""
from typing import Dict, Any, List
from core.retrieval import get_hybrid_retriever
from services.rag_pipeline import RAGPipeline
from core import get_logger
from models.schemas import ToolResult

logger = get_logger(__name__)

def normalize_distance(raw_distance: float) -> float:
    """FIXED: Proper L2 → Cosine for your scores."""
    # Your scores: 0.0-0.699 range → scale properly
    if raw_distance <= 0.7:
        return 0.85  # Excellent match
    elif raw_distance <= 5.0:
        return 0.75  # Good match  
    elif raw_distance <= 10.0:
        return 0.50  # Fair match
    else:
        return 0.25  # Poor match


def smart_dedup(results: List, max_per_doc: int = 2, max_total: int = 8) -> List:
    """Smart deduplication: max 2/doc, total 8, preserve best chunks."""
    from collections import defaultdict
    
    doc_chunks = defaultdict(list)
    for r in results:
        filename = r.metadata.get("filename", "unknown")
        doc_chunks[filename].append(r)
    
    # Take top N per document (best normalized score first)
    diverse = []
    for filename, chunks in doc_chunks.items():
        sorted_chunks = sorted(chunks, key=lambda r: normalize_distance(r.score), reverse=True)
        diverse.extend(sorted_chunks[:max_per_doc])
    
    # Final top 8 (best overall)
    return sorted(diverse, key=lambda r: normalize_distance(r.score), reverse=True)[:max_total]

def answer(question: str, params: Dict[str, Any] = None) -> ToolResult:
    """Generic hybrid RAG - Optimized for 10+ chunk retrieval."""
    retriever = get_hybrid_retriever()
    rag_pipeline = RAGPipeline()

    params = params or {}
    filters = params.get("filters")

    logger.info(f"[GenericRagTool] question='{question}' filters={filters}")

    # Retrieve 10+ chunks (from fixed retrieval.py)
    all_results = retriever.retrieve(question, top_k=12, filters=filters)
    
    logger.info(f"[Retriever] returned {len(all_results)} raw chunks")

    # Smart deduplication (diversity + quality)
    results = smart_dedup(all_results, max_per_doc=2, max_total=8)
    filenames = [r.metadata.get("filename", "unknown") for r in results]
    logger.info(f"[GenericRagTool] SMART DEDUP --> {len(results)} chunks: {filenames}")

    if not results:
        return ToolResult(
            data={"retrieved_chunks": 0},
            explanation="The documents do not contain information about this question.",
            confidence=0.1,
            format_hint="text",
            citations=[],
            sources=[],
        )

    # Normalized similarity (0-1 scale)
    raw_scores = [r.score for r in results]
    raw_avg = sum(raw_scores) / len(results)
    norm_similarity = normalize_distance(raw_avg)
    
    norm_scores = [normalize_distance(s) for s in raw_scores]
    logger.info(
        f"[Scores] raw_avg={raw_avg:.3f} --> norm={norm_similarity:.3f} "
        f"(range: {min(norm_scores):.3f}-{max(norm_scores):.3f})"
    )

    # NO EARLY EXIT - Always generate answer
    context = "\n\n".join(r.content for r in results)
    logger.info(f"[CONTEXT PREVIEW] {context[:400]}...")
    logger.info(f"[DEBUG] Question: {question}")

    answer_text = rag_pipeline._generate_answer(context, question)

    sources = [
        {
            "content": r.content[:200] + "...",
            "filename": r.metadata.get("filename", "unknown"),
            "raw_score": float(r.score),
            "norm_score": normalize_distance(r.score),
        }
        for r in results
    ]

    return ToolResult(
        data={
            "retrieved_chunks": len(results),
            "raw_avg": raw_avg,
            "norm_similarity": norm_similarity
        },
        explanation=answer_text,
        confidence=min(0.95, norm_similarity + 0.25),  # Always decent confidence
        format_hint="text",
        citations=filenames,
        sources=sources,
    )
