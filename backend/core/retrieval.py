"""
Hybrid Retrieval Engine: Vector + BM25 + Cross-Encoder Reranking.
Enhanced with relevance filtering and source diversity for universal quality.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import logging
from core import get_logger
from config import get_settings
from db.chromadb_manager import ChromaDBManager


logger = get_logger(__name__)


class RetrievalResult:
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, source: str = "vector"):
        self.content = content
        self.metadata = metadata or {}
        self.score = score
        self.source = source
        self.doc_id = metadata.get("chunkid") or f"{source}_{hash(content) % 10000}"


class HybridRetriever:
    """
    Universal high-quality retrieval with:
    - Hybrid search (vector + BM25)
    - Cross-encoder re-ranking
    - Relevance threshold filtering
    - Source diversity enforcement
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db = ChromaDBManager()
        
        # Cross-encoder for re-ranking
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("[Retriever] Cross-encoder loaded")
        except Exception as e:
            logger.warning(f"[Retriever] Reranker load failed: {e}")
            self.reranker = None
        
        self.bm25_index = None
        self.doc_mapping = {}
        self._build_bm25_index()
        logger.info("[Retriever] HybridRetriever ready")


    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in ChromaDB."""
        try:
            all_docs = self.db.collection.get(include=["documents", "metadatas"])
            corpus = [doc for doc in all_docs["documents"] if doc]
            self.doc_mapping = {i: meta for i, meta in enumerate(all_docs["metadatas"])}
            self.bm25_index = BM25Okapi([doc.split() for doc in corpus])
            logger.info(f"[Retriever]  BM25: {len(corpus)} docs")
        except Exception as e:
            logger.warning(f"[Retriever] BM25 failed: {e}")
            self.bm25_index = None


    def retrieve(
        self, 
        query: str, 
        top_k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Universal high-quality retrieval pipeline.
        
        Flow:
        1. Hybrid search (vector + BM25) --> Over-fetch candidates
        2. Reciprocal rank fusion --> Merge results
        3. Cross-encoder re-rank --> Score relevance
        4. Relevance filtering --> Remove low-quality (score < 0.35)
        5. Source diversity --> Max 2 chunks per document
        6. Return top_k high-quality, diverse chunks
        """
        
        # Stage 1: Over-fetch for better final quality
        chroma_filters = None
        if filters:
            # Remove None values and empty entries
            cleaned = {k: v for k, v in filters.items() if v is not None and v != ""}
            if cleaned:  # Only use if non-empty after cleaning
                chroma_filters = cleaned        
        
        fetch_k = max(36, top_k * 3)  # Get 30-90 candidates
        
        vector_results = self._vector_search(query, fetch_k, chroma_filters)
        bm25_results = self._bm25_search(query, fetch_k) if self.bm25_index else []
        
        logger.info(
            f"[Retriever] query='{query[:50]}...' --> "
            f"{len(vector_results)} vector + {len(bm25_results)} bm25"
        )
        
        # Stage 2: Fusion
        fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        
        # Stage 3 & 4: Re-rank + Filter relevance
        reranked_results = self._rerank_with_quality_filter(query, fused_results, top_k * 2)
        
        # Stage 5: Enforce source diversity
        final_results = self._enforce_source_diversity(reranked_results, max_per_source=2, target_count=top_k)
        
        logger.info(
            f"[Retriever] --> {len(fused_results)} fused --> "
            f"{len(reranked_results)} quality --> {len(final_results)} final"
        )
        
        return final_results


    def _vector_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """Vector similarity search using ChromaDB embeddings."""
        results = self.db.query(query, top_k=top_k, filters=filters)
        return [
            RetrievalResult(
                r["content"], 
                r["metadata"], 
                1.0 - r["distance"], 
                "vector"
            ) for r in results
        ]


    def _bm25_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """BM25 keyword search for exact term matching."""
        if not self.bm25_index:
            return []
        
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Get actual documents from ChromaDB
        all_docs = self.db.collection.get()["documents"]
        
        return [
            RetrievalResult(
                content=all_docs[idx] if idx < len(all_docs) else "BM25 match",
                metadata=self.doc_mapping.get(idx, {}),
                score=float(scores[idx]),
                source="bm25"
            ) for idx in top_indices if idx < len(self.doc_mapping)
        ]


    def _reciprocal_rank_fusion(
        self, 
        list1: List[RetrievalResult], 
        list2: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Fuse results from multiple retrievers using reciprocal rank."""
        scores: Dict[str, float] = {}
        k = 60.0
        
        for rank, result in enumerate(list1, 1):
            scores[result.doc_id] = scores.get(result.doc_id, 0) + 1.0 / (k + rank)
        for rank, result in enumerate(list2, 1):
            scores[result.doc_id] = scores.get(result.doc_id, 0) + 1.0 / (k + rank)
        
        result_map = {r.doc_id: r for r in list1 + list2}
        sorted_results = sorted(
            [(doc_id, score, result_map.get(doc_id)) for doc_id, score in scores.items()],
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [result for _, _, result in sorted_results if result]


    def _rerank_with_quality_filter(
        self, 
        query: str, 
        candidates: List[RetrievalResult], 
        max_rerank: int = 20
    ) -> List[RetrievalResult]:
        """
        Re-rank with cross-encoder and filter low-relevance chunks.
        
        CRITICAL: This is the QUALITY GATE that prevents irrelevant chunks
        from reaching the LLM (e.g., bor_json for calendar queries).
        """
        if not candidates:
            return []
        
        # If no reranker, return top candidates by score
        if not self.reranker or len(candidates) < 2:
            return sorted(candidates, key=lambda x: x.score, reverse=True)[:max_rerank]
        
        # Limit reranking to top candidates (computational efficiency)
        candidates_to_rerank = candidates[:max_rerank]
        
        # Prepare query-document pairs
        pairs = [(query, cand.content[:512]) for cand in candidates_to_rerank]  # Limit to 512 chars
        
        try:
            # Get cross-encoder relevance scores
            scores = self.reranker.predict(pairs)
            
            # Update results with rerank scores
            for score, cand in zip(scores, candidates_to_rerank):
                cand.metadata['rerank_score'] = float(score)
                cand.score = float(score)  # Update primary score
            
            # QUALITY FILTER: Remove chunks with low relevance
            RELEVANCE_THRESHOLD = 0.35  # Tuned for balance
            
            quality_results = [
                cand for cand in candidates_to_rerank
                if cand.metadata.get('rerank_score', 0.0) >= RELEVANCE_THRESHOLD
            ]
            
            # Sort by rerank score
            quality_results.sort(key=lambda x: x.metadata.get('rerank_score', 0.0), reverse=True)
            
            filtered_count = len(candidates_to_rerank) - len(quality_results)
            if filtered_count > 0:
                logger.info(
                    f"[Retriever] Quality filter: removed {filtered_count} "
                    f"low-relevance chunks (threshold={RELEVANCE_THRESHOLD})"
                )
            
            # If all filtered out, return top candidates anyway (safety)
            return quality_results if quality_results else candidates_to_rerank[:10]
            
        except Exception as e:
            logger.error(f"[Retriever] Reranking failed: {e}")
            return sorted(candidates, key=lambda x: x.score, reverse=True)[:max_rerank]


    def _enforce_source_diversity(
        self,
        results: List[RetrievalResult],
        max_per_source: int = 2,
        target_count: int = 10
    ) -> List[RetrievalResult]:
        """
        ADAPTIVE source diversity enforcement.
        
        Logic:
        - If top chunks are from SAME source with HIGH scores --> Allow up to 4 chunks (sequential data)
        - If top chunks are from DIFFERENT sources --> Enforce strict limit (2 per source)
        
        This handles cases like calendars where data spans multiple sequential pages.
        """
        if not results:
            return []
        
        # Analyze top 5 results
        top_sources = [r.metadata.get('filename', 'unknown') for r in results[:5]]
        dominant_source = max(set(top_sources), key=top_sources.count)
        dominant_count = top_sources.count(dominant_source)
        
        # ADAPTIVE: If top results are dominated by ONE source (3+ out of 5)
        # AND they have high scores --> Likely sequential/related content
        if dominant_count >= 3:
            avg_top_score = np.mean([r.score for r in results[:5]])
            
            if avg_top_score > 0.5:  # High relevance
                # RELAXED: Allow up to 4 chunks from dominant source
                adaptive_max = 4
                logger.info(
                    f"[Retriever] Adaptive diversity: {dominant_source} dominates "
                    f"({dominant_count}/5 top results, avg_score={avg_top_score:.2f}) "
                    f"--> Allowing up to {adaptive_max} chunks (likely sequential data)"
                )
            else:
                # Medium relaxation
                adaptive_max = 3
                logger.info(
                    f"[Retriever] Adaptive diversity: Allowing up to {adaptive_max} chunks from {dominant_source}"
                )
        else:
            # STRICT: Multiple sources competing --> Enforce diversity
            adaptive_max = max_per_source
            logger.info(f"[Retriever] Strict diversity: Max {adaptive_max} chunks per source")
        
        # Apply adaptive limit
        source_counts = {}
        diverse_results = []
        
        for result in results:
            source = result.metadata.get('filename', 'unknown')
            count = source_counts.get(source, 0)
            
            # Use adaptive limit for dominant source, strict for others
            limit = adaptive_max if source == dominant_source else max_per_source
            
            if count < limit:
                diverse_results.append(result)
                source_counts[source] = count + 1
            
            if len(diverse_results) >= target_count:
                break
        
        # Fallback: if still not enough, add remaining
        if len(diverse_results) < target_count:
            for result in results:
                if result not in diverse_results:
                    diverse_results.append(result)
                    if len(diverse_results) >= target_count:
                        break
        
        # Log final distribution
        source_distribution = {}
        for result in diverse_results:
            source = result.metadata.get('filename', 'unknown')
            source_distribution[source] = source_distribution.get(source, 0) + 1
        
        logger.info(f"[Retriever] Final distribution: {source_distribution}")
        
        return diverse_results


# Singleton instance
_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get or create singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
