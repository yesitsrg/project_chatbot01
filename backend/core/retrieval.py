"""
Hybrid Retrieval Engine: Vector + BM25 + Cross-Encoder Reranking.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import logging
from core import get_logger
from config import get_settings  # ← FIXED
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
    def __init__(self):
        self.settings = get_settings()  # ← FIXED
        self.db = ChromaDBManager()
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logger.warning(f"Reranker load failed: {e}")
            self.reranker = None
        self.bm25_index = None
        self.doc_mapping = {}
        self._build_bm25_index()
        logger.info("HybridRetriever ready")

    def _build_bm25_index(self) -> None:
        try:
            all_docs = self.db.collection.get(include=["documents", "metadatas"])
            corpus = [doc for doc in all_docs["documents"] if doc]
            self.doc_mapping = {i: meta for i, meta in enumerate(all_docs["metadatas"])}
            self.bm25_index = BM25Okapi([doc.split() for doc in corpus])
            logger.info(f" BM25: {len(corpus)} docs")
        except Exception as e:
            logger.warning(f"BM25 failed: {e}")
            self.bm25_index = None

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """
        Retrieve with higher default + better over-fetching.
        """
        # Over-fetch MORE for better final quality
        fetch_k = max(30, top_k * 3)  # 30-90 chunks
        
        vector_results = self._vector_search(query, fetch_k, filters)
        bm25_results = self._bm25_search(query, fetch_k) if self.bm25_index else []
        
        fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        final_results = self._rerank(query, fused_results, top_k)  # Respect top_k param
        
        logger.info(f"[Retriever] query='{query[:50]}...' --> {len(vector_results)} vector + {len(bm25_results)} bm25 → {len(final_results)} final")
        
        return final_results


    def _vector_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[RetrievalResult]:
        results = self.db.query(query, top_k=top_k, filters=filters or {})
        return [RetrievalResult(r["content"], r["metadata"], 1.0 - r["distance"], "vector") for r in results]

    # Fix _bm25_search method in core/retrieval.py
    def _bm25_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        if not self.bm25_index:
            return []
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # FIXED: Use actual documents from ChromaDB
        all_docs = self.db.collection.get()["documents"]  # Get real content
        return [
            RetrievalResult(
                content=all_docs[idx] if idx < len(all_docs) else "BM25 match",
                metadata=self.doc_mapping.get(idx, {}),
                score=float(scores[idx]),
                source="bm25"
            ) for idx in top_indices if idx < len(self.doc_mapping)
        ]


    def _reciprocal_rank_fusion(self, list1: List[RetrievalResult], list2: List[RetrievalResult]) -> List[RetrievalResult]:
        scores: Dict[str, float] = {}
        k = 60.0
        
        for rank, result in enumerate(list1, 1):
            scores[result.doc_id] = scores.get(result.doc_id, 0) + 1.0 / (k + rank)
        for rank, result in enumerate(list2, 1):
            scores[result.doc_id] = scores.get(result.doc_id, 0) + 1.0 / (k + rank)
        
        result_map = {r.doc_id: r for r in list1 + list2}
        sorted_results = sorted(
            [(doc_id, score, result_map.get(doc_id)) for doc_id, score in scores.items()],
            key=lambda x: x[1], reverse=True
        )
        return [result for _, _, result in sorted_results if result][:10]

    def _rerank(self, query: str, candidates: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        if not self.reranker or len(candidates) < 2:
            return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]
        pairs = [(query, cand.content) for cand in candidates[:20]]
        try:
            scores = self.reranker.predict(pairs)
            scored = [(score, cand) for score, cand in zip(scores, candidates[:20])]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [result for _, result in scored[:top_k]]
        except:
            return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]

_retriever: Optional[HybridRetriever] = None

def get_hybrid_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
