"""
ChromaDB vector store manager with metadata filtering.

Supports incremental indexing, deduplication, domain filtering, and embeddings.
"""

from typing import List, Dict, Any, Optional

from pathlib import Path
import chromadb
from chromadb.config import Settings

from core import get_logger, CHROMADB_COLLECTION_NAME
from config import get_settings
from services.text_processor import TextChunk
from core.embeddings import EmbeddingPipeline  # embedding pipeline

logger = get_logger(__name__)


class ChromaDBManager:
    """Enterprise ChromaDB with metadata filtering and deduplication."""

    def __init__(self):
        self.settings = get_settings()

        self.client = chromadb.PersistentClient(
            path=str(self.settings.vectorstore_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=CHROMADB_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Embedding pipeline (sentence-transformers)
        self.embedder = EmbeddingPipeline()

        # Hash-based deduplication
        self.indexed_hashes: set[str] = set()
        self._load_existing_hashes()

        logger.info(f"ChromaDB ready: {self.collection.count()} chunks")

    def add_chunks(self, chunks: List[TextChunk]) -> Dict[str, int]:
        """Add chunks with deduplication and embeddings."""
        if not chunks:
            return {"added": 0, "skipped": 0}

        # Deduplicate by file_hash
        new_chunks: List[TextChunk] = []
        skipped = 0
        for chunk in chunks:
            fh = chunk.metadata.get("file_hash")
            if fh not in self.indexed_hashes:
                new_chunks.append(chunk)
            else:
                skipped += 1

        if not new_chunks:
            return {"added": 0, "skipped": skipped}

        def _clean_meta(m: Dict[str, Any]) -> Dict[str, Any]:
            cleaned: Dict[str, Any] = {}
            for k, v in (m or {}).items():
                # Chroma only allows str, int, float, bool
                if v is None:
                    cleaned[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    cleaned[k] = v
                else:
                    cleaned[k] = str(v)
            return cleaned

        ids = [c.chunk_id for c in new_chunks]
        documents = [c.content for c in new_chunks]
        metadatas = [_clean_meta(c.metadata) for c in new_chunks]

        # Compute embeddings for new chunks
        embeddings = self.embedder.embed_texts(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        # Update dedup index
        self._update_hashes([c.metadata.get("file_hash", "") for c in new_chunks])

        logger.info(f"Added {len(new_chunks)} chunks (skipped {skipped})")
        return {"added": len(new_chunks), "skipped": skipped}

    def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic query with metadata filtering.
        
        ChromaDB 0.5.17+ rejects empty where clause {} - omit parameter entirely if no filters.

        Args:
            query: Search query text.
            top_k: Number of results.
            filters: Metadata filters e.g. {"domain": "hr"} or None.
        
        Returns:
            List of {content, metadata, distance} dicts.
        """
        # Build query params dynamically
        query_params = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # Only add 'where' if we have filters
        if filters and len(filters) > 0:
            query_params["where"] = filters
        
        results = self.collection.query(**query_params)

        formatted: List[Dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i in range(len(docs)):
            formatted.append(
                {
                    "content": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                }
            )

        return formatted

    def delete_by_hash(self, file_hash: str):
        """Delete all chunks for a given file hash."""
        self.collection.delete_where({"file_hash": file_hash})
        self.indexed_hashes.discard(file_hash)
        logger.info(f"Deleted document: {file_hash}")

    def get_stats(self) -> Dict[str, Any]:
        """Collection statistics."""
        count = self.collection.count()
        try:
            metas = self.collection.get(include=["metadatas"]).get("metadatas", []) or []
            domains = {m.get("domain") for m in metas if m.get("domain") is not None}
        except Exception:
            metas = []
            domains = set()

        return {
            "total_chunks": count,
            "domains": len(domains),
            "unique_docs": len(self.indexed_hashes),
        }

    def _load_existing_hashes(self):
        """Load existing file hashes for deduplication."""
        try:
            results = self.collection.get(include=["metadatas"])
            metas = results.get("metadatas", []) or []
            self.indexed_hashes = {m.get("file_hash") for m in metas if m.get("file_hash")}
            logger.info(f"Loaded {len(self.indexed_hashes)} existing doc hashes")
        except Exception:
            self.indexed_hashes = set()

    def _update_hashes(self, new_hashes: List[str]):
        """Update deduplication index."""
        self.indexed_hashes.update(h for h in new_hashes if h)
