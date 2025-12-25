"""
Smart text chunking with semantic boundaries + metadata preservation + UNIVERSAL CLEANING.

Hybrid strategy: Fixed-size fallback + semantic splits (paragraphs/sentences).
Preserves document hierarchy for accurate citations.
UNIVERSAL OCR CLEANUP applied BEFORE chunking (fixes re-contamination).
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

from core import get_logger, ChunkingStrategy, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
from config import get_settings
from services.document_parser import DocumentParser, ParsedDocument   # ← NEW: Reuse universal cleaner

logger = get_logger(__name__)

@dataclass
class TextChunk:
    """Single chunk with full metadata for retrieval/citations."""
    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = None  # Document metadata + chunk-specific
    
    def __post_init__(self):
        self.metadata = self.metadata or {}

class TextProcessor:
    """Enterprise-grade chunking with multiple strategies + UNIVERSAL CLEANING."""
    
    def __init__(self):
        self.settings = get_settings()
        self.chunk_size = self.settings.chunk_size or CHUNK_SIZE
        self.chunk_overlap = self.settings.chunk_overlap or CHUNK_OVERLAP
        self.min_chunk_size = MIN_CHUNK_SIZE
        self.strategy = ChunkingStrategy.HYBRID
        self.parser = DocumentParser()  # ← NEW: Reuse universal OCR cleaner
        logger.info(f"TextProcessor ready: {self.strategy.value} chunks={self.chunk_size}")

    def process_document(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Main entry point: ParsedDocument → List[TextChunk].
        
        Args:
            parsed_doc: Output from DocumentParser
            
        Returns:
            List of chunks ready for embedding/indexing
        """
        if not parsed_doc.content:
            logger.warning(f"No content to chunk: {parsed_doc.filename}")
            return []

        # CRITICAL LAYER 2 CLEANING: TextProcessor MUST clean AGAIN
        cleaned_content = []
        original_count = len(parsed_doc.content)
        for i, block in enumerate(parsed_doc.content):
            clean_block = self.parser._universal_ocr_cleanup(block)
            if clean_block and len(clean_block) > 30:  # Enterprise quality gate
                cleaned_content.append(clean_block)
        
        logger.info(f"TextProcessor cleaned {original_count} --> {len(cleaned_content)} blocks ({parsed_doc.filename})")
        
        # Chunk ONLY clean content
        chunks = self._chunk_content(parsed_doc, cleaned_content)
        logger.info(f" Chunked {parsed_doc.filename}: {len(chunks)} chunks")
        return chunks

    def _chunk_content(self, parsed_doc: ParsedDocument, content: List[str]) -> List[TextChunk]:
        """Apply hybrid chunking strategy to ALREADY-CLEANED content."""
        if self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(parsed_doc, content)
        elif self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunking(parsed_doc, content)
        else:
            # HYBRID: Semantic first, fixed-size fallback
            semantic_chunks = self._semantic_chunking(parsed_doc, content)
            if len(semantic_chunks) > 0 and all(len(c.content) >= self.min_chunk_size for c in semantic_chunks):
                return semantic_chunks
            return self._fixed_size_chunking(parsed_doc, content)

    def _semantic_chunking(self, parsed_doc: ParsedDocument, content: List[str]) -> List[TextChunk]:
        """Chunk on semantic boundaries (paragraphs → sentences)."""
        chunks = []
        current_chunk = ""
        current_page = None
        
        # Semantic separators (per spec)
        separators = r'\n\n|\n|\.\s+|\?\s+|\!\s+'
        
        for i, block in enumerate(content):  # ← FIXED: Use cleaned content
            page_num = parsed_doc.pages[i] if parsed_doc.pages and i < len(parsed_doc.pages) else None
            
            # Try semantic split first
            sentences = re.split(separators, block)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip tiny fragments
                    continue
                    
                # Add to current chunk
                if len(current_chunk) + len(sentence) > self.chunk_size:
                    # Chunk full → save
                    if len(current_chunk) >= self.min_chunk_size:
                        chunks.append(self._create_chunk(current_chunk, len(chunks), len(content), 
                                                        parsed_doc, current_page))
                    # Start new chunk with overlap
                    current_chunk = current_chunk[-self.chunk_overlap:] + " " + sentence
                else:
                    current_chunk += " " + sentence
                
                current_page = page_num
        
        # Final chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(current_chunk, len(chunks), len(content), 
                                            parsed_doc, current_page))
        
        return chunks

    def _fixed_size_chunking(self, parsed_doc: ParsedDocument, content: List[str]) -> List[TextChunk]:
        """Fixed-size chunking with overlap (fallback)."""
        chunks = []
        full_text = " ".join(content)  # ← FIXED: Use cleaned content
        
        for i in range(0, len(full_text), self.chunk_size - self.chunk_overlap):
            chunk_text = full_text[i:i + self.chunk_size]
            if len(chunk_text) >= self.min_chunk_size:
                # Estimate page from character position
                page_estimate = self._estimate_page(i, parsed_doc)
                chunks.append(self._create_chunk(chunk_text.strip(), len(chunks), 
                                                len(content), parsed_doc, page_estimate))
        
        return chunks

    def _create_chunk(self, content: str, chunk_index: int, total_chunks: int, 
                     parsed_doc: ParsedDocument, page_num: Optional[int]) -> TextChunk:
        """Create standardized TextChunk with full metadata."""
        chunk_id = f"{parsed_doc.document_id}_c{chunk_index:04d}"
        
        # Preserve ALL document metadata
        metadata = {
            **(parsed_doc.metadata or {}),
            "filename": parsed_doc.filename,
            "document_type": parsed_doc.document_type.value,
            "user_id": parsed_doc.user_id,
            "file_hash": parsed_doc.file_hash,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "is_public": parsed_doc.user_id is None,
        }
        
        return TextChunk(
            content=content.strip(),
            chunk_id=chunk_id,
            document_id=parsed_doc.document_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            page_number=page_num,
            metadata=metadata
        )

    def _estimate_page(self, char_pos: int, parsed_doc: ParsedDocument) -> Optional[int]:
        """Estimate page number from character position."""
        if not parsed_doc.pages:
            return None
        
        # Simple proportional estimation
        total_chars = sum(len(block) for block in parsed_doc.content)
        if total_chars == 0:
            return None
        
        page_ratio = char_pos / total_chars
        page_idx = min(int(page_ratio * len(parsed_doc.pages)), len(parsed_doc.pages) - 1)
        return parsed_doc.pages[page_idx]

    def validate_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Validate chunk quality (size, content)."""
        valid_chunks = []
        for chunk in chunks:
            if (self.min_chunk_size <= len(chunk.content) <= self.chunk_size * 1.5 and 
                len(chunk.content.strip()) > 10):
                valid_chunks.append(chunk)
            else:
                logger.debug(f"Filtered bad chunk {chunk.chunk_id}: {len(chunk.content)} chars")
        return valid_chunks
