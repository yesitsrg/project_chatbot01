"""
SINGLE FILE INGEST - ONLY Residence Hall for fast testing
"""
from core import init_logger
init_logger()

"""
SINGLE FILE INGEST - ONLY Residence Hall (NO rag_pipeline dependency)
"""

import sys
sys.path.append(".")

from pathlib import Path
import json
from services.document_parser import DocumentParser
from services.text_processor import TextProcessor
from db.chromadb_manager import ChromaDBManager
from core.embeddings import EmbeddingPipeline
from core import get_logger

logger = get_logger(__name__)

PDF_PATH = Path("D:/jericho/data/documents/hr_policies/FINAL-Rev2024-DC RES LIFE HANDBOOK (7).pdf")

def ingest_single_file():
    """Process ONLY Residence Hall - Full pipeline"""
    
    if not PDF_PATH.exists():
        print(f"❌ File not found: {PDF_PATH}")
        return
    
    print(f"🎯 SINGLE FILE: {PDF_PATH.name}")
    print("🚀 Starting targeted ingest...")
    
    # Force re-process
    PDF_PATH.touch()
    
    # 1. Parse
    print("📄 STEP 1: DocumentParser...")
    parser = DocumentParser()
    parsed_doc = parser.parse_file(PDF_PATH)
    print(f"   Parsed: {len(parsed_doc.content)} blocks")
    
    # 2. Chunk
    print("🔪 STEP 2: TextProcessor...")
    processor = TextProcessor()
    chunks = processor.process_document(parsed_doc)
    print(f"   Chunked: {len(chunks)} chunks")
    
    # 3. Embed + Store (skip if testing only)
    print("💾 STEP 3: ChromaDB (skipped for speed)")
    print(f"\n✅ SUCCESS: {len(chunks)} clean chunks ready!")
    print("🎯 Start server + test query NOW")
    
    # Save chunks for inspection
    with open("residence_hall_clean_chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks[:10]):  # First 10
            f.write(f"CHUNK {i}: {len(chunk.content)} chars\n")
            f.write(chunk.content[:200] + "\n\n")
    print("💾 Saved: residence_hall_clean_chunks.txt")

if __name__ == "__main__":
    ingest_single_file()
