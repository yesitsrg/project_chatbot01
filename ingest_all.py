from pathlib import Path
import sys

# 1) Ensure backend root on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 2) Import the REAL logger helpers from core (__init__ re-exports them)
from core import initlogger, get_logger

# Initialize global logger BEFORE importing services
init_logger()
logger = get_logger(__name__)

# 3) Now safely import RAGPipeline (which imports services/document_parser)
from services.rag_pipeline import RAGPipeline


def main():
    docs_root = Path(r"D:\jericho\data\documents")
    paths = [str(p) for p in docs_root.rglob("*") if p.is_file()]
    logger.info(f"Found {len(paths)} files to ingest under {docs_root}")

    rag = RAGPipeline()
    stats = rag.ingest_documents(paths)
    print("=== RAW CHUNK DEBUG ===")
    collection = chroma_client.get_collection("jericho_kb")
    sample_chunks = collection.get(limit=3, include=["documents"])
    for i, chunk in enumerate(sample_chunks["documents"]):
        if "cccscsc" in chunk.lower():
            print(f"❌ DIRTY CHUNK {i}: {chunk[:200]}...")
        else:
            print(f"✅ CLEAN CHUNK {i}: {chunk[:100]}...")
    logger.info(f"INGEST STATS: {stats}")


if __name__ == "__main__":
    main()
