# live_check.py - uses EXACT app initialization
from app import app  # Your app.py imports everything correctly
from core import init_logger, get_logger
from services.rag_pipeline import RAGPipeline
from core.retrieval import get_hybrid_retriever

init_logger()
logger = get_logger(__name__)

print("🔍 LIVE RAG PIPELINE (same as your app)...")

try:
    # Exact same as your routes.py
    rag_pipeline = RAGPipeline()
    retriever = get_hybrid_retriever()
    
    print("✅ RAGPipeline loaded (same as app)")
    print("✅ HybridRetriever ready")
    
    # Test retrieval (BM25 + vector)
    print("\n=== RESIDENCE HALL SEARCH ===")
    results = retriever.retrieve("residence hall check in", top_k=5)
    
    print(f"Retrieved {len(results)} chunks:")
    filenames = []
    for i, doc in enumerate(results):
        filename = doc.metadata.get('filename', 'unknown') if doc.metadata else 'no metadata'
        filenames.append(filename)
        print(f"  {i+1}. {filename:<60} | score: {doc.score}")
    
    print(f"\nUnique files: {len(set(filenames))}")
    print("Files found:", ', '.join(set(filenames)))
    
    residence_found = any('res life' in f.lower() or 'residence' in f.lower() for f in filenames)
    print(f"✅ Residence hall doc: {'YES' if residence_found else 'NO'}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
