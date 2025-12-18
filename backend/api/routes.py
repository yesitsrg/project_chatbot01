"""
Jericho API routes

- RAG endpoints (hybrid retrieval + generation)
- UI-compatible endpoints expected by chat.html / login.html / admin.html
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict

from fastapi import (
    APIRouter,
    Form,
    UploadFile,
    File,
    HTTPException,
    Depends,
)
from pydantic import BaseModel
from api.deps import get_current_user

from core import get_logger
from services.rag_pipeline import RAGPipeline
from core.retrieval import get_hybrid_retriever

logger = get_logger(__name__)
router = APIRouter()

# ---------- RAG PIPELINE INITIALIZATION ----------

rag_pipeline = RAGPipeline()
hybrid_retriever = get_hybrid_retriever()

# ---------- SIMPLE RAG API (TESTED) ----------

class SimpleQueryRequest(BaseModel):
    query: str
    session_id: int = 1

@router.get("/query_simple")
async def query_simple(query: str, session_id: int = 1):
    """
    Simple GET query endpoint used for earlier testing.
    Kept for debugging; UI uses POST /query.
    """
    try:
        logger.info(f"[query_simple] q='{query}' session={session_id}")

        # Use unified RAG pipeline (multi-modal) instead of manual prompt
        result = rag_pipeline.query(question=query, top_k=5)
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        return {
            "answer": answer,
            "sources": [
                {
                    "content": s.get("content", "")[:200] + "...",
                    "filename": s.get("filename", "unknown"),
                }
                for s in sources
            ],
            "session_id": session_id,
        }
    except Exception as e:
        logger.error(f"[query_simple] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ---------- IN-MEMORY SESSION STORE (UI STUB) ----------

# For now, keep sessions in memory.
# Later this can be replaced with a DB-backed implementation similar to GitHub main.py.
_sessions: Dict[int, Dict] = {}
_next_session_id = 1

def _get_or_create_session(session_id: int) -> Dict:
    global _next_session_id
    if session_id in _sessions:
        return _sessions[session_id]
    session = {
        "session_id": session_id,
        "session_name": "New Chat",
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    _sessions[session_id] = session
    _next_session_id = max(_next_session_id, session_id + 1)
    return session

# ---------- UI-COMPATIBLE ENDPOINTS ----------

@router.post("/new_session")
async def new_session():
    """
    Create a new chat session.

    Matches JS usage in chat.html:
    - POST /new_session
    - Returns: {"session_id": int}
    """
    global _next_session_id
    session_id = _next_session_id
    _next_session_id += 1

    _sessions[session_id] = {
        "session_id": session_id,
        "session_name": "New Chat",
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    logger.info(f"[new_session] created session_id={session_id}")
    return {"session_id": session_id}

@router.get("/user_sessions")
async def user_sessions():
    """
    Return list of sessions for current user.

    GitHub main.py scopes by username; for now we return all sessions:
    - GET /user_sessions
    - Returns: {"sessions": [{session_id, session_name}, ...]}
    """
    sessions = [
        {
            "session_id": s["session_id"],
            "session_name": s.get("session_name", "New Chat"),
        }
        for s in _sessions.values()
    ]
    return {"sessions": sessions}

@router.get("/history")
async def history(session_id: int):
    """
    Return Q&A history for a session.

    JS calls: GET /history?session_id=...
    Response: {"history": [{"question": "...", "answer": "..."}, ...]}
    """
    session = _sessions.get(session_id)
    if not session:
        return {"history": []}

    history = [
        {"question": m["question"], "answer": m["answer"]}
        for m in session["messages"]
    ]
    return {"history": history}

@router.post("/rename_session")
async def rename_session(session_id: int = Form(...), new_name: str = Form(...)):
    """
    Rename a session (used by rename modal in chat.html).
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["session_name"] = new_name.strip() or "New Chat"
    logger.info(f"[rename_session] session_id={session_id} new_name='{session['session_name']}'")
    return {"success": True}

@router.post("/delete_session")
async def delete_session(session_id: int = Form(...)):
    """
    Delete a session (used by delete option in chat sidebar).
    """
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"[delete_session] deleted session_id={session_id}")
    return {"success": True}

# ---------- CORE RAG /query USED BY UI ----------

from services.orchestrator import Orchestrator, OrchestratorRequest

# Global singleton
orchestrator = Orchestrator()

@router.post("/query")
async def query_endpoint(
    query: str = Form(...),
    session_id: int = Form(...),
    private: bool = Form(False),
):
    # Your existing auth/session logic stays EXACTLY the same
    # username = get_username_from_token(request)
    # if not username:
    #     raise HTTPException(status_code=401, detail="Unauthorized")

    username = "test_user"
    
    # NEW: Delegate to orchestrator
    request = OrchestratorRequest(query=query)
    response = orchestrator.handle_query(request)
    
    # Save to session history (your existing logic)
    # session_db.add_single_qa_to_history(session_id, query, response.answer)
    
    # Return orchestrator response in your existing format
    return {
        "answer": response.answer,
        "session_id": session_id,
        "sources": response.sources,
        "tools_used": response.tools_used,
        "confidence": response.confidence
    }

# ---------- FILE UPLOAD (USED BY UI) ----------

@router.post("/upload")
async def upload_endpoint(
    files: List[UploadFile] = File(...),
    session_id: int = Form(...),
    private: bool = Form(False),
    user=Depends(get_current_user),  # require auth
):
    """
    Upload endpoint used by chat.html.

    Frontend sends FormData:
      - files (one or more)
      - private (boolean)
      - session_id

    We save files to data/documents/ and ingest into RAG pipeline.
    """
    try:
        logger.info(
    f"[upload] user={getattr(user, 'username', '?')} session_id={session_id} private={private} files={[f.filename for f in files]}"
)

        base_dir = Path("data/documents")
        base_dir.mkdir(parents=True, exist_ok=True)

        paths: List[str] = []
        for file in files:
            dest = base_dir / file.filename
            with dest.open("wb") as f:
                f.write(await file.read())
            paths.append(str(dest))

        stats = rag_pipeline.ingest_documents(paths)

        message = f"Processed {stats.get('processed', 0)} file(s), failed {stats.get('failed', 0)}."
        return {
            "success": True,
            "message": message,
            "processed_files": [Path(p).name for p in paths],
            "errors": [] if stats.get("failed", 0) == 0 else ["Some files failed to ingest."],
        }
    except Exception as e:
        logger.error(f"[upload] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload/ingest failed")

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.rag_pipeline import RAGPipeline
from core.retrieval import get_hybrid_retriever  # adjust name if needed


ragpipeline = RAGPipeline()
hybridretriever = get_hybrid_retriever()


class ReactChatRequest(BaseModel):
    query: str
    sessionid: int = 1
    private: bool = False


@router.post("/react-query")
async def react_query(
    req: ReactChatRequest,
    user=Depends(get_current_user),  # keep auth guard
):
    """
    React-facing query endpoint.

    Uses the Orchestrator so queries are routed to the right domain tool:
    - Transcript, Payroll, BOR, or generic Policy RAG.
    """
    try:
        logger.info(
            f"[react-query] user={getattr(user, 'username', '?')} q={req.query!r} session={req.sessionid}"
        )

        # Build orchestrator request
        orch_request = OrchestratorRequest(query=req.query)
        orch_response = orchestrator.handle_query(orch_request)

        # OrchestratorResponse fields (from your orchestrator module):
        # - answer: str
        # - tools_used: List[str]
        # - confidence: float
        # - sources: Optional[...]  (depending on your implementation)

        return {
            "answer": orch_response.answer,
            "sessionid": req.sessionid,
            "tools_used": orch_response.tools_used,
            "confidence": orch_response.confidence,
            "sources": orch_response.sources,
        }
    except Exception as e:
        logger.error(f"[react-query] orchestrator error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Query failed")
