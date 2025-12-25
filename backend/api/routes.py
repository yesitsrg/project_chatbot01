"""
Jericho API routes

- RAG endpoints (hybrid retrieval + generation)
- UI-compatible endpoints expected by chat.html / login.html / admin.html
"""

from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter,
    Form,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Response,
)
from pydantic import BaseModel

from api.deps import get_current_user
from core import get_logger
from services.rag_pipeline import RAGPipeline
from core.retrieval import get_hybrid_retriever
from db import session_db  # NEW: DB-backed sessions/history

from services.orchestrator import Orchestrator, OrchestratorRequest

logger = get_logger(__name__)
router = APIRouter()

# ---------- RAG PIPELINE INITIALIZATION ----------

rag_pipeline = RAGPipeline()
hybrid_retriever = get_hybrid_retriever()

# Global singleton orchestrator
orchestrator = Orchestrator(rag_pipeline=rag_pipeline)

# ---------- SIMPLE RAG API (TEST / DEBUG) ----------

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


# ---------- UI-COMPATIBLE SESSION ENDPOINTS (DB-BACKED) ----------

@router.post("/new_session")
async def new_session(user=Depends(get_current_user)):
    """
    Create a new chat session for the current user.

    JS usage in chat.html:
    - POST /new_session
    - Returns: {"session_id": int}
    """
    username = getattr(user, "username", "guest")
    session_id = session_db.create_session(username=username)
    logger.info(f"[new_session] user={username} session_id={session_id}")
    return {"session_id": session_id}


@router.get("/user_sessions")
async def user_sessions(user=Depends(get_current_user)):
    """
    Return list of sessions for current user.

    - GET /user_sessions
    - Returns: {"sessions": [{session_id, session_name}, ...]}
    """
    username = getattr(user, "username", "guest")
    sessions = session_db.list_sessions_for_user(username=username)
    return {"sessions": sessions}


@router.get("/history")
async def history(session_id: int, user=Depends(get_current_user)):
    """
    Return Q&A history for a session.

    JS calls: GET /history?session_id=...
    Response: {"history": [{"question": "...", "answer": "..."}, ...]}
    """
    # (Optional) ownership check can be added later
    records = session_db.get_history(session_id)
    return {
        "history": [
            {"question": q, "answer": a}
            for (q, a) in records
        ]
    }


@router.post("/rename_session")
async def rename_session(
    session_id: int = Form(...),
    new_name: str = Form(...),
    user=Depends(get_current_user),
):
    """
    Rename a session (used by rename modal in chat.html).
    """
    new_name_clean = new_name.strip() or "New Chat"
    session_db.rename_session(session_id, new_name_clean)
    logger.info(f"[rename_session] session_id={session_id} new_name='{new_name_clean}'")
    return {"success": True}


@router.post("/delete_session")
async def delete_session(
    session_id: int = Form(...),
    user=Depends(get_current_user),
):
    """
    Delete a session (used by delete option in chat sidebar).
    """
    session_db.delete_session(session_id)
    logger.info(f"[delete_session] deleted session_id={session_id}")
    return {"success": True}


# ---------- CORE RAG /query USED BY UI (WITH HISTORY) ----------

@router.post("/query")
async def query_endpoint(
    query: str = Form(...),
    session_id: int = Form(...),
    private: bool = Form(False),
    user=Depends(get_current_user),
):
    """
    Main chat endpoint used by chat.html JS.

    Frontend sends FormData:
      - query
      - session_id
      - private

    Response is expected to contain:
      - answer
      - session_id
      - sources
      - tools_used
      - confidence
      - sessionnameupdated (bool)  <-- name chosen to match old UI
    """
    username = getattr(user, "username", "test_user")
    logger.info(f"[query] user={username} q={query!r} session_id={session_id} private={private}")

    try:
        # Delegate to orchestrator
        request = OrchestratorRequest(query=query)
        response = orchestrator.handle_query(request)
        answer_text = response.answer

        # Persist message in DB
        before_count = session_db.count_messages(session_id)
        session_db.add_message(session_id=session_id, question=query, answer=answer_text)

        # Auto-rename session on first question (similar to old logic)
        sessionnameupdated = False
        if before_count == 0:
            words = query.strip().split()
            base = " ".join(words[:6]) if words else "New Chat"
            nice_name = base[:40].rstrip()
            session_db.rename_session(session_id, nice_name)
            sessionnameupdated = True

        return {
            "answer": answer_text,
            "session_id": session_id,
            "sources": response.sources,
            "tools_used": response.tools_used,
            "confidence": response.confidence,
            "sessionnameupdated": sessionnameupdated,
        }
    except Exception as e:
        logger.error(f"[query] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error while answering query")


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
            f"[upload] user={getattr(user, 'username', '?')} "
            f"session_id={session_id} private={private} "
            f"files={[f.filename for f in files]}"
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


# ---------- REACT-FACING QUERY (ORCHESTRATOR) ----------

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

    Uses the Orchestrator so queries are routed to the right domain tool,
    and persists Q&A into session_db keyed by sessionid.
    """
    try:
        username = getattr(user, "username", "admin")
        logger.info(
            f"[react-query] user={username} q={req.query!r} session={req.sessionid}"
        )

        # Orchestrator call
        orch_request = OrchestratorRequest(query=req.query)
        orch_response = orchestrator.handle_query(orch_request)
        answer_text = orch_response.answer

        # Persist message in DB for this session
        before_count = session_db.count_messages(req.sessionid)
        session_db.add_message(
            session_id=req.sessionid,
            question=req.query,
            answer=answer_text,
        )

        # Auto-rename session on first question
        sessionnameupdated = False
        if before_count == 0:
            words = req.query.strip().split()
            base = " ".join(words[:6]) if words else "New Chat"
            nice_name = base[:40].rstrip()
            session_db.rename_session(req.sessionid, nice_name)
            sessionnameupdated = True

        return {
            "answer": answer_text,
            "sessionid": req.sessionid,
            "tools_used": orch_response.tools_used,
            "confidence": orch_response.confidence,
            "sources": orch_response.sources,
            "sessionnameupdated": sessionnameupdated,
        }
    except Exception as e:
        logger.error(f"[react-query] orchestrator error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Query failed")

# ---------- LOGOUT ----------

@router.post("/logout")
async def logout(response: Response):
    # Adjust cookie names if different
    response.delete_cookie("access_token")
    return {"success": True}

@router.post("/feedback")
async def submit_feedback(
    message_id: int = Form(...),
    rating: str = Form(...),
    session_id: int = Form(...),
    user: dict = Depends(get_current_user),
):
    """Submit user feedback (like/dislike) for answer"""
    username = getattr(user, 'username', 'guest')
    try:
        session_db.add_feedback(message_id, session_id, username, rating)
        logger.info(f"Feedback: {username} rated message {message_id} as {rating}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Feedback failed")

