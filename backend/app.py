"""
Jericho Chatbot - FastAPI entrypoint.

- Serves HTML pages (login, admin login, chat) via Jinja2 templates
- Serves static assets from ./static
- Exposes RAG API under /api/v1/...
- Also exposes UI-compatible endpoints with NO prefix
  (new_session, user_sessions, history, query, upload)
"""

from pathlib import Path
import sys

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# --- LOGGER INIT (must be BEFORE importing services/api) ---
from core import init_logger  # uses your existing core/__init__.py
init_logger()

# --- ROUTERS ---
from api.routes import router as api_router
from api.health import router as health_router

from api.auth_api import router as auth_router

# --- APP ---
app = FastAPI(
    title="Jericho Chatbot",
    version="1.0.0",
    description="Enterprise multi-domain RAG chatbot for Jericho.",
)

app.include_router(auth_router, prefix="/apiv1", tags=["auth"])

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- TEMPLATES & STATIC ---
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# --- API ROUTERS ---

# a) JSON API under /api/v1/...
app.include_router(api_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")

# b) UI-compatible endpoints with NO prefix
#    chat.html, admin.html, etc. call /new_session, /user_sessions, /query, /upload directly.
app.include_router(api_router)


# ====== PAGE ROUTES (UI) ======

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "user_authenticated": False,
            "error": None,
        },
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "user_authenticated": False,
            "error": None,
        },
    )


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return templates.TemplateResponse(
        "admin_login.html",
        {
            "request": request,
            "user_authenticated": False,
            "error": None,
        },
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    # For now, stub auth + sessions; UI will call /new_session + /user_sessions
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user_authenticated": True,
            "username": "User",
            "sessions": [],
        },
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    # Stub data; admin.html expects these keys
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "user_authenticated": True,
            "users": [],
            "active_sessions": [],
            "csrf_token": "",
            "add_success": False,
            "error": None,
            "edit_success": False,
            "edit_error": False,
            "delete_success": False,
            "delete_error": False,
            "session_kill_success": False,
            "session_kill_error": False,
        },
    )


# --- DEV ENTRYPOINT ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
