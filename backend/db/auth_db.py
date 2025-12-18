from pathlib import Path
from typing import Optional
import hashlib

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .auth_models import Base, User

DB_PATH = Path("backend/data/auth.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def init_db() -> None:
    """Create tables and ensure default admin user exists."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            hashed = _sha256("admin123")
            admin = User(username="admin", password_hash=hashed, is_admin=True)
            db.add(admin)
            db.commit()
    finally:
        db.close()


def get_db() -> Session:
    return SessionLocal()


def verify_password(plain: str, hashed: str) -> bool:
    return _sha256(plain) == hashed


def hash_password(plain: str) -> str:
    return _sha256(plain)


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()
