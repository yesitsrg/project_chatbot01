from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
import jwt

from db.auth_db import (
    get_db,
    get_user_by_username,
    verify_password,
    init_db,
)
from sqlalchemy.orm import Session

SECRET_KEY = "change-this-in-.env"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str
    role: str


def create_access_token(sub: str, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = {"sub": sub}
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


@router.on_event("startup")
def startup_auth() -> None:
    init_db()


@router.post("/react-login", response_model=LoginResponse)
def react_login(body: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = get_user_by_username(db, body.username)
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(sub=user.username)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,  # set True in production with HTTPS
        samesite="lax",
    )

    return LoginResponse(
        username=user.username,
        role="admin" if user.is_admin else "user",
    )
