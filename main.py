# main.py
import os
import datetime as dt
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, status, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from jose import jwt, JWTError

# -----------------------------
# Config & Environment
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./househive.db")
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PROD")
JWT_ALG = "HS256"
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "60"))  # 60 minutes default

# -----------------------------
# DB Setup (SQLAlchemy)
# -----------------------------
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, future=True, echo=False, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# Helper: Extract Bearer Token
# -----------------------------
def bearer_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


# -----------------------------
# Security
# -----------------------------
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_access_token(sub: str) -> str:
    exp = dt.datetime.utcnow() + dt.timedelta(minutes=JWT_EXPIRES_MIN)
    payload = {"sub": sub, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def get_current_user(db: Session = Depends(get_db), token: Optional[str] = Depends(bearer_token)):
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# -----------------------------
# Schemas
# -----------------------------
class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: Optional[str] = None
    created_at: dt.datetime

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI(title="HouseHive Backend", version="3.0.0")

origins = [
    "https://househive-frontend.vercel.app",
    "https://www.househive.ai",
    "https://househive.ai",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,   # allows Authorization headers
    allow_methods=["*"],      # includes OPTIONS
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": "househive-backend-v3", "time": dt.datetime.utcnow().isoformat()}


# -----------------------------
# Auth Router: /api/auth/*
# -----------------------------
auth = APIRouter(prefix="/api/auth", tags=["auth"])


@auth.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        email=payload.email.lower(),
        name=payload.name or "",
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@auth.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(sub=user.email)
    return TokenResponse(access_token=token, user=user)


@auth.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


app.include_router(auth)
# -----------------------------
# AI Router: /api/ai/*
# -----------------------------
from fastapi import Body
import openai

AI_API_KEY = os.getenv("OPENAI_API_KEY", "")

ai = APIRouter(prefix="/api/ai", tags=["ai"])

@ai.post("/chat")
def ai_chat(payload: dict = Body(...)):
    """Simple AI relay route"""
    if not AI_API_KEY:
        raise HTTPException(status_code=500, detail="AI API key not configured")
    try:
        user_message = payload.get("message", "")
        if not user_message:
            raise HTTPException(status_code=400, detail="Missing message")

        # Initialize OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=AI_API_KEY)

        # Chat completion
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are HiveBot, a helpful AI property assistant."},
                {"role": "user", "content": user_message},
            ],
        )

        reply = completion.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")

# Include AI router
app.include_router(ai)

# -----------------------------
# Notes:
# - Fixes NameError by defining bearer_token first.
# - Keeps same logic, routes, and dependencies.
# - Fully compatible with your frontend api.js.
# -----------------------------
