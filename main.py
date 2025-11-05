# main.py
import os
import datetime as dt
from typing import Optional, List

from fastapi import FastAPI, Depends, Header, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
import requests

# CONFIG
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./househive.db")
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PROD")
JWT_ALG = "HS256"
JWT_EXPIRES_MIN = 60

CORS_ORIGINS = [
    "*",  # ✅ Allow all for now (fixes CORS instantly)
]

# DB
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, future=True, echo=False, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# MODELS
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255))
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

Base.metadata.create_all(bind=engine)

# SECURITY
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(p): return pwd_ctx.hash(p)
def verify_password(p, h): return pwd_ctx.verify(p, h)

def create_access_token(sub: str):
    exp = dt.datetime.utcnow() + dt.timedelta(minutes=JWT_EXPIRES_MIN)
    return jwt.encode({"sub": sub, "exp": exp}, JWT_SECRET, algorithm=JWT_ALG)

def bearer_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        return None
    parts = authorization.split()
    return parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else None

def get_current_user(db: Session = Depends(get_db), token: str = Depends(bearer_token)):
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    try:
        email = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG]).get("sub")
    except:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# SCHEMAS
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = ""

class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: Optional[str]
    created_at: dt.datetime
    class Config: from_attributes = True

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    user: UserOut

# =============================
# APP + CORS
# =============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://househive-frontend.vercel.app",
        "https://househive.ai",
        "https://www.househive.ai"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(): return {"status": "ok"}

# AUTH ROUTES
auth = APIRouter(prefix="/api/auth", tags=["auth"])

@auth.post("/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email.lower()).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=payload.email.lower(), name=payload.name, password_hash=hash_password(payload.password))
    db.add(user); db.commit(); db.refresh(user)
    return user

@auth.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == payload.email.lower()).first()
    if not u or not verify_password(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(u.email), user=u)

app.include_router(auth)
