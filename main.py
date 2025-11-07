# main.py
import os
import datetime as dt
from typing import Optional, List

from fastapi import Cookie
from fastapi import FastAPI, Depends, Header, HTTPException, APIRouter, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, func
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
import requests

# =============================
# CONFIG
# =============================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://househive_db_user:853bkQc9s9y7oWVmGnpjqt8G8zlWRDJp@dpg-d45u9hvdiees738h3f80-a.oregon-postgres.render.com/househive_db",
)

# ✅ Safe database configuration
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
    echo=False
)

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PROD")
JWT_ALG = "HS256"
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "60"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# =============================
# APP + CORS
# =============================
# =============================
# APP + CORS
# =============================
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://househive.ai",
    "https://www.househive.ai",
    "https://househive-frontend.vercel.app",
]

app = FastAPI(title="HouseHive Backend", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# =============================
# DB SETUP
# =============================
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================
# MODELS
# =============================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255))
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Property(Base):
    __tablename__ = "properties"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    address = Column(String(255))
    owner_email = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255))
    phone = Column(String(50))
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False, index=True)
    owner_email = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(String(1024))
    status = Column(String(50), default="open", index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=True, index=True)
    owner_email = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Reminder(Base):
    __tablename__ = "reminders"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    due_date = Column(DateTime(timezone=True))
    completed = Column(Boolean, default=False)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True, index=True)
    owner_email = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

Base.metadata.create_all(bind=engine)

# =============================
# SECURITY
# =============================
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str: return pwd_ctx.hash(plain)
def verify_password(plain: str, hashed: str) -> bool: return pwd_ctx.verify(plain, hashed)

def create_access_token(sub: str) -> str:
    exp = dt.datetime.utcnow() + dt.timedelta(minutes=JWT_EXPIRES_MIN)
    return jwt.encode({"sub": sub, "exp": exp}, JWT_SECRET, algorithm=JWT_ALG)

def bearer_token(authorization: Optional[str] = Header(None)):
    if not authorization: return None
    parts = authorization.split()
    return parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else None

def get_current_user(db: Session = Depends(get_db), token: Optional[str] = Depends(bearer_token)):
    if not token: raise HTTPException(status_code=401, detail="Missing token")
    try:
        email = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG]).get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.email == email).first()
    if not user: raise HTTPException(status_code=401, detail="User not found")
    return user

# =============================
# SCHEMAS
# =============================
class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: Optional[str]
    created_at: dt.datetime
    class Config: orm_mode = True

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

# =============================
# AUTH ROUTES
# =============================
auth = APIRouter(prefix="/api/auth", tags=["auth"])


@auth.post("/refresh")
def refresh_token(
    refresh_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")

    try:
        email = jwt.decode(refresh_token, JWT_SECRET, algorithms=[JWT_ALG]).get("sub")
    except:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    new_access = create_access_token(email)
    return {"access_token": new_access}


@auth.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    try:
        user = User(
            email=payload.email.lower(),
            name=payload.name,
            password_hash=hash_password(payload.password)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return UserOut.from_orm(user)

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered")


from fastapi.responses import JSONResponse

@auth.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == payload.email.lower()).first()

    if not u or not verify_password(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access = create_access_token(u.email)
    refresh = jwt.encode({"sub": u.email}, JWT_SECRET, algorithm=JWT_ALG)

    token_payload = TokenResponse(
        access_token=access,
        user=UserOut.from_orm(u)
    )

    response = JSONResponse(jsonable_encoder(token_payload))

    response.set_cookie(
        key="refresh_token",
        value=refresh,
        httponly=True,
        secure=True,
        samesite="None",
        max_age=60 * 60 * 24 * 30,  # 30 days
        path="/"
    )

    return response


@auth.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut.from_orm(user)


# =============================
# PASSWORD RESET
# =============================
def send_reset_email(email: str, token: str):
    print(f"Password reset link for {email}: https://househive.ai/reset-password?token={token}")

class ForgotRequest(BaseModel):
    email: EmailStr

@auth.post("/forgot")
def forgot_password(
    data: ForgotRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db)
):
    email = data.email.lower().strip()

    user = db.query(User).filter(User.email == email).first()

    if user:
        reset_token = create_access_token(user.email)
        background.add_task(send_reset_email, user.email, reset_token)

    # Always return success so we don't leak who has an account
    return {"message": "If that email is registered, a reset link was sent."}


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=6)

@auth.post("/reset", response_model=UserOut)
def reset_password(data: ResetPasswordRequest, db: Session = Depends(get_db)):
    try:
        email = jwt.decode(data.token, JWT_SECRET, algorithms=[JWT_ALG]).get("sub")
    except:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.password_hash = hash_password(data.new_password)
    db.commit()
    db.refresh(user)
    return UserOut.from_orm(user)

# =============================
# AI CHAT ROUTE
# =============================
from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI

ai = APIRouter(prefix="/api/ai", tags=["ai"])
client = OpenAI(api_key=OPENAI_API_KEY)

class ChatMessage(BaseModel):
    message: str
    history: list = []

@ai.post("/chat")
def chat(payload: ChatMessage, user: User = Depends(get_current_user)):
    messages = []

    # Load past conversation
    for m in payload.history:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    # Add new user message
    messages.append({"role": "user", "content": payload.message})

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
    )

    reply = completion.choices[0].message.content.strip()

    return {
        "reply": reply,
        "history": messages + [{"role": "assistant", "content": reply}]
    }

app.include_router(ai)

# =============================
# INSIGHTS
# =============================
insights = APIRouter(prefix="/api/insights", tags=["insights"])

@insights.get("/")
def get_insights(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    property_count = db.query(Property).filter(Property.owner_email == user.email).count()
    tenant_count = db.query(Tenant).filter(Tenant.owner_email == user.email).count()
    open_tasks = db.query(Task).filter(Task.owner_email == user.email, Task.status == "open").count()
    reminders = db.query(Reminder).filter(Reminder.owner_email == user.email).count()

    summary = (
        f"You have {property_count} properties, {tenant_count} tenants, "
        f"{open_tasks} open tasks, and {reminders} reminders."
    )

    return {
        "summary": summary,
        "property_count": property_count,
        "tenant_count": tenant_count,
        "open_tasks": open_tasks,
        "reminders": reminders
    }

from fastapi import APIRouter

properties = APIRouter(prefix="/api/properties", tags=["properties"])

@properties.get("/")
def list_properties(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Property).filter(Property.owner_email == user.email).order_by(Property.created_at.desc()).all()

@properties.post("/")
def add_property(data: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = Property(
        name=data.get("name"),
        address=data.get("address", ""),
        owner_email=user.email
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p

app.include_router(properties)

app.include_router(auth)
app.include_router(insights)
