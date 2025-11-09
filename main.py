# main.py
import os
import datetime as dt
from typing import Optional, List
from fastapi import Depends

from fastapi import Cookie
from fastapi import FastAPI, Depends, Header, HTTPException, APIRouter, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, func
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from jose import jwt, JWTError

# =============================
# CONFIG
# =============================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://househive_db_user:853bkQc9s9y7oWVmGnpjqt8G8zlWRDJp@dpg-d45u9hvdiees738h3f80-a.oregon-postgres.render.com/househive_db?sslmode=require",
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require"},
    pool_pre_ping=True,
    future=True,
)

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PROD")
JWT_ALG = "HS256"
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "60"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

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
# AUTH ROUTES
# =============================
auth = APIRouter(prefix="/api/auth", tags=["auth"])

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

@auth.post("/refresh")
def refresh_token(refresh_token: Optional[str] = Cookie(None), db: Session = Depends(get_db)):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    try:
        email = jwt.decode(refresh_token, JWT_SECRET, algorithms=[JWT_ALG]).get("sub")
    except:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"access_token": create_access_token(email)}

@auth.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    try:
        user = User(email=payload.email.lower(), name=payload.name, password_hash=hash_password(payload.password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return UserOut.from_orm(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered")

@auth.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == payload.email.lower()).first()
    if not u or not verify_password(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access = create_access_token(u.email)
    refresh = jwt.encode({"sub": u.email}, JWT_SECRET, algorithm=JWT_ALG)
    res = JSONResponse(jsonable_encoder(TokenResponse(access_token=access, user=UserOut.from_orm(u))))
    res.set_cookie("refresh_token", refresh, httponly=True, secure=True, samesite="None", max_age=60*60*24*30, path="/")
    return res

@auth.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut.from_orm(user)

app.include_router(auth)

# =============================
# CONTEXT
# =============================
def get_context_for_user(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"properties": [], "tenants": [], "open_tasks": []}

    properties = db.query(Property).filter(Property.owner_email == user.email).all()
    tenants = db.query(Tenant).filter(Tenant.owner_email == user.email).all()
    tasks = db.query(Task).filter(Task.owner_email == user.email, Task.status == "open").all()

    return {
        "properties": [{"id": p.id, "name": p.name, "address": p.address} for p in properties],
        "tenants": [{"id": t.id, "name": t.name, "property": t.property_id, "phone": t.phone, "email": t.email} for t in tenants],
        "open_tasks": [{"id": t.id, "title": t.title, "property": t.property_id} for t in tasks],
    }

# =============================
# PROPERTIES
# =============================
properties = APIRouter(prefix="/api/properties", tags=["properties"])

@properties.get("/")
def list_properties(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Property).filter(Property.owner_email == user.email).order_by(Property.created_at.desc()).all()

@properties.post("/")
def add_property(data: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = Property(name=data.get("name"), address=data.get("address", ""), owner_email=user.email)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p

app.include_router(properties)

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
    return {
        "summary": f"You have {property_count} properties, {tenant_count} tenants, {open_tasks} open tasks, and {reminders} reminders.",
        "property_count": property_count,
        "tenant_count": tenant_count,
        "open_tasks": open_tasks,
        "reminders": reminders,
    }

app.include_router(insights)

# =============================
# HEALTH CHECK (Render)
# =============================
core = APIRouter(prefix="/api", tags=["core"])

@core.get("/health")
def health():
    return {"status": "ok"}

app.include_router(core)

# =============================
# AI CHAT + DRAFT (Correct + Working)
# =============================
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

ai = APIRouter(prefix="/api/ai", tags=["ai"]) 
@api.get("/dashboard/summary") 
@api.get("/landlord/overview") 
MODEL_NAME = OPENAI_MODEL or "gpt-4o-mini"

class ChatMessage(BaseModel):
    message: str
    history: list = []

@ai.post("/chat")
def chat(payload: ChatMessage, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if not client:
        raise HTTPException(status_code=502, detail="Missing OPENAI_API_KEY")

    context = get_context_for_user(db, user.id)

    system_prompt = (
        "You are HIVEBOT — property management AI.\n"
        "Be clear, friendly, and concise.\n\n"
        f"Properties: {context['properties']}\n"
        f"Tenants: {context['tenants']}\n"
        f"Open Tasks: {context['open_tasks']}\n"
    )

    messages = [{"role": "system", "content": system_prompt}] + payload.history
    messages.append({"role": "user", "content": payload.message})

    try:
        resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.6)
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        print("[AI ERROR /chat]", e)
        raise HTTPException(status_code=500, detail="HiveBot could not reach OpenAI")

    return {"reply": reply, "history": messages + [{"role": "assistant", "content": reply}]}

class DraftRequest(BaseModel):
    recipient: str
    context: str
    tone: str = "friendly"

@ai.post("/draft")
def draft(payload: DraftRequest, user: User = Depends(get_current_user)):
    if not client:
        raise HTTPException(status_code=502, detail="Missing OPENAI_API_KEY")

    sys = (
        f"You are HIVEBOT — a property communication assistant.\n"
        f"Tone: {payload.tone}\n"
        f"Context: {payload.context}\n"
        f"Recipient: {payload.recipient}\n"
        "Write the message now.\n"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": sys}],
            temperature=0.4,
        )
        text = resp.choices[0].message.content.strip()
        return {"draft": text}
    except Exception as e:
        print("[AI ERROR /draft]", e)
        raise HTTPException(status_code=500, detail="Draft unavailable")

app.include_router(ai)

# =============================
# TEST DB
# =============================
@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1;")
        return {"db": "ok"}
    except Exception as e:
        return {"db_error": str(e)}
