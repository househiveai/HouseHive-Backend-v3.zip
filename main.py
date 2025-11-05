# main.py
import os
import datetime as dt
from typing import Optional, List

from fastapi import FastAPI, Depends, Header, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, func
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
import requests  # used for optional OpenAI call


# =============================
# CONFIG
# =============================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./househive.db")
JWT_SECRET   = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PROD")
JWT_ALG      = "HS256"
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "60"))

# ✅ Your real frontend domains **must be here**
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://househive.ai",
    "https://www.househive.ai",
    # ✅ Your Vercel deployment:
    "https://househive-frontend-c6g16o9yc-househives-projects.vercel.app",
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


# =============================
# DB SETUP
# =============================
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, future=True, echo=False, connect_args=connect_args)
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
    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(255), unique=True, index=True, nullable=False)
    name          = Column(String(255))
    password_hash = Column(String(255), nullable=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Property(Base):
    __tablename__ = "properties"
    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(255), nullable=False)
    address     = Column(String(255))
    owner_email = Column(String(255), nullable=False, index=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Tenant(Base):
    __tablename__ = "tenants"
    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(255), nullable=False)
    email       = Column(String(255))
    phone       = Column(String(50))
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=False, index=True)
    owner_email = Column(String(255), nullable=False, index=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Task(Base):
    __tablename__ = "tasks"
    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String(255), nullable=False)
    description = Column(String(1024))
    status      = Column(String(50), default="open", index=True)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=True, index=True)
    owner_email = Column(String(255), nullable=False, index=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Reminder(Base):
    __tablename__ = "reminders"
    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String(255), nullable=False)
    due_date    = Column(DateTime(timezone=True))
    completed   = Column(Boolean, default=False)
    property_id = Column(Integer, ForeignKey("properties.id"), nullable=True, index=True)
    tenant_id   = Column(Integer, ForeignKey("tenants.id"), nullable=True, index=True)
    owner_email = Column(String(255), nullable=False, index=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

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

def bearer_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization: return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None

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
    class Config: from_attributes = True

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

class PropertyCreate(BaseModel):
    name: str
    address: Optional[str] = None

class PropertyOut(BaseModel):
    id: int
    name: str
    address: Optional[str]
    owner_email: str
    created_at: dt.datetime
    class Config: from_attributes = True

class TenantCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    property_id: int

class TenantOut(BaseModel):
    id: int
    name: str
    email: Optional[str]
    phone: Optional[str]
    property_id: int
    owner_email: str
    created_at: dt.datetime
    class Config: from_attributes = True

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    property_id: Optional[int] = None

class TaskOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    status: str
    property_id: Optional[int]
    owner_email: str
    created_at: dt.datetime
    class Config: from_attributes = True

class ReminderCreate(BaseModel):
    title: str
    due_date: Optional[dt.datetime] = None
    property_id: Optional[int] = None
    tenant_id: Optional[int] = None

class ReminderOut(BaseModel):
    id: int
    title: str
    due_date: Optional[dt.datetime]
    completed: bool
    property_id: Optional[int]
    tenant_id: Optional[int]
    owner_email: str
    created_at: dt.datetime
    class Config: from_attributes = True

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


# =============================
# ✅ CREATE APP
# =============================
app = FastAPI()

# ✅ APPLY CORS ONCE — CORRECTLY
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "househive-backend", "status": "running"}


# =============================
# AUTH ROUTES
# =============================
auth = APIRouter(prefix="/api/auth", tags=["auth"])

@auth.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email.lower()).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=payload.email.lower(), name=payload.name, password_hash=hash_password(payload.password))
    db.add(user); db.commit(); db.refresh(user)

    token = create_access_token(user.email)
    return TokenResponse(access_token=token, user=user)


@auth.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == payload.email.lower()).first()
    if not u or not verify_password(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(u.email), user=u)

@auth.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user

app.include_router(auth)


# =============================
# PROPERTY ROUTES
# =============================
prop = APIRouter(prefix="/api/properties", tags=["properties"])

@prop.post("/", response_model=PropertyOut)
def create_property(payload: PropertyCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = Property(name=payload.name, address=payload.address, owner_email=user.email)
    db.add(p); db.commit(); db.refresh(p)
    return p

@prop.get("/", response_model=List[PropertyOut])
def list_properties(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Property).filter(Property.owner_email == user.email).order_by(Property.created_at.desc()).all()

app.include_router(prop)


# =============================
# TENANTS
# =============================
ten = APIRouter(prefix="/api/tenants", tags=["tenants"])

@ten.post("/", response_model=TenantOut)
def create_tenant(payload: TenantCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    prop_obj = db.query(Property).filter(Property.id == payload.property_id, Property.owner_email == user.email).first()
    if not prop_obj: raise HTTPException(status_code=404, detail="Property not found")
    t = Tenant(name=payload.name, email=payload.email, phone=payload.phone, property_id=payload.property_id, owner_email=user.email)
    db.add(t); db.commit(); db.refresh(t)
    return t

@ten.get("/", response_model=List[TenantOut])
def list_tenants(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Tenant).filter(Tenant.owner_email == user.email).order_by(Tenant.created_at.desc()).all()

app.include_router(ten)


# =============================
# TASKS
# =============================
tsk = APIRouter(prefix="/api/tasks", tags=["tasks"])

@tsk.post("/", response_model=TaskOut)
def create_task(payload: TaskCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if payload.property_id:
        prop_obj = db.query(Property).filter(Property.id == payload.property_id, Property.owner_email == user.email).first()
        if not prop_obj: raise HTTPException(status_code=404, detail="Property not found")
    t = Task(title=payload.title, description=payload.description, property_id=payload.property_id, owner_email=user.email)
    db.add(t); db.commit(); db.refresh(t)
    return t

@tsk.get("/", response_model=List[TaskOut])
def list_tasks(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Task).filter(Task.owner_email == user.email).order_by(Task.created_at.desc()).all()

@tsk.post("/{task_id}/done", response_model=TaskOut)
def mark_done(task_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    t = db.query(Task).filter(Task.id == task_id, Task.owner_email == user.email).first()
    if not t: raise HTTPException(status_code=404, detail="Task not found")
    t.status = "done"; db.commit(); db.refresh(t); return t

app.include_router(tsk)


# =============================
# REMINDERS
# =============================
rem = APIRouter(prefix="/api/reminders", tags=["reminders"])

@rem.post("/", response_model=ReminderOut)
def create_reminder(payload: ReminderCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if payload.property_id and not db.query(Property).filter(Property.id == payload.property_id, Property.owner_email == user.email).first():
        raise HTTPException(status_code=404, detail="Property not found")
    if payload.tenant_id and not db.query(Tenant).filter(Tenant.id == payload.tenant_id, Tenant.owner_email == user.email).first():
        raise HTTPException(status_code=404, detail="Tenant not found")
    r = Reminder(
        title=payload.title, due_date=payload.due_date,
        property_id=payload.property_id, tenant_id=payload.tenant_id,
        owner_email=user.email
    )
    db.add(r); db.commit(); db.refresh(r)
    return r

@rem.get("/", response_model=List[ReminderOut])
def list_reminders(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Reminder).filter(Reminder.owner_email == user.email).order_by(Reminder.created_at.desc()).all()

app.include_router(rem)


# =============================
# INSIGHTS
# =============================
ins = APIRouter(prefix="/api/insights", tags=["insights"])

@ins.get("/")
def get_insights(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    pc = db.query(Property).filter(Property.owner_email == user.email).count()
    tc = db.query(Tenant).filter(Tenant.owner_email == user.email).count()
    oc = db.query(Task).filter(Task.owner_email == user.email, Task.status == "open").count()
    rc = db.query(Reminder).filter(Reminder.owner_email == user.email).count()

    if pc == 0:
        return {"summary": "No data yet. Add your first property to get started!", "property_count": 0, "tenant_count": 0, "open_tasks": 0, "reminders": 0}

    summary = f"{pc} propert{'y' if pc == 1 else 'ies'} with {oc} open maintenance request{'s' if oc != 1 else ''}. {rc} reminder{'s' if rc != 1 else ''} scheduled."
    return {"summary": summary, "property_count": pc, "tenant_count": tc, "open_tasks": oc, "reminders": rc}

app.include_router(ins)


# =============================
# AI CHAT
# =============================
ai = APIRouter(prefix="/api/ai", tags=["ai"])

@ai.post("/chat", response_model=ChatResponse)
def ai_chat(req: ChatRequest, user: User = Depends(get_current_user)):
    if not OPENAI_API_KEY:
        return ChatResponse(reply="HiveBot is online. Add your OPENAI_API_KEY on the backend to enable real AI responses.")

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You are HiveBot, a concise property management assistant."},
                    {"role": "user", "content": req.message}
                ]
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        return ChatResponse(reply=reply.strip())
    except Exception as e:
        return ChatResponse(reply=f"HiveBot could not reach the AI service: {e}")

app.include_router(ai)
