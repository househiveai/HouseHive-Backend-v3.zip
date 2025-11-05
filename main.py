# main.py
import os
import datetime as dt
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from jose import jwt, JWTError

# -----------------------------
# CONFIG
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./househive.db")
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PROD")
JWT_ALG = "HS256"
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "60"))
CORS_ORIGINS = [
    *[o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()],
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://househive-frontend.vercel.app",
]

# -----------------------------
# DATABASE SETUP
# -----------------------------
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

# -----------------------------
# MODELS
# -----------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255))
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Property(Base):
    __tablename__ = "properties"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    address = Column(String(255))
    owner_email = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255))
    property_id = Column(Integer)
    owner_email = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(String(255))
    status = Column(String(50), default="open")
    property_id = Column(Integer)
    owner_email = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Reminder(Base):
    __tablename__ = "reminders"
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    due_date = Column(DateTime(timezone=True))
    completed = Column(Boolean, default=False)
    owner_email = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

Base.metadata.create_all(bind=engine)

# -----------------------------
# SECURITY
# -----------------------------
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p): return pwd_ctx.hash(p)
def verify_password(p, h): return pwd_ctx.verify(p, h)
def create_access_token(sub): 
    exp = dt.datetime.utcnow() + dt.timedelta(minutes=JWT_EXPIRES_MIN)
    return jwt.encode({"sub": sub, "exp": exp}, JWT_SECRET, algorithm=JWT_ALG)

def bearer_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization: return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer": return parts[1]
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

# -----------------------------
# SCHEMAS
# -----------------------------
class UserOut(BaseModel):
    id: int; email: EmailStr; name: Optional[str]; created_at: dt.datetime
    class Config: from_attributes = True

class UserCreate(BaseModel):
    email: EmailStr; password: str = Field(min_length=6); name: Optional[str]

class LoginRequest(BaseModel): email: EmailStr; password: str
class TokenResponse(BaseModel): access_token: str; token_type: str = "bearer"; user: UserOut

class PropertyCreate(BaseModel): name: str; address: Optional[str]
class PropertyOut(PropertyCreate): id: int; owner_email: str; created_at: dt.datetime
class TenantCreate(BaseModel): name: str; email: Optional[str]; property_id: Optional[int]
class TaskCreate(BaseModel): title: str; description: Optional[str]; property_id: Optional[int]
class ReminderCreate(BaseModel): title: str; due_date: Optional[dt.datetime]
class SimpleOut(BaseModel): id: int; title: str; created_at: dt.datetime

# -----------------------------
# APP
# -----------------------------
app = FastAPI(title="HouseHive Backend", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health(): return {"ok": True, "service": "househive-backend", "time": dt.datetime.utcnow().isoformat()}

# -----------------------------
# AUTH
# -----------------------------
auth = APIRouter(prefix="/api/auth", tags=["auth"])

@auth.post("/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email.lower()).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=payload.email.lower(), name=payload.name, password_hash=hash_password(payload.password))
    db.add(user); db.commit(); db.refresh(user); return user

@auth.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == payload.email.lower()).first()
    if not u or not verify_password(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(u.email), user=u)

@auth.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)): return user
app.include_router(auth)

# -----------------------------
# PROPERTIES
# -----------------------------
prop = APIRouter(prefix="/api/properties", tags=["properties"])
@prop.post("/", response_model=PropertyOut)
def add_property(payload: PropertyCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = Property(name=payload.name, address=payload.address, owner_email=user.email)
    db.add(p); db.commit(); db.refresh(p); return p
@prop.get("/", response_model=List[PropertyOut])
def get_properties(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Property).filter(Property.owner_email == user.email).order_by(Property.created_at.desc()).all()
app.include_router(prop)

# -----------------------------
# TENANTS
# -----------------------------
ten = APIRouter(prefix="/api/tenants", tags=["tenants"])
@ten.post("/", response_model=TenantCreate)
def add_tenant(payload: TenantCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    t = Tenant(name=payload.name, email=payload.email, property_id=payload.property_id, owner_email=user.email)
    db.add(t); db.commit(); return payload
@ten.get("/", response_model=List[TenantCreate])
def get_tenants(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Tenant).filter(Tenant.owner_email == user.email).all()
app.include_router(ten)

# -----------------------------
# TASKS
# -----------------------------
tsk = APIRouter(prefix="/api/tasks", tags=["tasks"])
@tsk.post("/", response_model=SimpleOut)
def add_task(payload: TaskCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    t = Task(title=payload.title, description=payload.description, property_id=payload.property_id, owner_email=user.email)
    db.add(t); db.commit(); db.refresh(t); return t
@tsk.get("/", response_model=List[SimpleOut])
def get_tasks(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Task).filter(Task.owner_email == user.email).order_by(Task.created_at.desc()).all()
app.include_router(tsk)

# -----------------------------
# REMINDERS
# -----------------------------
rem = APIRouter(prefix="/api/reminders", tags=["reminders"])
@rem.post("/", response_model=SimpleOut)
def add_rem(payload: ReminderCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    r = Reminder(title=payload.title, due_date=payload.due_date, owner_email=user.email)
    db.add(r); db.commit(); db.refresh(r); return r
@rem.get("/", response_model=List[SimpleOut])
def get_rems(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Reminder).filter(Reminder.owner_email == user.email).all()
app.include_router(rem)

# -----------------------------
# INSIGHTS
# -----------------------------
ins = APIRouter(prefix="/api/insights", tags=["insights"])
@ins.get("/")
def insights(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    pc = db.query(Property).filter(Property.owner_email == user.email).count()
    tc = db.query(Tenant).filter(Tenant.owner_email == user.email).count()
    oc = db.query(Task).filter(Task.owner_email == user.email, Task.status == "open").count()
    rc = db.query(Reminder).filter(Reminder.owner_email == user.email).count()
    if pc == 0: 
        return {"summary": "No data yet. Add your first property to get started!", "property_count": 0, "tenant_count": 0, "open_tasks": 0, "reminders": 0}
    summary = f"{pc} propert{'y' if pc == 1 else 'ies'} with {oc} open maintenance request{'s' if oc != 1 else ''}. {rc} reminder{'s' if rc != 1 else ''} scheduled."
    return {"summary": summary, "property_count": pc, "tenant_count": tc, "open_tasks": oc, "reminders": rc}
app.include_router(ins)
