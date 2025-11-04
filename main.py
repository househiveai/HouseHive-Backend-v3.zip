from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
app = FastAPI(title="HouseHive Backend v3", version="1.0.0")

SECRET_KEY = "househive_secret_key_123"  # change this before production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# CORS
origins = [
    "https://househive.ai",
    "https://www.househive.ai",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_users_db = {}  # Temporary in-memory store


# -------------------------------------------------------
# MODELS
# -------------------------------------------------------
class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def create_access_token(data: dict):
    """Generate a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "HouseHive Backend v3"}


@app.get("/api/healthz")
def health():
    return {"status": "healthy"}


@app.get("/api/readyz")
def ready():
    return {"status": "ready"}


@app.post("/auth/register")
def register_user(req: RegisterRequest):
    """Register a new user (temporary in-memory)."""
    if req.email in fake_users_db:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = get_password_hash(req.password)
    fake_users_db[req.email] = {"password": hashed_pw}
    return {"message": "User registered successfully"}


@app.post("/auth/login", response_model=TokenResponse)
def login_user(req: LoginRequest):
    """Authenticate user and return JWT token."""
    user = fake_users_db.get(req.email)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    token = create_access_token({"sub": req.email})
    return {"token": token, "token_type": "bearer"}


@app.get("/auth/me")
def get_me(token: str):
    """Verify a provided JWT and return the user email."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"email": email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
