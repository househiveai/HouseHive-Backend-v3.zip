import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import openai
import stripe
import jwt

# ---------------------------
# ENVIRONMENT CONFIGURATION
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
openai.api_key = OPENAI_API_KEY

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
stripe.api_key = STRIPE_SECRET_KEY

JWT_SECRET = os.getenv("JWT_SECRET", "change_me_in_prod")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXP_MIN = int(os.getenv("JWT_EXP_MIN", "60"))

# ---------------------------
# APP + CORS FIX
# ---------------------------
app = FastAPI(title="HouseHive Backend", version="2.0")

# ✅ Explicit CORS Fix (final)
app = FastAPI(title="HouseHive Backend", version="1.0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://househive.ai",
        "https://www.househive.ai",
        "https://house-hive-frontend-js-brand-zip.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auth_scheme = HTTPBearer(auto_error=False)

# ---------------------------
# MODELS
# ---------------------------
class HealthResp(BaseModel):
    status: str
    time: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    token: str
    token_type: str = "Bearer"
    expires_in: int

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    reply: str
    model: str
    usage: Optional[Dict[str, Any]] = None

# ---------------------------
# AUTH HELPERS
# ---------------------------
def create_jwt(payload: dict, exp_minutes: int = JWT_EXP_MIN) -> str:
    payload = payload.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=exp_minutes)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing or invalid auth token")
    return verify_jwt(creds.credentials)

# ---------------------------
# BASIC ROUTES
# ---------------------------
@app.get("/api/healthz", response_model=HealthResp)
def healthz():
    return HealthResp(status="healthy", time=datetime.utcnow().isoformat() + "Z")

@app.get("/api/readyz", response_model=HealthResp)
def readyz():
    return HealthResp(status="ready", time=datetime.utcnow().isoformat() + "Z")

# ---------------------------
# AUTH ROUTES
# ---------------------------
@app.post("/api/auth/register", response_model=LoginResponse)
def register(body: LoginRequest):
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Missing email or password")

    token = create_jwt({"sub": body.email, "email": body.email})
    return LoginResponse(token=token, expires_in=JWT_EXP_MIN * 60)

@app.post("/api/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Missing email or password")

    token = create_jwt({"sub": body.email, "email": body.email})
    return LoginResponse(token=token, expires_in=JWT_EXP_MIN * 60)

@app.get("/api/auth/verify")
def verify_token(_user=Depends(get_current_user)):
    return {"ok": True}

# ---------------------------
# AI CHAT ENDPOINT
# ---------------------------
@app.post("/api/ai/chat", response_model=ChatResponse)
def ai_chat(req: ChatRequest, _user=Depends(get_current_user)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    try:
        response = openai.ChatCompletion.create(
            model=req.model or OPENAI_MODEL,
            messages=[m.dict() for m in req.messages],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        reply = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        return ChatResponse(reply=reply, model=req.model or OPENAI_MODEL, usage=usage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")

# ---------------------------
# RENDER LOCAL RUNNER
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        reload=True
    )
