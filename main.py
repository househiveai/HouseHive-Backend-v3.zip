import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

import uvicorn
import openai
import stripe
import jwt

# ---------------------------
# Environment configuration
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

VERCEL_URL = os.getenv("VERCEL_URL", "")
CORS_ORIGINS = list(
    filter(
        None,
        [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            VERCEL_URL,
            os.getenv("CORS_EXTRA_ORIGIN", ""),
        ],
    )
)

# ---------------------------
# App bootstrap
# ---------------------------
app = FastAPI(title="HouseHive Backend", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auth_scheme = HTTPBearer(auto_error=False)

# ---------------------------
# Pydantic models
# ---------------------------
class HealthResp(BaseModel):
    status: str
    time: str

class LoginRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User ID (optional)")
    email: Optional[str] = Field(None, description="Email (optional)")
    name: Optional[str] = None

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

class CheckoutSessionRequest(BaseModel):
    price_id: str
    mode: str = "subscription"
    success_url: str
    cancel_url: str
    customer_email: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

class CheckoutSessionResponse(BaseModel):
    id: str
    url: str

# ---------------------------
# Helpers
# ---------------------------
def create_jwt(payload: dict, exp_minutes: int = JWT_EXP_MIN) -> str:
    to_encode = payload.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(minutes=exp_minutes)
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)

def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="Auth token required")
    return verify_jwt(creds.credentials)

# ---------------------------
# Health & root
# ---------------------------
@app.get("/", response_model=HealthResp)
def root():
    return HealthResp(status="ok", time=datetime.utcnow().isoformat() + "Z")

@app.get("/healthz", response_model=HealthResp)
def healthz():
    return HealthResp(status="healthy", time=datetime.utcnow().isoformat() + "Z")

@app.get("/readyz", response_model=HealthResp)
def readyz():
    return HealthResp(status="ready", time=datetime.utcnow().isoformat() + "Z")

# ---------------------------
# Auth (frontend compatible)
# ---------------------------
@app.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    """
    Accepts either user_id or email and returns a JWT.
    Compatible with simple frontend login forms (email + password).
    """
    user_identifier = body.user_id or body.email
    if not user_identifier:
        raise HTTPException(status_code=400, detail="Missing user_id or email")

    token = create_jwt(
        {
            "sub": user_identifier,
            "email": body.email,
            "name": body.name,
            "scope": "user",
        }
    )
    return LoginResponse(token=token, expires_in=JWT_EXP_MIN * 60)

# ---------------------------
# AI Chat (OpenAI legacy SDK)
# ---------------------------
@app.post("/ai/chat", response_model=ChatResponse)
def ai_chat(req: ChatRequest, _user=Depends(get_current_user)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    model = req.model or OPENAI_MODEL
    try:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[m.dict() for m in req.messages],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        reply = completion["choices"][0]["message"]["content"]
        usage = completion.get("usage")
        return ChatResponse(reply=reply, model=model, usage=usage)
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {str(e)}")

# ---------------------------
# Stripe: Checkout Session
# ---------------------------
@app.post("/billing/create-checkout-session", response_model=CheckoutSessionResponse)
def create_checkout_session(body: CheckoutSessionRequest, _user=Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Missing STRIPE_SECRET_KEY")

    try:
        session = stripe.checkout.Session.create(
            mode=body.mode,
            line_items=[{"price": body.price_id, "quantity": 1}],
            success_url=body.success_url,
            cancel_url=body.cancel_url,
            customer_email=body.customer_email,
            metadata=body.metadata or {},
            allow_promotion_codes=True,
            automatic_tax={"enabled": True},
        )
        return CheckoutSessionResponse(id=session["id"], url=session["url"])
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=502, detail=f"Stripe error: {str(e)}")

# ---------------------------
# Stripe: Webhook
# ---------------------------
@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        payload = await request.body()
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            data = {}
        event_type = data.get("type")
        return {"received": True, "unchecked": True, "type": event_type}

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
    elif event["type"] == "customer.subscription.updated":
        sub = event["data"]["object"]
    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]

    return {"received": True, "type": event["type"]}

# ---------------------------
# Utility: verify token
# ---------------------------
@app.get("/auth/verify")
def verify_token(_user=Depends(get_current_user)):
    return {"ok": True}

# ---------------------------
# Local dev
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        reload=bool(os.getenv("DEV_RELOAD", "0") == "1"),
    )
