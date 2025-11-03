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
# FRONTEND DOMAINS (CORS)
# ---------------------------
CORS_ORIGINS = [
    "https://househive.ai",
    "https://www.househive.ai",
    "https://househive-frontend.vercel.app",
    "http://localhost:3000",
]


# ---------------------------
# APP INITIALIZATION
# ---------------------------
app = FastAPI(title="HouseHive Backend", version="1.0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auth_scheme = HTTPBearer(auto_error=False)


# ---------------------------
# DATA MODELS
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
# AUTH HELPERS
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
# BASIC ROUTES
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
# AUTH ROUTES  (✅ FIXED PATH)
# ---------------------------
@app.post("/api/login", response_model=LoginResponse)
def login(body: LoginRequest):
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Missing email or password")

    token = create_jwt({
        "sub": body.email,
        "email": body.email,
        "scope": "user"
    })

    return LoginResponse(token=token, expires_in=JWT_EXP_MIN * 60)


@app.get("/api/verify")
def verify_token(_user=Depends(get_current_user)):
    return {"ok": True}


# ---------------------------
# AI CHAT ENDPOINT
# ---------------------------
@app.post("/api/ai/chat", response_model=ChatResponse)
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
# STRIPE CHECKOUT
# ---------------------------
@app.post("/api/billing/create-checkout-session", response_model=CheckoutSessionResponse)
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
# STRIPE WEBHOOK
# ---------------------------
@app.post("/api/billing/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    if not STRIPE_WEBHOOK_SECRET:
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            data = {}
        return {"received": True, "unchecked": True, "type": data.get("type")}

    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET
        )
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid signature or payload")

    return {"received": True, "type": event["type"]}


# ---------------------------
# RENDER LOCAL RUNNER
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        reload=bool(os.getenv("DEV_RELOAD", "0") == "1"),
    )
