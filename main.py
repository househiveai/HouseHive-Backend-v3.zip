# -------------------------------
# HouseHive Backend API v5
# -------------------------------

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import stripe
import os
from dotenv import load_dotenv

# ✅ Load environment variables (for Stripe keys)
load_dotenv()

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Allow frontend connections (Vercel + custom domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://househive.vercel.app",
        "https://househive.ai"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Stripe setup
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# -------------------------------
# BASIC ROUTES
# -------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/login")
@app.post("/auth/login")
async def login(request: Request):
    """Demo login route — accepts JSON or Form"""
    try:
        data = await request.json()
        email = data.get("email")
        password = data.get("password")
    except:
        form = await request.form()
        email = form.get("email")
        password = form.get("password")

    if email == "demo@househive.ai" and password == "password123":
        return {"success": True, "token": "househive-demo-token"}

    return {"success": False, "error": "Invalid credentials"}

@app.get("/auth/me")
def get_user():
    return {
        "email": "demo@househive.ai",
        "name": "Demo User",
        "plan": "Premium",
        "role": "Owner"
    }

# -------------------------------
# MOCK DATA ROUTES
# -------------------------------

@app.get("/api/properties")
def get_properties():
    return [
        {"id": 1, "name": "Luxury Condo", "address": "123 Ocean Ave"},
        {"id": 2, "name": "Downtown Loft", "address": "456 City St"},
        {"id": 3, "name": "Beach House", "address": "789 Sunset Blvd"},
    ]

@app.get("/api/maintenance")
def get_maintenance():
    return [
        {"id": 1, "task": "Fix leaky faucet", "status": "Pending"},
        {"id": 2, "task": "Replace smoke detector", "status": "Completed"},
        {"id": 3, "task": "Check HVAC filter", "status": "In Progress"},
    ]

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {"id": 101, "property": "Downtown Condo", "task": "Fix sink leak", "status": "Open"},
            {"id": 102, "property": "Beach House", "task": "Replace smoke alarm", "status": "Completed"},
        ]
    }

# -------------------------------
# STRIPE PAYMENT ROUTES
# -------------------------------

@app.post("/api/create-checkout-session")
async def create_checkout_session():
    """Create a Stripe Checkout session"""
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "HouseHive Premium Plan"},
                        "unit_amount": 1999,  # $19.99/month
                    },
                    "quantity": 1,
                }
            ],
            success_url="https://househive.vercel.app/billing/success",
            cancel_url="https://househive.vercel.app/billing/cancel",
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/billing-portal")
async def billing_portal():
    """Create Stripe Billing Portal session"""
    try:
        session = stripe.billing_portal.Session.create(
            customer=os.getenv("STRIPE_CUSTOMER_ID", "cus_demo123"),
            return_url="https://househive.vercel.app/billing",
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
from openai import OpenAI
import json

# ✅ AI Assistant endpoint (HiveBot)
from openai import OpenAI
from fastapi import Request, HTTPException
from typing import Dict, List
import os

# Store chat history per user session
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

@app.post("/api/ai/chat")
async def ai_chat(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id", "guest")  # later can link to actual login
        user_msg = data.get("message", "").strip()
        system_prompt = data.get(
            "system_prompt",
            "You are HiveBot, an intelligent AI co-host for HouseHive.ai. You help manage properties, tasks, and guests in a friendly and efficient tone."
        )

        if not user_msg:
            raise HTTPException(status_code=400, detail="Missing message content")

        # Initialize session if not exists
        if user_id not in chat_sessions:
            chat_sessions[user_id] = [{"role": "system", "content": system_prompt}]

        # Add user message
        chat_sessions[user_id].append({"role": "user", "content": user_msg})

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-5",
            messages=chat_sessions[user_id],
            temperature=0.7,
            max_tokens=500
        )

        ai_reply = response.choices[0].message.content.strip()

        # Save AI reply to history
        chat_sessions[user_id].append({"role": "assistant", "content": ai_reply})

        return {"reply": ai_reply, "history": chat_sessions[user_id][-10:]}  # return last 10 messages

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------
# ROOT
# -------------------------------

@app.get("/")
def home():
    return {"message": "Welcome to HouseHive Backend API v5!"}

