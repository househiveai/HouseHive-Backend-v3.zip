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
import sqlite3
from pathlib import Path
import jwt
from datetime import datetime, timedelta

JWT_SECRET = os.getenv("JWT_SECRET", "househive_secret_fallback")
JWT_ALGORITHM = "HS256"

def create_token(data: dict, expires_delta: timedelta = timedelta(hours=6)):
    """Create a signed JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str):
    """Verify and decode JWT token."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- DATABASE SETUP ---
DB_PATH = Path("househive.db")

# Create tables if not exist
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS properties (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT,
        rent REAL,
        status TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_name TEXT,
        task TEXT,
        status TEXT
    )
    """)
    c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    password TEXT,
    plan TEXT DEFAULT 'Free'
)
""")

    conn.commit()
    conn.close()

init_db()

# ✅ Load environment variables (for Stripe keys)
load_dotenv()

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Allow frontend connections (Vercel + custom domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ temporarily allow all for testing
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

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import bcrypt, sqlite3, os

# ✅ REGISTER USER
@app.post("/api/register")
async def register(request: Request):
    data = await request.json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_pw))
        conn.commit()
        conn.close()
        return {"success": True, "message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="User already exists")

# ✅ LOGIN USER
@app.post("/api/login")
async def login(request: Request):
    try:
        # Try reading JSON
        data = await request.json()
        email = data.get("email")
        password = data.get("password")
    except:
        # Fallback to form data
        form = await request.form()
        email = form.get("email")
        password = form.get("password")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Missing email or password")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, password, plan FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()

   if not row or not bcrypt.checkpw(password.encode(), row[1].encode()):
    raise HTTPException(status_code=401, detail="Invalid email or password")

# ✅ Generate JWT token for the user
token = create_token({"user_id": row[0], "email": email, "plan": row[2]})
return {"success": True, "token": token, "plan": row[2]}


from fastapi import Header

@app.get("/auth/me")
def get_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]
    payload = verify_token(token)

    user_id = payload.get("user_id")
    email = payload.get("email")
    plan = payload.get("plan")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token data")

    # Optionally load user details from database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT email, plan FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "email": row[0],
        "plan": row[1],
        "role": "Owner",
        "name": email.split("@")[0].capitalize(),
    }


# -------------------------------
# MOCK DATA ROUTES
# -------------------------------

@app.get("/api/properties")
def get_properties():
    return get_all_properties()

@app.get("/api/maintenance")
def get_maintenance():
    return get_all_tasks()


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

from openai import OpenAI
from fastapi import Request, HTTPException
from typing import Dict, List
import os, re

# Memory for chat + temporary storage
chat_sessions: Dict[str, List[Dict[str, str]]] = {}
properties_db: List[Dict[str, str]] = []
tasks_db: List[Dict[str, str]] = []

from openai import OpenAI
from fastapi import Request, HTTPException
import json, os, re, sqlite3

chat_sessions = {}

def insert_property(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO properties (name, address, rent, status) VALUES (?, ?, ?, ?)",
        (data.get("name"), data.get("address"), data.get("rent"), data.get("status"))
    )
    conn.commit()
    conn.close()

def insert_task(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO tasks (property_name, task, status) VALUES (?, ?, ?)",
        (data.get("property_name"), data.get("task"), data.get("status"))
    )
    conn.commit()
    conn.close()

def get_all_properties():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, address, rent, status FROM properties")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "address": r[2], "rent": r[3], "status": r[4]} for r in rows]

def get_all_tasks():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, property_name, task, status FROM tasks")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "property": r[1], "task": r[2], "status": r[3]} for r in rows]

@app.post("/api/ai/chat")
async def ai_chat(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id", "guest")
        user_msg = data.get("message", "").strip()

        if not user_msg:
            raise HTTPException(status_code=400, detail="Missing message")

        # Initialize session memory
        if user_id not in chat_sessions:
            chat_sessions[user_id] = [
                {
                    "role": "system",
                    "content": (
                        "You are HiveBot, an AI co-host for HouseHive.ai. "
                        "You help manage rental properties, maintenance, and guests. "
                        "When a user requests to add a property or maintenance task, "
                        "respond normally but also output JSON after '##ACTION##' describing the action."
                    )
                }
            ]

        chat_sessions[user_id].append({"role": "user", "content": user_msg})

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=chat_sessions[user_id],
            temperature=0.7
        )

        reply = completion.choices[0].message.content.strip()
        chat_sessions[user_id].append({"role": "assistant", "content": reply})

        # Detect AI action
        match = re.search(r"##ACTION##(.*)", reply, re.DOTALL)
        if match:
            try:
                action = json.loads(match.group(1).strip())
                if action.get("type") == "property":
                    insert_property(action["data"])
                elif action.get("type") == "task":
                    insert_task(action["data"])
            except Exception as e:
                print("Action parse error:", e)

        return {
            "reply": reply,
            "history": chat_sessions[user_id][-10:],
            "properties": get_all_properties(),
            "tasks": get_all_tasks()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

        chat_sessions[user_id].append({"role": "user", "content": user_msg})

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        completion = client.chat.completions.create(
            model="gpt-5",
            messages=chat_sessions[user_id],
            temperature=0.7
        )

        ai_reply = completion.choices[0].message.content.strip()
        chat_sessions[user_id].append({"role": "assistant", "content": ai_reply})

        # Detect ##ACTION## JSON
        match = re.search(r"##ACTION##(.*)", ai_reply, re.DOTALL)
        action_data = None
        if match:
            try:
                action_data = eval(match.group(1).strip())
                if action_data.get("type") == "property":
                    properties_db.append(action_data["data"])
                elif action_data.get("type") == "task":
                    tasks_db.append(action_data["data"])
            except Exception as e:
                print("AI Action parse error:", e)

        return {
            "reply": ai_reply,
            "history": chat_sessions[user_id][-10:],
            "properties": properties_db[-5:],
            "tasks": tasks_db[-5:]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------
# ROOT
# -------------------------------

@app.get("/")
def home():
    return {"message": "Welcome to HouseHive Backend API v5!"}

