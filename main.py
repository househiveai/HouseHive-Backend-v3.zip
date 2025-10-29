# -------------------------------
# HouseHive Backend API v5
# -------------------------------

from fastapi import FastAPI, Form, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import stripe
import os
from dotenv import load_dotenv
import sqlite3
from pathlib import Path
import bcrypt
import jwt
from datetime import datetime, timedelta

# -------------------------------
# JWT CONFIG
# -------------------------------
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

# -------------------------------
# DATABASE
# -------------------------------
DB_PATH = Path("househive.db")

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

# -------------------------------
# FASTAPI SETUP
# -------------------------------
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can later replace "*" with your production domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# -------------------------------
# BASIC ROUTES
# -------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


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
        data = await request.json()
        email = data.get("email")
        password = data.get("password")
    except:
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

    token = create_token({"user_id": row[0], "email": email, "plan": row[2]})
    return {"success": True, "token": token, "plan": row[2]}


# ✅ GET AUTHENTICATED USER
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

@app.get("/api/properties")
def get_properties():
    return get_all_properties()

@app.get("/api/maintenance")
def get_maintenance():
    return get_all_tasks()


# -------------------------------
# STRIPE PAYMENT ROUTES
# -------------------------------

@app.post("/api/create-checkout-session")
async def create_checkout_session():
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "HouseHive Premium Plan"},
                        "unit_amount": 1999,
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
    try:
        session = stripe.billing_portal.Session.create(
            customer=os.getenv("STRIPE_CUSTOMER_ID", "cus_demo123"),
            return_url="https://househive.vercel.app/billing",
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------
# ROOT
# -------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to HouseHive Backend API v5!"}
