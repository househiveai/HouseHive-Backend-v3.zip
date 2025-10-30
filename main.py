# -------------------------------
# HouseHive Backend API v5 (JWT + CRUD + Stripe)
# -------------------------------

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3, os, bcrypt, jwt, stripe, json

# -------------------------------
# ENV / CONFIG
# -------------------------------
from dotenv import load_dotenv
load_dotenv()

DB_PATH = Path("househive.db")

JWT_SECRET = os.getenv("JWT_SECRET", "househive_secret_fallback")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://househive.ai")
VERCEL_URL   = os.getenv("VERCEL_URL",   "https://househive.vercel.app")

STRIPE_SECRET_KEY      = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET", "")
stripe.api_key = STRIPE_SECRET_KEY

# Optional: map plan codes -> Stripe Price IDs (recommended)
# If you already created Prices in Stripe Dashboard, put their IDs here:
# Example: price_123 for $15/mo, price_456 for $29/mo, price_789 for $99/mo
PLAN_PRICE_IDS = {
    "cohost": os.getenv("PRICE_COHOST_ID", ""),
    "pro":    os.getenv("PRICE_PRO_ID",    ""),
    "agency": os.getenv("PRICE_AGENCY_ID", ""),
}

# -------------------------------
# APP
# -------------------------------
app = FastAPI(title="HouseHive Backend API v5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://house-hive-frontend-js-brand-zip.vercel.app",
        "https://househive.ai",
        "https://www.househive.ai"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# -------------------------------
# DB INIT
# -------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # users
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password TEXT,
            plan TEXT DEFAULT 'Free',
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT
        )
    """)
    # properties (per user)
    c.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            address TEXT,
            rent REAL,
            status TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    # tasks (per user)
    c.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            property_id INTEGER,
            property_name TEXT,
            task TEXT,
            status TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(property_id) REFERENCES properties(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------------------
# JWT Helpers
# -------------------------------
def create_token(payload: Dict[str, Any]) -> str:
    data = payload.copy()
    data["exp"] = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(authorization: str = Header(None)) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    payload = verify_token(token)
    # Minimal checks
    if not payload.get("user_id") or not payload.get("email"):
        raise HTTPException(status_code=401, detail="Invalid token data")
    return payload  # includes user_id, email, plan maybe

# -------------------------------
# Models (lightweight)
# -------------------------------
class RegisterBody(BaseModel):
    email: str
    password: str

class LoginBody(BaseModel):
    email: str
    password: str

class PropertyBody(BaseModel):
    name: str
    address: str
    rent: float
    status: str = "Active"

class TaskBody(BaseModel):
    property_id: Optional[int] = None
    property_name: Optional[str] = None
    task: str
    status: str = "Open"

class CheckoutBody(BaseModel):
    plan: str  # 'cohost' | 'pro' | 'agency'

# -------------------------------
# Utils: DB
# -------------------------------
def db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def get_user_by_email(email: str):
    conn = db(); c = conn.cursor()
    c.execute("SELECT id, email, password, plan, stripe_customer_id, stripe_subscription_id FROM users WHERE email=?", (email,))
    row = c.fetchone(); conn.close()
    return row

def get_user_by_id(user_id: int):
    conn = db(); c = conn.cursor()
    c.execute("SELECT id, email, password, plan, stripe_customer_id, stripe_subscription_id FROM users WHERE id=?", (user_id,))
    row = c.fetchone(); conn.close()
    return row

def update_user_plan_and_stripe(user_id: int, plan: str, customer_id: Optional[str], subscription_id: Optional[str]):
    conn = db(); c = conn.cursor()
    c.execute("""
        UPDATE users
        SET plan = ?, stripe_customer_id = COALESCE(?, stripe_customer_id),
            stripe_subscription_id = COALESCE(?, stripe_subscription_id)
        WHERE id = ?
    """, (plan, customer_id, subscription_id, user_id))
    conn.commit(); conn.close()

# -------------------------------
# Health
# -------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

# -------------------------------
# Auth
# -------------------------------
@app.post("/api/register")
def register(body: RegisterBody):
    email = body.email.strip().lower()
    if not email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password required")
    if get_user_by_email(email):
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_pw = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()

    # Optional: create Stripe customer now
    stripe_customer_id = None
    if STRIPE_SECRET_KEY:
        sc = stripe.Customer.create(email=email, metadata={"househive": "true"})
        stripe_customer_id = sc["id"]

    conn = db(); c = conn.cursor()
    c.execute("INSERT INTO users (email, password, stripe_customer_id) VALUES (?, ?, ?)",
              (email, hashed_pw, stripe_customer_id))
    conn.commit(); user_id = c.lastrowid; conn.close()

    token = create_token({"user_id": user_id, "email": email, "plan": "Free"})
    return {"success": True, "token": token, "plan": "Free"}

@app.post("/api/login")
async def login(request: Request):
    # Accept JSON or form
    try:
        data = await request.json()
    except:
        data = await request.form()

    email = (data.get("email") or "").strip().lower()
    password = data.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Missing email or password")

    row = get_user_by_email(email)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_id, _, hashed_pw, plan, _, _ = row
    if not bcrypt.checkpw(password.encode(), hashed_pw.encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token({"user_id": user_id, "email": email, "plan": plan})
    return {"success": True, "token": token, "plan": plan}

@app.get("/auth/me")
def me(user = Depends(get_current_user)):
    row = get_user_by_id(int(user["user_id"]))
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    _, email, _, plan, stripe_customer_id, stripe_subscription_id = row
    return {
        "email": email,
        "plan": plan,
        "role": "Owner",
        "name": email.split("@")[0].capitalize(),
        "stripe_customer_id": stripe_customer_id,
        "stripe_subscription_id": stripe_subscription_id
    }

# -------------------------------
# Properties (CRUD, per user)
# -------------------------------
@app.get("/api/properties")
def list_properties(user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("SELECT id, name, address, rent, status FROM properties WHERE user_id=? ORDER BY id DESC", (user["user_id"],))
    rows = c.fetchall(); conn.close()
    return [{"id": r[0], "name": r[1], "address": r[2], "rent": r[3], "status": r[4]} for r in rows]

@app.post("/api/properties")
def create_property(body: PropertyBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        INSERT INTO properties (user_id, name, address, rent, status)
        VALUES (?, ?, ?, ?, ?)
    """, (user["user_id"], body.name, body.address, body.rent, body.status))
    conn.commit(); pid = c.lastrowid; conn.close()
    return {"success": True, "id": pid}

@app.put("/api/properties/{property_id}")
def update_property(property_id: int, body: PropertyBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        UPDATE properties
        SET name=?, address=?, rent=?, status=?
        WHERE id=? AND user_id=?
    """, (body.name, body.address, body.rent, body.status, property_id, user["user_id"]))
    conn.commit(); updated = (c.rowcount > 0); conn.close()
    if not updated: raise HTTPException(status_code=404, detail="Property not found")
    return {"success": True}

@app.delete("/api/properties/{property_id}")
def delete_property(property_id: int, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("DELETE FROM properties WHERE id=? AND user_id=?", (property_id, user["user_id"]))
    conn.commit(); deleted = (c.rowcount > 0); conn.close()
    if not deleted: raise HTTPException(status_code=404, detail="Property not found")
    return {"success": True}

# -------------------------------
# Tasks (CRUD, per user)
# -------------------------------
@app.get("/api/maintenance")
def list_tasks(user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        SELECT id, property_id, property_name, task, status
        FROM tasks WHERE user_id=? ORDER BY id DESC
    """, (user["user_id"],))
    rows = c.fetchall(); conn.close()
    return [{"id": r[0], "property_id": r[1], "property_name": r[2], "task": r[3], "status": r[4]} for r in rows]

@app.post("/api/maintenance")
def create_task(body: TaskBody, user = Depends(get_current_user)):
    # Allow either property_id (preferred) or property_name (legacy)
    # If property_id is provided, we can fetch and store its name for convenience
    pname = body.property_name
    if body.property_id and not pname:
        conn = db(); c = conn.cursor()
        c.execute("SELECT name FROM properties WHERE id=? AND user_id=?", (body.property_id, user["user_id"]))
        r = c.fetchone(); conn.close()
        if r: pname = r[0]

    conn = db(); c = conn.cursor()
    c.execute("""
        INSERT INTO tasks (user_id, property_id, property_name, task, status)
        VALUES (?, ?, ?, ?, ?)
    """, (user["user_id"], body.property_id, pname, body.task, body.status))
    conn.commit(); tid = c.lastrowid; conn.close()
    return {"success": True, "id": tid}

@app.put("/api/maintenance/{task_id}")
def update_task(task_id: int, body: TaskBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        UPDATE tasks SET property_id=?, property_name=?, task=?, status=?
        WHERE id=? AND user_id=?
    """, (body.property_id, body.property_name, body.task, body.status, task_id, user["user_id"]))
    conn.commit(); updated = (c.rowcount > 0); conn.close()
    if not updated: raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True}

@app.delete("/api/maintenance/{task_id}")
def delete_task(task_id: int, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("DELETE FROM tasks WHERE id=? AND user_id=?", (task_id, user["user_id"]))
    conn.commit(); deleted = (c.rowcount > 0); conn.close()
    if not deleted: raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True}

# -------------------------------
# Stripe: Checkout + Billing Portal + Webhook
# -------------------------------
@app.post("/api/create-checkout-session")
def create_checkout_session(body: CheckoutBody, user = Depends(get_current_user)):
    plan = body.plan.lower()
    price_id = PLAN_PRICE_IDS.get(plan, "")

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=400, detail="Stripe not configured")

    # Ensure customer exists
    uid = int(user["user_id"])
    urow = get_user_by_id(uid)
    if not urow:
        raise HTTPException(status_code=404, detail="User not found")

    _, email, _, current_plan, stripe_customer_id, _ = urow
    if not stripe_customer_id:
        sc = stripe.Customer.create(email=email, metadata={"househive_user_id": str(uid)})
        stripe_customer_id = sc["id"]
        update_user_plan_and_stripe(uid, current_plan, stripe_customer_id, None)

    # Prefer real price IDs if provided; else fall back to inline price_data
    if price_id:
        line_items = [{"price": price_id, "quantity": 1}]
    else:
        # Fallback demo price (change amount/name to your product)
        line_items = [{
            "price_data": {
                "currency": "usd",
                "product_data": {"name": f"HouseHive {plan.title()} Plan"},
                "recurring": {"interval": "month"},
                "unit_amount": 1999 if plan == "cohost" else (2900 if plan == "pro" else 9900),
            },
            "quantity": 1,
        }]

    session = stripe.checkout.Session.create(
        customer=stripe_customer_id,
        mode="subscription",
        line_items=line_items,
        success_url=f"{FRONTEND_URL}/billing/success",
        cancel_url=f"{FRONTEND_URL}/billing/cancel",
        metadata={
            "househive_user_id": str(uid),
            "plan_code": plan
        },
        subscription_data={
            "metadata": {
                "househive_user_id": str(uid),
                "plan_code": plan
            }
        }
    )
    return {"url": session.url}

@app.post("/api/billing-portal")
def billing_portal(user = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=400, detail="Stripe not configured")

    uid = int(user["user_id"])
    row = get_user_by_id(uid)
    if not row: raise HTTPException(status_code=404, detail="User not found")
    _, email, _, current_plan, stripe_customer_id, _ = row

    if not stripe_customer_id:
        sc = stripe.Customer.create(email=email, metadata={"househive_user_id": str(uid)})
        stripe_customer_id = sc["id"]
        update_user_plan_and_stripe(uid, current_plan, stripe_customer_id, None)

    portal = stripe.billing_portal.Session.create(
        customer=stripe_customer_id,
        return_url=f"{FRONTEND_URL}/billing"
    )
    return {"url": portal.url}

@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        # If not set, accept as plaintext (dev only)
        payload = await request.body()
        try:
            event = json.loads(payload.decode("utf-8"))
        except Exception:
            return PlainTextResponse("Invalid payload", status_code=400)
    else:
        payload = await request.body()
        sig = request.headers.get("Stripe-Signature")
        try:
            event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
        except Exception:
            return PlainTextResponse("Invalid signature", status_code=400)

    etype = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    # Subscription created/updated/deleted => set user's plan + subscription id
    if etype in ("customer.subscription.created", "customer.subscription.updated"):
        sub = data
        customer_id = sub.get("customer")
        sub_id = sub.get("id")
        plan_code = (sub.get("metadata", {}) or {}).get("plan_code", "")
        # Fallback to "plan.nickname" if needed
        if not plan_code:
            try:
                plan_code = (sub["plan"]["nickname"] or "").lower()
            except:
                plan_code = "premium"

        # Find user by stripe_customer_id
        conn = db(); c = conn.cursor()
        c.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,))
        r = c.fetchone(); conn.close()
        if r:
            uid = r[0]
            update_user_plan_and_stripe(uid, plan_code.title(), customer_id, sub_id)

    if etype == "customer.subscription.deleted":
        sub = data
        customer_id = sub.get("customer")
        conn = db(); c = conn.cursor()
        c.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,))
        r = c.fetchone(); conn.close()
        if r:
            uid = r[0]
            update_user_plan_and_stripe(uid, "Free", customer_id, None)

    return PlainTextResponse("OK", status_code=200)

# -------------------------------
# Root
# -------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to HouseHive Backend API v5!"}

