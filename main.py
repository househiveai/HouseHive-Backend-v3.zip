# -------------------------------
# HouseHive Backend API v5 (JWT + CRUD + Stripe + AI Chat)
# -------------------------------

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Generator
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3, os, bcrypt, jwt, stripe, json, asyncio

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

PRICE_COHOST = os.getenv("STRIPE_PRICE_COHOST", "")
PRICE_PRO    = os.getenv("STRIPE_PRICE_PRO", "")
PRICE_AGENCY = os.getenv("STRIPE_PRICE_AGENCY", "")

PLAN_PRICE_IDS = {
    "cohost": PRICE_COHOST,
    "pro":    PRICE_PRO,
    "agency": PRICE_AGENCY,
}

ADMIN_EMAIL = (os.getenv("ADMIN_EMAIL", "dntullo@yahoo.com")).strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_FAKE_AI = not bool(OPENAI_API_KEY)
if not USE_FAKE_AI:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# APP
# -------------------------------
app = FastAPI(title="HouseHive Backend API v5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        VERCEL_URL,
        "https://www.househive.ai",
        "https://househive.ai",
        "http://localhost:3000",
        "*",  # dev convenience; tighten later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# DB INIT / HELPERS
# -------------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db(); c = conn.cursor()

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
            notes TEXT,
            rent REAL,
            lease_months INTEGER,
            status TEXT DEFAULT 'Active',
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # tasks (per user)
    c.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            property_id INTEGER,
            title TEXT,
            description TEXT,
            urgent INTEGER DEFAULT 0,
            status TEXT DEFAULT 'open',
            assignee TEXT,
            priority TEXT DEFAULT 'normal',
            due_date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(property_id) REFERENCES properties(id)
        )
    """)

    # tenants (per user)
    c.execute("""
        CREATE TABLE IF NOT EXISTS tenants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            email TEXT,
            phone TEXT,
            property_id INTEGER,
            unit TEXT,
            start_date TEXT,
            end_date TEXT,
            rent REAL,
            frequency TEXT DEFAULT 'monthly',
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(property_id) REFERENCES properties(id)
        )
    """)

    # reminders (per user)
    c.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            tenant_id INTEGER,
            property_id INTEGER,
            title TEXT,
            due_date TEXT,
            amount REAL,
            method TEXT,
            is_paid INTEGER DEFAULT 0,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(tenant_id) REFERENCES tenants(id),
            FOREIGN KEY(property_id) REFERENCES properties(id)
        )
    """)

    conn.commit(); conn.close()

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
    if not payload.get("user_id") or not payload.get("email"):
        raise HTTPException(status_code=401, detail="Invalid token data")
    return payload

# -------------------------------
# Models
# -------------------------------
class RegisterBody(BaseModel):
    email: str
    password: str

class LoginBody(BaseModel):
    email: str
    password: str

class PropertyBody(BaseModel):
    name: str
    address: Optional[str] = ""
    notes: Optional[str] = ""
    rent: Optional[float] = 0.0
    lease_months: Optional[int] = 0
    status: Optional[str] = "Active"

class TaskBody(BaseModel):
    property_id: int
    title: str
    description: Optional[str] = ""
    urgent: Optional[bool] = False
    status: Optional[str] = "open"
    assignee: Optional[str] = ""
    priority: Optional[str] = "normal"
    due_date: Optional[str] = ""  # ISO YYYY-MM-DD

class TenantBody(BaseModel):
    name: str
    email: Optional[str] = ""
    phone: Optional[str] = ""
    property_id: Optional[int] = None
    unit: Optional[str] = ""
    start_date: Optional[str] = ""  # YYYY-MM-DD
    end_date: Optional[str] = ""    # YYYY-MM-DD
    rent: Optional[float] = 0.0
    frequency: Optional[str] = "monthly"

class ReminderBody(BaseModel):
    title: str
    due_date: str   # YYYY-MM-DD
    amount: Optional[float] = 0.0
    tenant_id: Optional[int] = None
    property_id: Optional[int] = None
    method: Optional[str] = "email"  # email/sms/app
    is_paid: Optional[bool] = False
    notes: Optional[str] = ""

class CheckoutBody(BaseModel):
    plan: str  # cohost | pro | agency

# -------------------------------
# Utilities (Users)
# -------------------------------
def get_user_by_email(email: str):
    conn = db(); c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    row = c.fetchone(); conn.close()
    return row

def get_user_by_id(user_id: int):
    conn = db(); c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = c.fetchone(); conn.close()
    return row

def update_user_plan_and_stripe(user_id: int, plan: str, customer_id: Optional[str], subscription_id: Optional[str]):
    conn = db(); c = conn.cursor()
    c.execute("""
        UPDATE users
        SET plan = ?, 
            stripe_customer_id = COALESCE(?, stripe_customer_id),
            stripe_subscription_id = COALESCE(?, stripe_subscription_id)
        WHERE id = ?
    """, (plan, customer_id, subscription_id, user_id))
    conn.commit(); conn.close()

# -------------------------------
# Health
# -------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "name": "HouseHive.ai API", "ts": datetime.utcnow().isoformat()}

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

    # Optional Stripe customer
    stripe_customer_id = None
    if STRIPE_SECRET_KEY:
        sc = stripe.Customer.create(email=email, metadata={"househive": "true"})
        stripe_customer_id = sc["id"]

    conn = db(); c = conn.cursor()
    c.execute("INSERT INTO users (email, password, stripe_customer_id) VALUES (?, ?, ?)",
              (email, hashed_pw, stripe_customer_id))
    conn.commit(); user_id = c.lastrowid; conn.close()

    # Admin backdoor: auto-admin plan if matches ADMIN_EMAIL
    plan = "Admin" if email == ADMIN_EMAIL else "Free"
    if plan == "Admin":
        update_user_plan_and_stripe(user_id, plan, stripe_customer_id, None)

    token = create_token({"user_id": user_id, "email": email, "plan": plan})
    return {"success": True, "token": token, "plan": plan}

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

    hashed_pw = row["password"]
    if not bcrypt.checkpw(password.encode(), hashed_pw.encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token({"user_id": row["id"], "email": email, "plan": row["plan"]})
    return {"success": True, "token": token, "plan": row["plan"]}

@app.get("/auth/me")
def me(user = Depends(get_current_user)):
    row = get_user_by_id(int(user["user_id"]))
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "email": row["email"],
        "plan": row["plan"],
        "role": "Owner" if row["plan"] != "Admin" else "Admin",
        "name": row["email"].split("@")[0].capitalize(),
        "stripe_customer_id": row["stripe_customer_id"],
        "stripe_subscription_id": row["stripe_subscription_id"]
    }

# -------------------------------
# Properties (CRUD)
# -------------------------------
@app.get("/api/properties")
def list_properties(user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        SELECT id, name, address, notes, rent, lease_months, status 
        FROM properties WHERE user_id=?
        ORDER BY id DESC
    """, (user["user_id"],))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

@app.post("/api/properties")
def create_property(body: PropertyBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        INSERT INTO properties (user_id, name, address, notes, rent, lease_months, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user["user_id"], body.name, body.address, body.notes, body.rent, body.lease_months, body.status))
    conn.commit(); pid = c.lastrowid; conn.close()
    return {"success": True, "id": pid}

# -------------------------------
# Tasks (CRUD)
# -------------------------------
@app.get("/api/maintenance")
def list_tasks(status: Optional[str] = None, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    if status:
        c.execute("""
            SELECT * FROM tasks WHERE user_id=? AND status=?
            ORDER BY id DESC
        """, (user["user_id"], status))
    else:
        c.execute("""
            SELECT * FROM tasks WHERE user_id=?
            ORDER BY id DESC
        """, (user["user_id"],))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

@app.post("/api/maintenance")
def create_task(body: TaskBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        INSERT INTO tasks (user_id, property_id, title, description, urgent, status, assignee, priority, due_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user["user_id"], body.property_id, body.title, body.description,
        1 if body.urgent else 0, body.status, body.assignee, body.priority, body.due_date
    ))
    conn.commit(); tid = c.lastrowid; conn.close()
    return {"success": True, "id": tid}

# -------------------------------
# Tenants (CRUD)
# -------------------------------
@app.get("/api/tenants")
def list_tenants(user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        SELECT * FROM tenants WHERE user_id=? ORDER BY id DESC
    """, (user["user_id"],))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

@app.post("/api/tenants")
def create_tenant(body: TenantBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        INSERT INTO tenants (user_id, name, email, phone, property_id, unit, start_date, end_date, rent, frequency)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user["user_id"], body.name, body.email, body.phone, body.property_id,
        body.unit, body.start_date, body.end_date, body.rent, body.frequency
    ))
    conn.commit(); tid = c.lastrowid; conn.close()
    return {"success": True, "id": tid}

# -------------------------------
# Reminders (CRUD)
# -------------------------------
@app.get("/api/reminders")
def list_reminders(user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        SELECT * FROM reminders WHERE user_id=? ORDER BY date(due_date) ASC, id DESC
    """, (user["user_id"],))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

@app.post("/api/reminders")
def create_reminder(body: ReminderBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        INSERT INTO reminders (user_id, tenant_id, property_id, title, due_date, amount, method, is_paid, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user["user_id"], body.tenant_id, body.property_id, body.title, body.due_date,
        body.amount, body.method, 1 if body.is_paid else 0, body.notes
    ))
    conn.commit(); rid = c.lastrowid; conn.close()
    return {"success": True, "id": rid}

# -------------------------------
# Admin (Backdoor)
# -------------------------------
@app.get("/api/admin/users")
def admin_users(user = Depends(get_current_user)):
    if user["email"].lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Forbidden")
    conn = db(); c = conn.cursor()
    c.execute("SELECT id, email, plan, stripe_customer_id, stripe_subscription_id FROM users ORDER BY id DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

class SetPlanBody(BaseModel):
    user_id: int
    plan: str

@app.put("/api/admin/set-plan")
def admin_set_plan(body: SetPlanBody, user = Depends(get_current_user)):
    if user["email"].lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Forbidden")
    update_user_plan_and_stripe(body.user_id, body.plan, None, None)
    return {"success": True}

@app.post("/api/admin/impersonate/{user_id}")
def admin_impersonate(user_id: int, user = Depends(get_current_user)):
    if user["email"].lower() != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Forbidden")
    row = get_user_by_id(user_id)
    if not row: raise HTTPException(status_code=404, detail="User not found")
    token = create_token({"user_id": row["id"], "email": row["email"], "plan": row["plan"]})
    return {"token": token, "plan": row["plan"]}

# -------------------------------
# Stripe: Checkout + Billing Portal + Webhook
# -------------------------------
@app.post("/api/create-checkout-session")
def create_checkout_session(body: CheckoutBody, user = Depends(get_current_user)):
    plan = (body.plan or "").lower()
    if plan not in ("cohost", "pro", "agency"):
        raise HTTPException(status_code=400, detail="Invalid plan")

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=400, detail="Stripe not configured")

    uid = int(user["user_id"])
    urow = get_user_by_id(uid)
    if not urow:
        raise HTTPException(status_code=404, detail="User not found")

    email = urow["email"]
    current_plan = urow["plan"]
    stripe_customer_id = urow["stripe_customer_id"]

    if not stripe_customer_id:
        sc = stripe.Customer.create(email=email, metadata={"househive_user_id": str(uid)})
        stripe_customer_id = sc["id"]
        update_user_plan_and_stripe(uid, current_plan, stripe_customer_id, None)

    price_id = PLAN_PRICE_IDS.get(plan, "")

    if price_id:
        line_items = [{"price": price_id, "quantity": 1}]
    else:
        # Fallback demo prices
        unit_amount = 1999 if plan == "cohost" else (2999 if plan == "pro" else 9999)
        line_items = [{
            "price_data": {
                "currency": "usd",
                "product_data": {"name": f"HouseHive {plan.title()} Plan"},
                "recurring": {"interval": "month"},
                "unit_amount": unit_amount,
            },
            "quantity": 1,
        }]

    session = stripe.checkout.Session.create(
        customer=stripe_customer_id,
        mode="subscription",
        line_items=line_items,
        success_url=f"{FRONTEND_URL}/billing/success",
        cancel_url=f"{FRONTEND_URL}/billing/cancel",
        metadata={"househive_user_id": str(uid), "plan_code": plan},
        subscription_data={"metadata": {"househive_user_id": str(uid), "plan_code": plan}},
    )
    return {"url": session.url}

@app.post("/api/billing-portal")
def billing_portal(user = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=400, detail="Stripe not configured")

    uid = int(user["user_id"])
    row = get_user_by_id(uid)
    if not row: raise HTTPException(status_code=404, detail="User not found")
    stripe_customer_id = row["stripe_customer_id"]
    if not stripe_customer_id:
        sc = stripe.Customer.create(email=row["email"], metadata={"househive_user_id": str(uid)})
        stripe_customer_id = sc["id"]
        update_user_plan_and_stripe(uid, row["plan"], stripe_customer_id, None)

    portal = stripe.billing_portal.Session.create(
        customer=stripe_customer_id,
        return_url=f"{FRONTEND_URL}/billing"
    )
    return {"url": portal.url}

@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    if STRIPE_WEBHOOK_SECRET:
        sig = request.headers.get("Stripe-Signature")
        try:
            event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
        except Exception:
            return PlainTextResponse("Invalid signature", status_code=400)
    else:
        try:
            event = json.loads(payload.decode("utf-8"))
        except Exception:
            return PlainTextResponse("Invalid payload", status_code=400)

    etype = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    if etype in ("customer.subscription.created", "customer.subscription.updated"):
        sub = data
        customer_id = sub.get("customer")
        sub_id = sub.get("id")
        plan_code = (sub.get("metadata", {}) or {}).get("plan_code", "")
        if not plan_code:
            try:
                plan_code = (sub["plan"]["nickname"] or "").lower()
            except:
                plan_code = "premium"

        conn = db(); c = conn.cursor()
        c.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,))
        r = c.fetchone(); conn.close()
        if r:
            uid = r["id"]
            update_user_plan_and_stripe(uid, plan_code.title(), customer_id, sub_id)

    if etype == "customer.subscription.deleted":
        sub = data
        customer_id = sub.get("customer")
        conn = db(); c = conn.cursor()
        c.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,))
        r = c.fetchone(); conn.close()
        if r:
            uid = r["id"]
            update_user_plan_and_stripe(uid, "Free", customer_id, None)

    return PlainTextResponse("OK", status_code=200)

# -------------------------------
# AI Chat (simple + stream)
# -------------------------------
@app.post("/chat")
async def chat_simple(payload: Dict[str, str]):
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    if USE_FAKE_AI:
        reply = f"✅ (Demo AI) I received: “{message}”. Here’s a quick plan:\n1) Capture details\n2) Assign vendor\n3) Track to completion."
        return {"reply": reply}

    # Real OpenAI call (concise)
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are HiveBot for HouseHive.ai. Be concise, actionable, and property-management savvy."},
            {"role": "user", "content": message},
        ],
        temperature=0.5,
    )
    reply = resp.choices[0].message.content.strip()
    return {"reply": reply}

@app.post("/chat/stream")
async def chat_stream(payload: Dict[str, str]):
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    async def fake_stream() -> Generator[bytes, None, None]:
        parts = [
            "Okay — here's your plan:\n",
            "• Step 1: Log the issue\n",
            "• Step 2: Assign a technician\n",
            "• Step 3: Confirm completion and close\n",
        ]
        for p in parts:
            yield p.encode("utf-8")
            await asyncio.sleep(0.2)

    if USE_FAKE_AI:
        return StreamingResponse(fake_stream(), media_type="text/plain; charset=utf-8")

    # For simplicity: just return non-SSE chunked text
    async def openai_stream() -> Generator[bytes, None, None]:
        # Minimal: single response chunk (OpenAI SDK streaming omitted for brevity)
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are HiveBot for HouseHive.ai. Be concise, actionable, and property-management savvy."},
                {"role": "user", "content": message},
            ],
            temperature=0.5,
        )
        out = resp.choices[0].message.content.strip()
        yield out.encode("utf-8")

    return StreamingResponse(openai_stream(), media_type="text/plain; charset=utf-8")

# -------------------------------
# Root
# -------------------------------
@app.get("/")
def root():
    return {"ok": True, "name": "HouseHive.ai API"}
