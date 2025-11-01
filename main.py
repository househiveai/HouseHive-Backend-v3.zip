# -------------------------------
# HouseHive Backend API v5 (JWT + CRUD + Tenants + Reminders + Stripe + Admin)
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

PLAN_PRICE_IDS = {
    "cohost": os.getenv("PRICE_COHOST_ID", ""),   # e.g. price_...
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
        FRONTEND_URL,
        VERCEL_URL,
        "https://www.househive.ai",
        "http://localhost:3000",
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
            rental_length INTEGER,
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

def db() -> sqlite3.Connection:
    # enable row factory if you want dicts
    return sqlite3.connect(DB_PATH)

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
# Admin Backdoor
# -------------------------------
ADMIN_EMAILS = {"dntullo@yahoo.com"}

def is_admin(email: str) -> bool:
    return (email or "").strip().lower() in ADMIN_EMAILS

def require_admin(user=Depends(get_current_user)):
    if not is_admin(user.get("email")):
        raise HTTPException(status_code=403, detail="Admin only")
    return user

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
    address: str
    rent: float
    rental_length: Optional[int] = None
    status: str = "Active"


class TaskBody(BaseModel):
    property_id: Optional[int] = None
    property_name: Optional[str] = None
    task: str
    status: str = "Open"

class TenantBody(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    property_id: Optional[int] = None
    unit: Optional[str] = None
    notes: Optional[str] = None

class ReminderBody(BaseModel):
    tenant_id: Optional[int] = None
    property_id: Optional[int] = None
    title: str
    message: str
    due_date: str  # ISO date string

class CheckoutBody(BaseModel):
    plan: str  # 'cohost' | 'pro' | 'agency'

# -------------------------------
# Helpers to read users
# -------------------------------
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
    return {"status": "ok", "name": "HouseHive.ai API", "ts": datetime.utcnow().isoformat()}

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
    role = "Admin" if is_admin(email) else "Owner"
    return {
        "email": email,
        "plan": plan,
        "role": role,
        "name": email.split("@")[0].capitalize(),
        "stripe_customer_id": stripe_customer_id,
        "stripe_subscription_id": stripe_subscription_id
    }

# -------------------------------
# Properties (CRUD)
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
        INSERT INTO properties (user_id, name, address, rent, rental_length, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user["user_id"], body.name, body.address, body.rent, body.rental_length, body.status))

    conn.commit(); pid = c.lastrowid; conn.close()
    return {"success": True, "id": pid}

@app.put("/api/properties/{property_id}")
def update_property(property_id: int, body: PropertyBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        UPDATE properties SET name=?, address=?, rent=?, status=?
        WHERE id=? AND user_id=?
    """, (body.name, body.address, body.rent, body.status, property_id, user["user_id"]))
    conn.commit(); ok = c.rowcount > 0; conn.close()
    if not ok: raise HTTPException(status_code=404, detail="Property not found")
    return {"success": True}

@app.delete("/api/properties/{property_id}")
def delete_property(property_id: int, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("DELETE FROM properties WHERE id=? AND user_id=?", (property_id, user["user_id"]))
    conn.commit(); ok = c.rowcount > 0; conn.close()
    if not ok: raise HTTPException(status_code=404, detail="Property not found")
    return {"success": True}

# -------------------------------
# Maintenance Tasks (CRUD) + “Active” shortcut
# -------------------------------
@app.get("/api/maintenance")
def list_tasks(user = Depends(get_current_user), status: Optional[str] = None):
    conn = db(); c = conn.cursor()
    if status:
        c.execute("""
            SELECT id, property_id, property_name, task, status, created_at
            FROM tasks WHERE user_id=? AND status=?
            ORDER BY id DESC
        """, (user["user_id"], status))
    else:
        c.execute("""
            SELECT id, property_id, property_name, task, status, created_at
            FROM tasks WHERE user_id=? ORDER BY id DESC
        """, (user["user_id"],))
    rows = c.fetchall(); conn.close()
    return [{"id": r[0], "property_id": r[1], "property_name": r[2], "task": r[3], "status": r[4], "created_at": r[5]} for r in rows]

@app.post("/api/maintenance")
def create_task(body: TaskBody, user = Depends(get_current_user)):
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
    conn.commit(); ok = c.rowcount > 0; conn.close()
    if not ok: raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True}

@app.delete("/api/maintenance/{task_id}")
def delete_task(task_id: int, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("DELETE FROM tasks WHERE id=? AND user_id=?", (task_id, user["user_id"]))
    conn.commit(); ok = c.rowcount > 0; conn.close()
    if not ok: raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True}

# -------------------------------
# Tenants (CRUD)
# -------------------------------
@app.get("/api/tenants")
def list_tenants(user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        SELECT id, name, email, phone, property_id, unit, notes
        FROM tenants WHERE user_id=? ORDER BY id DESC
    """, (user["user_id"],))
    rows = c.fetchall(); conn.close()
    return [{"id": r[0], "name": r[1], "email": r[2], "phone": r[3], "property_id": r[4], "unit": r[5], "notes": r[6]} for r in rows]

@app.post("/api/tenants")
def create_tenant(body: TenantBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        INSERT INTO tenants (user_id, name, email, phone, property_id, unit, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user["user_id"], body.name, body.email, body.phone, body.property_id, body.unit, body.notes))
    conn.commit(); tid = c.lastrowid; conn.close()
    return {"success": True, "id": tid}

@app.put("/api/tenants/{tenant_id}")
def update_tenant(tenant_id: int, body: TenantBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        UPDATE tenants SET name=?, email=?, phone=?, property_id=?, unit=?, notes=?
        WHERE id=? AND user_id=?
    """, (body.name, body.email, body.phone, body.property_id, body.unit, body.notes, tenant_id, user["user_id"]))
    conn.commit(); ok = c.rowcount > 0; conn.close()
    if not ok: raise HTTPException(status_code=404, detail="Tenant not found")
    return {"success": True}

@app.delete("/api/tenants/{tenant_id}")
def delete_tenant(tenant_id: int, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("DELETE FROM tenants WHERE id=? AND user_id=?", (tenant_id, user["user_id"]))
    conn.commit(); ok = c.rowcount > 0; conn.close()
    if not ok: raise HTTPException(status_code=404, detail="Tenant not found")
    return {"success": True}

# -------------------------------
# Reminders (CRUD) — Rent reminders, etc.
# -------------------------------
@app.get("/api/reminders")
def list_reminders(user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        SELECT id, tenant_id, property_id, title, message, due_date, sent, created_at
        FROM reminders WHERE user_id=? ORDER BY due_date ASC
    """, (user["user_id"],))
    rows = c.fetchall(); conn.close()
    return [
        {"id": r[0], "tenant_id": r[1], "property_id": r[2], "title": r[3],
         "message": r[4], "due_date": r[5], "sent": bool(r[6]), "created_at": r[7]}
        for r in rows
    ]

@app.post("/api/reminders")
def create_reminder(body: ReminderBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        INSERT INTO reminders (user_id, tenant_id, property_id, title, message, due_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user["user_id"], body.tenant_id, body.property_id, body.title, body.message, body.due_date))
    conn.commit(); rid = c.lastrowid; conn.close()
    return {"success": True, "id": rid}

@app.put("/api/reminders/{reminder_id}")
def update_reminder(reminder_id: int, body: ReminderBody, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("""
        UPDATE reminders SET tenant_id=?, property_id=?, title=?, message=?, due_date=?
        WHERE id=? AND user_id=?
    """, (body.tenant_id, body.property_id, body.title, body.message, body.due_date, reminder_id, user["user_id"]))
    conn.commit(); ok = c.rowcount > 0; conn.close()
    if not ok: raise HTTPException(status_code=404, detail="Reminder not found")
    return {"success": True}

@app.delete("/api/reminders/{reminder_id}")
def delete_reminder(reminder_id: int, user = Depends(get_current_user)):
    conn = db(); c = conn.cursor()
    c.execute("DELETE FROM reminders WHERE id=? AND user_id=?", (reminder_id, user["user_id"]))
    conn.commit(); ok = c.rowcount > 0; conn.close()
    if not ok: raise HTTPException(status_code=404, detail="Reminder not found")
    return {"success": True}

# -------------------------------
# Stripe: Checkout + Billing Portal + Webhook
# -------------------------------
@app.post("/api/create-checkout-session")
def create_checkout_session(body: CheckoutBody, user = Depends(get_current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=400, detail="Stripe not configured")

    plan = body.plan.lower()
    uid = int(user["user_id"])
    urow = get_user_by_id(uid)
    if not urow: raise HTTPException(status_code=404, detail="User not found")
    _, email, _, current_plan, stripe_customer_id, _ = urow

    if not stripe_customer_id:
        sc = stripe.Customer.create(email=email, metadata={"househive_user_id": str(uid)})
        stripe_customer_id = sc["id"]
        update_user_plan_and_stripe(uid, current_plan, stripe_customer_id, None)

    price_id = PLAN_PRICE_IDS.get(plan, "")
    if price_id:
        line_items = [{"price": price_id, "quantity": 1}]
    else:
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
        metadata={"househive_user_id": str(uid), "plan_code": plan},
        subscription_data={"metadata": {"househive_user_id": str(uid), "plan_code": plan}}
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
    payload = await request.body()

    if not STRIPE_WEBHOOK_SECRET:
        try:
            event = json.loads(payload.decode("utf-8"))
        except Exception:
            return PlainTextResponse("Invalid payload", status_code=400)
    else:
        sig = request.headers.get("Stripe-Signature")
        try:
            event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
        except Exception:
            return PlainTextResponse("Invalid signature", status_code=400)

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
# Admin API
# -------------------------------
@app.get("/api/admin/users")
def admin_list_users(_: dict = Depends(require_admin)):
    conn = db(); c = conn.cursor()
    c.execute("SELECT id, email, plan, stripe_customer_id, stripe_subscription_id FROM users ORDER BY id DESC")
    rows = c.fetchall(); conn.close()
    return [
        {"id": r[0], "email": r[1], "plan": r[2], "stripe_customer_id": r[3], "stripe_subscription_id": r[4]}
        for r in rows
    ]

@app.put("/api/admin/set-plan")
def admin_set_plan(data: dict, _: dict = Depends(require_admin)):
    uid = data.get("user_id")
    plan = data.get("plan", "Free")
    if not uid: raise HTTPException(status_code=400, detail="Missing user_id")
    update_user_plan_and_stripe(int(uid), plan, None, None)
    return {"success": True, "message": f"Plan updated to {plan}"}

@app.delete("/api/admin/delete-user/{user_id}")
def admin_delete_user(user_id: int, _: dict = Depends(require_admin)):
    conn = db(); c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit(); conn.close()
    return {"success": True, "message": "User deleted"}

@app.post("/api/admin/impersonate/{user_id}")
def admin_impersonate(user_id: int, _: dict = Depends(require_admin)):
    row = get_user_by_id(user_id)
    if not row: raise HTTPException(status_code=404, detail="User not found")
    uid, email, _, plan, _, _ = row
    token = create_token({"user_id": uid, "email": email, "plan": plan})
    return {"success": True, "token": token, "email": email, "plan": plan}

# -------------------------------
# Root
# -------------------------------
@app.get("/")
def root():
    return {"ok": True, "name": "HouseHive.ai API"}

