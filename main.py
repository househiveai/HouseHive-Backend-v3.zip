from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import stripe
import os
from dotenv import load_dotenv

# Load environment variables (for STRIPE keys)
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# ✅ CORS setup — allows frontend (Vercel) to talk to backend (Render)
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


# ----------------------------
#   BASIC ENDPOINTS
# ----------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}

# 🔹 Login (mocked)
class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/login")
@app.post("/auth/login")
async def login(request: Request):
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

# ----------------------------
#   MOCK DATA (for frontend)
# ----------------------------

@app.get("/auth/me")
def get_user():
    return {
        "email": "demo@househive.ai",
        "name": "Demo User",
        "plan": "Premium",
        "role": "Owner"
    }

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

# ----------------------------
#   STRIPE: Checkout + Billing
# ----------------------------

# ✅ Create Checkout Session
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
                        "unit_amount": 1500,  # $15.00/month
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


# ✅ Billing Portal
@app.post("/api/billing-portal")
async def billing_portal():
    try:
        # Temporary hardcoded test customer (replace later with your real Stripe customer ID)
        session = stripe.billing_portal.Session.create(
            customer="cus_QZz8Hb9EYjRzLe",
            return_url="https://househive.vercel.app/billing",
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ----------------------------
#   ROOT
# ----------------------------

@app.get("/")
def home():
    return {"message": "Welcome to HouseHive Backend API v5!"}

