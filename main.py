from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import stripe
import os

# Initialize app
app = FastAPI()

# ✅ CORS (connects to your frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://househive.ai",
        "https://househive.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Stripe setup
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
# Add Stripe dependency for backend payments

# 🔹 Health check endpoint
@app.get("/api/health")
def health():
    return {"status": "ok"}

# 🔹 Model for login
class LoginRequest(BaseModel):
    email: str
    password: str

# 🔹 Universal login (accepts both JSON + Form)
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
#   Mock API Endpoints
# ----------------------------

@app.get("/auth/me")
def get_user():
    return {
        "email": "demo@househive.ai",
        "name": "Demo User",
        "plan": "Premium",
        "role": "Owner"
    }

@app.get("/properties")
def get_properties():
    return {
        "properties": [
            {"id": 1, "name": "Downtown Condo", "status": "Active", "rent": 2200},
            {"id": 2, "name": "Beach House", "status": "Rented", "rent": 4100},
        ]
    }

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {"id": 101, "property": "Downtown Condo", "task": "Fix sink leak", "status": "Open"},
            {"id": 102, "property": "Beach House", "task": "Replace smoke alarm", "status": "Completed"},
        ]
    }

# --- TEMP DATA ROUTES (for Properties and Maintenance) ---

@app.get("/api/properties")
def get_properties():
    # return fake data for now so frontend doesn't crash
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
#   Stripe Checkout Endpoint
# ----------------------------

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
                        "unit_amount": 1999,  # $19.99 per month
                    },
                    "quantity": 1,
                }
            ],
            success_url="https://househive.ai/success",
            cancel_url="https://househive.ai/cancel",
        )
        return JSONResponse({"checkout_url": session.url})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
import os, stripe
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# ✅ Create checkout session
@app.post("/api/create-checkout-session")
async def create_checkout_session():
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "HouseHive Premium Plan"},
                        "unit_amount": 1500,  # $15.00
                    },
                    "quantity": 1,
                }
            ],
            mode="subscription",  # or "payment" for one-time charge
            success_url="https://househive.vercel.app/billing/success",
            cancel_url="https://househive.vercel.app/billing/cancel",
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root route
@app.get("/")
def home():
    return {"message": "Welcome to HouseHive Backend API v5!"}

