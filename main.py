from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

class LoginRequest(BaseModel):
    email: str
    password: str

# ✅ Allow Vercel frontends to connect
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

# 🔹 Health check endpoint
@app.get("/api/health")
def health():
    return {"status": "ok"}

# 🔹 Simple login simulation (for /api/login)
@app.post("/api/login")
def api_login(email: str = Form(...), password: str = Form(...)):
    if email == "demo@househive.ai" and password == "password123":
        return {"success": True, "token": "househive-demo-token"}
    else:
        return {"success": False, "error": "Invalid credentials"}

# 🔹 Add alias route (/auth/login)
@app.post("/api/login")
@app.post("/auth/login")
async def login(data: LoginRequest):
    if data.email == "demo@househive.ai" and data.password == "password123":
        return {"success": True, "token": "househive-demo-token"}
    return {"success": False, "error": "Invalid credentials"}

# Optional root message
@app.get("/")
def home():
    return {"message": "Welcome to HouseHive Backend API v3!"}
