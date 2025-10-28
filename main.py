from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# ✅ Allow your frontend origins
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

# 🔹 Model for JSON body
class LoginRequest(BaseModel):
    email: str
    password: str

# 🔹 Universal login route (handles both JSON + Form)
@app.post("/api/login")
@app.post("/auth/login")
async def login(request: Request):
    try:
        # Try JSON first
        data = await request.json()
        email = data.get("email")
        password = data.get("password")
    except:
        # Fall back to form data
        form = await request.form()
        email = form.get("email")
        password = form.get("password")

    if email == "demo@househive.ai" and password == "password123":
        return {"success": True, "token": "househive-demo-token"}

    return {"success": False, "error": "Invalid credentials"}

# Optional root route
@app.get("/")
def home():
    return {"message": "Welcome to HouseHive Backend API v4!"}

