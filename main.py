from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow your frontend origins
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

# Health check
@app.get("/api/health")
def health():
    return {"status": "ok"}

# Model for JSON body
class LoginRequest(BaseModel):
    email: str
    password: str

# Unified login handler
@app.post("/api/login")
@app.post("/auth/login")
async def login(data: LoginRequest):
    if data.email == "demo@househive.ai" and data.password == "password123":
        return {"success": True, "token": "househive-demo-token"}
    return {"success": False, "error": "Invalid credentials"}

@app.get("/")
def home():
    return {"message": "Welcome to HouseHive Backend API v4!"}

