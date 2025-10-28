# 🏠 HouseHive Backend v3

Now supports both `/api/login` and `/auth/login`.

### 🚀 Deploy on Render
1. Upload this folder to your GitHub repo or upload ZIP directly.
2. Create a new **Web Service** on [Render](https://render.com).
3. Set **Start Command:**
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
4. Set environment variables (optional):
   ```
   OPENAI_API_KEY=sk-yourkey
   SECRET_KEY=househive-secret
   ```
5. Click **Deploy**.

### ✅ Test endpoints
- `https://your-app.onrender.com/api/health`
- `https://your-app.onrender.com/api/login`
- `https://your-app.onrender.com/auth/login`
