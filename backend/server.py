"""
SwingAI Backend Server Entry Point
This file serves as the entry point for the uvicorn server.
"""
import sys
import os

# Add the src directory to path
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

# For now, create a minimal FastAPI app until Supabase is configured
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SwingAI Trading API",
    description="AI-powered swing trading signals for NSE/BSE",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "SwingAI API is running. Configure Supabase credentials to enable full functionality."
    }

# Import and include the actual routes when Supabase is configured
try:
    from src.backend.api.app import app as full_app
    # If successful, use the full app
    app = full_app
    print("✅ Full SwingAI API loaded")
except Exception as e:
    print(f"⚠️ Running in minimal mode: {e}")
    print("   Configure SUPABASE_URL and SUPABASE_SERVICE_KEY to enable full API")
