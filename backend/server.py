"""
SwingAI Backend Server Entry Point
This file serves as the entry point for the uvicorn server.
"""
import sys
import os

# Add the src directory to path
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        "message": "SwingAI API is running!"
    }

# ============================================================
# 📦 LOAD ALL API ROUTERS
# ============================================================

# 1. Market Data Router (existing)
try:
    from market_data import router as market_router
    app.include_router(market_router)
    print("✅ Market data endpoints loaded")
except Exception as e:
    print(f"⚠️ Market data endpoints not loaded: {e}")

# 2. Paper Trading Router (NEW!)
try:
    from paper_trading import router as paper_router
    app.include_router(paper_router, prefix="/api")
    print("✅ Paper trading endpoints loaded")
except Exception as e:
    print(f"⚠️ Paper trading endpoints not loaded: {e}")

# 3. Users Router (NEW!)
try:
    from users import router as users_router
    app.include_router(users_router, prefix="/api")
    print("✅ User management endpoints loaded")
except Exception as e:
    print(f"⚠️ User management endpoints not loaded: {e}")

# ============================================================
# 📝 API DOCUMENTATION
# ============================================================

@app.get("/api")
async def api_info():
    """List all available API endpoints"""
    return {
        "name": "SwingAI Trading API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "market": {
                "status": "/api/market/status",
                "stocks": "/api/market/stocks",
            },
            "paper_trading": {
                "portfolio": "/api/paper/portfolio/{user_id}",
                "place_order": "/api/paper/order",
                "orders": "/api/paper/orders/{user_id}",
                "price": "/api/paper/price/{symbol}",
                "reset": "/api/paper/reset/{user_id}",
                "leaderboard": "/api/paper/leaderboard",
            },
            "users": {
                "register": "/api/users/register",
                "login": "/api/users/login",
                "get_user": "/api/users/{user_id}",
                "stats": "/api/users/{user_id}/stats",
            }
        }
    }

