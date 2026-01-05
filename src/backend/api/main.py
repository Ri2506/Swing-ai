"""
================================================================================
SWINGAI PRODUCTION BACKEND - COMPLETE
================================================================================
FastAPI + Supabase + Razorpay + F&O Trading + Real-time WebSocket
================================================================================
"""

import os
import json
import hmac
import hashlib
import asyncio
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from decimal import Decimal
import logging

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
from supabase import create_client, Client
import httpx
import razorpay

# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings:
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    
    # Razorpay
    RAZORPAY_KEY_ID: str = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET: str = os.getenv("RAZORPAY_KEY_SECRET", "")
    
    # App
    APP_ENV: str = os.getenv("APP_ENV", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://swingai.vercel.app",
        "https://*.vercel.app",
    ]

settings = Settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CLIENTS
# ============================================================================

def get_supabase() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

def get_supabase_admin() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

def get_razorpay() -> razorpay.Client:
    return razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

# Auth
class UserSignup(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Profile
class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    capital: Optional[float] = None
    risk_profile: Optional[str] = None
    trading_mode: Optional[str] = None
    max_positions: Optional[int] = None
    risk_per_trade: Optional[float] = None
    fo_enabled: Optional[bool] = None
    preferred_option_type: Optional[str] = None
    daily_loss_limit: Optional[float] = None

# Broker
class BrokerConnect(BaseModel):
    broker_name: str
    api_key: str
    api_secret: Optional[str] = None
    client_id: Optional[str] = None
    totp_secret: Optional[str] = None
    access_token: Optional[str] = None

# Payment
class CreateOrder(BaseModel):
    plan_id: str
    billing_period: str  # monthly, quarterly, yearly

class VerifyPayment(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

# Trading
class ExecuteTrade(BaseModel):
    signal_id: str
    quantity: Optional[int] = None
    custom_sl: Optional[float] = None
    custom_target: Optional[float] = None

class CloseTrade(BaseModel):
    trade_id: str
    exit_price: Optional[float] = None
    reason: str = "manual"

# ============================================================================
# AUTH DEPENDENCY
# ============================================================================

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        supabase = get_supabase()
        user = supabase.auth.get_user(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user.user
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

async def get_user_profile(user = Depends(get_current_user)):
    supabase = get_supabase_admin()
    result = supabase.table("user_profiles").select("*, subscription_plans(*)").eq("id", user.id).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Profile not found")
    return result.data

# ============================================================================
# APP INITIALIZATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 SwingAI Backend starting...")
    yield
    logger.info("SwingAI Backend shutting down...")

app = FastAPI(
    title="SwingAI API",
    description="AI-Powered Swing Trading Platform",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.get("/")
async def root():
    return {"service": "SwingAI API", "version": "2.0.0", "status": "running"}

@app.get("/health")
async def health():
    try:
        supabase = get_supabase_admin()
        supabase.table("market_data").select("date").limit(1).execute()
        db_status = "connected"
    except:
        db_status = "error"
    
    return {
        "status": "healthy",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# AUTH ROUTES
# ============================================================================

@app.post("/api/auth/signup")
async def signup(data: UserSignup):
    try:
        supabase = get_supabase()
        response = supabase.auth.sign_up({
            "email": data.email,
            "password": data.password,
            "options": {"data": {"full_name": data.full_name, "phone": data.phone}}
        })
        
        if response.user:
            return {"success": True, "message": "Account created. Check email for verification.", "user_id": response.user.id}
        raise HTTPException(status_code=400, detail="Signup failed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login")
async def login(data: UserLogin):
    try:
        supabase = get_supabase()
        response = supabase.auth.sign_in_with_password({"email": data.email, "password": data.password})
        
        if response.user and response.session:
            # Update last login
            supabase_admin = get_supabase_admin()
            supabase_admin.table("user_profiles").update({
                "last_login": datetime.utcnow().isoformat(),
                "last_active": datetime.utcnow().isoformat()
            }).eq("id", response.user.id).execute()
            
            return {
                "success": True,
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "expires_at": response.session.expires_at,
                "user": {"id": response.user.id, "email": response.user.email}
            }
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/api/auth/refresh")
async def refresh(refresh_token: str):
    try:
        supabase = get_supabase()
        response = supabase.auth.refresh_session(refresh_token)
        if response.session:
            return {
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token
            }
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/api/auth/logout")
async def logout(user = Depends(get_current_user)):
    return {"success": True}

# ============================================================================
# USER PROFILE ROUTES
# ============================================================================

@app.get("/api/user/profile")
async def get_profile(profile = Depends(get_user_profile)):
    return profile

@app.put("/api/user/profile")
async def update_profile(data: ProfileUpdate, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        update_data = {k: v for k, v in data.dict().items() if v is not None}
        result = supabase.table("user_profiles").update(update_data).eq("id", user.id).execute()
        return {"success": True, "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/user/stats")
async def get_user_stats(user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        
        # Get profile with stats
        profile = supabase.table("user_profiles").select("*").eq("id", user.id).single().execute()
        
        # Get open positions
        positions = supabase.table("positions").select("*").eq("user_id", user.id).eq("is_active", True).execute()
        
        # Get today's P&L
        today = date.today().isoformat()
        today_trades = supabase.table("trades").select("net_pnl").eq("user_id", user.id).eq("status", "closed").gte("closed_at", today).execute()
        
        # Get this week's trades
        week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        week_trades = supabase.table("trades").select("net_pnl, status").eq("user_id", user.id).gte("created_at", week_start).execute()
        
        p = profile.data
        pos = positions.data or []
        
        unrealized_pnl = sum(float(p.get("unrealized_pnl", 0) or 0) for p in pos)
        today_pnl = sum(float(t.get("net_pnl", 0) or 0) for t in today_trades.data or [])
        week_pnl = sum(float(t.get("net_pnl", 0) or 0) for t in week_trades.data or [] if t.get("status") == "closed")
        
        win_rate = (p["winning_trades"] / p["total_trades"] * 100) if p["total_trades"] > 0 else 0
        
        return {
            "capital": p["capital"],
            "total_pnl": p["total_pnl"],
            "total_trades": p["total_trades"],
            "winning_trades": p["winning_trades"],
            "win_rate": round(win_rate, 2),
            "open_positions": len(pos),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "today_pnl": round(today_pnl, 2),
            "week_pnl": round(week_pnl, 2),
            "subscription_status": p["subscription_status"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SUBSCRIPTION & PAYMENT ROUTES
# ============================================================================

@app.get("/api/plans")
async def get_plans():
    try:
        supabase = get_supabase_admin()
        result = supabase.table("subscription_plans").select("*").eq("is_active", True).order("sort_order").execute()
        return {"plans": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/payments/create-order")
async def create_payment_order(data: CreateOrder, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        rzp = get_razorpay()
        
        # Get plan
        plan = supabase.table("subscription_plans").select("*").eq("id", data.plan_id).single().execute()
        if not plan.data:
            raise HTTPException(status_code=404, detail="Plan not found")
        
        # Calculate amount
        if data.billing_period == "monthly":
            amount = plan.data["price_monthly"]
        elif data.billing_period == "quarterly":
            amount = plan.data["price_quarterly"]
        else:
            amount = plan.data["price_yearly"]
        
        # Create Razorpay order
        order_data = {
            "amount": amount,
            "currency": "INR",
            "receipt": f"order_{user.id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "notes": {
                "user_id": user.id,
                "plan_id": data.plan_id,
                "billing_period": data.billing_period
            }
        }
        
        rzp_order = rzp.order.create(order_data)
        
        # Save to database
        supabase.table("payments").insert({
            "user_id": user.id,
            "razorpay_order_id": rzp_order["id"],
            "amount": amount,
            "plan_id": data.plan_id,
            "billing_period": data.billing_period,
            "status": "pending"
        }).execute()
        
        return {
            "order_id": rzp_order["id"],
            "amount": amount,
            "currency": "INR",
            "key_id": settings.RAZORPAY_KEY_ID
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/payments/verify")
async def verify_payment(data: VerifyPayment, user = Depends(get_current_user)):
    try:
        rzp = get_razorpay()
        supabase = get_supabase_admin()
        
        # Verify signature
        message = f"{data.razorpay_order_id}|{data.razorpay_payment_id}"
        expected_signature = hmac.new(
            settings.RAZORPAY_KEY_SECRET.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if expected_signature != data.razorpay_signature:
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        # Get payment record
        payment = supabase.table("payments").select("*").eq("razorpay_order_id", data.razorpay_order_id).single().execute()
        
        if not payment.data:
            raise HTTPException(status_code=404, detail="Payment not found")
        
        # Update payment
        supabase.table("payments").update({
            "razorpay_payment_id": data.razorpay_payment_id,
            "razorpay_signature": data.razorpay_signature,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        }).eq("id", payment.data["id"]).execute()
        
        # Calculate subscription end date
        billing_period = payment.data["billing_period"]
        if billing_period == "monthly":
            end_date = datetime.utcnow() + timedelta(days=30)
        elif billing_period == "quarterly":
            end_date = datetime.utcnow() + timedelta(days=90)
        else:
            end_date = datetime.utcnow() + timedelta(days=365)
        
        # Update user subscription
        supabase.table("user_profiles").update({
            "subscription_plan_id": payment.data["plan_id"],
            "subscription_status": "active",
            "subscription_start": datetime.utcnow().isoformat(),
            "subscription_end": end_date.isoformat()
        }).eq("id", user.id).execute()
        
        return {"success": True, "message": "Payment verified and subscription activated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/payments/history")
async def get_payment_history(user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        result = supabase.table("payments").select("*, subscription_plans(display_name)").eq("user_id", user.id).order("created_at", desc=True).execute()
        return {"payments": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SIGNALS ROUTES
# ============================================================================

@app.get("/api/signals/today")
async def get_today_signals(
    segment: Optional[str] = None,
    direction: Optional[str] = None,
    profile = Depends(get_user_profile)
):
    try:
        supabase = get_supabase_admin()
        today = date.today().isoformat()
        
        query = supabase.table("signals").select("*").eq("date", today).eq("status", "active")
        
        if segment:
            query = query.eq("segment", segment)
        if direction:
            query = query.eq("direction", direction)
        
        # Check subscription for premium signals
        is_premium = profile.get("subscription_status") in ["active", "trial"]
        if not is_premium:
            query = query.eq("is_premium", False)
        
        result = query.order("confidence", desc=True).execute()
        
        signals = result.data
        
        return {
            "date": today,
            "total": len(signals),
            "long_signals": [s for s in signals if s["direction"] == "LONG"],
            "short_signals": [s for s in signals if s["direction"] == "SHORT"],
            "equity_signals": [s for s in signals if s["segment"] == "EQUITY"],
            "futures_signals": [s for s in signals if s["segment"] == "FUTURES"],
            "options_signals": [s for s in signals if s["segment"] == "OPTIONS"],
            "all_signals": signals
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals/{signal_id}")
async def get_signal(signal_id: str, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        result = supabase.table("signals").select("*").eq("id", signal_id).single().execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Signal not found")
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals/performance")
async def get_signal_performance(days: int = 30, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        start_date = (date.today() - timedelta(days=days)).isoformat()
        
        result = supabase.table("model_performance").select("*").gte("date", start_date).order("date", desc=True).execute()
        
        return {"performance": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TRADES ROUTES
# ============================================================================

@app.get("/api/trades")
async def get_trades(
    status: Optional[str] = None,
    segment: Optional[str] = None,
    limit: int = 50,
    user = Depends(get_current_user)
):
    try:
        supabase = get_supabase_admin()
        query = supabase.table("trades").select("*, signals(symbol, direction, confidence)").eq("user_id", user.id)
        
        if status:
            query = query.eq("status", status)
        if segment:
            query = query.eq("segment", segment)
        
        result = query.order("created_at", desc=True).limit(limit).execute()
        return {"trades": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trades/execute")
async def execute_trade(data: ExecuteTrade, profile = Depends(get_user_profile)):
    try:
        supabase = get_supabase_admin()
        user_id = profile["id"]
        
        # Get signal
        signal = supabase.table("signals").select("*").eq("id", data.signal_id).single().execute()
        if not signal.data:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        sig = signal.data
        
        # Check subscription for premium signals
        if sig.get("is_premium") and profile.get("subscription_status") not in ["active", "trial"]:
            raise HTTPException(status_code=403, detail="Premium subscription required")
        
        # Check if F&O is enabled for F&O signals
        if sig["segment"] in ["FUTURES", "OPTIONS"] and not profile.get("fo_enabled"):
            raise HTTPException(status_code=403, detail="F&O trading not enabled")
        
        # Check trading mode
        if profile["trading_mode"] == "signal_only":
            raise HTTPException(status_code=400, detail="Auto-trading not enabled")
        
        # Check max positions
        positions = supabase.table("positions").select("id").eq("user_id", user_id).eq("is_active", True).execute()
        plan = profile.get("subscription_plans") or {}
        max_positions = plan.get("max_positions", profile.get("max_positions", 5))
        
        if len(positions.data) >= max_positions:
            raise HTTPException(status_code=400, detail=f"Max positions ({max_positions}) reached")
        
        # Calculate position size
        capital = float(profile["capital"])
        risk_per_trade = float(profile["risk_per_trade"])
        entry_price = float(data.custom_sl or sig["entry_price"])
        stop_loss = float(data.custom_sl or sig["stop_loss"])
        target = float(data.custom_target or sig["target_1"])
        
        risk_amount = capital * (risk_per_trade / 100)
        risk_per_unit = abs(entry_price - stop_loss)
        
        if sig["segment"] == "EQUITY":
            quantity = data.quantity or int(risk_amount / risk_per_unit) if risk_per_unit > 0 else 0
            lots = 1
            margin_used = quantity * entry_price
        else:  # FUTURES/OPTIONS
            lot_size = sig.get("lot_size", 1)
            lots = data.quantity or 1
            quantity = lots * lot_size
            margin_used = quantity * entry_price * 0.2  # ~20% margin for F&O
        
        if quantity < 1:
            raise HTTPException(status_code=400, detail="Position size too small")
        
        # Create trade
        trade = {
            "user_id": user_id,
            "signal_id": data.signal_id,
            "symbol": sig["symbol"],
            "exchange": sig.get("exchange", "NSE"),
            "segment": sig["segment"],
            "expiry_date": sig.get("expiry_date"),
            "strike_price": sig.get("strike_price"),
            "option_type": sig.get("option_type"),
            "lot_size": sig.get("lot_size"),
            "lots": lots,
            "direction": sig["direction"],
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "risk_amount": risk_amount,
            "position_value": quantity * entry_price,
            "margin_used": margin_used,
            "product_type": "CNC" if sig["segment"] == "EQUITY" else "NRML",
            "status": "pending" if profile["trading_mode"] == "semi_auto" else "open"
        }
        
        result = supabase.table("trades").insert(trade).execute()
        trade_id = result.data[0]["id"]
        
        # If full auto, create position immediately
        if profile["trading_mode"] == "full_auto":
            position = {
                "user_id": user_id,
                "trade_id": trade_id,
                "symbol": sig["symbol"],
                "exchange": sig.get("exchange", "NSE"),
                "segment": sig["segment"],
                "expiry_date": sig.get("expiry_date"),
                "strike_price": sig.get("strike_price"),
                "option_type": sig.get("option_type"),
                "direction": sig["direction"],
                "quantity": quantity,
                "lots": lots,
                "average_price": entry_price,
                "current_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "margin_used": margin_used,
                "is_active": True
            }
            supabase.table("positions").insert(position).execute()
            
            supabase.table("trades").update({
                "status": "open",
                "executed_at": datetime.utcnow().isoformat()
            }).eq("id", trade_id).execute()
        
        return {
            "success": True,
            "trade_id": trade_id,
            "status": trade["status"],
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trades/{trade_id}/approve")
async def approve_trade(trade_id: str, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        
        trade = supabase.table("trades").select("*").eq("id", trade_id).eq("user_id", user.id).single().execute()
        if not trade.data:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        if trade.data["status"] != "pending":
            raise HTTPException(status_code=400, detail="Trade not pending")
        
        t = trade.data
        
        # Create position
        position = {
            "user_id": user.id,
            "trade_id": trade_id,
            "symbol": t["symbol"],
            "exchange": t.get("exchange", "NSE"),
            "segment": t["segment"],
            "expiry_date": t.get("expiry_date"),
            "strike_price": t.get("strike_price"),
            "option_type": t.get("option_type"),
            "direction": t["direction"],
            "quantity": t["quantity"],
            "lots": t.get("lots", 1),
            "average_price": t["entry_price"],
            "current_price": t["entry_price"],
            "stop_loss": t["stop_loss"],
            "target": t["target"],
            "margin_used": t.get("margin_used", 0),
            "is_active": True
        }
        supabase.table("positions").insert(position).execute()
        
        supabase.table("trades").update({
            "status": "open",
            "approved_at": datetime.utcnow().isoformat(),
            "executed_at": datetime.utcnow().isoformat()
        }).eq("id", trade_id).execute()
        
        return {"success": True, "message": "Trade approved and executed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trades/{trade_id}/close")
async def close_trade(trade_id: str, data: CloseTrade, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        
        trade = supabase.table("trades").select("*").eq("id", trade_id).eq("user_id", user.id).single().execute()
        if not trade.data or trade.data["status"] != "open":
            raise HTTPException(status_code=400, detail="Trade not found or not open")
        
        t = trade.data
        exit_price = data.exit_price or t["entry_price"]
        
        # Calculate P&L
        if t["direction"] == "LONG":
            gross_pnl = (exit_price - t["average_price"]) * t["quantity"]
        else:
            gross_pnl = (t["average_price"] - exit_price) * t["quantity"]
        
        # Estimate charges (~0.1% for equity, ~0.05% for F&O)
        charge_rate = 0.001 if t["segment"] == "EQUITY" else 0.0005
        charges = abs(t["position_value"]) * charge_rate
        net_pnl = gross_pnl - charges
        pnl_percent = (net_pnl / t["position_value"]) * 100 if t["position_value"] else 0
        
        # Update trade
        supabase.table("trades").update({
            "status": "closed",
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "charges": charges,
            "net_pnl": net_pnl,
            "pnl_percent": pnl_percent,
            "exit_reason": data.reason,
            "closed_at": datetime.utcnow().isoformat()
        }).eq("id", trade_id).execute()
        
        # Deactivate position
        supabase.table("positions").update({"is_active": False}).eq("trade_id", trade_id).execute()
        
        return {
            "success": True,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "pnl_percent": round(pnl_percent, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PORTFOLIO ROUTES
# ============================================================================

@app.get("/api/portfolio")
async def get_portfolio(profile = Depends(get_user_profile)):
    try:
        supabase = get_supabase_admin()
        user_id = profile["id"]
        
        positions = supabase.table("positions").select("*").eq("user_id", user_id).eq("is_active", True).execute()
        
        pos_list = positions.data or []
        
        total_invested = sum(p["quantity"] * p["average_price"] for p in pos_list)
        total_current = sum(p["quantity"] * (p["current_price"] or p["average_price"]) for p in pos_list)
        unrealized_pnl = total_current - total_invested
        margin_used = sum(p.get("margin_used", 0) or 0 for p in pos_list)
        
        return {
            "capital": profile["capital"],
            "deployed": round(total_invested, 2),
            "available": round(profile["capital"] - total_invested, 2),
            "margin_used": round(margin_used, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "positions": pos_list,
            "equity_positions": [p for p in pos_list if p["segment"] == "EQUITY"],
            "fo_positions": [p for p in pos_list if p["segment"] in ["FUTURES", "OPTIONS"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/history")
async def get_portfolio_history(days: int = 30, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        start_date = (date.today() - timedelta(days=days)).isoformat()
        
        result = supabase.table("portfolio_history").select("*").eq("user_id", user.id).gte("date", start_date).order("date").execute()
        
        return {"history": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/performance")
async def get_performance_metrics(user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        
        # Get all closed trades
        trades = supabase.table("trades").select("*").eq("user_id", user.id).eq("status", "closed").execute()
        
        if not trades.data:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "total_pnl": 0,
                "best_trade": 0,
                "worst_trade": 0
            }
        
        t_list = trades.data
        winners = [t for t in t_list if (t.get("net_pnl") or 0) > 0]
        losers = [t for t in t_list if (t.get("net_pnl") or 0) < 0]
        
        total_wins = sum(t.get("net_pnl", 0) for t in winners)
        total_losses = abs(sum(t.get("net_pnl", 0) for t in losers))
        
        return {
            "total_trades": len(t_list),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(t_list) * 100, 2) if t_list else 0,
            "avg_win": round(total_wins / len(winners), 2) if winners else 0,
            "avg_loss": round(total_losses / len(losers), 2) if losers else 0,
            "profit_factor": round(total_wins / total_losses, 2) if total_losses > 0 else 0,
            "total_pnl": round(sum(t.get("net_pnl", 0) for t in t_list), 2),
            "best_trade": round(max(t.get("net_pnl", 0) for t in t_list), 2),
            "worst_trade": round(min(t.get("net_pnl", 0) for t in t_list), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MARKET DATA ROUTES
# ============================================================================

@app.get("/api/market/status")
async def get_market_status():
    now = datetime.now()
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    is_open = market_open <= now.time() <= market_close and now.weekday() < 5
    
    return {
        "timestamp": now.isoformat(),
        "is_open": is_open,
        "is_trading_day": now.weekday() < 5,
        "market_hours": "09:15 - 15:30 IST"
    }

@app.get("/api/market/data")
async def get_market_data(user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        today = date.today().isoformat()
        
        result = supabase.table("market_data").select("*").eq("date", today).single().execute()
        
        return result.data or {
            "date": today,
            "nifty_close": 0,
            "vix_close": 0,
            "market_trend": "UNKNOWN",
            "risk_level": "UNKNOWN"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/risk")
async def get_risk_assessment(user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        today = date.today().isoformat()
        
        result = supabase.table("market_data").select("*").eq("date", today).single().execute()
        data = result.data or {}
        
        vix = data.get("vix_close", 15)
        
        if vix < 15:
            risk_level, recommendation = "LOW", "Normal trading - full position sizes"
        elif vix < 20:
            risk_level, recommendation = "MODERATE", "Reduce position sizes by 25%"
        elif vix < 25:
            risk_level, recommendation = "HIGH", "Reduce position sizes by 50%, only high-confidence trades"
        else:
            risk_level, recommendation = "EXTREME", "Stop all new trades, consider hedging"
        
        return {
            "vix": vix,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "nifty_change": data.get("nifty_change_percent", 0),
            "fii_net": data.get("fii_cash", 0),
            "market_trend": data.get("market_trend", "UNKNOWN"),
            "circuit_breaker": risk_level == "EXTREME"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# BROKER ROUTES
# ============================================================================

@app.post("/api/broker/connect")
async def connect_broker(data: BrokerConnect, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        
        # Store credentials (encrypted in production)
        credentials = {k: v for k, v in data.dict().items() if v is not None and k != "broker_name"}
        
        supabase.table("user_profiles").update({
            "broker_name": data.broker_name,
            "broker_credentials": credentials,
            "broker_connected": True,
            "broker_last_sync": datetime.utcnow().isoformat()
        }).eq("id", user.id).execute()
        
        return {"success": True, "broker": data.broker_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/broker/disconnect")
async def disconnect_broker(user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        
        supabase.table("user_profiles").update({
            "broker_name": None,
            "broker_credentials": {},
            "broker_connected": False
        }).eq("id", user.id).execute()
        
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# NOTIFICATIONS ROUTES
# ============================================================================

@app.get("/api/notifications")
async def get_notifications(unread_only: bool = False, limit: int = 50, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        
        query = supabase.table("notifications").select("*").eq("user_id", user.id)
        if unread_only:
            query = query.eq("is_read", False)
        
        result = query.order("created_at", desc=True).limit(limit).execute()
        
        return {"notifications": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/notifications/{notification_id}/read")
async def mark_read(notification_id: str, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        supabase.table("notifications").update({
            "is_read": True,
            "read_at": datetime.utcnow().isoformat()
        }).eq("id", notification_id).eq("user_id", user.id).execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WATCHLIST ROUTES
# ============================================================================

@app.get("/api/watchlist")
async def get_watchlist(user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        result = supabase.table("watchlist").select("*").eq("user_id", user.id).order("added_at", desc=True).execute()
        return {"watchlist": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist")
async def add_to_watchlist(symbol: str, segment: str = "EQUITY", user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        supabase.table("watchlist").insert({
            "user_id": user.id,
            "symbol": symbol.upper(),
            "segment": segment
        }).execute()
        return {"success": True}
    except:
        return {"success": False, "message": "Already in watchlist"}

@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str, user = Depends(get_current_user)):
    try:
        supabase = get_supabase_admin()
        supabase.table("watchlist").delete().eq("user_id", user.id).eq("symbol", symbol.upper()).execute()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected: {user_id}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected: {user_id}")
    
    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except:
                self.disconnect(user_id)
    
    async def broadcast(self, message: dict):
        for user_id, ws in list(self.active_connections.items()):
            try:
                await ws.send_json(message)
            except:
                self.disconnect(user_id)

manager = ConnectionManager()

@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    try:
        supabase = get_supabase()
        user = supabase.auth.get_user(token)
        
        if not user:
            await websocket.close(code=4001)
            return
        
        user_id = user.user.id
        await manager.connect(websocket, user_id)
        
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
        except WebSocketDisconnect:
            manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=4000)

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def send_signal_notification(signal: dict):
    """Send notification to all relevant users when new signal is generated"""
    supabase = get_supabase_admin()
    
    # Get all users with notifications enabled
    users = supabase.table("user_profiles").select("id, telegram_chat_id").eq("notifications_enabled", True).execute()
    
    for user in users.data:
        # Create in-app notification
        supabase.table("notifications").insert({
            "user_id": user["id"],
            "type": "signal",
            "title": f"New {signal['direction']} Signal: {signal['symbol']}",
            "message": f"Confidence: {signal['confidence']}% | Entry: ₹{signal['entry_price']}",
            "data": signal
        }).execute()
        
        # Send via WebSocket
        await manager.send_to_user(user["id"], {
            "type": "new_signal",
            "data": signal
        })

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
