"""
👤 USER MANAGEMENT API
======================
Simple user registration and login for paper trading.

BEGINNER TIP:
- This is a simplified auth system for paper trading
- For production, use Supabase Auth (more secure)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
import uuid

from database import get_supabase, get_user_by_email, create_user, get_user_by_id

router = APIRouter(prefix="/users", tags=["Users"])


# ============================================================
# 📝 DATA MODELS
# ============================================================

class RegisterRequest(BaseModel):
    """Request to register a new user"""
    email: EmailStr
    name: Optional[str] = None


class UserResponse(BaseModel):
    """User data response"""
    id: str
    email: str
    name: str
    paper_balance: float
    is_paper_mode: bool


# ============================================================
# 🎯 API ENDPOINTS
# ============================================================

@router.post("/register")
async def register_user(request: RegisterRequest):
    """
    Register a new user for paper trading.
    
    Each new user gets:
    - ₹10 Lakh virtual balance
    - Paper trading mode enabled
    
    Example:
        POST /api/users/register
        {"email": "trader@example.com", "name": "John"}
    """
    try:
        # Check if user already exists
        existing = await get_user_by_email(request.email)
        if existing:
            # Return existing user (simple login)
            return {
                "success": True,
                "message": "Welcome back!",
                "user": {
                    "id": existing["id"],
                    "email": existing["email"],
                    "name": existing["name"],
                    "paper_balance": float(existing["paper_balance"]),
                    "is_paper_mode": existing["is_paper_mode"],
                }
            }
        
        # Create new user
        user = await create_user(request.email, request.name)
        
        if not user:
            raise HTTPException(
                status_code=500, 
                detail="Could not create user. Is Supabase configured?"
            )
        
        return {
            "success": True,
            "message": "Account created! You have ₹10,00,000 virtual money to practice trading.",
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "paper_balance": float(user["paper_balance"]),
                "is_paper_mode": user["is_paper_mode"],
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_user(user_id: str):
    """
    Get user details by ID.
    
    Example:
        GET /api/users/123e4567-e89b-12d3-a456-426614174000
    """
    try:
        user = await get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "paper_balance": float(user["paper_balance"]),
            "is_paper_mode": user["is_paper_mode"],
            "created_at": user["created_at"],
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login")
async def login_user(request: RegisterRequest):
    """
    Simple login - just checks if email exists.
    
    For paper trading, we keep it simple:
    - If email exists, return user
    - If not, create new account
    
    Example:
        POST /api/users/login
        {"email": "trader@example.com"}
    """
    # Just use register - it handles both cases
    return await register_user(request)


@router.get("/email/{email}")
async def get_user_by_email_endpoint(email: str):
    """
    Get user by email address.
    
    Example:
        GET /api/users/email/trader@example.com
    """
    try:
        user = await get_user_by_email(email)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "paper_balance": float(user["paper_balance"]),
            "is_paper_mode": user["is_paper_mode"],
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 📊 STATS ENDPOINT
# ============================================================

@router.get("/{user_id}/stats")
async def get_user_stats(user_id: str):
    """
    Get user's trading statistics.
    
    Shows:
    - Total trades
    - Win rate
    - Best trade
    - Worst trade
    """
    try:
        db = get_supabase()
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        # Get user
        user = await get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get orders
        orders = db.table("paper_orders").select("*").eq(
            "user_id", user_id
        ).execute()
        
        orders_data = orders.data or []
        
        total_orders = len(orders_data)
        buy_orders = [o for o in orders_data if o["action"] == "BUY"]
        sell_orders = [o for o in orders_data if o["action"] == "SELL"]
        
        # Calculate total invested and current value
        total_invested = sum(float(o.get("total_value", 0) or 0) for o in buy_orders)
        total_sold = sum(float(o.get("total_value", 0) or 0) for o in sell_orders)
        
        # Starting balance
        starting_balance = 1000000.00
        current_balance = float(user["paper_balance"])
        
        # Get holdings value
        holdings = db.table("paper_holdings").select("*").eq(
            "user_id", user_id
        ).execute()
        
        holdings_value = sum(
            h["quantity"] * float(h["avg_price"]) 
            for h in (holdings.data or [])
        )
        
        portfolio_value = current_balance + holdings_value
        total_pnl = portfolio_value - starting_balance
        
        return {
            "user_id": user_id,
            "total_orders": total_orders,
            "buy_orders": len(buy_orders),
            "sell_orders": len(sell_orders),
            "starting_balance": starting_balance,
            "current_balance": current_balance,
            "holdings_value": round(holdings_value, 2),
            "portfolio_value": round(portfolio_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_percent": round(total_pnl / starting_balance * 100, 2),
            "total_invested": round(total_invested, 2),
            "total_sold": round(total_sold, 2),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
