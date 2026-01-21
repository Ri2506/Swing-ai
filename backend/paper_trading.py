"""
📈 PAPER TRADING API
====================
This handles all paper trading functionality.

BEGINNER TIP:
- Paper trading = Fake money trading for practice
- Users get ₹10 Lakh virtual money to learn trading
- All trades are simulated with real market prices
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import yfinance as yf

# Import our simple database functions
from database import (
    get_user_by_id,
    update_user_balance,
    create_paper_order,
    get_user_orders,
    get_user_holdings,
    update_holding,
    get_supabase,
)

# Create router for paper trading endpoints
router = APIRouter(prefix="/paper", tags=["Paper Trading"])


# ============================================================
# 📝 DATA MODELS (What data looks like)
# ============================================================

class PlaceOrderRequest(BaseModel):
    """Request to place a paper trade"""
    user_id: str
    symbol: str           # Stock symbol like "RELIANCE.NS"
    action: str           # "BUY" or "SELL"
    quantity: int         # Number of shares
    order_type: str = "MARKET"  # "MARKET" or "LIMIT"
    limit_price: Optional[float] = None  # Price for limit orders


class OrderResponse(BaseModel):
    """Response after placing an order"""
    success: bool
    message: str
    order_id: Optional[str] = None
    executed_price: Optional[float] = None
    total_value: Optional[float] = None
    new_balance: Optional[float] = None


class PortfolioResponse(BaseModel):
    """User's portfolio summary"""
    user_id: str
    cash_balance: float
    holdings: List[dict]
    total_invested: float
    total_current_value: float
    total_pnl: float
    total_pnl_percent: float


# ============================================================
# 📊 HELPER FUNCTIONS
# ============================================================

def get_live_price(symbol: str) -> float:
    """
    Get live/current price of a stock using yfinance.
    
    Example:
        price = get_live_price("RELIANCE.NS")
        print(price)  # 2850.50
    """
    try:
        # Add .NS for NSE stocks if not present
        if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        
        if data.empty:
            # Try BSE
            symbol = symbol.replace(".NS", ".BO")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
        
        if not data.empty:
            return round(float(data['Close'].iloc[-1]), 2)
        
        return None
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
        return None


def calculate_charges(value: float, action: str) -> dict:
    """
    Calculate realistic brokerage charges for Indian market.
    
    Returns breakdown of all charges.
    """
    brokerage = value * 0.0003  # 0.03% brokerage
    stt = value * 0.001 if action == "SELL" else 0  # STT only on sell
    exchange_charges = value * 0.0000345  # NSE charges
    gst = (brokerage + exchange_charges) * 0.18  # 18% GST
    sebi_charges = value * 0.000001  # SEBI charges
    stamp_duty = value * 0.00015 if action == "BUY" else 0  # Stamp duty on buy
    
    total_charges = brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty
    
    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_charges": round(exchange_charges, 2),
        "gst": round(gst, 2),
        "sebi_charges": round(sebi_charges, 2),
        "stamp_duty": round(stamp_duty, 2),
        "total_charges": round(total_charges, 2),
    }


# ============================================================
# 🎯 API ENDPOINTS
# ============================================================

@router.get("/portfolio/{user_id}")
async def get_portfolio(user_id: str):
    """
    Get user's complete paper trading portfolio.
    
    Shows:
    - Cash balance
    - All stock holdings
    - Total P&L (profit/loss)
    """
    try:
        # Get user data
        user = await get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get holdings
        holdings = await get_user_holdings(user_id)
        
        # Calculate current values with live prices
        total_invested = 0
        total_current_value = 0
        holdings_with_prices = []
        
        for holding in holdings:
            symbol = holding["symbol"]
            quantity = holding["quantity"]
            avg_price = float(holding["avg_price"])
            invested = quantity * avg_price
            
            # Get live price
            live_price = get_live_price(symbol)
            if live_price is None:
                live_price = avg_price  # Fallback to avg price
            
            current_value = quantity * live_price
            pnl = current_value - invested
            pnl_percent = (pnl / invested * 100) if invested > 0 else 0
            
            total_invested += invested
            total_current_value += current_value
            
            holdings_with_prices.append({
                "symbol": symbol.replace(".NS", "").replace(".BO", ""),
                "quantity": quantity,
                "avg_price": avg_price,
                "live_price": live_price,
                "invested": round(invested, 2),
                "current_value": round(current_value, 2),
                "pnl": round(pnl, 2),
                "pnl_percent": round(pnl_percent, 2),
            })
        
        # Calculate totals
        total_pnl = total_current_value - total_invested
        total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        return {
            "user_id": user_id,
            "cash_balance": float(user["paper_balance"]),
            "holdings": holdings_with_prices,
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_percent": round(total_pnl_percent, 2),
            "portfolio_value": round(float(user["paper_balance"]) + total_current_value, 2),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/order")
async def place_order(request: PlaceOrderRequest):
    """
    Place a paper trading order (BUY or SELL).
    
    This simulates a real trade:
    - Gets live market price
    - Deducts money for BUY
    - Adds money for SELL
    - Updates holdings
    - Records the order
    """
    try:
        # Validate action
        if request.action.upper() not in ["BUY", "SELL"]:
            raise HTTPException(status_code=400, detail="Action must be BUY or SELL")
        
        if request.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        
        # Get user
        user = await get_user_by_id(request.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        cash_balance = float(user["paper_balance"])
        action = request.action.upper()
        
        # Get live price
        symbol = request.symbol
        if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            symbol = f"{symbol}.NS"
        
        if request.order_type == "LIMIT" and request.limit_price:
            execution_price = request.limit_price
        else:
            execution_price = get_live_price(symbol)
            if execution_price is None:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Could not get price for {request.symbol}"
                )
        
        # Calculate total value and charges
        total_value = request.quantity * execution_price
        charges = calculate_charges(total_value, action)
        total_with_charges = total_value + charges["total_charges"]
        
        # Get current holdings
        holdings = await get_user_holdings(request.user_id)
        current_holding = next(
            (h for h in holdings if h["symbol"] == symbol), 
            None
        )
        current_qty = current_holding["quantity"] if current_holding else 0
        current_avg = float(current_holding["avg_price"]) if current_holding else 0
        
        # Process BUY order
        if action == "BUY":
            if total_with_charges > cash_balance:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient balance. Need ₹{total_with_charges:.2f}, have ₹{cash_balance:.2f}"
                )
            
            # Update balance
            new_balance = cash_balance - total_with_charges
            await update_user_balance(request.user_id, new_balance)
            
            # Update holdings (calculate new average price)
            new_qty = current_qty + request.quantity
            new_avg = ((current_qty * current_avg) + (request.quantity * execution_price)) / new_qty
            await update_holding(request.user_id, symbol, new_qty, new_avg)
            
        # Process SELL order
        else:  # SELL
            if request.quantity > current_qty:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient holdings. Have {current_qty} shares of {request.symbol}"
                )
            
            # Update balance (add money from sale minus charges)
            new_balance = cash_balance + total_value - charges["total_charges"]
            await update_user_balance(request.user_id, new_balance)
            
            # Update holdings
            new_qty = current_qty - request.quantity
            await update_holding(
                request.user_id, 
                symbol, 
                new_qty, 
                current_avg if new_qty > 0 else 0
            )
        
        # Record the order
        order = await create_paper_order(request.user_id, {
            "symbol": symbol,
            "action": action,
            "quantity": request.quantity,
            "price": execution_price,
            "order_type": request.order_type,
        })
        
        return {
            "success": True,
            "message": f"{action} order executed successfully!",
            "order_id": order["id"] if order else None,
            "symbol": request.symbol,
            "action": action,
            "quantity": request.quantity,
            "executed_price": execution_price,
            "total_value": round(total_value, 2),
            "charges": charges,
            "total_with_charges": round(total_with_charges, 2),
            "new_balance": round(new_balance, 2),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders/{user_id}")
async def get_orders(user_id: str, limit: int = 50):
    """
    Get user's paper trading order history.
    """
    try:
        orders = await get_user_orders(user_id, limit)
        
        # Format orders for display
        formatted_orders = []
        for order in orders:
            formatted_orders.append({
                "id": order["id"],
                "symbol": order["symbol"].replace(".NS", "").replace(".BO", ""),
                "action": order["action"],
                "quantity": order["quantity"],
                "price": float(order["price"]),
                "total_value": float(order["total_value"]) if order.get("total_value") else None,
                "status": order["status"],
                "order_type": order["order_type"],
                "created_at": order["created_at"],
            })
        
        return {
            "user_id": user_id,
            "orders": formatted_orders,
            "total_count": len(formatted_orders),
        }
        
    except Exception as e:
        print(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price/{symbol}")
async def get_stock_price(symbol: str):
    """
    Get current price for a stock.
    
    Example: /api/paper/price/RELIANCE
    """
    try:
        price = get_live_price(symbol)
        
        if price is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Could not find price for {symbol}"
            )
        
        # Get additional info
        full_symbol = symbol if symbol.endswith(".NS") or symbol.endswith(".BO") else f"{symbol}.NS"
        ticker = yf.Ticker(full_symbol)
        info = ticker.info
        
        return {
            "symbol": symbol.replace(".NS", "").replace(".BO", ""),
            "price": price,
            "name": info.get("shortName", symbol),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "prev_close": info.get("previousClose"),
            "change": round(price - info.get("previousClose", price), 2) if info.get("previousClose") else 0,
            "change_percent": round(
                ((price - info.get("previousClose", price)) / info.get("previousClose", price) * 100), 2
            ) if info.get("previousClose") else 0,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting price: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset/{user_id}")
async def reset_paper_account(user_id: str):
    """
    Reset paper trading account to starting balance.
    
    This will:
    - Reset balance to ₹10 Lakh
    - Clear all holdings
    - Keep order history (for learning)
    """
    try:
        # Reset balance
        await update_user_balance(user_id, 1000000.00)
        
        # Clear holdings
        db = get_supabase()
        if db:
            db.table("paper_holdings").delete().eq("user_id", user_id).execute()
        
        return {
            "success": True,
            "message": "Paper trading account reset successfully!",
            "new_balance": 1000000.00,
        }
        
    except Exception as e:
        print(f"Error resetting account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🏆 LEADERBOARD (Gamification!)
# ============================================================

@router.get("/leaderboard")
async def get_leaderboard():
    """
    Get paper trading leaderboard.
    
    Shows top performers by portfolio value.
    """
    try:
        db = get_supabase()
        if not db:
            return {"leaderboard": [], "message": "Database not connected"}
        
        # Get all users with their balances
        result = db.table("users").select("id, name, paper_balance").order(
            "paper_balance", desc=True
        ).limit(10).execute()
        
        leaderboard = []
        for i, user in enumerate(result.data or []):
            # Get user's holdings value (simplified - without live prices for speed)
            holdings = db.table("paper_holdings").select("*").eq(
                "user_id", user["id"]
            ).execute()
            
            holdings_value = sum(
                h["quantity"] * float(h["avg_price"]) 
                for h in (holdings.data or [])
            )
            
            total_value = float(user["paper_balance"]) + holdings_value
            pnl = total_value - 1000000  # Starting balance was 10 Lakh
            
            leaderboard.append({
                "rank": i + 1,
                "name": user["name"] or "Anonymous",
                "total_value": round(total_value, 2),
                "pnl": round(pnl, 2),
                "pnl_percent": round(pnl / 1000000 * 100, 2),
            })
        
        return {
            "leaderboard": leaderboard,
            "updated_at": datetime.now().isoformat(),
        }
        
    except Exception as e:
        print(f"Error getting leaderboard: {e}")
        return {"leaderboard": [], "error": str(e)}
