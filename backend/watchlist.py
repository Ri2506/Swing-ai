"""
🔖 WATCHLIST API
================
User watchlist management for tracking favorite stocks.

Features:
- Add/remove stocks from watchlist
- Get user's watchlist with live prices
- Bulk add from scan results
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import yfinance as yf

from database import get_supabase

router = APIRouter(prefix="/watchlist", tags=["Watchlist"])


# ============================================================
# 📝 DATA MODELS
# ============================================================

class AddToWatchlistRequest(BaseModel):
    """Request to add stock to watchlist"""
    user_id: str
    symbol: str
    notes: Optional[str] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None


class BulkAddRequest(BaseModel):
    """Request to add multiple stocks"""
    user_id: str
    symbols: List[str]


class WatchlistItem(BaseModel):
    """Watchlist item with price data"""
    id: str
    symbol: str
    name: Optional[str]
    current_price: float
    change_percent: float
    added_at: str
    notes: Optional[str]
    target_price: Optional[float]
    stop_loss: Optional[float]


# ============================================================
# 📊 HELPER FUNCTIONS
# ============================================================

def get_live_price(symbol: str) -> dict:
    """Get live price and info for a stock"""
    try:
        full_symbol = symbol if symbol.endswith(".NS") or symbol.endswith(".BO") else f"{symbol}.NS"
        ticker = yf.Ticker(full_symbol)
        data = ticker.history(period="2d")
        
        if data.empty:
            return None
        
        current_price = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
        
        info = ticker.info
        
        return {
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "name": info.get("shortName", symbol),
            "sector": info.get("sector", "Unknown"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "volume": int(data['Volume'].iloc[-1]) if 'Volume' in data else 0,
        }
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
        return None


# ============================================================
# 🎯 API ENDPOINTS
# ============================================================

@router.get("/{user_id}")
async def get_watchlist(user_id: str):
    """
    Get user's complete watchlist with live prices.
    """
    try:
        db = get_supabase()
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        # Get watchlist items - select only columns that exist
        result = db.table("watchlist").select("id, user_id, symbol, created_at").eq(
            "user_id", user_id
        ).order("created_at", desc=True).execute()
        
        watchlist_items = []
        
        for item in (result.data or []):
            symbol = item["symbol"].replace(".NS", "").replace(".BO", "")
            
            # Get live price
            price_data = get_live_price(symbol)
            
            watchlist_items.append({
                "id": item["id"],
                "symbol": symbol,
                "name": price_data.get("name", symbol) if price_data else symbol,
                "current_price": price_data.get("current_price", 0) if price_data else 0,
                "change_percent": price_data.get("change_percent", 0) if price_data else 0,
                "volume": price_data.get("volume", 0) if price_data else 0,
                "sector": price_data.get("sector", "Unknown") if price_data else "Unknown",
                "notes": None,  # Column may not exist
                "target_price": None,
                "stop_loss": None,
                "added_at": item["created_at"],
            })
        
        return {
            "success": True,
            "user_id": user_id,
            "count": len(watchlist_items),
            "watchlist": watchlist_items,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add")
async def add_to_watchlist(request: AddToWatchlistRequest):
    """
    Add a stock to user's watchlist.
    """
    try:
        db = get_supabase()
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        # Normalize symbol
        symbol = request.symbol.upper().replace(".NS", "").replace(".BO", "")
        full_symbol = f"{symbol}.NS"
        
        # Check if already in watchlist
        existing = db.table("watchlist").select("id").eq(
            "user_id", request.user_id
        ).eq("symbol", full_symbol).execute()
        
        if existing.data and len(existing.data) > 0:
            return {
                "success": False,
                "message": f"{symbol} is already in your watchlist",
                "already_exists": True
            }
        
        # Verify stock exists
        price_data = get_live_price(symbol)
        if not price_data:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Add to watchlist - simple insert without optional fields
        # The table may not have notes/target/stop_loss columns
        insert_data = {
            "user_id": request.user_id,
            "symbol": full_symbol,
        }
        
        result = db.table("watchlist").insert(insert_data).execute()
        
        return {
            "success": True,
            "message": f"{symbol} added to watchlist",
            "item": {
                "id": result.data[0]["id"] if result.data else None,
                "symbol": symbol,
                "name": price_data.get("name", symbol),
                "current_price": price_data.get("current_price", 0),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-add")
async def bulk_add_to_watchlist(request: BulkAddRequest):
    """
    Add multiple stocks to watchlist at once.
    Useful for adding stocks from scan results.
    """
    try:
        db = get_supabase()
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        added = []
        skipped = []
        
        for symbol in request.symbols:
            # Normalize symbol
            symbol = symbol.upper().replace(".NS", "").replace(".BO", "")
            full_symbol = f"{symbol}.NS"
            
            # Check if already exists
            existing = db.table("watchlist").select("id").eq(
                "user_id", request.user_id
            ).eq("symbol", full_symbol).execute()
            
            if existing.data and len(existing.data) > 0:
                skipped.append(symbol)
                continue
            
            # Add to watchlist
            try:
                db.table("watchlist").insert({
                    "user_id": request.user_id,
                    "symbol": full_symbol,
                }).execute()
                added.append(symbol)
            except:
                skipped.append(symbol)
        
        return {
            "success": True,
            "message": f"Added {len(added)} stocks to watchlist",
            "added": added,
            "skipped": skipped,
            "added_count": len(added),
            "skipped_count": len(skipped)
        }
        
    except Exception as e:
        print(f"Error bulk adding to watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}/{symbol}")
async def remove_from_watchlist(user_id: str, symbol: str):
    """
    Remove a stock from user's watchlist.
    """
    try:
        db = get_supabase()
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        # Normalize symbol
        symbol_clean = symbol.upper().replace(".NS", "").replace(".BO", "")
        full_symbol = f"{symbol_clean}.NS"
        
        # Delete from watchlist
        result = db.table("watchlist").delete().eq(
            "user_id", user_id
        ).eq("symbol", full_symbol).execute()
        
        return {
            "success": True,
            "message": f"{symbol_clean} removed from watchlist"
        }
        
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{user_id}/{symbol}")
async def update_watchlist_item(
    user_id: str, 
    symbol: str,
    notes: Optional[str] = Query(None),
    target_price: Optional[float] = Query(None),
    stop_loss: Optional[float] = Query(None)
):
    """
    Update watchlist item (notes, target, stop loss).
    """
    try:
        db = get_supabase()
        if not db:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        # Normalize symbol
        symbol_clean = symbol.upper().replace(".NS", "").replace(".BO", "")
        full_symbol = f"{symbol_clean}.NS"
        
        # Build update data
        update_data = {}
        if notes is not None:
            update_data["notes"] = notes
        if target_price is not None:
            update_data["target_price"] = target_price
        if stop_loss is not None:
            update_data["stop_loss"] = stop_loss
        
        if not update_data:
            return {"success": False, "message": "No updates provided"}
        
        # Update
        db.table("watchlist").update(update_data).eq(
            "user_id", user_id
        ).eq("symbol", full_symbol).execute()
        
        return {
            "success": True,
            "message": f"{symbol_clean} updated",
            "updates": update_data
        }
        
    except Exception as e:
        print(f"Error updating watchlist item: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/check/{symbol}")
async def check_in_watchlist(user_id: str, symbol: str):
    """
    Check if a stock is in user's watchlist.
    """
    try:
        db = get_supabase()
        if not db:
            return {"in_watchlist": False}
        
        symbol_clean = symbol.upper().replace(".NS", "").replace(".BO", "")
        full_symbol = f"{symbol_clean}.NS"
        
        result = db.table("watchlist").select("id").eq(
            "user_id", user_id
        ).eq("symbol", full_symbol).execute()
        
        return {
            "symbol": symbol_clean,
            "in_watchlist": len(result.data or []) > 0
        }
        
    except Exception as e:
        return {"in_watchlist": False, "error": str(e)}
