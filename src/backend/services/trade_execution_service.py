"""
Trade execution service for scheduler-driven fills.
Creates positions and updates trades when broker execution is not integrated.
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TradeExecutionService:
    def __init__(self, supabase_admin):
        self.supabase = supabase_admin

    async def execute(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a pending trade by opening a position and updating trade status.
        This is a DB-level execution for environments without broker integration.
        """
        try:
            trade_id = trade.get("id")
            user_id = trade.get("user_id")

            if not trade_id or not user_id:
                return {"success": False, "message": "Missing trade or user id"}

            if trade.get("status") not in ["pending", "approved"]:
                return {"success": False, "message": "Trade is not pending/approved"}

            existing = self.supabase.table("positions").select("id").eq("trade_id", trade_id).execute()
            if existing.data:
                return {"success": True, "message": "Position already open"}

            entry_price = float(trade.get("average_price") or trade.get("entry_price") or 0)
            if entry_price <= 0:
                return {"success": False, "message": "Invalid entry price"}

            quantity = int(trade.get("quantity") or 0)
            if quantity <= 0:
                return {"success": False, "message": "Invalid quantity"}

            position = {
                "user_id": user_id,
                "trade_id": trade_id,
                "symbol": trade.get("symbol"),
                "exchange": trade.get("exchange", "NSE"),
                "segment": trade.get("segment", "EQUITY"),
                "expiry_date": trade.get("expiry_date"),
                "strike_price": trade.get("strike_price"),
                "option_type": trade.get("option_type"),
                "direction": trade.get("direction"),
                "quantity": quantity,
                "lots": trade.get("lots", 1),
                "average_price": entry_price,
                "current_price": entry_price,
                "current_value": quantity * entry_price,
                "stop_loss": trade.get("stop_loss"),
                "target": trade.get("target"),
                "margin_used": trade.get("margin_used"),
                "risk_amount": trade.get("risk_amount"),
                "is_active": True,
                "last_updated": datetime.utcnow().isoformat(),
            }

            self.supabase.table("positions").insert(position).execute()

            self.supabase.table("trades").update({
                "status": "open",
                "executed_at": datetime.utcnow().isoformat(),
                "average_price": entry_price,
                "filled_quantity": quantity,
                "pending_quantity": 0,
            }).eq("id", trade_id).execute()

            return {"success": True, "message": "Trade executed"}
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"success": False, "message": str(e)}
