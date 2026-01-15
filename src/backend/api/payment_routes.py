"""
================================================================================
SWINGAI - PAYMENT ROUTES (RAZORPAY)
================================================================================
Complete Razorpay integration:
- Create orders
- Verify payments
- Webhook handling (payment.captured, payment.failed, refund.created)
- Subscription management
================================================================================
"""

import os
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Header
from pydantic import BaseModel

from ..core.database import supabase_admin
from ..core.security import get_current_user
from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["payments"])

# ============================================================================
# RAZORPAY CONFIG
# ============================================================================

RAZORPAY_KEY_ID = settings.RAZORPAY_KEY_ID
RAZORPAY_KEY_SECRET = settings.RAZORPAY_KEY_SECRET
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

# Initialize Razorpay client lazily
_razorpay_client = None

def get_razorpay_client():
    global _razorpay_client
    if _razorpay_client is None:
        try:
            import razorpay
            _razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        except ImportError:
            logger.warning("razorpay package not installed")
            return None
    return _razorpay_client


# ============================================================================
# SUBSCRIPTION PLANS (Prices in paise)
# ============================================================================

SUBSCRIPTION_PLANS = {
    "free": {
        "id": "free",
        "name": "free",
        "display_name": "Free",
        "description": "Basic access with limited signals",
        "price_monthly": 0,
        "price_quarterly": 0,
        "price_yearly": 0,
        "max_signals_per_day": 2,
        "max_positions": 2,
        "max_capital": 10000000,  # 1 lakh in paise
        "signal_only": True,
        "semi_auto": False,
        "full_auto": False,
        "equity_trading": True,
        "futures_trading": False,
        "options_trading": False,
        "telegram_alerts": False,
        "priority_support": False,
        "api_access": False,
    },
    "starter": {
        "id": "starter",
        "name": "starter",
        "display_name": "Starter",
        "description": "For beginners starting their trading journey",
        "price_monthly": 49900,  # ₹499
        "price_quarterly": 129900,  # ₹1,299 (save ₹198)
        "price_yearly": 399900,  # ₹3,999 (save ₹1,989)
        "max_signals_per_day": 5,
        "max_positions": 5,
        "max_capital": 50000000,  # 5 lakh
        "signal_only": True,
        "semi_auto": True,
        "full_auto": False,
        "equity_trading": True,
        "futures_trading": False,
        "options_trading": False,
        "telegram_alerts": True,
        "priority_support": False,
        "api_access": False,
    },
    "pro": {
        "id": "pro",
        "name": "pro",
        "display_name": "Pro",
        "description": "For serious traders who want edge",
        "price_monthly": 149900,  # ₹1,499
        "price_quarterly": 399900,  # ₹3,999 (save ₹498)
        "price_yearly": 1199900,  # ₹11,999 (save ₹5,989)
        "max_signals_per_day": 15,
        "max_positions": 10,
        "max_capital": 200000000,  # 20 lakh
        "signal_only": True,
        "semi_auto": True,
        "full_auto": True,
        "equity_trading": True,
        "futures_trading": True,
        "options_trading": True,
        "telegram_alerts": True,
        "priority_support": True,
        "api_access": False,
    },
    "elite": {
        "id": "elite",
        "name": "elite",
        "display_name": "Elite",
        "description": "For professional traders & HNIs",
        "price_monthly": 299900,  # ₹2,999
        "price_quarterly": 799900,  # ₹7,999 (save ₹998)
        "price_yearly": 2499900,  # ₹24,999 (save ₹10,989)
        "max_signals_per_day": -1,  # Unlimited
        "max_positions": 25,
        "max_capital": -1,  # Unlimited
        "signal_only": True,
        "semi_auto": True,
        "full_auto": True,
        "equity_trading": True,
        "futures_trading": True,
        "options_trading": True,
        "telegram_alerts": True,
        "priority_support": True,
        "api_access": True,
    },
}


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreateOrderRequest(BaseModel):
    plan_id: str
    billing_period: str  # monthly, quarterly, yearly


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def verify_razorpay_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Verify Razorpay payment signature.
    """
    if not RAZORPAY_KEY_SECRET:
        logger.warning("Razorpay secret not configured")
        return False
    
    message = f"{order_id}|{payment_id}"
    expected_signature = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


def verify_webhook_signature(body: bytes, signature: str) -> bool:
    """
    Verify Razorpay webhook signature.
    """
    if not RAZORPAY_WEBHOOK_SECRET:
        logger.warning("Razorpay webhook secret not configured")
        return False
    
    expected_signature = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


def calculate_subscription_dates(billing_period: str) -> tuple:
    """
    Calculate subscription start and end dates.
    """
    start_date = datetime.utcnow()
    
    if billing_period == "monthly":
        end_date = start_date + timedelta(days=30)
    elif billing_period == "quarterly":
        end_date = start_date + timedelta(days=90)
    elif billing_period == "yearly":
        end_date = start_date + timedelta(days=365)
    else:
        end_date = start_date + timedelta(days=30)
    
    return start_date, end_date


def activate_subscription(user_id: str, plan_id: str, billing_period: str, payment_id: str, order_id: str):
    """
    Activate user subscription after successful payment.
    """
    plan = SUBSCRIPTION_PLANS.get(plan_id)
    if not plan:
        raise ValueError(f"Unknown plan: {plan_id}")
    
    start_date, end_date = calculate_subscription_dates(billing_period)
    
    # Get price based on billing period
    price_key = f"price_{billing_period}"
    amount = plan.get(price_key, plan.get("price_monthly", 0))
    
    # Create/update subscription record
    subscription_data = {
        "user_id": user_id,
        "plan_id": plan_id,
        "plan_name": plan["name"],
        "billing_period": billing_period,
        "amount": amount,
        "status": "active",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "razorpay_payment_id": payment_id,
        "razorpay_order_id": order_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    supabase_admin.table("subscriptions").upsert(
        subscription_data,
        on_conflict="user_id"
    ).execute()
    
    # Update user profile
    supabase_admin.table("user_profiles").update({
        "subscription_status": "active",
        "subscription_plan_id": plan_id,
        "subscription_end_date": end_date.isoformat(),
        "max_positions": plan["max_positions"],
        "fo_enabled": plan["futures_trading"] or plan["options_trading"],
    }).eq("id", user_id).execute()
    
    logger.info(f"Subscription activated for user {user_id}: {plan_id} ({billing_period})")


def deactivate_subscription(user_id: str, reason: str = "cancelled"):
    """
    Deactivate user subscription.
    """
    # Update subscription status
    supabase_admin.table("subscriptions").update({
        "status": reason,
        "cancelled_at": datetime.utcnow().isoformat(),
    }).eq("user_id", user_id).eq("status", "active").execute()
    
    # Update user profile to free tier
    supabase_admin.table("user_profiles").update({
        "subscription_status": reason,
        "subscription_plan_id": "free",
        "max_positions": 2,
        "fo_enabled": False,
    }).eq("id", user_id).execute()
    
    logger.info(f"Subscription deactivated for user {user_id}: {reason}")


# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================

@router.get("/plans")
async def get_plans():
    """
    Get all available subscription plans.
    """
    return {"plans": list(SUBSCRIPTION_PLANS.values())}


# ============================================================================
# AUTHENTICATED ENDPOINTS
# ============================================================================

@router.post("/create-order")
async def create_order(
    request: CreateOrderRequest,
    user: Any = Depends(get_current_user)
):
    """
    Create Razorpay order for subscription payment.
    """
    plan = SUBSCRIPTION_PLANS.get(request.plan_id)
    if not plan:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    if plan["name"] == "free":
        raise HTTPException(status_code=400, detail="Cannot purchase free plan")
    
    # Get price based on billing period
    price_key = f"price_{request.billing_period}"
    amount = plan.get(price_key)
    
    if amount is None:
        raise HTTPException(status_code=400, detail="Invalid billing period")
    
    if amount == 0:
        raise HTTPException(status_code=400, detail="Plan has no cost")
    
    # Create Razorpay order
    client = get_razorpay_client()
    
    if client:
        try:
            order = client.order.create({
                "amount": amount,
                "currency": "INR",
                "receipt": f"sub_{user.id}_{request.plan_id}",
                "notes": {
                    "user_id": user.id,
                    "plan_id": request.plan_id,
                    "billing_period": request.billing_period
                }
            })
            
            # Store order in database
            supabase_admin.table("payment_orders").insert({
                "user_id": user.id,
                "razorpay_order_id": order["id"],
                "plan_id": request.plan_id,
                "billing_period": request.billing_period,
                "amount": amount,
                "currency": "INR",
                "status": "created",
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            
            return {
                "order_id": order["id"],
                "amount": amount,
                "currency": "INR",
                "key_id": RAZORPAY_KEY_ID
            }
            
        except Exception as e:
            logger.error(f"Razorpay order creation failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to create order")
    else:
        # Mock order for development
        mock_order_id = f"order_mock_{datetime.now().timestamp()}"
        
        supabase_admin.table("payment_orders").insert({
            "user_id": user.id,
            "razorpay_order_id": mock_order_id,
            "plan_id": request.plan_id,
            "billing_period": request.billing_period,
            "amount": amount,
            "currency": "INR",
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        return {
            "order_id": mock_order_id,
            "amount": amount,
            "currency": "INR",
            "key_id": RAZORPAY_KEY_ID or "rzp_test_mock"
        }


@router.post("/verify")
async def verify_payment(
    request: VerifyPaymentRequest,
    user: Any = Depends(get_current_user)
):
    """
    Verify Razorpay payment signature and activate subscription.
    """
    # Verify signature
    if RAZORPAY_KEY_SECRET:
        is_valid = verify_razorpay_signature(
            request.razorpay_order_id,
            request.razorpay_payment_id,
            request.razorpay_signature
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail="Invalid payment signature")
    
    # Get order details
    order_result = supabase_admin.table("payment_orders").select(
        "*"
    ).eq("razorpay_order_id", request.razorpay_order_id).single().execute()
    
    if not order_result.data:
        raise HTTPException(status_code=404, detail="Order not found")
    
    order = order_result.data
    
    # Verify order belongs to user
    if order["user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Order does not belong to user")
    
    # Update order status
    supabase_admin.table("payment_orders").update({
        "status": "paid",
        "razorpay_payment_id": request.razorpay_payment_id,
        "paid_at": datetime.utcnow().isoformat()
    }).eq("razorpay_order_id", request.razorpay_order_id).execute()
    
    # Record payment transaction
    supabase_admin.table("payment_transactions").insert({
        "user_id": user.id,
        "razorpay_order_id": request.razorpay_order_id,
        "razorpay_payment_id": request.razorpay_payment_id,
        "amount": order["amount"],
        "currency": order["currency"],
        "status": "captured",
        "plan_id": order["plan_id"],
        "created_at": datetime.utcnow().isoformat()
    }).execute()
    
    # Activate subscription
    try:
        activate_subscription(
            user_id=user.id,
            plan_id=order["plan_id"],
            billing_period=order["billing_period"],
            payment_id=request.razorpay_payment_id,
            order_id=request.razorpay_order_id
        )
        
        return {"success": True, "message": "Subscription activated"}
        
    except Exception as e:
        logger.error(f"Subscription activation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate subscription")


@router.get("/subscription")
async def get_subscription(user: Any = Depends(get_current_user)):
    """
    Get user's current subscription details.
    """
    result = supabase_admin.table("subscriptions").select(
        "*"
    ).eq("user_id", user.id).order("created_at", desc=True).limit(1).execute()
    
    if not result.data:
        return {
            "plan_id": "free",
            "status": "active",
            "plan_details": SUBSCRIPTION_PLANS["free"]
        }
    
    subscription = result.data[0]
    plan_details = SUBSCRIPTION_PLANS.get(subscription["plan_id"], SUBSCRIPTION_PLANS["free"])
    
    return {
        **subscription,
        "plan_details": plan_details
    }


@router.post("/subscription/cancel")
async def cancel_subscription(user: Any = Depends(get_current_user)):
    """
    Cancel user's subscription (effective at end of billing period).
    """
    try:
        # Mark for cancellation (don't immediately deactivate)
        supabase_admin.table("subscriptions").update({
            "status": "cancelling",
            "cancel_at_period_end": True,
            "cancelled_at": datetime.utcnow().isoformat()
        }).eq("user_id", user.id).eq("status", "active").execute()
        
        return {"success": True, "message": "Subscription will be cancelled at the end of billing period"}
        
    except Exception as e:
        logger.error(f"Subscription cancellation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel subscription")


# ============================================================================
# WEBHOOK ENDPOINT
# ============================================================================

@router.post("/webhook")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: Optional[str] = Header(None)
):
    """
    Handle Razorpay webhooks:
    - payment.captured: Payment successful
    - payment.failed: Payment failed
    - refund.created: Refund initiated
    """
    body = await request.body()
    
    # Verify webhook signature (if configured)
    if RAZORPAY_WEBHOOK_SECRET and x_razorpay_signature:
        if not verify_webhook_signature(body, x_razorpay_signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=400, detail="Invalid signature")
    
    try:
        payload = await request.json()
        event = payload.get("event")
        
        logger.info(f"Received Razorpay webhook: {event}")
        
        if event == "payment.captured":
            await handle_payment_captured(payload)
        elif event == "payment.failed":
            await handle_payment_failed(payload)
        elif event == "refund.created":
            await handle_refund_created(payload)
        elif event == "subscription.cancelled":
            await handle_subscription_cancelled(payload)
        else:
            logger.info(f"Unhandled webhook event: {event}")
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        # Return 200 to prevent retries for processing errors
        return {"status": "error", "message": str(e)}


async def handle_payment_captured(payload: Dict):
    """
    Handle successful payment capture.
    """
    payment = payload.get("payload", {}).get("payment", {}).get("entity", {})
    
    order_id = payment.get("order_id")
    payment_id = payment.get("id")
    amount = payment.get("amount")
    
    logger.info(f"Payment captured: {payment_id} for order {order_id}")
    
    # Get order details
    order_result = supabase_admin.table("payment_orders").select(
        "*"
    ).eq("razorpay_order_id", order_id).single().execute()
    
    if not order_result.data:
        logger.warning(f"Order not found for payment: {order_id}")
        return
    
    order = order_result.data
    
    # Update order status if not already paid
    if order["status"] != "paid":
        supabase_admin.table("payment_orders").update({
            "status": "paid",
            "razorpay_payment_id": payment_id,
            "paid_at": datetime.utcnow().isoformat()
        }).eq("razorpay_order_id", order_id).execute()
        
        # Activate subscription
        activate_subscription(
            user_id=order["user_id"],
            plan_id=order["plan_id"],
            billing_period=order["billing_period"],
            payment_id=payment_id,
            order_id=order_id
        )


async def handle_payment_failed(payload: Dict):
    """
    Handle failed payment.
    """
    payment = payload.get("payload", {}).get("payment", {}).get("entity", {})
    
    order_id = payment.get("order_id")
    payment_id = payment.get("id")
    error_code = payment.get("error_code")
    error_description = payment.get("error_description")
    
    logger.warning(f"Payment failed: {payment_id} - {error_code}: {error_description}")
    
    # Update order status
    supabase_admin.table("payment_orders").update({
        "status": "failed",
        "error_code": error_code,
        "error_description": error_description,
        "failed_at": datetime.utcnow().isoformat()
    }).eq("razorpay_order_id", order_id).execute()
    
    # Record failed transaction
    order_result = supabase_admin.table("payment_orders").select(
        "user_id, plan_id, amount, currency"
    ).eq("razorpay_order_id", order_id).single().execute()
    
    if order_result.data:
        supabase_admin.table("payment_transactions").insert({
            "user_id": order_result.data["user_id"],
            "razorpay_order_id": order_id,
            "razorpay_payment_id": payment_id,
            "amount": order_result.data["amount"],
            "currency": order_result.data["currency"],
            "status": "failed",
            "error_code": error_code,
            "error_description": error_description,
            "plan_id": order_result.data["plan_id"],
            "created_at": datetime.utcnow().isoformat()
        }).execute()


async def handle_refund_created(payload: Dict):
    """
    Handle refund creation.
    """
    refund = payload.get("payload", {}).get("refund", {}).get("entity", {})
    
    payment_id = refund.get("payment_id")
    refund_id = refund.get("id")
    amount = refund.get("amount")
    
    logger.info(f"Refund created: {refund_id} for payment {payment_id}, amount: {amount}")
    
    # Find original transaction
    tx_result = supabase_admin.table("payment_transactions").select(
        "user_id, plan_id"
    ).eq("razorpay_payment_id", payment_id).single().execute()
    
    if not tx_result.data:
        logger.warning(f"Transaction not found for refund: {payment_id}")
        return
    
    user_id = tx_result.data["user_id"]
    
    # Record refund transaction
    supabase_admin.table("payment_transactions").insert({
        "user_id": user_id,
        "razorpay_payment_id": payment_id,
        "razorpay_refund_id": refund_id,
        "amount": -amount,  # Negative for refund
        "currency": "INR",
        "status": "refunded",
        "plan_id": tx_result.data["plan_id"],
        "created_at": datetime.utcnow().isoformat()
    }).execute()
    
    # Deactivate subscription if full refund
    order_result = supabase_admin.table("payment_orders").select(
        "amount"
    ).eq("razorpay_payment_id", payment_id).single().execute()
    
    if order_result.data and amount >= order_result.data["amount"]:
        deactivate_subscription(user_id, "refunded")


async def handle_subscription_cancelled(payload: Dict):
    """
    Handle subscription cancellation from Razorpay.
    """
    subscription = payload.get("payload", {}).get("subscription", {}).get("entity", {})
    
    subscription_id = subscription.get("id")
    
    logger.info(f"Subscription cancelled: {subscription_id}")
    
    # Find subscription by Razorpay ID
    sub_result = supabase_admin.table("subscriptions").select(
        "user_id"
    ).eq("razorpay_subscription_id", subscription_id).single().execute()
    
    if sub_result.data:
        deactivate_subscription(sub_result.data["user_id"], "cancelled")
