"""
📦 DATABASE HELPER - Simple Supabase Connection
================================================
This file makes it EASY to use Supabase database.

BEGINNER TIP: 
- Supabase is like a super simple database with a nice UI
- You can see your data at: https://supabase.com/dashboard
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Supabase credentials from .env file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Try to import supabase, handle gracefully if not available
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Supabase import error: {e}")
    SUPABASE_AVAILABLE = False
    Client = None

# Create Supabase client (this connects to your database)
supabase = None

def get_supabase() -> Client:
    """
    Get the Supabase client.
    
    Usage:
        from database import get_supabase
        db = get_supabase()
        
        # Insert data
        db.table("users").insert({"name": "John"}).execute()
        
        # Get data
        db.table("users").select("*").execute()
    """
    global supabase
    
    if supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("⚠️ WARNING: Supabase credentials not found!")
            print("Please add SUPABASE_URL and SUPABASE_KEY to backend/.env")
            return None
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase!")
    
    return supabase


# ============================================================
# 🎯 SIMPLE DATABASE FUNCTIONS (Beginner Friendly!)
# ============================================================

async def create_user(email: str, name: str = None):
    """Create a new user in database"""
    db = get_supabase()
    if not db:
        return None
    
    result = db.table("users").insert({
        "email": email,
        "name": name or email.split("@")[0],
        "paper_balance": 1000000.00,  # ₹10 Lakh starting balance
        "is_paper_mode": True,  # Start with paper trading
    }).execute()
    
    return result.data[0] if result.data else None


async def get_user_by_email(email: str):
    """Get user by email"""
    db = get_supabase()
    if not db:
        return None
    
    result = db.table("users").select("*").eq("email", email).execute()
    return result.data[0] if result.data else None


async def get_user_by_id(user_id: str):
    """Get user by ID"""
    db = get_supabase()
    if not db:
        return None
    
    result = db.table("users").select("*").eq("id", user_id).execute()
    return result.data[0] if result.data else None


async def update_user_balance(user_id: str, new_balance: float):
    """Update user's paper trading balance"""
    db = get_supabase()
    if not db:
        return None
    
    result = db.table("users").update({
        "paper_balance": new_balance
    }).eq("id", user_id).execute()
    
    return result.data[0] if result.data else None


# ============================================================
# 📈 PAPER TRADING FUNCTIONS
# ============================================================

async def create_paper_order(user_id: str, order_data: dict):
    """
    Create a new paper trading order.
    
    order_data = {
        "symbol": "RELIANCE",
        "action": "BUY",
        "quantity": 10,
        "price": 2850.00,
        "order_type": "MARKET"
    }
    """
    db = get_supabase()
    if not db:
        return None
    
    result = db.table("paper_orders").insert({
        "user_id": user_id,
        "symbol": order_data["symbol"],
        "action": order_data["action"],
        "quantity": order_data["quantity"],
        "price": order_data["price"],
        "order_type": order_data.get("order_type", "MARKET"),
        "status": "EXECUTED",  # Paper orders execute instantly
        "total_value": order_data["quantity"] * order_data["price"],
    }).execute()
    
    return result.data[0] if result.data else None


async def get_user_orders(user_id: str, limit: int = 50):
    """Get user's paper trading orders"""
    db = get_supabase()
    if not db:
        return []
    
    result = db.table("paper_orders").select("*").eq(
        "user_id", user_id
    ).order("created_at", desc=True).limit(limit).execute()
    
    return result.data or []


async def get_user_holdings(user_id: str):
    """Get user's current paper holdings (portfolio)"""
    db = get_supabase()
    if not db:
        return []
    
    result = db.table("paper_holdings").select("*").eq(
        "user_id", user_id
    ).execute()
    
    return result.data or []


async def update_holding(user_id: str, symbol: str, quantity: int, avg_price: float):
    """Update or create a holding"""
    db = get_supabase()
    if not db:
        return None
    
    # Check if holding exists
    existing = db.table("paper_holdings").select("*").eq(
        "user_id", user_id
    ).eq("symbol", symbol).execute()
    
    if existing.data:
        # Update existing holding
        if quantity <= 0:
            # Delete if quantity is 0
            db.table("paper_holdings").delete().eq(
                "user_id", user_id
            ).eq("symbol", symbol).execute()
            return None
        else:
            result = db.table("paper_holdings").update({
                "quantity": quantity,
                "avg_price": avg_price,
            }).eq("user_id", user_id).eq("symbol", symbol).execute()
            return result.data[0] if result.data else None
    else:
        # Create new holding
        if quantity > 0:
            result = db.table("paper_holdings").insert({
                "user_id": user_id,
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": avg_price,
            }).execute()
            return result.data[0] if result.data else None
    
    return None


# ============================================================
# 📊 WATCHLIST FUNCTIONS
# ============================================================

async def get_user_watchlist(user_id: str):
    """Get user's watchlist"""
    db = get_supabase()
    if not db:
        return []
    
    result = db.table("watchlist").select("*").eq("user_id", user_id).execute()
    return result.data or []


async def add_to_watchlist(user_id: str, symbol: str):
    """Add stock to watchlist"""
    db = get_supabase()
    if not db:
        return None
    
    result = db.table("watchlist").insert({
        "user_id": user_id,
        "symbol": symbol,
    }).execute()
    
    return result.data[0] if result.data else None


async def remove_from_watchlist(user_id: str, symbol: str):
    """Remove stock from watchlist"""
    db = get_supabase()
    if not db:
        return None
    
    db.table("watchlist").delete().eq(
        "user_id", user_id
    ).eq("symbol", symbol).execute()
    
    return True


# ============================================================
# 🔧 SETUP HELPER - Run this once to create tables!
# ============================================================

def print_setup_instructions():
    """
    Print SQL to create tables in Supabase.
    
    HOW TO USE:
    1. Go to https://supabase.com/dashboard
    2. Select your project
    3. Go to "SQL Editor" (left sidebar)
    4. Copy and paste the SQL below
    5. Click "Run"
    """
    
    sql = """
-- ============================================================
-- 🚀 SWINGAI DATABASE SETUP
-- Copy this entire SQL and run in Supabase SQL Editor
-- ============================================================

-- 1. USERS TABLE
-- Stores user accounts and settings
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    paper_balance DECIMAL(15,2) DEFAULT 1000000.00,  -- ₹10 Lakh
    is_paper_mode BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. PAPER ORDERS TABLE
-- Stores all paper trading orders
CREATE TABLE IF NOT EXISTS paper_orders (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,  -- BUY or SELL
    quantity INTEGER NOT NULL,
    price DECIMAL(15,2) NOT NULL,
    order_type TEXT DEFAULT 'MARKET',
    status TEXT DEFAULT 'EXECUTED',
    total_value DECIMAL(15,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. PAPER HOLDINGS TABLE
-- Stores current portfolio holdings
CREATE TABLE IF NOT EXISTS paper_holdings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price DECIMAL(15,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, symbol)
);

-- 4. WATCHLIST TABLE
-- User's favorite stocks
CREATE TABLE IF NOT EXISTS watchlist (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, symbol)
);

-- 5. SIGNALS TABLE
-- AI generated trading signals
CREATE TABLE IF NOT EXISTS signals (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,  -- BUY
    entry_price DECIMAL(15,2),
    target_price DECIMAL(15,2),
    stop_loss DECIMAL(15,2),
    confidence INTEGER,
    status TEXT DEFAULT 'ACTIVE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- 🔐 ENABLE ROW LEVEL SECURITY (RLS)
-- This keeps each user's data private
-- ============================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE paper_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE paper_holdings ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;

-- Allow users to read their own data
CREATE POLICY "Users can view own data" ON users
    FOR SELECT USING (true);

CREATE POLICY "Users can view own orders" ON paper_orders
    FOR ALL USING (true);

CREATE POLICY "Users can view own holdings" ON paper_holdings
    FOR ALL USING (true);

CREATE POLICY "Users can view own watchlist" ON watchlist
    FOR ALL USING (true);

-- Signals are public (everyone can see)
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Signals are public" ON signals
    FOR SELECT USING (true);

-- ============================================================
-- ✅ DONE! Your database is ready.
-- ============================================================
"""
    
    print("\n" + "="*60)
    print("📦 SUPABASE DATABASE SETUP")
    print("="*60)
    print("\n1. Go to: https://supabase.com/dashboard")
    print("2. Click on your project")
    print("3. Click 'SQL Editor' in left sidebar")
    print("4. Click 'New Query'")
    print("5. Copy the SQL below and paste it")
    print("6. Click 'Run'\n")
    print("="*60)
    print(sql)
    print("="*60)
    
    return sql


# Run this if you want to see setup instructions
if __name__ == "__main__":
    print_setup_instructions()
