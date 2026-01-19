"""
Market Data API Endpoints
Real-time NSE/BSE market data, stock information, and AI analysis
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
import random

router = APIRouter(prefix="/api/market", tags=["Market Data"])

# Sample NSE/BSE stock data
SAMPLE_STOCKS = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "sector": "Energy", "market_cap": 1720000},
    {"symbol": "TCS", "name": "Tata Consultancy Services Ltd", "sector": "IT", "market_cap": 1380000},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "sector": "Banking", "market_cap": 1240000},
    {"symbol": "INFY", "name": "Infosys Ltd", "sector": "IT", "market_cap": 720000},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "sector": "Banking", "market_cap": 670000},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd", "sector": "FMCG", "market_cap": 620000},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "sector": "Telecom", "market_cap": 920000},
    {"symbol": "SBIN", "name": "State Bank of India", "sector": "Banking", "market_cap": 560000},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd", "sector": "Finance", "market_cap": 450000},
    {"symbol": "ADANIENT", "name": "Adani Enterprises Ltd", "sector": "Diversified", "market_cap": 380000},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank Ltd", "sector": "Banking", "market_cap": 350000},
    {"symbol": "LT", "name": "Larsen & Toubro Ltd", "sector": "Infrastructure", "market_cap": 490000},
    {"symbol": "AXISBANK", "name": "Axis Bank Ltd", "sector": "Banking", "market_cap": 330000},
    {"symbol": "ITC", "name": "ITC Ltd", "sector": "FMCG", "market_cap": 570000},
    {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd", "sector": "Automobile", "market_cap": 380000},
]

def generate_price_data(base_price: float, symbol: str):
    """Generate realistic price data with some randomness"""
    # Use symbol hash for consistent randomness
    seed = sum(ord(c) for c in symbol)
    random.seed(seed + int(datetime.now().timestamp() / 3600))  # Changes hourly
    
    current_price = base_price * (1 + random.uniform(-0.05, 0.05))
    day_change = random.uniform(-5, 5)
    volume = random.randint(500000, 15000000)
    high_52w = base_price * random.uniform(1.1, 1.4)
    low_52w = base_price * random.uniform(0.7, 0.9)
    
    return {
        "current_price": round(current_price, 2),
        "day_change": round(day_change, 2),
        "day_change_percent": round(day_change / current_price * 100, 2),
        "volume": volume,
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "last_updated": datetime.now().isoformat()
    }

@router.get("/overview")
async def get_market_overview():
    """Get NSE/BSE market overview with indices"""
    return {
        "timestamp": datetime.now().isoformat(),
        "indices": {
            "nifty50": {
                "value": 21678.90,
                "change": 145.30,
                "change_percent": 0.67,
                "high": 21720.50,
                "low": 21550.20
            },
            "sensex": {
                "value": 71752.40,
                "change": 223.50,
                "change_percent": 0.31,
                "high": 71890.80,
                "low": 71620.10
            },
            "banknifty": {
                "value": 46342.80,
                "change": -124.60,
                "change_percent": -0.27,
                "high": 46510.20,
                "low": 46280.50
            }
        },
        "market_status": "OPEN",
        "market_sentiment": "BULLISH",
        "top_gainers": 127,
        "top_losers": 98,
        "unchanged": 25
    }

@router.get("/stocks")
async def get_all_stocks(
    sector: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0
):
    """Get all NSE/BSE stocks with prices"""
    stocks = SAMPLE_STOCKS.copy()
    
    if sector:
        stocks = [s for s in stocks if s["sector"].lower() == sector.lower()]
    
    # Generate price data for each stock
    base_prices = {
        "RELIANCE": 2847, "TCS": 3679, "HDFCBANK": 1678, "INFY": 1523,
        "ICICIBANK": 1089, "HINDUNILVR": 2456, "BHARTIARTL": 1547, "SBIN": 789,
        "BAJFINANCE": 6789, "ADANIENT": 2387, "KOTAKBANK": 1834, "LT": 3567,
        "AXISBANK": 1123, "ITC": 456, "MARUTI": 12456
    }
    
    result = []
    for stock in stocks[offset:offset+limit]:
        base_price = base_prices.get(stock["symbol"], 1000)
        price_data = generate_price_data(base_price, stock["symbol"])
        
        result.append({
            **stock,
            **price_data,
            "ai_score": random.randint(70, 95)
        })
    
    return {
        "total": len(SAMPLE_STOCKS),
        "count": len(result),
        "stocks": result
    }

@router.get("/stocks/{symbol}")
async def get_stock_detail(symbol: str):
    """Get detailed stock information"""
    stock = next((s for s in SAMPLE_STOCKS if s["symbol"] == symbol), None)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
    
    base_prices = {
        "RELIANCE": 2847, "TCS": 3679, "HDFCBANK": 1678, "INFY": 1523,
        "ICICIBANK": 1089, "HINDUNILVR": 2456, "BHARTIARTL": 1547, "SBIN": 789,
        "BAJFINANCE": 6789, "ADANIENT": 2387, "KOTAKBANK": 1834, "LT": 3567,
        "AXISBANK": 1123, "ITC": 456, "MARUTI": 12456
    }
    
    base_price = base_prices.get(symbol, 1000)
    price_data = generate_price_data(base_price, symbol)
    
    return {
        **stock,
        **price_data,
        "ai_score": random.randint(75, 95),
        "ai_analysis": {
            "recommendation": random.choice(["BUY", "HOLD", "SELL"]),
            "confidence": random.randint(70, 95),
            "target_price": round(price_data["current_price"] * random.uniform(1.05, 1.15), 2),
            "stop_loss": round(price_data["current_price"] * random.uniform(0.92, 0.96), 2),
            "risk_reward": round(random.uniform(2.0, 4.5), 1)
        },
        "fundamentals": {
            "pe_ratio": round(random.uniform(15, 35), 2),
            "market_cap": stock["market_cap"],
            "dividend_yield": round(random.uniform(0.5, 3.5), 2),
            "roe": round(random.uniform(12, 28), 2)
        }
    }

@router.get("/trending")
async def get_trending_stocks(limit: int = 10):
    """Get trending stocks based on AI analysis"""
    stocks = SAMPLE_STOCKS[:limit]
    
    base_prices = {
        "RELIANCE": 2847, "TCS": 3679, "HDFCBANK": 1678, "INFY": 1523,
        "ICICIBANK": 1089, "HINDUNILVR": 2456, "BHARTIARTL": 1547, "SBIN": 789,
        "BAJFINANCE": 6789, "ADANIENT": 2387, "KOTAKBANK": 1834, "LT": 3567,
    }
    
    result = []
    for stock in stocks:
        base_price = base_prices.get(stock["symbol"], 1000)
        price_data = generate_price_data(base_price, stock["symbol"])
        
        result.append({
            "symbol": stock["symbol"],
            "name": stock["name"],
            "current_price": price_data["current_price"],
            "day_change_percent": price_data["day_change_percent"],
            "volume": price_data["volume"],
            "ai_score": random.randint(80, 95),
            "trading_volume_rank": random.randint(1, 100)
        })
    
    # Sort by AI score
    result.sort(key=lambda x: x["ai_score"], reverse=True)
    
    return {"trending_stocks": result}

@router.get("/top-movers")
async def get_top_movers():
    """Get top gainers and losers"""
    base_prices = {
        "RELIANCE": 2847, "TCS": 3679, "HDFCBANK": 1678, "INFY": 1523,
        "ICICIBANK": 1089, "HINDUNILVR": 2456, "BHARTIARTL": 1547, "SBIN": 789,
        "BAJFINANCE": 6789, "ADANIENT": 2387, "KOTAKBANK": 1834, "LT": 3567,
        "AXISBANK": 1123, "ITC": 456, "MARUTI": 12456
    }
    
    gainers = []
    losers = []
    
    for stock in SAMPLE_STOCKS[:10]:
        base_price = base_prices.get(stock["symbol"], 1000)
        # Force some stocks to be gainers and some losers
        is_gainer = len(gainers) < 5
        
        if is_gainer:
            change_percent = random.uniform(2, 8)
        else:
            change_percent = random.uniform(-8, -2)
        
        current_price = base_price * (1 + change_percent / 100)
        
        data = {
            "symbol": stock["symbol"],
            "name": stock["name"],
            "current_price": round(current_price, 2),
            "day_change_percent": round(change_percent, 2),
            "volume": random.randint(1000000, 10000000)
        }
        
        if is_gainer:
            gainers.append(data)
        else:
            losers.append(data)
    
    return {
        "gainers": sorted(gainers, key=lambda x: x["day_change_percent"], reverse=True),
        "losers": sorted(losers, key=lambda x: x["day_change_percent"])
    }

@router.get("/sectors")
async def get_sector_performance():
    """Get sector-wise performance"""
    sectors = [
        "IT", "Banking", "Energy", "FMCG", "Automobile", 
        "Pharma", "Telecom", "Infrastructure", "Finance", "Diversified"
    ]
    
    result = []
    for sector in sectors:
        change = random.uniform(-3, 5)
        result.append({
            "sector": sector,
            "change_percent": round(change, 2),
            "stocks_count": random.randint(15, 50),
            "market_cap": random.randint(50000, 500000),
            "top_stock": random.choice([s["symbol"] for s in SAMPLE_STOCKS if s["sector"] == sector] or ["N/A"])
        })
    
    return {"sectors": sorted(result, key=lambda x: x["change_percent"], reverse=True)}
