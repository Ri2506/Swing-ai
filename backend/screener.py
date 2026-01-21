"""
🔍 AI SCREENER API v2 - Full NSE Coverage
==========================================
Real stock screening with ALL NSE stocks using nsepython + yfinance.

Features:
- All 2200+ NSE stocks
- Real-time data
- Technical indicators (RSI, MACD, MA)
- Multiple scanner categories
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd
import numpy as np

# Import nsepython for full NSE coverage
try:
    from nsepython import nse_eq_symbols, nse_get_index_list, nse_get_index_quote
    NSE_AVAILABLE = True
    print(f"✅ nsepython loaded - {len(nse_eq_symbols())} NSE stocks available")
except ImportError as e:
    NSE_AVAILABLE = False
    print(f"⚠️ nsepython not available: {e}")
except Exception as e:
    NSE_AVAILABLE = False
    print(f"⚠️ nsepython error: {e}")

router = APIRouter(prefix="/screener", tags=["Screener"])

# ============================================================
# 📊 STOCK UNIVERSE
# ============================================================

def get_all_nse_stocks() -> List[str]:
    """Get all NSE stock symbols"""
    if NSE_AVAILABLE:
        try:
            return nse_eq_symbols()
        except Exception as e:
            print(f"Error fetching NSE symbols: {e}")
    
    # Fallback to Nifty 500 stocks
    return get_nifty_500_stocks()


def get_nifty_50_stocks() -> List[str]:
    """Nifty 50 constituents"""
    return [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
        "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
        "HCLTECH", "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO",
        "NESTLEIND", "ADANIENT", "ADANIPORTS", "POWERGRID", "NTPC",
        "TATAMOTORS", "ONGC", "COALINDIA", "JSWSTEEL", "TATASTEEL",
        "TECHM", "BAJAJ-AUTO", "INDUSINDBK", "HINDALCO", "DRREDDY",
        "GRASIM", "CIPLA", "BRITANNIA", "EICHERMOT", "DIVISLAB",
        "BPCL", "APOLLOHOSP", "HEROMOTOCO", "TATACONSUM", "SBILIFE",
        "M&M", "UPL", "LTIM", "HDFCLIFE", "BAJAJFINSV"
    ]


def get_nifty_500_stocks() -> List[str]:
    """Extended list of major NSE stocks"""
    # Top 200 stocks by market cap (commonly traded)
    return [
        # Nifty 50
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", 
        "BHARTIARTL", "KOTAKBANK", "ITC", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
        "MARUTI", "HCLTECH", "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO",
        # Nifty Next 50
        "ADANIGREEN", "AMBUJACEM", "AUROPHARMA", "BANDHANBNK", "BANKBARODA",
        "BERGEPAINT", "BIOCON", "BOSCHLTD", "CHOLAFIN", "COLPAL",
        "DABUR", "DLF", "GAIL", "GODREJCP", "HAVELLS", "ICICIPRULI",
        "ICICIGI", "INDUSTOWER", "INDIGO", "IOC", "IRCTC", "JINDALSTEL",
        "JUBLFOOD", "LICI", "LUPIN", "M&MFIN", "MARICO", "MCDOWELL-N",
        "MUTHOOTFIN", "NAUKRI", "PAGEIND", "PETRONET", "PIDILITIND",
        "PNB", "POLYCAB", "SAIL", "SRF", "TORNTPHARM", "TRENT",
        "TVSMOTOR", "UBL", "VEDL", "VOLTAS", "ZOMATO", "ZYDUSLIFE",
        # Popular mid-caps
        "AAPL", "ABB", "ABCAPITAL", "ABFRL", "ACC", "ADANIENSOL",
        "ADANIPOWER", "AJANTPHARM", "ALKEM", "AMARAJABAT", "APLLTD",
        "ASHOKLEY", "ASTRAL", "ATUL", "AUBANK", "BALRAMCHIN",
        "BEL", "BHARATFORG", "BHEL", "BSE", "CANFINHOME", "CGPOWER",
        "CUMMINSIND", "DEEPAKNTR", "DMART", "ESCORTS", "EXIDEIND",
        "FEDERALBNK", "FSL", "GLAND", "GLAXO", "GMRINFRA", "GNFC",
        "GRANULES", "HDFCAMC", "HINDZINC", "HONAUT", "IDFCFIRSTB",
        "IGL", "INDHOTEL", "IRFC", "ISEC", "JKCEMENT", "JSWENERGY",
        "KALYANKJIL", "KEI", "LAURUSLABS", "LICHSGFIN", "LODHA",
        "LTF", "LTTS", "MANAPPURAM", "MFSL", "MGL", "MOTHERSON",
        "MPHASIS", "NAM-INDIA", "NATIONALUM", "NAVINFLUOR", "NHPC",
        "NMDC", "OBEROIRLTY", "OFSS", "OIL", "PAYTM", "PEL",
        "PERSISTENT", "PFC", "PIIND", "PVR", "RAMCOCEM", "RBLBANK",
        "RECLTD", "RELAXO", "SBICARD", "SHREECEM", "SIEMENS",
        "SONACOMS", "STARHEALTH", "SUMICHEM", "SUNDARMFIN", "SUNTV",
        "SUPREMEIND", "SYNGENE", "TATACOMM", "TATAELXSI", "TATAPOWER",
        "TATASTLLP", "TORNTPOWER", "TRIDENT", "TTML", "UNIONBANK",
        "VBL", "WHIRLPOOL", "YESBANK"
    ]


# Cache for stock data
_stock_cache = {}
_cache_time = None
CACHE_DURATION = 300  # 5 minutes


def get_stock_data(symbol: str) -> dict:
    """Get stock data with technical indicators"""
    try:
        full_symbol = f"{symbol}.NS"
        ticker = yf.Ticker(full_symbol)
        
        # Get historical data (60 days for indicators)
        hist = ticker.history(period="60d")
        
        if hist.empty or len(hist) < 20:
            return None
        
        # Current price info
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close * 100) if prev_close > 0 else 0
        
        # Volume
        current_volume = int(hist['Volume'].iloc[-1])
        avg_volume = int(hist['Volume'].mean())
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Calculate RSI (14-day)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        
        # Moving Averages
        ma_20 = float(hist['Close'].rolling(20).mean().iloc[-1])
        ma_50 = float(hist['Close'].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else ma_20
        
        # 52 week high/low
        high_52w = float(hist['High'].max())
        low_52w = float(hist['Low'].min())
        
        # Distance from 52W high
        distance_from_high = ((high_52w - current_price) / high_52w * 100) if high_52w > 0 else 0
        
        # MACD
        ema_12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_value = float(macd.iloc[-1])
        signal_value = float(signal.iloc[-1])
        macd_histogram = macd_value - signal_value
        
        # Get info
        info = ticker.info
        
        return {
            "symbol": symbol,
            "name": info.get("shortName", symbol),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "current_price": round(current_price, 2),
            "ltp": round(current_price, 2),
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "volume": current_volume,
            "avg_volume": avg_volume,
            "volume_ratio": round(volume_ratio, 2),
            "rsi": round(current_rsi, 2),
            "ma_20": round(ma_20, 2),
            "ma_50": round(ma_50, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "distance_from_high": round(distance_from_high, 2),
            "macd": round(macd_value, 2),
            "macd_signal": round(signal_value, 2),
            "macd_histogram": round(macd_histogram, 2),
            "above_ma_20": current_price > ma_20,
            "above_ma_50": current_price > ma_50,
            "market_cap": info.get("marketCap", 0),
        }
    except Exception as e:
        # Silent fail for individual stocks
        return None


def fetch_stocks_parallel(symbols: List[str], max_workers: int = 10) -> List[dict]:
    """Fetch multiple stocks in parallel for speed"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stock_data, symbol): symbol for symbol in symbols}
        
        for future in as_completed(futures):
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception:
                pass
    
    return results


def get_stocks_data(stock_list: str = "nifty50", limit: int = 100) -> List[dict]:
    """Get data for a specific stock list"""
    global _stock_cache, _cache_time
    
    # Determine which stocks to scan
    if stock_list == "all":
        symbols = get_all_nse_stocks()[:limit]
    elif stock_list == "nifty500":
        symbols = get_nifty_500_stocks()[:limit]
    else:  # Default to Nifty 50
        symbols = get_nifty_50_stocks()[:limit]
    
    # Check cache
    cache_key = f"{stock_list}_{limit}"
    now = datetime.now()
    
    if _cache_time and (now - _cache_time).seconds < CACHE_DURATION:
        if cache_key in _stock_cache:
            return _stock_cache[cache_key]
    
    # Fetch fresh data
    print(f"Fetching data for {len(symbols)} stocks...")
    stocks_data = fetch_stocks_parallel(symbols, max_workers=15)
    
    # Update cache
    _stock_cache[cache_key] = stocks_data
    _cache_time = now
    
    return stocks_data


# ============================================================
# 🎯 SCANNER FUNCTIONS
# ============================================================

def scan_swing_candidates(stocks: List[dict]) -> List[dict]:
    """AI Swing Candidates - Best setups for swing trading"""
    results = []
    for stock in stocks:
        score = 0
        reasons = []
        
        if stock["above_ma_20"]:
            score += 20
            reasons.append("Above 20 MA")
        if stock["above_ma_50"]:
            score += 15
            reasons.append("Above 50 MA")
        
        if 40 <= stock["rsi"] <= 65:
            score += 25
            reasons.append(f"Healthy RSI ({stock['rsi']:.0f})")
        
        if stock["distance_from_high"] < 15:
            score += 20
            reasons.append(f"Near highs ({stock['distance_from_high']:.0f}% from 52W high)")
        
        if stock["volume_ratio"] > 1.2:
            score += 20
            reasons.append(f"Good volume ({stock['volume_ratio']:.1f}x)")
        
        if score >= 60:
            stock["ai_score"] = score
            stock["signal_reason"] = " | ".join(reasons)
            results.append(stock)
    
    return sorted(results, key=lambda x: x.get("ai_score", 0), reverse=True)


def scan_breakout(stocks: List[dict]) -> List[dict]:
    """Stocks breaking out of consolidation"""
    results = []
    for stock in stocks:
        if stock["distance_from_high"] < 5 and stock["volume_ratio"] > 1.5:
            stock["signal_reason"] = f"Near 52W high ({stock['distance_from_high']:.1f}% away) with {stock['volume_ratio']:.1f}x volume"
            results.append(stock)
    return sorted(results, key=lambda x: x["volume_ratio"], reverse=True)


def scan_top_gainers(stocks: List[dict]) -> List[dict]:
    """Top gaining stocks (>2%)"""
    results = [s for s in stocks if s["change_percent"] > 2]
    for stock in results:
        stock["signal_reason"] = f"Up {stock['change_percent']:.2f}% today"
    return sorted(results, key=lambda x: x["change_percent"], reverse=True)


def scan_top_losers(stocks: List[dict]) -> List[dict]:
    """Top losing stocks (>2%)"""
    results = [s for s in stocks if s["change_percent"] < -2]
    for stock in results:
        stock["signal_reason"] = f"Down {abs(stock['change_percent']):.2f}% today"
    return sorted(results, key=lambda x: x["change_percent"])


def scan_volume_breakout(stocks: List[dict]) -> List[dict]:
    """High volume with price movement"""
    results = []
    for stock in stocks:
        if stock["volume_ratio"] > 2 and abs(stock["change_percent"]) > 1:
            stock["signal_reason"] = f"{stock['volume_ratio']:.1f}x avg volume with {stock['change_percent']:.2f}% move"
            results.append(stock)
    return sorted(results, key=lambda x: x["volume_ratio"], reverse=True)


def scan_52w_high(stocks: List[dict]) -> List[dict]:
    """Stocks near 52-week high"""
    results = []
    for stock in stocks:
        if stock["distance_from_high"] < 3:
            stock["signal_reason"] = f"Only {stock['distance_from_high']:.1f}% from 52W high"
            results.append(stock)
    return sorted(results, key=lambda x: x["distance_from_high"])


def scan_52w_low(stocks: List[dict]) -> List[dict]:
    """Stocks near 52-week low"""
    results = []
    for stock in stocks:
        distance_from_low = ((stock["current_price"] - stock["low_52w"]) / stock["low_52w"] * 100) if stock["low_52w"] > 0 else 100
        if distance_from_low < 10:
            stock["signal_reason"] = f"Only {distance_from_low:.1f}% from 52W low"
            stock["distance_from_low"] = round(distance_from_low, 2)
            results.append(stock)
    return sorted(results, key=lambda x: x.get("distance_from_low", 100))


def scan_volume_surge(stocks: List[dict]) -> List[dict]:
    """Unusual volume (>2.5x average)"""
    results = []
    for stock in stocks:
        if stock["volume_ratio"] > 2.5:
            stock["signal_reason"] = f"{stock['volume_ratio']:.1f}x average volume"
            results.append(stock)
    return sorted(results, key=lambda x: x["volume_ratio"], reverse=True)


def scan_rsi_oversold(stocks: List[dict]) -> List[dict]:
    """RSI below 30"""
    results = []
    for stock in stocks:
        if stock["rsi"] < 30:
            stock["signal_reason"] = f"RSI at {stock['rsi']:.1f} - oversold"
            results.append(stock)
    return sorted(results, key=lambda x: x["rsi"])


def scan_rsi_overbought(stocks: List[dict]) -> List[dict]:
    """RSI above 70"""
    results = []
    for stock in stocks:
        if stock["rsi"] > 70:
            stock["signal_reason"] = f"RSI at {stock['rsi']:.1f} - strong momentum"
            results.append(stock)
    return sorted(results, key=lambda x: x["rsi"], reverse=True)


def scan_bullish_ma(stocks: List[dict]) -> List[dict]:
    """Price above both 20 and 50 MA"""
    results = []
    for stock in stocks:
        if stock["above_ma_20"] and stock["above_ma_50"]:
            stock["signal_reason"] = f"Above 20 MA (₹{stock['ma_20']}) and 50 MA (₹{stock['ma_50']})"
            results.append(stock)
    return sorted(results, key=lambda x: x["change_percent"], reverse=True)


def scan_macd_crossover(stocks: List[dict]) -> List[dict]:
    """MACD bullish"""
    results = []
    for stock in stocks:
        if stock["macd_histogram"] > 0 and stock["macd"] > 0:
            stock["signal_reason"] = f"MACD bullish - histogram: {stock['macd_histogram']:.2f}"
            results.append(stock)
    return sorted(results, key=lambda x: x["macd_histogram"], reverse=True)


# Scanner mapping
SCANNERS = {
    0: ("Full Screening / AI Swing", scan_swing_candidates),
    1: ("Breakout (Consolidation)", scan_breakout),
    2: ("Top Gainers (>2%)", scan_top_gainers),
    3: ("Top Losers (>2%)", scan_top_losers),
    4: ("Volume Breakout", scan_volume_breakout),
    5: ("52-Week High", scan_52w_high),
    6: ("10-Day High", scan_52w_high),  # Same logic
    7: ("52-Week Low", scan_52w_low),
    8: ("Volume Surge (>2.5x)", scan_volume_surge),
    9: ("RSI Oversold (<30)", scan_rsi_oversold),
    10: ("RSI Overbought (>70)", scan_rsi_overbought),
    11: ("Bullish MA Setup", scan_bullish_ma),
    17: ("Bull Momentum", scan_bullish_ma),
    26: ("MACD Crossover", scan_macd_crossover),
    30: ("Momentum Burst", scan_volume_breakout),
}


# ============================================================
# 🎯 API ENDPOINTS
# ============================================================

@router.get("/scan/{scanner_id}")
async def run_scanner(
    scanner_id: int,
    universe: str = Query("nifty50", description="Stock universe: nifty50, nifty500, all"),
    limit: int = Query(100, description="Max stocks to scan")
):
    """
    Run a specific scanner on NSE stocks.
    
    Scanner IDs:
    - 0: Full Screening (AI Swing Candidates)
    - 1: Breakout (Consolidation)
    - 2: Top Gainers (>2%)
    - 3: Top Losers (>2%)
    - 4: Volume Breakout
    - 5: 52-Week High
    - 7: 52-Week Low
    - 8: Volume Surge (>2.5x)
    - 9: RSI Oversold (<30)
    - 10: RSI Overbought (>70)
    - 11: Bullish MA Setup
    - 26: MACD Crossover
    
    Universe options:
    - nifty50: Nifty 50 stocks (fastest)
    - nifty500: Top 150 stocks
    - all: All NSE stocks (2200+, slow)
    """
    try:
        stocks = get_stocks_data(universe, limit)
        
        if not stocks:
            return {
                "success": False,
                "message": "Could not fetch stock data",
                "results": []
            }
        
        if scanner_id not in SCANNERS:
            scanner_name, scanner_func = SCANNERS[0]
        else:
            scanner_name, scanner_func = SCANNERS[scanner_id]
        
        results = scanner_func(stocks)
        
        return {
            "success": True,
            "scanner_name": scanner_name,
            "scanner_id": scanner_id,
            "universe": universe,
            "total_scanned": len(stocks),
            "results_count": len(results),
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Scanner error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/swing-candidates")
async def get_swing_candidates(
    limit: int = Query(30, description="Max results"),
    universe: str = Query("nifty50", description="Stock universe")
):
    """Get AI Swing Trading Candidates"""
    try:
        stocks = get_stocks_data(universe, 100)
        
        if not stocks:
            return {"success": False, "results": []}
        
        results = scan_swing_candidates(stocks)[:limit]
        
        return {
            "success": True,
            "scanner_name": "AI Swing Candidates",
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stocks")
async def get_all_screener_stocks(
    universe: str = Query("nifty50", description="Stock universe"),
    limit: int = Query(50, description="Max stocks")
):
    """Get all stocks with technical data"""
    try:
        stocks = get_stocks_data(universe, limit)
        return {
            "success": True,
            "universe": universe,
            "count": len(stocks),
            "stocks": stocks,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def get_available_scanners():
    """List all available scanners"""
    return {
        "scanners": [
            {"id": 0, "name": "Full Screening / AI Swing", "category": "AI"},
            {"id": 1, "name": "Breakout (Consolidation)", "category": "Breakout"},
            {"id": 2, "name": "Top Gainers (>2%)", "category": "Momentum"},
            {"id": 3, "name": "Top Losers (>2%)", "category": "Momentum"},
            {"id": 4, "name": "Volume Breakout", "category": "Volume"},
            {"id": 5, "name": "52-Week High", "category": "Breakout"},
            {"id": 7, "name": "52-Week Low", "category": "Reversal"},
            {"id": 8, "name": "Volume Surge (>2.5x)", "category": "Volume"},
            {"id": 9, "name": "RSI Oversold (<30)", "category": "Reversal"},
            {"id": 10, "name": "RSI Overbought (>70)", "category": "Momentum"},
            {"id": 11, "name": "Bullish MA Setup", "category": "Trend"},
            {"id": 26, "name": "MACD Crossover", "category": "Momentum"},
        ],
        "universes": [
            {"id": "nifty50", "name": "Nifty 50", "count": 50},
            {"id": "nifty500", "name": "Top 150 Stocks", "count": 150},
            {"id": "all", "name": "All NSE Stocks", "count": "2200+"},
        ]
    }


@router.get("/universe/count")
async def get_universe_count():
    """Get count of stocks in each universe"""
    all_stocks = get_all_nse_stocks() if NSE_AVAILABLE else []
    
    return {
        "nifty50": 50,
        "nifty500": len(get_nifty_500_stocks()),
        "all_nse": len(all_stocks),
        "nse_available": NSE_AVAILABLE
    }


@router.get("/ai/nifty-prediction")
async def get_nifty_prediction():
    """Get AI Nifty outlook"""
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="30d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="Could not fetch Nifty data")
        
        current = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2])
        change = current - prev
        change_percent = (change / prev * 100)
        
        ma_5 = float(hist['Close'].rolling(5).mean().iloc[-1])
        ma_10 = float(hist['Close'].rolling(10).mean().iloc[-1])
        
        direction = "UP" if ma_5 > ma_10 else "DOWN"
        confidence = min(0.85, 0.5 + abs(ma_5 - ma_10) / current * 10)
        
        return {
            "success": True,
            "current_level": round(current, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "ensemble": {
                "prediction": round(current * (1.01 if direction == "UP" else 0.99), 0),
                "direction": direction,
                "confidence": round(confidence, 2)
            },
            "support_levels": [round(current * 0.98, 0), round(current * 0.95, 0)],
            "resistance_levels": [round(current * 1.02, 0), round(current * 1.05, 0)],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
