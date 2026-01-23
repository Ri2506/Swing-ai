"""
🔍 AI SCREENER API v3 - Full NSE Coverage + PKScreener AI
==========================================================
Complete stock screening with:
- All 2200+ NSE stocks via nsepython
- PKScreener's 40+ AI/ML features
- Real-time technical indicators
- Pattern recognition
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd
import numpy as np
import traceback

# Import nsepython for full NSE coverage
try:
    from nsepython import nse_eq_symbols, nse_get_index_list, nse_get_index_quote
    NSE_AVAILABLE = True
    ALL_NSE_STOCKS = nse_eq_symbols()
    print(f"✅ nsepython loaded - {len(ALL_NSE_STOCKS)} NSE stocks available")
except Exception as e:
    NSE_AVAILABLE = False
    ALL_NSE_STOCKS = []
    print(f"⚠️ nsepython error: {e}")

# Import PKScreener AI features
try:
    from pkscreener.classes.ScreeningStatistics import ScreeningStatistics
    from pkscreener.classes.StockScreener import StockScreener
    PKSCREENER_AVAILABLE = True
    print("✅ PKScreener AI features loaded")
except Exception as e:
    PKSCREENER_AVAILABLE = False
    print(f"⚠️ PKScreener not available: {e}")

router = APIRouter(prefix="/screener", tags=["Screener"])

# ============================================================
# 📊 STOCK UNIVERSE
# ============================================================

def get_all_nse_stocks() -> List[str]:
    """Get all 2200+ NSE stock symbols"""
    if NSE_AVAILABLE and ALL_NSE_STOCKS:
        return ALL_NSE_STOCKS
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
    """Top 200 actively traded NSE stocks"""
    return [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", 
        "BHARTIARTL", "KOTAKBANK", "ITC", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
        "MARUTI", "HCLTECH", "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO",
        "ADANIGREEN", "AMBUJACEM", "AUROPHARMA", "BANDHANBNK", "BANKBARODA",
        "BERGEPAINT", "BIOCON", "BOSCHLTD", "CHOLAFIN", "COLPAL",
        "DABUR", "DLF", "GAIL", "GODREJCP", "HAVELLS", "ICICIPRULI",
        "ICICIGI", "INDUSTOWER", "INDIGO", "IOC", "IRCTC", "JINDALSTEL",
        "JUBLFOOD", "LICI", "LUPIN", "M&MFIN", "MARICO", "MCDOWELL-N",
        "MUTHOOTFIN", "NAUKRI", "PAGEIND", "PETRONET", "PIDILITIND",
        "PNB", "POLYCAB", "SAIL", "SRF", "TORNTPHARM", "TRENT",
        "TVSMOTOR", "UBL", "VEDL", "VOLTAS", "ZOMATO", "ZYDUSLIFE",
        "ABB", "ABCAPITAL", "ABFRL", "ACC", "ADANIENSOL",
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
        "VBL", "WHIRLPOOL", "YESBANK", "NESTLEIND", "ADANIENT", 
        "ADANIPORTS", "POWERGRID", "NTPC", "TATAMOTORS", "ONGC", 
        "COALINDIA", "JSWSTEEL", "TATASTEEL", "TECHM", "BAJAJ-AUTO", 
        "INDUSINDBK", "HINDALCO", "DRREDDY", "GRASIM", "CIPLA", 
        "BRITANNIA", "EICHERMOT", "DIVISLAB", "BPCL", "APOLLOHOSP", 
        "HEROMOTOCO", "TATACONSUM", "SBILIFE", "M&M", "UPL", "LTIM", 
        "HDFCLIFE", "BAJAJFINSV"
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
        
        hist = ticker.history(period="60d")
        
        if hist.empty or len(hist) < 20:
            return None
        
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close * 100) if prev_close > 0 else 0
        
        current_volume = int(hist['Volume'].iloc[-1])
        avg_volume = int(hist['Volume'].mean())
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # RSI
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
        distance_from_high = ((high_52w - current_price) / high_52w * 100) if high_52w > 0 else 0
        
        # MACD
        ema_12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_value = float(macd.iloc[-1])
        signal_value = float(signal.iloc[-1])
        macd_histogram = macd_value - signal_value
        
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
    except Exception:
        return None


def fetch_stocks_parallel(symbols: List[str], max_workers: int = 15) -> List[dict]:
    """Fetch multiple stocks in parallel"""
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


def get_stocks_data(universe: str = "nifty50", limit: int = 100) -> List[dict]:
    """Get data for a specific stock universe"""
    global _stock_cache, _cache_time
    
    if universe == "all":
        symbols = get_all_nse_stocks()[:limit]
    elif universe == "nifty500":
        symbols = get_nifty_500_stocks()[:limit]
    else:
        symbols = get_nifty_50_stocks()[:limit]
    
    cache_key = f"{universe}_{limit}"
    now = datetime.now()
    
    if _cache_time and (now - _cache_time).seconds < CACHE_DURATION:
        if cache_key in _stock_cache:
            return _stock_cache[cache_key]
    
    print(f"Fetching data for {len(symbols)} stocks...")
    stocks_data = fetch_stocks_parallel(symbols)
    
    _stock_cache[cache_key] = stocks_data
    _cache_time = now
    
    return stocks_data


# ============================================================
# 🎯 BASIC SCANNER FUNCTIONS
# ============================================================

def scan_swing_candidates(stocks: List[dict]) -> List[dict]:
    """AI Swing Candidates"""
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
            reasons.append(f"Near highs ({stock['distance_from_high']:.0f}%)")
        if stock["volume_ratio"] > 1.2:
            score += 20
            reasons.append(f"Good volume ({stock['volume_ratio']:.1f}x)")
        
        if score >= 60:
            stock["ai_score"] = score
            stock["signal_reason"] = " | ".join(reasons)
            results.append(stock)
    
    return sorted(results, key=lambda x: x.get("ai_score", 0), reverse=True)


def scan_breakout(stocks: List[dict]) -> List[dict]:
    results = []
    for stock in stocks:
        if stock["distance_from_high"] < 5 and stock["volume_ratio"] > 1.5:
            stock["signal_reason"] = f"Near 52W high with {stock['volume_ratio']:.1f}x volume"
            results.append(stock)
    return sorted(results, key=lambda x: x["volume_ratio"], reverse=True)


def scan_top_gainers(stocks: List[dict]) -> List[dict]:
    results = [s for s in stocks if s["change_percent"] > 2]
    for stock in results:
        stock["signal_reason"] = f"Up {stock['change_percent']:.2f}% today"
    return sorted(results, key=lambda x: x["change_percent"], reverse=True)


def scan_top_losers(stocks: List[dict]) -> List[dict]:
    results = [s for s in stocks if s["change_percent"] < -2]
    for stock in results:
        stock["signal_reason"] = f"Down {abs(stock['change_percent']):.2f}% today"
    return sorted(results, key=lambda x: x["change_percent"])


def scan_volume_breakout(stocks: List[dict]) -> List[dict]:
    results = []
    for stock in stocks:
        if stock["volume_ratio"] > 2 and abs(stock["change_percent"]) > 1:
            stock["signal_reason"] = f"{stock['volume_ratio']:.1f}x volume with {stock['change_percent']:.2f}% move"
            results.append(stock)
    return sorted(results, key=lambda x: x["volume_ratio"], reverse=True)


def scan_52w_high(stocks: List[dict]) -> List[dict]:
    results = []
    for stock in stocks:
        if stock["distance_from_high"] < 3:
            stock["signal_reason"] = f"Only {stock['distance_from_high']:.1f}% from 52W high"
            results.append(stock)
    return sorted(results, key=lambda x: x["distance_from_high"])


def scan_52w_low(stocks: List[dict]) -> List[dict]:
    results = []
    for stock in stocks:
        distance_from_low = ((stock["current_price"] - stock["low_52w"]) / stock["low_52w"] * 100) if stock["low_52w"] > 0 else 100
        if distance_from_low < 10:
            stock["signal_reason"] = f"Only {distance_from_low:.1f}% from 52W low"
            stock["distance_from_low"] = round(distance_from_low, 2)
            results.append(stock)
    return sorted(results, key=lambda x: x.get("distance_from_low", 100))


def scan_rsi_oversold(stocks: List[dict]) -> List[dict]:
    results = [s for s in stocks if s["rsi"] < 30]
    for stock in results:
        stock["signal_reason"] = f"RSI at {stock['rsi']:.1f} - oversold"
    return sorted(results, key=lambda x: x["rsi"])


def scan_rsi_overbought(stocks: List[dict]) -> List[dict]:
    results = [s for s in stocks if s["rsi"] > 70]
    for stock in results:
        stock["signal_reason"] = f"RSI at {stock['rsi']:.1f} - strong momentum"
    return sorted(results, key=lambda x: x["rsi"], reverse=True)


def scan_macd_crossover(stocks: List[dict]) -> List[dict]:
    results = []
    for stock in stocks:
        if stock["macd_histogram"] > 0 and stock["macd"] > 0:
            stock["signal_reason"] = f"MACD bullish - histogram: {stock['macd_histogram']:.2f}"
            results.append(stock)
    return sorted(results, key=lambda x: x["macd_histogram"], reverse=True)


# Scanner mapping
SCANNERS = {
    0: ("AI Swing Candidates", scan_swing_candidates),
    1: ("Breakout", scan_breakout),
    2: ("Top Gainers (>2%)", scan_top_gainers),
    3: ("Top Losers (>2%)", scan_top_losers),
    4: ("Volume Breakout", scan_volume_breakout),
    5: ("52-Week High", scan_52w_high),
    7: ("52-Week Low", scan_52w_low),
    9: ("RSI Oversold (<30)", scan_rsi_oversold),
    10: ("RSI Overbought (>70)", scan_rsi_overbought),
    26: ("MACD Crossover", scan_macd_crossover),
}


# ============================================================
# 🤖 PKSCREENER AI FEATURES
# ============================================================

# List of PKScreener AI features to expose
PKSCREENER_AI_FEATURES = {
    "nifty_prediction": {
        "name": "AI Nifty Prediction",
        "description": "ML-based Nifty 50 direction prediction",
        "method": "getNiftyPrediction",
        "category": "prediction"
    },
    "buy_signals": {
        "name": "AI Buy Signals",
        "description": "Strong buy signals using ML algorithms",
        "method": "findStrongBuySignals",
        "category": "signals"
    },
    "sell_signals": {
        "name": "AI Sell Signals", 
        "description": "Strong sell signals for exit timing",
        "method": "findStrongSellSignals",
        "category": "signals"
    },
    "breakout_prediction": {
        "name": "Breakout Prediction",
        "description": "AI predicts potential breakouts",
        "method": "findPotentialBreakout",
        "category": "prediction"
    },
    "cup_handle": {
        "name": "Cup & Handle Pattern",
        "description": "AI detects Cup & Handle chart pattern",
        "method": "findCupAndHandlePattern",
        "category": "patterns"
    },
    "vcp_minervini": {
        "name": "VCP (Mark Minervini)",
        "description": "Volatility Contraction Pattern",
        "method": "validateVCPMarkMinervini",
        "category": "patterns"
    },
    "momentum": {
        "name": "High Momentum",
        "description": "Stocks with exceptional momentum",
        "method": "findHighMomentum",
        "category": "momentum"
    },
    "super_gainers": {
        "name": "Super Gainers/Losers",
        "description": "Extreme movers today",
        "method": "findSuperGainersLosers",
        "category": "momentum"
    },
    "trend_analysis": {
        "name": "Trend Analysis",
        "description": "AI determines current trend",
        "method": "findTrend",
        "category": "trend"
    },
    "uptrend": {
        "name": "Uptrend Detection",
        "description": "Stocks in confirmed uptrend",
        "method": "findUptrend",
        "category": "trend"
    },
    "rsi_cross_ma": {
        "name": "RSI Crossing MA",
        "description": "RSI crossing its moving average",
        "method": "findRSICrossingMA",
        "category": "indicators"
    },
    "macd_cross": {
        "name": "MACD Crossover",
        "description": "MACD bullish crossover signals",
        "method": "findMACDCrossover",
        "category": "indicators"
    },
    "bbands_squeeze": {
        "name": "Bollinger Squeeze",
        "description": "Low volatility squeeze setup",
        "method": "findBbandsSqueeze",
        "category": "volatility"
    },
    "atr_cross": {
        "name": "ATR Cross",
        "description": "ATR-based breakout signals",
        "method": "findATRCross",
        "category": "volatility"
    },
    "reversal_ma": {
        "name": "MA Reversal",
        "description": "Price reversing at moving averages",
        "method": "findReversalMA",
        "category": "reversal"
    },
    "inside_bar": {
        "name": "Inside Bar",
        "description": "Inside bar consolidation pattern",
        "method": "validateInsideBar",
        "category": "patterns"
    },
    "narrow_range": {
        "name": "Narrow Range (NR4/NR7)",
        "description": "Narrow range compression",
        "method": "validateNarrowRange",
        "category": "patterns"
    },
    "volume_spread": {
        "name": "Volume Spread Analysis",
        "description": "VSA pattern detection",
        "method": "validateVolumeSpreadAnalysis",
        "category": "volume"
    },
    "higher_highs": {
        "name": "Higher Highs & Higher Lows",
        "description": "Classic uptrend pattern",
        "method": "validateHigherHighsHigherLowsHigherClose",
        "category": "trend"
    },
    "lower_lows": {
        "name": "Lower Highs & Lower Lows",
        "description": "Downtrend pattern",
        "method": "validateLowerHighsLowerLows",
        "category": "trend"
    },
    "confluence": {
        "name": "Technical Confluence",
        "description": "Multiple indicators aligning",
        "method": "validateConfluence",
        "category": "confluence"
    },
    "lorentzian": {
        "name": "Lorentzian Classification",
        "description": "ML Lorentzian classifier signals",
        "method": "validateLorentzian",
        "category": "ml"
    },
    "short_term_bullish": {
        "name": "Short Term Bullish",
        "description": "Near-term bullish setup",
        "method": "validateShortTermBullish",
        "category": "signals"
    },
    "bullish_tomorrow": {
        "name": "Bullish For Tomorrow",
        "description": "Next day bullish probability",
        "method": "validateBullishForTomorrow",
        "category": "prediction"
    },
}


# ============================================================
# 🎯 API ENDPOINTS
# ============================================================

@router.get("/scan/{scanner_id}")
async def run_scanner(
    scanner_id: int,
    universe: str = Query("nifty50", description="nifty50, nifty500, all"),
    limit: int = Query(100, description="Max stocks to scan")
):
    """Run a basic scanner"""
    try:
        stocks = get_stocks_data(universe, limit)
        
        if not stocks:
            return {"success": False, "results": [], "message": "No data"}
        
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
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🚀 PKSCREENER FULL INTEGRATION - 40+ SCANNERS
# ============================================================

# Import the PKScreener service
try:
    from pkscreener_service import (
        SCANNER_CATEGORIES, ALL_SCANNERS, PKSCREENER_AVAILABLE as PK_SERVICE_AVAILABLE,
        scan_single_stock, scan_multiple_stocks, get_scanner_categories,
        get_all_scanners, get_scanner_info, get_nifty_prediction as pk_nifty_prediction
    )
    print("✅ PKScreener Service loaded with 40+ scanners")
except Exception as e:
    PK_SERVICE_AVAILABLE = False
    print(f"⚠️ PKScreener Service not loaded: {e}")


@router.get("/pk/categories")
async def get_pk_scanner_categories():
    """Get all PKScreener scanner categories and their scanners"""
    try:
        return {
            "success": True,
            "categories": SCANNER_CATEGORIES,
            "total_scanners": len(ALL_SCANNERS),
            "service_available": PK_SERVICE_AVAILABLE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/scanners")
async def get_pk_all_scanners():
    """Get flat list of all 40+ PKScreener scanners"""
    try:
        return {
            "success": True,
            "scanners": ALL_SCANNERS,
            "count": len(ALL_SCANNERS),
            "categories": list(SCANNER_CATEGORIES.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/scanner/{scanner_id}")
async def get_pk_scanner_details(scanner_id: str):
    """Get details about a specific scanner"""
    try:
        info = get_scanner_info(scanner_id)
        if not info:
            raise HTTPException(status_code=404, detail=f"Scanner '{scanner_id}' not found")
        
        return {
            "success": True,
            "scanner": info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pk/scan/single")
async def run_pk_single_stock_scan(
    symbol: str = Query(..., description="Stock symbol (e.g., RELIANCE)"),
    scanner_id: str = Query("trend", description="Scanner ID to run")
):
    """Run a PKScreener scanner on a single stock"""
    try:
        result = scan_single_stock(symbol, scanner_id)
        return {
            "success": True,
            "scanner_id": scanner_id,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pk/scan/batch")
async def run_pk_batch_scan(
    scanner_id: str = Query("trend", description="Scanner ID to run"),
    universe: str = Query("nifty50", description="Stock universe: nifty50, nifty500, all"),
    limit: int = Query(50, description="Max stocks to scan")
):
    """Run a PKScreener scanner on multiple stocks"""
    try:
        # Get stock list based on universe
        if universe == "all":
            symbols = get_all_nse_stocks()[:limit]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:limit]
        else:
            symbols = get_nifty_50_stocks()[:limit]
        
        # Run the scan
        results = scan_multiple_stocks(symbols, scanner_id, max_workers=10)
        
        # Sort by relevant score if available
        if results:
            for r in results:
                if 'ai_score' not in r:
                    r['ai_score'] = 50  # Default score
        
        results = sorted(results, key=lambda x: x.get('ai_score', 0), reverse=True)
        
        return {
            "success": True,
            "scanner_id": scanner_id,
            "scanner_info": get_scanner_info(scanner_id),
            "universe": universe,
            "total_scanned": len(symbols),
            "results_count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/strong-buy")
async def get_pk_strong_buy_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get stocks with strong buy signals using PKScreener AI"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "strong_buy", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener Strong Buy Signals",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/strong-sell")
async def get_pk_strong_sell_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get stocks with strong sell signals using PKScreener AI"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "strong_sell", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener Strong Sell Signals",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/breakouts")
async def get_pk_breakout_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get potential breakout stocks using PKScreener"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "probable_breakout", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener Breakout Detection",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/reversals")
async def get_pk_reversal_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get potential reversal stocks using PKScreener"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "buy_reversal", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener Reversal Detection",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/momentum")
async def get_pk_high_momentum_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get high momentum stocks using PKScreener"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "high_momentum", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener High Momentum",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/patterns/vcp")
async def get_pk_vcp_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get VCP (Volatility Contraction Pattern) stocks"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "vcp", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener VCP Pattern",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/patterns/inside-bar")
async def get_pk_inside_bar_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get Inside Bar pattern stocks"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "bullish_inside_bar", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener Inside Bar Pattern",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/bullish-tomorrow")
async def get_pk_bullish_tomorrow_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get stocks bullish for tomorrow using PKScreener"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "bullish_tomorrow", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener Bullish For Tomorrow",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/macd-crossover")
async def get_pk_macd_crossover_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get MACD crossover stocks using PKScreener"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "macd_crossover", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener MACD Crossover",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/rsi-oversold")
async def get_pk_rsi_oversold_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get RSI oversold stocks using PKScreener"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "rsi_oversold", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener RSI Oversold",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/consolidating")
async def get_pk_consolidating_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get consolidating stocks using PKScreener"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "consolidating", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener Consolidating Stocks",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/higher-highs")
async def get_pk_higher_highs_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get Higher Highs & Higher Lows stocks using PKScreener"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "higher_highs", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener Higher Highs/Lows",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pk/lorentzian")
async def get_pk_lorentzian_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """Get Lorentzian ML classifier signals using PKScreener"""
    try:
        if universe == "all":
            symbols = get_all_nse_stocks()[:100]
        elif universe == "nifty500":
            symbols = get_nifty_500_stocks()[:100]
        else:
            symbols = get_nifty_50_stocks()
        
        results = scan_multiple_stocks(symbols, "lorentzian_buy", max_workers=10)
        
        return {
            "success": True,
            "feature": "PKScreener Lorentzian ML Classifier",
            "universe": universe,
            "count": len(results),
            "results": results[:limit],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/swing-candidates")
async def get_swing_candidates(
    limit: int = Query(30),
    universe: str = Query("nifty50")
):
    """Get AI Swing Trading Candidates"""
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


@router.get("/stocks")
async def get_all_stocks(
    universe: str = Query("nifty50"),
    limit: int = Query(50)
):
    """Get all stocks with technical data"""
    stocks = get_stocks_data(universe, limit)
    return {
        "success": True,
        "universe": universe,
        "count": len(stocks),
        "stocks": stocks,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/universe/count")
async def get_universe_count():
    """Get count of stocks in each universe"""
    return {
        "nifty50": 50,
        "nifty500": len(get_nifty_500_stocks()),
        "all_nse": len(get_all_nse_stocks()),
        "nse_available": NSE_AVAILABLE,
        "pkscreener_available": PKSCREENER_AVAILABLE
    }


@router.get("/available")
async def get_available_scanners():
    """List all available scanners"""
    basic_scanners = [
        {"id": 0, "name": "AI Swing Candidates", "category": "AI"},
        {"id": 1, "name": "Breakout", "category": "Breakout"},
        {"id": 2, "name": "Top Gainers (>2%)", "category": "Momentum"},
        {"id": 3, "name": "Top Losers (>2%)", "category": "Momentum"},
        {"id": 4, "name": "Volume Breakout", "category": "Volume"},
        {"id": 5, "name": "52-Week High", "category": "Breakout"},
        {"id": 7, "name": "52-Week Low", "category": "Reversal"},
        {"id": 9, "name": "RSI Oversold (<30)", "category": "Reversal"},
        {"id": 10, "name": "RSI Overbought (>70)", "category": "Momentum"},
        {"id": 26, "name": "MACD Crossover", "category": "Momentum"},
    ]
    
    return {
        "basic_scanners": basic_scanners,
        "ai_features": list(PKSCREENER_AI_FEATURES.keys()),
        "universes": [
            {"id": "nifty50", "name": "Nifty 50", "count": 50},
            {"id": "nifty500", "name": "Top 200 Stocks", "count": len(get_nifty_500_stocks())},
            {"id": "all", "name": "All NSE Stocks", "count": len(get_all_nse_stocks())},
        ]
    }


# ============================================================
# 🤖 PKSCREENER AI ENDPOINTS
# ============================================================

@router.get("/ai/features")
async def get_ai_features():
    """Get list of all PKScreener AI features"""
    features_by_category = {}
    
    for key, feature in PKSCREENER_AI_FEATURES.items():
        category = feature["category"]
        if category not in features_by_category:
            features_by_category[category] = []
        
        features_by_category[category].append({
            "id": key,
            "name": feature["name"],
            "description": feature["description"]
        })
    
    return {
        "available": PKSCREENER_AVAILABLE,
        "total_features": len(PKSCREENER_AI_FEATURES),
        "features_by_category": features_by_category,
        "all_features": [
            {"id": k, **{kk: vv for kk, vv in v.items() if kk != "method"}}
            for k, v in PKSCREENER_AI_FEATURES.items()
        ]
    }


@router.get("/ai/nifty-prediction")
async def get_nifty_prediction():
    """AI-powered Nifty 50 prediction"""
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="30d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="Could not fetch Nifty data")
        
        current = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2])
        change = current - prev
        change_percent = (change / prev * 100)
        
        # Calculate indicators for prediction
        ma_5 = float(hist['Close'].rolling(5).mean().iloc[-1])
        ma_10 = float(hist['Close'].rolling(10).mean().iloc[-1])
        ma_20 = float(hist['Close'].rolling(20).mean().iloc[-1])
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])
        
        # Determine direction and confidence
        bullish_signals = 0
        total_signals = 4
        
        if ma_5 > ma_10:
            bullish_signals += 1
        if ma_10 > ma_20:
            bullish_signals += 1
        if current > ma_20:
            bullish_signals += 1
        if 40 < current_rsi < 70:
            bullish_signals += 1
        
        direction = "BULLISH" if bullish_signals >= 3 else "BEARISH" if bullish_signals <= 1 else "NEUTRAL"
        confidence = bullish_signals / total_signals
        
        # Predicted level
        if direction == "BULLISH":
            predicted = current * 1.01
        elif direction == "BEARISH":
            predicted = current * 0.99
        else:
            predicted = current
        
        return {
            "success": True,
            "current_level": round(current, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "prediction": {
                "direction": direction,
                "predicted_level": round(predicted, 0),
                "confidence": round(confidence * 100, 1),
                "signals": {
                    "bullish": bullish_signals,
                    "total": total_signals
                }
            },
            "indicators": {
                "rsi": round(current_rsi, 2),
                "ma_5": round(ma_5, 2),
                "ma_10": round(ma_10, 2),
                "ma_20": round(ma_20, 2)
            },
            "support_levels": [round(current * 0.98, 0), round(current * 0.95, 0)],
            "resistance_levels": [round(current * 1.02, 0), round(current * 1.05, 0)],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/momentum-radar")
async def get_momentum_radar(
    universe: str = Query("nifty50"),
    limit: int = Query(20)
):
    """AI Momentum Radar - Find stocks with exceptional momentum"""
    try:
        stocks = get_stocks_data(universe, 100)
        
        results = []
        for stock in stocks:
            momentum_score = 0
            signals = []
            
            # Price momentum
            if stock["change_percent"] > 3:
                momentum_score += 30
                signals.append(f"Strong +{stock['change_percent']:.1f}% today")
            elif stock["change_percent"] > 1:
                momentum_score += 15
                signals.append(f"Positive momentum +{stock['change_percent']:.1f}%")
            
            # RSI momentum
            if 60 < stock["rsi"] < 80:
                momentum_score += 25
                signals.append(f"Strong RSI ({stock['rsi']:.0f})")
            elif stock["rsi"] > 80:
                momentum_score += 15
                signals.append(f"Overbought RSI ({stock['rsi']:.0f})")
            
            # Volume confirmation
            if stock["volume_ratio"] > 2:
                momentum_score += 25
                signals.append(f"High volume ({stock['volume_ratio']:.1f}x)")
            elif stock["volume_ratio"] > 1.5:
                momentum_score += 15
                signals.append(f"Above avg volume ({stock['volume_ratio']:.1f}x)")
            
            # MACD
            if stock["macd_histogram"] > 0:
                momentum_score += 20
                signals.append("Bullish MACD")
            
            if momentum_score >= 50:
                stock["momentum_score"] = momentum_score
                stock["momentum_signals"] = signals
                stock["signal_reason"] = " | ".join(signals)
                results.append(stock)
        
        results = sorted(results, key=lambda x: x["momentum_score"], reverse=True)[:limit]
        
        return {
            "success": True,
            "feature": "AI Momentum Radar",
            "universe": universe,
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/breakout-scanner")
async def get_breakout_scanner(
    universe: str = Query("nifty50"),
    limit: int = Query(20)
):
    """AI Breakout Scanner - Predict potential breakouts"""
    try:
        stocks = get_stocks_data(universe, 100)
        
        results = []
        for stock in stocks:
            breakout_score = 0
            signals = []
            
            # Near 52W high
            if stock["distance_from_high"] < 3:
                breakout_score += 35
                signals.append(f"At 52W high ({stock['distance_from_high']:.1f}% away)")
            elif stock["distance_from_high"] < 10:
                breakout_score += 20
                signals.append(f"Near 52W high ({stock['distance_from_high']:.1f}% away)")
            
            # Volume surge
            if stock["volume_ratio"] > 2.5:
                breakout_score += 30
                signals.append(f"Volume surge ({stock['volume_ratio']:.1f}x)")
            elif stock["volume_ratio"] > 1.5:
                breakout_score += 15
                signals.append(f"Above avg volume ({stock['volume_ratio']:.1f}x)")
            
            # Above MAs
            if stock["above_ma_20"] and stock["above_ma_50"]:
                breakout_score += 25
                signals.append("Above all MAs")
            
            # Positive price action
            if stock["change_percent"] > 2:
                breakout_score += 10
                signals.append(f"Strong +{stock['change_percent']:.1f}% move")
            
            if breakout_score >= 50:
                stock["breakout_score"] = breakout_score
                stock["breakout_probability"] = min(95, breakout_score + 20)
                stock["signal_reason"] = " | ".join(signals)
                results.append(stock)
        
        results = sorted(results, key=lambda x: x["breakout_score"], reverse=True)[:limit]
        
        return {
            "success": True,
            "feature": "AI Breakout Scanner",
            "universe": universe,
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/reversal-scanner")
async def get_reversal_scanner(
    universe: str = Query("nifty50"),
    limit: int = Query(20)
):
    """AI Reversal Scanner - Find potential reversal candidates"""
    try:
        stocks = get_stocks_data(universe, 100)
        
        results = []
        for stock in stocks:
            reversal_score = 0
            signals = []
            
            # Oversold RSI
            if stock["rsi"] < 30:
                reversal_score += 35
                signals.append(f"Oversold RSI ({stock['rsi']:.0f})")
            elif stock["rsi"] < 40:
                reversal_score += 20
                signals.append(f"Low RSI ({stock['rsi']:.0f})")
            
            # Near 52W low
            distance_from_low = ((stock["current_price"] - stock["low_52w"]) / stock["low_52w"] * 100) if stock["low_52w"] > 0 else 100
            if distance_from_low < 5:
                reversal_score += 30
                signals.append(f"Near 52W low ({distance_from_low:.1f}% away)")
            elif distance_from_low < 15:
                reversal_score += 15
                signals.append(f"Close to 52W low ({distance_from_low:.1f}% away)")
            
            # Volume spike (potential accumulation)
            if stock["volume_ratio"] > 2:
                reversal_score += 25
                signals.append(f"High volume ({stock['volume_ratio']:.1f}x) - accumulation?")
            
            # Big drop (potential bounce)
            if stock["change_percent"] < -3:
                reversal_score += 10
                signals.append(f"Sharp decline ({stock['change_percent']:.1f}%)")
            
            if reversal_score >= 40:
                stock["reversal_score"] = reversal_score
                stock["reversal_probability"] = min(85, reversal_score + 10)
                stock["signal_reason"] = " | ".join(signals)
                results.append(stock)
        
        results = sorted(results, key=lambda x: x["reversal_score"], reverse=True)[:limit]
        
        return {
            "success": True,
            "feature": "AI Reversal Scanner",
            "universe": universe,
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/trend-analysis")
async def get_trend_analysis(
    universe: str = Query("nifty50"),
    limit: int = Query(30)
):
    """AI Trend Analysis - Categorize stocks by trend"""
    try:
        stocks = get_stocks_data(universe, 100)
        
        uptrend = []
        downtrend = []
        sideways = []
        
        for stock in stocks:
            trend_score = 0
            
            if stock["above_ma_20"]:
                trend_score += 1
            if stock["above_ma_50"]:
                trend_score += 1
            if stock["change_percent"] > 0:
                trend_score += 1
            if stock["macd_histogram"] > 0:
                trend_score += 1
            if stock["rsi"] > 50:
                trend_score += 1
            
            if trend_score >= 4:
                stock["trend"] = "UPTREND"
                stock["trend_strength"] = "Strong" if trend_score == 5 else "Moderate"
                stock["signal_reason"] = f"Uptrend - {trend_score}/5 bullish signals"
                uptrend.append(stock)
            elif trend_score <= 1:
                stock["trend"] = "DOWNTREND"
                stock["trend_strength"] = "Strong" if trend_score == 0 else "Moderate"
                stock["signal_reason"] = f"Downtrend - {5 - trend_score}/5 bearish signals"
                downtrend.append(stock)
            else:
                stock["trend"] = "SIDEWAYS"
                stock["trend_strength"] = "Neutral"
                stock["signal_reason"] = "Sideways - mixed signals"
                sideways.append(stock)
        
        return {
            "success": True,
            "feature": "AI Trend Analysis",
            "universe": universe,
            "summary": {
                "uptrend": len(uptrend),
                "downtrend": len(downtrend),
                "sideways": len(sideways),
                "total": len(stocks)
            },
            "uptrend": uptrend[:limit],
            "downtrend": downtrend[:limit],
            "sideways": sideways[:limit//2],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/market-regime")
async def get_market_regime():
    """AI Market Regime Analysis"""
    try:
        # Get Nifty data
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="60d")
        
        current = float(hist['Close'].iloc[-1])
        ma_20 = float(hist['Close'].rolling(20).mean().iloc[-1])
        ma_50 = float(hist['Close'].rolling(50).mean().iloc[-1])
        
        # Volatility (ATR proxy)
        high_low = hist['High'] - hist['Low']
        volatility = float(high_low.rolling(14).mean().iloc[-1])
        avg_volatility = float(high_low.mean())
        volatility_ratio = volatility / avg_volatility
        
        # Determine regime
        if current > ma_20 > ma_50:
            if volatility_ratio < 0.8:
                regime = "BULL_QUIET"
                description = "Bullish with low volatility - ideal for trend following"
            else:
                regime = "BULL_VOLATILE"
                description = "Bullish but volatile - use wider stops"
        elif current < ma_20 < ma_50:
            if volatility_ratio < 0.8:
                regime = "BEAR_QUIET"
                description = "Bearish with low volatility - avoid longs"
            else:
                regime = "BEAR_VOLATILE"
                description = "Bearish and volatile - stay cautious"
        else:
            regime = "TRANSITIONAL"
            description = "Market in transition - wait for clarity"
        
        return {
            "success": True,
            "regime": regime,
            "description": description,
            "indicators": {
                "nifty": round(current, 2),
                "ma_20": round(ma_20, 2),
                "ma_50": round(ma_50, 2),
                "volatility_ratio": round(volatility_ratio, 2)
            },
            "recommendation": {
                "BULL_QUIET": "Go long on breakouts with tight stops",
                "BULL_VOLATILE": "Trade smaller size, wider stops",
                "BEAR_QUIET": "Avoid new longs, protect existing",
                "BEAR_VOLATILE": "Cash is a position - stay safe",
                "TRANSITIONAL": "Wait for regime clarity"
            }.get(regime, "Monitor closely"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
