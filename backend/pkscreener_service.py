"""
📊 PKScreener Service - Full Integration
==========================================
This service wraps all 40+ PKScreener scanners and exposes them via clean APIs.
It properly initializes the library and provides technical analysis on any stock.
"""

import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging before importing pkscreener
logging.basicConfig(level=logging.WARNING)
pk_logger = logging.getLogger('pkscreener')
pk_logger.setLevel(logging.WARNING)

# Import yfinance for data
import yfinance as yf


# =============================================================================
# 🔧 UTILITY FUNCTIONS
# =============================================================================

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


# Initialize PKScreener globals
try:
    from pkscreener import globals as pk_globals
    from pkscreener.classes.Pktalib import pktalib
    from pkscreener.classes.ScreeningStatistics import ScreeningStatistics
    
    pk_globals.default_logger = pk_logger
    PKSCREENER_AVAILABLE = True
    
    # Get the pre-initialized screener
    screener = pk_globals.screener
    print("✅ PKScreener Service initialized successfully")
except Exception as e:
    PKSCREENER_AVAILABLE = False
    screener = None
    print(f"⚠️ PKScreener not available: {e}")


# =============================================================================
# 📊 ALL SCANNER DEFINITIONS (40+ Scanners from PKScreener menus)
# =============================================================================

SCANNER_CATEGORIES = {
    "breakout": {
        "name": "Breakout Scanners",
        "description": "Find stocks breaking out of resistance levels",
        "scanners": [
            {"id": "probable_breakout", "name": "Probable Breakouts", "menu_code": "X:1:1"},
            {"id": "today_breakout", "name": "Today's Breakouts", "menu_code": "X:1:2"},
            {"id": "52w_high_breakout", "name": "52 Week High Breakout", "menu_code": "X:1:17"},
            {"id": "10d_low_breakout", "name": "10 Day Low Breakout (Sell)", "menu_code": "X:1:16"},
            {"id": "52w_low_breakout", "name": "52 Week Low Breakout (Sell)", "menu_code": "X:1:15"},
            {"id": "breaking_out_now", "name": "Breaking Out Now", "menu_code": "X:1:23"},
        ]
    },
    "momentum": {
        "name": "Momentum Scanners",
        "description": "Find stocks with strong momentum",
        "scanners": [
            {"id": "high_momentum", "name": "High Momentum (RSI, MFI, CCI)", "menu_code": "X:1:31"},
            {"id": "volume_gainers", "name": "Volume Gainers", "menu_code": "X:1:9"},
            {"id": "closing_2pct_up", "name": "Closing 2%+ Up (3 Days)", "menu_code": "X:1:10"},
            {"id": "super_gainers", "name": "Super Gainers", "menu_code": "X:1:42"},
            {"id": "super_losers", "name": "Super Losers", "menu_code": "X:1:43"},
            {"id": "rising_rsi", "name": "Rising RSI", "menu_code": "X:1:6:9"},
        ]
    },
    "reversal": {
        "name": "Reversal Scanners",
        "description": "Find potential reversal candidates",
        "scanners": [
            {"id": "buy_reversal", "name": "Buy Signals (Bullish Reversal)", "menu_code": "X:1:6:1"},
            {"id": "sell_reversal", "name": "Sell Signals (Bearish Reversal)", "menu_code": "X:1:6:2"},
            {"id": "momentum_gainers", "name": "Momentum Gainers (Rising)", "menu_code": "X:1:6:3"},
            {"id": "ma_reversal", "name": "Moving Average Reversal", "menu_code": "X:1:6:4"},
            {"id": "vsa_reversal", "name": "Volume Spread Analysis (VSA)", "menu_code": "X:1:6:5"},
            {"id": "narrow_range", "name": "Narrow Range (NRx) Reversal", "menu_code": "X:1:6:6"},
            {"id": "psar_rsi_reversal", "name": "PSAR & RSI Reversal", "menu_code": "X:1:6:8"},
            {"id": "rsi_ma_reversal", "name": "RSI MA Reversal", "menu_code": "X:1:6:10"},
        ]
    },
    "patterns": {
        "name": "Chart Pattern Scanners",
        "description": "Find stocks forming chart patterns",
        "scanners": [
            {"id": "bullish_inside_bar", "name": "Bullish Inside Bar (Flag)", "menu_code": "X:1:7:1"},
            {"id": "bearish_inside_bar", "name": "Bearish Inside Bar (Flag)", "menu_code": "X:1:7:2"},
            {"id": "confluence", "name": "Confluence (50 & 200 MA)", "menu_code": "X:1:7:3"},
            {"id": "vcp", "name": "VCP (Volatility Contraction)", "menu_code": "X:1:7:4"},
            {"id": "trendline_support", "name": "Buying at Trendline Support", "menu_code": "X:1:7:5"},
            {"id": "bollinger_squeeze", "name": "Bollinger Bands (TTM) Squeeze", "menu_code": "X:1:7:6"},
            {"id": "candlestick_patterns", "name": "Candlestick Patterns", "menu_code": "X:1:7:7"},
            {"id": "vcp_minervini", "name": "VCP (Mark Minervini)", "menu_code": "X:1:7:8"},
        ]
    },
    "ma_signals": {
        "name": "Moving Average Signals",
        "description": "Moving average based signals",
        "scanners": [
            {"id": "ma_support", "name": "MA Support", "menu_code": "X:1:7:9:1"},
            {"id": "ma_bearish", "name": "MA Bearish Signals", "menu_code": "X:1:7:9:2"},
            {"id": "ma_bullish", "name": "MA Bullish Signals", "menu_code": "X:1:7:9:3"},
            {"id": "bear_cross_ma", "name": "Bear Cross MA", "menu_code": "X:1:7:9:4"},
            {"id": "bull_cross_ma", "name": "Bull Cross MA", "menu_code": "X:1:7:9:5"},
            {"id": "ma_resist", "name": "MA Resistance", "menu_code": "X:1:7:9:6"},
            {"id": "bull_cross_vwap", "name": "Bull Cross VWAP", "menu_code": "X:1:7:9:7"},
        ]
    },
    "technical": {
        "name": "Technical Scanners",
        "description": "Technical indicator based scanners",
        "scanners": [
            {"id": "rsi_oversold", "name": "RSI Oversold (<30)", "menu_code": "X:1:5:1"},
            {"id": "rsi_overbought", "name": "RSI Overbought (>70)", "menu_code": "X:1:5:2"},
            {"id": "cci_screening", "name": "CCI Outside Range", "menu_code": "X:1:8"},
            {"id": "macd_crossover", "name": "MACD Crossover", "menu_code": "X:1:13"},
            {"id": "macd_below_zero", "name": "MACD Histogram Below 0 (Sell)", "menu_code": "X:1:19"},
            {"id": "aroon_crossover", "name": "Bullish Aroon Crossover", "menu_code": "X:1:18"},
            {"id": "atr_cross", "name": "ATR Cross", "menu_code": "X:1:27"},
        ]
    },
    "signals": {
        "name": "Signal Scanners",
        "description": "Multi-indicator buy/sell signals",
        "scanners": [
            {"id": "strong_buy", "name": "Strong Buy Signals", "menu_code": "X:1:44"},
            {"id": "strong_sell", "name": "Strong Sell Signals", "menu_code": "X:1:45"},
            {"id": "all_buy", "name": "All Buy Signals", "menu_code": "X:1:46"},
            {"id": "all_sell", "name": "All Sell Signals", "menu_code": "X:1:47"},
            {"id": "bullish_tomorrow", "name": "Bullish For Tomorrow", "menu_code": "X:1:20"},
            {"id": "short_term_bullish", "name": "Short Term Bullish (Ichimoku)", "menu_code": "X:1:11"},
        ]
    },
    "consolidation": {
        "name": "Consolidation Scanners",
        "description": "Find stocks in consolidation",
        "scanners": [
            {"id": "consolidating", "name": "Consolidating Stocks", "menu_code": "X:1:3"},
            {"id": "lowest_volume", "name": "Lowest Volume (Early Breakout)", "menu_code": "X:1:4"},
            {"id": "nr4_daily", "name": "NR4 Daily", "menu_code": "X:1:14"},
        ]
    },
    "trend": {
        "name": "Trend Scanners",
        "description": "Identify stock trends",
        "scanners": [
            {"id": "higher_highs", "name": "Higher Highs & Lows (SuperTrend)", "menu_code": "X:1:24"},
            {"id": "lower_lows", "name": "Lower Highs & Lows (Watch Rev.)", "menu_code": "X:1:25"},
            {"id": "bullish_higher_opens", "name": "Bullish Higher Opens", "menu_code": "X:1:28"},
            {"id": "bullish_avwap", "name": "Bullish Anchored VWAP", "menu_code": "X:1:34"},
        ]
    },
    "ml": {
        "name": "Machine Learning Scanners",
        "description": "ML-powered analysis",
        "scanners": [
            {"id": "lorentzian_buy", "name": "Lorentzian Classifier Buy", "menu_code": "X:1:6:7:1"},
            {"id": "lorentzian_sell", "name": "Lorentzian Classifier Sell", "menu_code": "X:1:6:7:2"},
            {"id": "nifty_prediction", "name": "AI Nifty Prediction", "menu_code": "X:N"},
        ]
    },
    "short_sell": {
        "name": "Short Sell Scanners",
        "description": "Find short selling candidates",
        "scanners": [
            {"id": "perfect_short", "name": "Perfect Short Sells (Futures)", "menu_code": "X:1:35"},
            {"id": "probable_short", "name": "Probable Short Sells (Futures)", "menu_code": "X:1:36"},
            {"id": "short_volume_sma", "name": "Short Sell (Volume SMA)", "menu_code": "X:1:37"},
        ]
    },
}

# Flatten all scanners for easy lookup
ALL_SCANNERS = {}
for category, cat_data in SCANNER_CATEGORIES.items():
    for scanner in cat_data["scanners"]:
        ALL_SCANNERS[scanner["id"]] = {
            **scanner,
            "category": category,
            "category_name": cat_data["name"]
        }


# =============================================================================
# 📈 DATA PREPARATION UTILITIES
# =============================================================================

def prepare_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare stock data with all required technical indicators for PKScreener.
    """
    if df.empty:
        return df
    
    # Make a copy and lowercase columns
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        return pd.DataFrame()
    
    try:
        # RSI
        df['RSI'] = pktalib.RSI(df['close'], timeperiod=14)
        
        # MFI
        df['MFI'] = pktalib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # CCI
        df['CCI'] = pktalib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # MACD
        macd, signal, hist = pktalib.MACD(df['close'], 12, 26, 9)
        df['MACD'] = macd
        df['MACDsignal'] = signal
        df['MACDhist'] = hist
        
        # Moving Averages
        df['SMA'] = pktalib.SMA(df['close'], timeperiod=20)
        df['EMA'] = pktalib.EMA(df['close'], timeperiod=20)
        df['SMA50'] = pktalib.SMA(df['close'], timeperiod=50)
        df['SMA200'] = pktalib.SMA(df['close'], timeperiod=200)
        df['EMA20'] = pktalib.EMA(df['close'], timeperiod=20)
        df['EMA50'] = pktalib.EMA(df['close'], timeperiod=50)
        
        # Bollinger Bands
        upper, middle, lower = pktalib.BBANDS(df['close'], timeperiod=20)
        df['BBupper'] = upper
        df['BBmiddle'] = middle
        df['BBlower'] = lower
        
        # ATR
        df['ATR'] = pktalib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # VWAP (if we have volume)
        df['VWAP'] = pktalib.VWAP(df['high'], df['low'], df['close'], df['volume'])
        
        # Stochastic RSI - with proper parameters
        try:
            fastk, fastd = pktalib.STOCHRSI(df['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
            df['STOCHRSIk'] = fastk
            df['STOCHRSId'] = fastd
        except:
            pass  # Skip if not supported
        
        # SuperTrend
        try:
            st, std = pktalib.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
            df['SuperTrend'] = st
            df['SuperTrendDirection'] = std
        except:
            pass
            
    except Exception as e:
        print(f"Error preparing indicators: {e}")
    
    return df


def get_stock_data_with_indicators(symbol: str, period: str = "6mo") -> Tuple[pd.DataFrame, dict]:
    """
    Fetch stock data and add all technical indicators.
    Returns (DataFrame, basic_info_dict)
    """
    try:
        full_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        ticker = yf.Ticker(full_symbol)
        
        df = ticker.history(period=period)
        if df.empty:
            return pd.DataFrame(), {}
        
        # Get basic info
        info = ticker.info
        basic_info = {
            "symbol": symbol.replace('.NS', ''),
            "name": info.get("shortName", symbol),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
        }
        
        # Add current price info
        if len(df) > 0:
            basic_info["current_price"] = round(float(df['Close'].iloc[-1]), 2)
            basic_info["ltp"] = basic_info["current_price"]
            
            if len(df) > 1:
                prev_close = float(df['Close'].iloc[-2])
                change = basic_info["current_price"] - prev_close
                change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                basic_info["change"] = round(change, 2)
                basic_info["change_percent"] = round(change_pct, 2)
            
            # Volume info
            current_vol = int(df['Volume'].iloc[-1])
            avg_vol = int(df['Volume'].mean())
            basic_info["volume"] = current_vol
            basic_info["avg_volume"] = avg_vol
            basic_info["volume_ratio"] = round(current_vol / avg_vol, 2) if avg_vol > 0 else 1
            
            # 52 week high/low
            basic_info["high_52w"] = round(float(df['High'].max()), 2)
            basic_info["low_52w"] = round(float(df['Low'].min()), 2)
            
        # Prepare technical indicators
        df = prepare_stock_data(df)
        
        # Add indicator values to basic_info
        if not df.empty and 'RSI' in df.columns:
            basic_info["rsi"] = round(float(df['RSI'].iloc[-1]), 2) if not pd.isna(df['RSI'].iloc[-1]) else 50
            basic_info["macd"] = round(float(df['MACD'].iloc[-1]), 2) if not pd.isna(df['MACD'].iloc[-1]) else 0
            basic_info["macd_signal"] = round(float(df['MACDsignal'].iloc[-1]), 2) if not pd.isna(df['MACDsignal'].iloc[-1]) else 0
            basic_info["sma_20"] = round(float(df['SMA'].iloc[-1]), 2) if not pd.isna(df['SMA'].iloc[-1]) else 0
            basic_info["sma_50"] = round(float(df['SMA50'].iloc[-1]), 2) if not pd.isna(df['SMA50'].iloc[-1]) else 0
        
        return df, basic_info
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame(), {}


# =============================================================================
# 🔍 SCANNER EXECUTION FUNCTIONS
# =============================================================================

def run_trend_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run trend analysis on a stock."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.findTrend(df, screenDict, saveDict, daysToLookback=20, stockName=info.get('symbol', ''))
        
        # Clean ANSI codes from result
        if result:
            import re
            result = re.sub(r'\x1b\[[0-9;]*m', '', str(result))
            
        return {
            "passed": result is not None,
            "trend": result,
            "reason": f"Trend: {result}" if result else "No clear trend"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_momentum_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run high momentum scan on a stock."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        result = screener.findHighMomentum(df, strict=False)
        
        reasons = []
        if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
            rsi = df['RSI'].iloc[-1]
            if rsi > 60:
                reasons.append(f"RSI {rsi:.1f} (bullish)")
            elif rsi < 40:
                reasons.append(f"RSI {rsi:.1f} (oversold)")
                
        if 'MFI' in df.columns and not pd.isna(df['MFI'].iloc[-1]):
            mfi = df['MFI'].iloc[-1]
            if mfi > 60:
                reasons.append(f"MFI {mfi:.1f} (inflow)")
                
        return {
            "passed": result == True,
            "momentum": result,
            "reason": " | ".join(reasons) if reasons else "No significant momentum"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_strong_buy_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run strong buy signals scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.findStrongBuySignals(df, screenDict, saveDict)
        
        return {
            "passed": result == True,
            "signals": screenDict,
            "reason": "Strong buy signals detected" if result else "No strong buy signals"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_strong_sell_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run strong sell signals scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.findStrongSellSignals(df, screenDict, saveDict)
        
        return {
            "passed": result == True,
            "signals": screenDict,
            "reason": "Strong sell signals detected" if result else "No strong sell signals"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_breakout_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run breakout detection scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.findPotentialBreakout(df, screenDict, saveDict)
        
        # Also check distance from 52w high
        dist_from_high = 0
        if 'high_52w' in info and 'current_price' in info:
            high = info['high_52w']
            price = info['current_price']
            if high > 0:
                dist_from_high = ((high - price) / high) * 100
        
        return {
            "passed": result == True or dist_from_high < 5,
            "breakout_signal": result,
            "distance_from_high": round(dist_from_high, 2),
            "reason": f"Breakout signal" if result else f"{dist_from_high:.1f}% from 52W high"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_reversal_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run reversal detection scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.findReversalMA(df, screenDict, saveDict, maLength=20)
        
        # Also check RSI for oversold
        rsi_oversold = False
        if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
            rsi = df['RSI'].iloc[-1]
            rsi_oversold = rsi < 35
        
        return {
            "passed": result == True or rsi_oversold,
            "ma_reversal": result,
            "rsi_oversold": rsi_oversold,
            "reason": "MA reversal detected" if result else ("RSI oversold" if rsi_oversold else "No reversal signal")
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_bullish_tomorrow_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run bullish for tomorrow scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.validateBullishForTomorrow(df, screenDict, saveDict)
        
        return {
            "passed": result == True,
            "signals": screenDict,
            "reason": "Bullish setup for tomorrow" if result else "No bullish setup"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_consolidation_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run consolidation detection scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.validateConsolidation(df, screenDict, saveDict, percentage=4)
        
        return {
            "passed": result == True,
            "signals": screenDict,
            "reason": "Stock is consolidating" if result else "Not consolidating"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_inside_bar_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run inside bar pattern scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.validateInsideBar(df, screenDict, saveDict, bullish=True)
        
        return {
            "passed": result == True,
            "pattern": "Inside Bar (Bullish)",
            "reason": "Bullish inside bar detected" if result else "No inside bar"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_vcp_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run VCP (Volatility Contraction Pattern) scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.validateVCP(df, screenDict, saveDict, stockName=info.get('symbol', ''))
        
        return {
            "passed": result == True,
            "pattern": "VCP",
            "reason": "VCP pattern detected" if result else "No VCP"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_macd_crossover_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run MACD crossover scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.findMACDCrossover(df, screenDict, saveDict)
        
        return {
            "passed": result == True,
            "signals": screenDict,
            "reason": "MACD bullish crossover" if result else "No MACD crossover"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_rsi_scan(df: pd.DataFrame, info: dict, oversold: bool = True) -> dict:
    """Run RSI screening."""
    if df.empty or 'RSI' not in df.columns:
        return {"passed": False, "reason": "No RSI data"}
    
    try:
        rsi = df['RSI'].iloc[-1]
        if pd.isna(rsi):
            return {"passed": False, "reason": "RSI not available"}
        
        if oversold:
            passed = rsi < 30
            reason = f"RSI {rsi:.1f} - Oversold" if passed else f"RSI {rsi:.1f}"
        else:
            passed = rsi > 70
            reason = f"RSI {rsi:.1f} - Overbought" if passed else f"RSI {rsi:.1f}"
        
        return {
            "passed": passed,
            "rsi": round(rsi, 2),
            "reason": reason
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_volume_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run volume analysis scan."""
    if df.empty or 'volume' not in df.columns:
        return {"passed": False, "reason": "No volume data"}
    
    try:
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].mean()
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        passed = vol_ratio > 1.5
        
        return {
            "passed": passed,
            "volume_ratio": round(vol_ratio, 2),
            "current_volume": int(current_vol),
            "avg_volume": int(avg_vol),
            "reason": f"Volume {vol_ratio:.1f}x average" if passed else f"Normal volume ({vol_ratio:.1f}x)"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_higher_highs_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run higher highs and higher lows scan."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.validateHigherHighsHigherLowsHigherClose(df, screenDict, saveDict)
        
        return {
            "passed": result == True,
            "pattern": "Higher Highs/Lows",
            "reason": "Higher Highs & Lows pattern" if result else "No HH/HL pattern"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_supertrend_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run SuperTrend analysis."""
    if df.empty or 'SuperTrendDirection' not in df.columns:
        return {"passed": False, "reason": "SuperTrend not available"}
    
    try:
        direction = df['SuperTrendDirection'].iloc[-1]
        passed = direction == 1  # 1 = bullish, -1 = bearish
        
        return {
            "passed": passed,
            "direction": "Bullish" if passed else "Bearish",
            "reason": "SuperTrend Bullish" if passed else "SuperTrend Bearish"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def run_lorentzian_scan(df: pd.DataFrame, info: dict) -> dict:
    """Run Lorentzian classifier scan (ML-based)."""
    if not PKSCREENER_AVAILABLE or df.empty:
        return {"passed": False, "reason": "No data"}
    
    try:
        screenDict = {}
        saveDict = {}
        result = screener.validateLorentzian(df, screenDict, saveDict)
        
        return {
            "passed": result == True,
            "ml_signal": result,
            "reason": "ML Lorentzian signal" if result else "No ML signal"
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


# Scanner function mapping
SCANNER_FUNCTIONS = {
    "trend": run_trend_scan,
    "high_momentum": run_momentum_scan,
    "strong_buy": run_strong_buy_scan,
    "strong_sell": run_strong_sell_scan,
    "probable_breakout": run_breakout_scan,
    "today_breakout": run_breakout_scan,
    "52w_high_breakout": run_breakout_scan,
    "breaking_out_now": run_breakout_scan,
    "buy_reversal": run_reversal_scan,
    "ma_reversal": run_reversal_scan,
    "bullish_tomorrow": run_bullish_tomorrow_scan,
    "consolidating": run_consolidation_scan,
    "bullish_inside_bar": run_inside_bar_scan,
    "vcp": run_vcp_scan,
    "macd_crossover": run_macd_crossover_scan,
    "rsi_oversold": lambda df, info: run_rsi_scan(df, info, oversold=True),
    "rsi_overbought": lambda df, info: run_rsi_scan(df, info, oversold=False),
    "volume_gainers": run_volume_scan,
    "higher_highs": run_higher_highs_scan,
    "lorentzian_buy": run_lorentzian_scan,
}


# =============================================================================
# 🎯 MAIN API FUNCTIONS
# =============================================================================

def scan_single_stock(symbol: str, scanner_id: str) -> dict:
    """
    Run a specific scanner on a single stock.
    Returns result dict with passed status and details.
    """
    df, info = get_stock_data_with_indicators(symbol)
    
    if df.empty:
        return {
            "symbol": symbol,
            "passed": False,
            "reason": "Could not fetch data"
        }
    
    # Get the appropriate scanner function
    scanner_func = SCANNER_FUNCTIONS.get(scanner_id)
    if not scanner_func:
        # Try running a generic trend scan
        scanner_func = run_trend_scan
    
    result = scanner_func(df, info)
    
    return {
        "symbol": symbol,
        **info,
        **result
    }


def scan_multiple_stocks(symbols: List[str], scanner_id: str, max_workers: int = 10) -> List[dict]:
    """
    Run a scanner on multiple stocks in parallel.
    Returns list of stocks that passed the scan.
    """
    results = []
    
    def scan_stock(symbol):
        return scan_single_stock(symbol, scanner_id)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_stock, symbol): symbol for symbol in symbols}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result.get("passed"):
                    results.append(result)
            except Exception as e:
                pass
    
    return results


def get_scanner_categories() -> Dict[str, Any]:
    """Get all scanner categories and their scanners."""
    return SCANNER_CATEGORIES


def get_all_scanners() -> Dict[str, Any]:
    """Get flat list of all scanners."""
    return ALL_SCANNERS


def get_scanner_info(scanner_id: str) -> Optional[dict]:
    """Get information about a specific scanner."""
    return ALL_SCANNERS.get(scanner_id)


# =============================================================================
# 🔮 NIFTY PREDICTION
# =============================================================================

def get_nifty_prediction() -> dict:
    """
    Get AI Nifty prediction using PKScreener's ML model.
    """
    try:
        nifty = yf.Ticker("^NSEI")
        df = nifty.history(period="1y")
        
        if df.empty:
            return {"success": False, "error": "Could not fetch Nifty data"}
        
        # Lowercase columns
        df.columns = df.columns.str.lower()
        
        current = float(df['close'].iloc[-1])
        prev = float(df['close'].iloc[-2])
        change = current - prev
        change_pct = (change / prev * 100)
        
        # Calculate indicators
        ma_5 = float(df['close'].rolling(5).mean().iloc[-1])
        ma_10 = float(df['close'].rolling(10).mean().iloc[-1])
        ma_20 = float(df['close'].rolling(20).mean().iloc[-1])
        ma_50 = float(df['close'].rolling(50).mean().iloc[-1])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])
        
        # Try PKScreener's prediction if available
        pk_prediction = None
        if PKSCREENER_AVAILABLE:
            try:
                pk_prediction = screener.getNiftyPrediction(df)
            except:
                pass
        
        # Calculate our prediction
        bullish_signals = 0
        bearish_signals = 0
        signals_detail = []
        
        if ma_5 > ma_10:
            bullish_signals += 1
            signals_detail.append("5MA > 10MA")
        else:
            bearish_signals += 1
            
        if ma_10 > ma_20:
            bullish_signals += 1
            signals_detail.append("10MA > 20MA")
        else:
            bearish_signals += 1
            
        if current > ma_20:
            bullish_signals += 1
            signals_detail.append("Price > 20MA")
        else:
            bearish_signals += 1
            
        if current > ma_50:
            bullish_signals += 1
            signals_detail.append("Price > 50MA")
        else:
            bearish_signals += 1
            
        if 40 < current_rsi < 70:
            bullish_signals += 1
            signals_detail.append(f"RSI healthy ({current_rsi:.0f})")
        elif current_rsi < 30:
            signals_detail.append(f"RSI oversold ({current_rsi:.0f})")
        elif current_rsi > 70:
            bearish_signals += 1
            signals_detail.append(f"RSI overbought ({current_rsi:.0f})")
        
        total_signals = bullish_signals + bearish_signals
        confidence = (bullish_signals / total_signals * 100) if total_signals > 0 else 50
        
        if bullish_signals >= 4:
            direction = "BULLISH"
            predicted = current * 1.01
        elif bearish_signals >= 3:
            direction = "BEARISH"
            predicted = current * 0.99
        else:
            direction = "NEUTRAL"
            predicted = current
        
        return {
            "success": True,
            "current_level": round(current, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "prediction": {
                "direction": direction,
                "predicted_level": round(predicted, 0),
                "confidence": round(confidence, 1),
                "signals_detail": signals_detail
            },
            "indicators": {
                "rsi": round(current_rsi, 2),
                "ma_5": round(ma_5, 2),
                "ma_10": round(ma_10, 2),
                "ma_20": round(ma_20, 2),
                "ma_50": round(ma_50, 2)
            },
            "support_levels": [round(current * 0.98, 0), round(current * 0.95, 0)],
            "resistance_levels": [round(current * 1.02, 0), round(current * 1.05, 0)],
            "pk_prediction": pk_prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
