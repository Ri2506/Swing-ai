"""
================================================================================
SWINGAI - MARKET REGIME FILTER
================================================================================

Rule-based filters applied AFTER AI model prediction.
These use external data (VIX, Nifty, FII/DII) that are harder to get real-time.

Benefits of this approach:
1. AI model runs on pure OHLCV - easy to get, fast inference
2. Filters can be delayed (EOD) without breaking model
3. Can adjust filter thresholds without retraining
4. More robust production system

Usage:
    from ml.filters.market_regime_filter import MarketRegimeFilter
    
    filter = MarketRegimeFilter()
    regime = filter.get_regime()
    filtered_signals = filter.apply_filters(ai_signals, regime)

================================================================================
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None
    print("Warning: yfinance not installed. Install with: pip install yfinance")

logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMS & DATA CLASSES
# ==============================================================================

class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_BULL = "STRONG_BULL"   # VIX low, Nifty up, FII buying
    BULL = "BULL"                  # Normal bullish conditions
    NEUTRAL = "NEUTRAL"            # Mixed signals
    BEAR = "BEAR"                  # Bearish conditions
    STRONG_BEAR = "STRONG_BEAR"   # VIX high, Nifty down, FII selling
    CRISIS = "CRISIS"              # Extreme volatility, avoid trading


@dataclass
class MarketData:
    """Container for market context data"""
    # VIX Data
    vix_current: float = 15.0
    vix_5d_avg: float = 15.0
    vix_20d_avg: float = 15.0
    vix_percentile: float = 50.0  # Percentile rank (0-100)
    vix_trend: str = "STABLE"     # RISING, FALLING, STABLE
    
    # Nifty Data
    nifty_close: float = 22000.0
    nifty_sma_20: float = 22000.0
    nifty_sma_50: float = 21500.0
    nifty_sma_200: float = 21000.0
    nifty_rsi: float = 50.0
    nifty_return_1d: float = 0.0
    nifty_return_5d: float = 0.0
    nifty_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    
    # Institutional Flows (in Crores)
    fii_net_today: float = 0.0
    fii_net_5d: float = 0.0
    fii_net_10d: float = 0.0
    dii_net_today: float = 0.0
    dii_net_5d: float = 0.0
    
    # Market Breadth
    advance_decline_ratio: float = 1.0
    pct_above_sma_20: float = 50.0
    pct_above_sma_50: float = 50.0
    
    # Timestamp
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeDecision:
    """Output from regime filter"""
    regime: MarketRegime
    confidence: float  # 0-100
    allow_longs: bool
    allow_shorts: bool
    position_size_multiplier: float  # 0.25 to 1.0
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data: Optional[MarketData] = None


@dataclass
class FilteredSignal:
    """Signal after applying market filters"""
    symbol: str
    direction: str  # LONG, SHORT
    original_confidence: float
    adjusted_confidence: float
    position_size_multiplier: float
    regime: str
    regime_notes: List[str]
    approved: bool
    rejection_reason: Optional[str] = None


# ==============================================================================
# MARKET REGIME FILTER
# ==============================================================================

class MarketRegimeFilter:
    """
    Rule-based market regime filter
    
    Analyzes:
    1. VIX (volatility/fear)
    2. Nifty trend (market direction)
    3. FII/DII flows (institutional sentiment)
    4. Market breadth (participation)
    
    And adjusts signals accordingly.
    """
    
    def __init__(self):
        # VIX Thresholds
        self.vix_low = 12          # Very calm, potential complacency
        self.vix_normal = 15       # Normal volatility
        self.vix_elevated = 20     # Elevated, be cautious
        self.vix_high = 25         # High fear, reduce position size
        self.vix_extreme = 30      # Crisis mode, avoid new positions
        
        # Nifty Thresholds
        self.nifty_rsi_overbought = 70
        self.nifty_rsi_oversold = 30
        
        # FII/DII Thresholds (in Crores)
        self.fii_heavy_buying = 3000    # Heavy buying
        self.fii_heavy_selling = -3000  # Heavy selling
        self.fii_extreme_selling = -5000  # Panic selling
        
        # Breadth Thresholds
        self.breadth_bullish = 60   # % stocks above SMA20
        self.breadth_bearish = 40
        
        # Cache
        self._market_data_cache: Optional[MarketData] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration_minutes = 15
    
    # ==========================================================================
    # DATA FETCHING
    # ==========================================================================
    
    def fetch_market_data(self, use_cache: bool = True) -> MarketData:
        """
        Fetch current market data from various sources
        
        Sources:
        - VIX: Yahoo Finance (^INDIAVIX)
        - Nifty: Yahoo Finance (^NSEI)
        - FII/DII: NSE website or proxy
        """
        
        # Check cache
        if use_cache and self._market_data_cache is not None:
            cache_age = (datetime.now() - self._cache_time).total_seconds() / 60
            if cache_age < self._cache_duration_minutes:
                return self._market_data_cache
        
        data = MarketData()
        
        try:
            # Fetch VIX
            vix_data = self._fetch_vix()
            if vix_data:
                data.vix_current = vix_data['current']
                data.vix_5d_avg = vix_data['avg_5d']
                data.vix_20d_avg = vix_data['avg_20d']
                data.vix_percentile = vix_data['percentile']
                data.vix_trend = vix_data['trend']
            
            # Fetch Nifty
            nifty_data = self._fetch_nifty()
            if nifty_data:
                data.nifty_close = nifty_data['close']
                data.nifty_sma_20 = nifty_data['sma_20']
                data.nifty_sma_50 = nifty_data['sma_50']
                data.nifty_sma_200 = nifty_data['sma_200']
                data.nifty_rsi = nifty_data['rsi']
                data.nifty_return_1d = nifty_data['return_1d']
                data.nifty_return_5d = nifty_data['return_5d']
                data.nifty_trend = nifty_data['trend']
            
            # Fetch FII/DII (placeholder - in production use NSE API)
            fii_dii = self._fetch_fii_dii()
            if fii_dii:
                data.fii_net_today = fii_dii['fii_today']
                data.fii_net_5d = fii_dii['fii_5d']
                data.fii_net_10d = fii_dii['fii_10d']
                data.dii_net_today = fii_dii['dii_today']
                data.dii_net_5d = fii_dii['dii_5d']
            
            data.last_updated = datetime.now()
            
            # Update cache
            self._market_data_cache = data
            self._cache_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
        
        return data
    
    def _fetch_vix(self) -> Optional[Dict]:
        """Fetch India VIX data"""
        if yf is None:
            return None
        
        try:
            vix = yf.download("^INDIAVIX", period="3mo", progress=False)
            if len(vix) == 0:
                return None
            
            current = vix['Close'].iloc[-1]
            avg_5d = vix['Close'].tail(5).mean()
            avg_20d = vix['Close'].tail(20).mean()
            
            # Percentile rank over 1 year
            percentile = (vix['Close'].iloc[-1] <= vix['Close']).mean() * 100
            
            # Trend
            if current > avg_5d * 1.1:
                trend = "RISING"
            elif current < avg_5d * 0.9:
                trend = "FALLING"
            else:
                trend = "STABLE"
            
            return {
                'current': round(current, 2),
                'avg_5d': round(avg_5d, 2),
                'avg_20d': round(avg_20d, 2),
                'percentile': round(percentile, 1),
                'trend': trend
            }
        except Exception as e:
            logger.warning(f"Error fetching VIX: {e}")
            return None
    
    def _fetch_nifty(self) -> Optional[Dict]:
        """Fetch Nifty 50 data"""
        if yf is None:
            return None
        
        try:
            nifty = yf.download("^NSEI", period="1y", progress=False)
            if len(nifty) == 0:
                return None
            
            close = nifty['Close'].iloc[-1]
            sma_20 = nifty['Close'].tail(20).mean()
            sma_50 = nifty['Close'].tail(50).mean()
            sma_200 = nifty['Close'].tail(200).mean()
            
            # RSI
            delta = nifty['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Returns
            return_1d = (close / nifty['Close'].iloc[-2] - 1) * 100
            return_5d = (close / nifty['Close'].iloc[-6] - 1) * 100
            
            # Trend
            if close > sma_20 > sma_50:
                trend = "BULLISH"
            elif close < sma_20 < sma_50:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
            
            return {
                'close': round(close, 2),
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'sma_200': round(sma_200, 2),
                'rsi': round(rsi, 2),
                'return_1d': round(return_1d, 2),
                'return_5d': round(return_5d, 2),
                'trend': trend
            }
        except Exception as e:
            logger.warning(f"Error fetching Nifty: {e}")
            return None
    
    def _fetch_fii_dii(self) -> Optional[Dict]:
        """
        Fetch FII/DII data
        
        Note: In production, use NSE API or web scraping.
        This is a placeholder that returns synthetic data based on market movement.
        """
        try:
            # Placeholder - in production, fetch from NSE
            # https://www.nseindia.com/reports/fii-dii
            
            # For now, return neutral values
            return {
                'fii_today': 0,
                'fii_5d': 0,
                'fii_10d': 0,
                'dii_today': 0,
                'dii_5d': 0
            }
        except Exception as e:
            logger.warning(f"Error fetching FII/DII: {e}")
            return None
    
    # ==========================================================================
    # REGIME DETECTION
    # ==========================================================================
    
    def get_regime(self, data: Optional[MarketData] = None) -> RegimeDecision:
        """
        Determine current market regime based on multiple factors
        
        Returns RegimeDecision with:
        - regime: Market regime classification
        - allow_longs: Whether to allow long positions
        - allow_shorts: Whether to allow short positions
        - position_size_multiplier: How much to scale position size
        - reasons: Why this regime was determined
        """
        
        if data is None:
            data = self.fetch_market_data()
        
        reasons = []
        warnings = []
        
        # Initialize scores
        bull_score = 0  # Positive = bullish, negative = bearish
        volatility_score = 0  # Higher = more risky
        
        # ==========================================================================
        # 1. VIX Analysis
        # ==========================================================================
        
        vix = data.vix_current
        
        if vix >= self.vix_extreme:
            volatility_score += 4
            reasons.append(f"🔴 VIX extreme: {vix:.1f} (Crisis mode)")
        elif vix >= self.vix_high:
            volatility_score += 3
            reasons.append(f"🟠 VIX high: {vix:.1f} (High fear)")
        elif vix >= self.vix_elevated:
            volatility_score += 2
            reasons.append(f"🟡 VIX elevated: {vix:.1f}")
        elif vix <= self.vix_low:
            volatility_score += 1
            warnings.append(f"⚠️ VIX very low: {vix:.1f} (Potential complacency)")
        else:
            reasons.append(f"🟢 VIX normal: {vix:.1f}")
        
        # VIX trend
        if data.vix_trend == "RISING":
            volatility_score += 1
            reasons.append("📈 VIX rising (fear increasing)")
        elif data.vix_trend == "FALLING":
            bull_score += 1
            reasons.append("📉 VIX falling (fear decreasing)")
        
        # ==========================================================================
        # 2. Nifty Trend Analysis
        # ==========================================================================
        
        if data.nifty_trend == "BULLISH":
            bull_score += 2
            reasons.append(f"📈 Nifty bullish: {data.nifty_close:.0f} > SMA20 > SMA50")
        elif data.nifty_trend == "BEARISH":
            bull_score -= 2
            reasons.append(f"📉 Nifty bearish: {data.nifty_close:.0f} < SMA20 < SMA50")
        else:
            reasons.append(f"↔️ Nifty neutral: {data.nifty_close:.0f}")
        
        # Nifty RSI
        if data.nifty_rsi >= self.nifty_rsi_overbought:
            bull_score -= 1
            warnings.append(f"⚠️ Nifty overbought: RSI={data.nifty_rsi:.0f}")
        elif data.nifty_rsi <= self.nifty_rsi_oversold:
            bull_score += 1
            reasons.append(f"📉 Nifty oversold: RSI={data.nifty_rsi:.0f} (Potential bounce)")
        
        # Nifty vs SMA200
        if data.nifty_close > data.nifty_sma_200:
            bull_score += 1
            reasons.append("✅ Nifty above 200 SMA (Long-term uptrend)")
        else:
            bull_score -= 1
            reasons.append("❌ Nifty below 200 SMA (Long-term downtrend)")
        
        # ==========================================================================
        # 3. FII/DII Analysis
        # ==========================================================================
        
        fii_5d = data.fii_net_5d
        
        if fii_5d >= self.fii_heavy_buying:
            bull_score += 2
            reasons.append(f"🏦 FII heavy buying: ₹{fii_5d:,.0f} Cr (5d)")
        elif fii_5d <= self.fii_extreme_selling:
            bull_score -= 3
            reasons.append(f"🔴 FII extreme selling: ₹{fii_5d:,.0f} Cr (5d)")
        elif fii_5d <= self.fii_heavy_selling:
            bull_score -= 2
            reasons.append(f"🟠 FII heavy selling: ₹{fii_5d:,.0f} Cr (5d)")
        
        # FII vs DII divergence
        if data.fii_net_5d > 0 and data.dii_net_5d > 0:
            bull_score += 1
            reasons.append("✅ Both FII & DII buying")
        elif data.fii_net_5d < 0 and data.dii_net_5d > 0:
            reasons.append("⚖️ FII selling, DII buying (Defensive)")
        elif data.fii_net_5d < 0 and data.dii_net_5d < 0:
            bull_score -= 2
            reasons.append("🔴 Both FII & DII selling")
        
        # ==========================================================================
        # 4. Market Breadth Analysis
        # ==========================================================================
        
        breadth = data.pct_above_sma_20
        
        if breadth >= self.breadth_bullish:
            bull_score += 1
            reasons.append(f"📊 Breadth strong: {breadth:.0f}% above SMA20")
        elif breadth <= self.breadth_bearish:
            bull_score -= 1
            reasons.append(f"📊 Breadth weak: {breadth:.0f}% above SMA20")
        
        # ==========================================================================
        # DETERMINE REGIME
        # ==========================================================================
        
        # Crisis check (VIX extreme)
        if vix >= self.vix_extreme:
            regime = MarketRegime.CRISIS
            allow_longs = False
            allow_shorts = False
            position_multiplier = 0.0
        
        # High volatility check
        elif volatility_score >= 3:
            if bull_score >= 2:
                regime = MarketRegime.BULL
            elif bull_score <= -2:
                regime = MarketRegime.STRONG_BEAR
            else:
                regime = MarketRegime.BEAR
            
            allow_longs = bull_score >= 0
            allow_shorts = bull_score <= 0
            position_multiplier = 0.5
        
        # Normal conditions
        else:
            if bull_score >= 4:
                regime = MarketRegime.STRONG_BULL
                allow_longs = True
                allow_shorts = False
                position_multiplier = 1.0
            elif bull_score >= 2:
                regime = MarketRegime.BULL
                allow_longs = True
                allow_shorts = True
                position_multiplier = 1.0
            elif bull_score <= -4:
                regime = MarketRegime.STRONG_BEAR
                allow_longs = False
                allow_shorts = True
                position_multiplier = 0.75
            elif bull_score <= -2:
                regime = MarketRegime.BEAR
                allow_longs = False
                allow_shorts = True
                position_multiplier = 0.75
            else:
                regime = MarketRegime.NEUTRAL
                allow_longs = True
                allow_shorts = True
                position_multiplier = 0.75
        
        # Confidence based on how clear the signals are
        signal_clarity = abs(bull_score) + (4 - volatility_score)
        confidence = min(100, max(0, signal_clarity * 15 + 40))
        
        return RegimeDecision(
            regime=regime,
            confidence=confidence,
            allow_longs=allow_longs,
            allow_shorts=allow_shorts,
            position_size_multiplier=position_multiplier,
            reasons=reasons,
            warnings=warnings,
            data=data
        )
    
    # ==========================================================================
    # SIGNAL FILTERING
    # ==========================================================================
    
    def filter_signals(
        self, 
        signals: List[Dict], 
        regime: Optional[RegimeDecision] = None
    ) -> List[FilteredSignal]:
        """
        Filter AI-generated signals based on market regime
        
        Args:
            signals: List of signals from AI model
                     Each signal should have: symbol, direction, confidence
            regime: Pre-computed regime (optional, will fetch if not provided)
        
        Returns:
            List of FilteredSignal objects
        """
        
        if regime is None:
            regime = self.get_regime()
        
        filtered = []
        
        for signal in signals:
            symbol = signal.get('symbol', 'UNKNOWN')
            direction = signal.get('direction', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            
            # Start with signal's original values
            approved = True
            rejection_reason = None
            adjusted_confidence = confidence
            position_multiplier = regime.position_size_multiplier
            
            # Apply regime filters
            if direction == 'LONG':
                if not regime.allow_longs:
                    approved = False
                    rejection_reason = f"Longs not allowed in {regime.regime.value} regime"
                elif regime.regime == MarketRegime.BEAR:
                    adjusted_confidence *= 0.8  # Reduce confidence
                    position_multiplier *= 0.7
            
            elif direction == 'SHORT':
                if not regime.allow_shorts:
                    approved = False
                    rejection_reason = f"Shorts not allowed in {regime.regime.value} regime"
                elif regime.regime == MarketRegime.STRONG_BULL:
                    adjusted_confidence *= 0.8
                    position_multiplier *= 0.7
            
            # Crisis mode - reject all
            if regime.regime == MarketRegime.CRISIS:
                approved = False
                rejection_reason = "Crisis mode - no new positions"
            
            # Minimum confidence threshold after adjustment
            if approved and adjusted_confidence < 60:
                approved = False
                rejection_reason = f"Adjusted confidence too low: {adjusted_confidence:.1f}%"
            
            filtered.append(FilteredSignal(
                symbol=symbol,
                direction=direction,
                original_confidence=confidence,
                adjusted_confidence=round(adjusted_confidence, 2),
                position_size_multiplier=round(position_multiplier, 2),
                regime=regime.regime.value,
                regime_notes=regime.reasons[:3],  # Top 3 reasons
                approved=approved,
                rejection_reason=rejection_reason
            ))
        
        return filtered
    
    # ==========================================================================
    # DISPLAY HELPERS
    # ==========================================================================
    
    def print_regime_report(self, regime: Optional[RegimeDecision] = None):
        """Print a formatted regime report"""
        
        if regime is None:
            regime = self.get_regime()
        
        print("\n" + "=" * 60)
        print("📊 MARKET REGIME REPORT")
        print("=" * 60)
        
        # Regime with emoji
        regime_emojis = {
            MarketRegime.STRONG_BULL: "🚀",
            MarketRegime.BULL: "📈",
            MarketRegime.NEUTRAL: "↔️",
            MarketRegime.BEAR: "📉",
            MarketRegime.STRONG_BEAR: "🔻",
            MarketRegime.CRISIS: "🚨"
        }
        
        emoji = regime_emojis.get(regime.regime, "❓")
        print(f"\n{emoji} Current Regime: {regime.regime.value}")
        print(f"   Confidence: {regime.confidence:.0f}%")
        
        print(f"\n📋 Trading Permissions:")
        print(f"   Allow Longs:  {'✅ Yes' if regime.allow_longs else '❌ No'}")
        print(f"   Allow Shorts: {'✅ Yes' if regime.allow_shorts else '❌ No'}")
        print(f"   Position Size: {regime.position_size_multiplier * 100:.0f}% of normal")
        
        if regime.data:
            print(f"\n📈 Market Data:")
            print(f"   VIX: {regime.data.vix_current:.1f} ({regime.data.vix_trend})")
            print(f"   Nifty: {regime.data.nifty_close:,.0f} ({regime.data.nifty_trend})")
            print(f"   FII 5d: ₹{regime.data.fii_net_5d:,.0f} Cr")
        
        print(f"\n📝 Analysis:")
        for reason in regime.reasons:
            print(f"   {reason}")
        
        if regime.warnings:
            print(f"\n⚠️ Warnings:")
            for warning in regime.warnings:
                print(f"   {warning}")
        
        print("\n" + "=" * 60)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_current_regime() -> RegimeDecision:
    """Quick function to get current market regime"""
    filter = MarketRegimeFilter()
    return filter.get_regime()


def should_trade_today() -> Tuple[bool, str]:
    """
    Quick check if we should trade today
    
    Returns:
        (should_trade, reason)
    """
    regime = get_current_regime()
    
    if regime.regime == MarketRegime.CRISIS:
        return False, "Crisis mode - VIX extremely high"
    
    if not regime.allow_longs and not regime.allow_shorts:
        return False, f"No positions allowed in {regime.regime.value}"
    
    return True, f"Trading allowed - {regime.regime.value} market"


# ==============================================================================
# MAIN (TESTING)
# ==============================================================================

if __name__ == "__main__":
    # Test the filter
    filter = MarketRegimeFilter()
    
    # Get and print regime
    regime = filter.get_regime()
    filter.print_regime_report(regime)
    
    # Test signal filtering
    test_signals = [
        {'symbol': 'RELIANCE', 'direction': 'LONG', 'confidence': 78},
        {'symbol': 'TCS', 'direction': 'SHORT', 'confidence': 72},
        {'symbol': 'INFY', 'direction': 'LONG', 'confidence': 65},
        {'symbol': 'HDFC', 'direction': 'SHORT', 'confidence': 55},
    ]
    
    print("\n📊 Signal Filtering Test:")
    print("-" * 60)
    
    filtered = filter.filter_signals(test_signals, regime)
    
    for sig in filtered:
        status = "✅ APPROVED" if sig.approved else f"❌ REJECTED: {sig.rejection_reason}"
        print(f"{sig.symbol} {sig.direction}: {sig.original_confidence}% → {sig.adjusted_confidence}% | {status}")
