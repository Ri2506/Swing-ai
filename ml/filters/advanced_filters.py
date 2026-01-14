"""
================================================================================
SWINGAI - ADVANCED FILTERING SYSTEM
================================================================================
Includes:
1. Market Regime Detector (4 regimes)
2. Confidence Decay System
3. Premium Signal Filter (8-point validation)
4. Regime-Aware Strategy Selection
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class MarketRegime(Enum):
    BULLISH = "BULLISH"      # Trending up
    BEARISH = "BEARISH"      # Trending down
    RANGE = "RANGE"          # Consolidating
    CHOPPY = "CHOPPY"        # Volatile, no direction


class SignalGrade(Enum):
    PREMIUM = "PREMIUM"      # 95%+ (full size)
    EXCELLENT = "EXCELLENT"  # 88-95% (normal size)
    GOOD = "GOOD"            # 80-88% (reduced size)
    SKIP = "SKIP"            # <80% (don't trade)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RegimeDetection:
    """Market regime detection result"""
    regime: MarketRegime
    confidence: float  # 0-100
    characteristics: Dict[str, float]
    
    # Trend metrics
    trend_strength: float
    volatility: float
    directional_bias: float  # -100 to +100


@dataclass
class ValidationResult:
    """8-point validation result"""
    ai_confidence: float
    strategy_confluence: float
    smc_confirmation: float
    price_action_score: float
    technical_alignment: float
    regime_fit: float
    volume_confirmation: bool
    entry_precision: float
    
    # Computed
    reliability_score: float
    grade: SignalGrade
    passed: bool


@dataclass
class SignalWithDecay:
    """Signal with confidence decay tracking"""
    signal_id: str
    symbol: str
    direction: str
    
    # Original values
    initial_confidence: float
    created_at: datetime
    
    # Current values (with decay)
    current_confidence: float
    days_held: int
    decay_applied: float  # % reduction
    
    # Status
    should_exit: bool
    exit_reason: Optional[str]


# ============================================================================
# MARKET REGIME DETECTOR
# ============================================================================

class MarketRegimeDetector:
    """
    Detect current market regime using multiple indicators
    
    Regimes:
    - BULLISH: Trending up (ADX >25, price above MAs, higher highs)
    - BEARISH: Trending down (ADX >25, price below MAs, lower lows)
    - RANGE: Consolidating (ADX <20, price within range)
    - CHOPPY: Volatile, no direction (High ATR, no clear trend)
    """
    
    def __init__(self):
        self.lookback = 50
    
    def detect_regime(self, df: pd.DataFrame, vix: Optional[float] = None) -> RegimeDetection:
        """
        Detect current market regime
        
        Args:
            df: OHLCV DataFrame
            vix: Current VIX level (optional)
            
        Returns:
            RegimeDetection with regime and characteristics
        """
        if len(df) < self.lookback:
            return self._default_regime()
        
        # Calculate indicators
        characteristics = self._calculate_regime_characteristics(df, vix)
        
        # Determine regime
        regime, confidence = self._classify_regime(characteristics)
        
        return RegimeDetection(
            regime=regime,
            confidence=confidence,
            characteristics=characteristics,
            trend_strength=characteristics['trend_strength'],
            volatility=characteristics['volatility'],
            directional_bias=characteristics['directional_bias']
        )
    
    def _calculate_regime_characteristics(
        self, 
        df: pd.DataFrame,
        vix: Optional[float]
    ) -> Dict[str, float]:
        """Calculate characteristics used for regime classification"""
        
        chars = {}
        
        # 1. Trend Strength (ADX)
        from ta.trend import ADXIndicator
        adx = ADXIndicator(df['high'], df['low'], df['close'])
        chars['adx'] = adx.adx().iloc[-1]
        chars['plus_di'] = adx.adx_pos().iloc[-1]
        chars['minus_di'] = adx.adx_neg().iloc[-1]
        
        # 2. Moving Average Alignment
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            chars['ma_alignment'] = 100  # Bullish
        elif current_price < sma_20 < sma_50:
            chars['ma_alignment'] = 0    # Bearish
        else:
            chars['ma_alignment'] = 50   # Mixed
        
        # 3. Price Range vs Trend
        high_20 = df['high'].rolling(20).max().iloc[-1]
        low_20 = df['low'].rolling(20).min().iloc[-1]
        range_size = ((high_20 - low_20) / current_price) * 100
        chars['range_percentage'] = range_size
        
        # 4. Volatility (ATR %)
        from ta.volatility import AverageTrueRange
        atr = AverageTrueRange(df['high'], df['low'], df['close'])
        chars['atr_percentage'] = (atr.average_true_range().iloc[-1] / current_price) * 100
        
        # 5. VIX Level (if provided)
        chars['vix'] = vix if vix else 15.0
        
        # 6. Directional Bias (-100 to +100)
        direction_score = 0
        if chars['plus_di'] > chars['minus_di']:
            direction_score = (chars['plus_di'] - chars['minus_di'])
        else:
            direction_score = -(chars['minus_di'] - chars['plus_di'])
        chars['directional_bias'] = direction_score
        
        # 7. Higher Highs / Lower Lows
        highs = df['high'].iloc[-10:].values
        lows = df['low'].iloc[-10:].values
        
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        chars['hh_count'] = hh_count
        chars['ll_count'] = ll_count
        
        # 8. Trend Strength (combined)
        if chars['adx'] > 25:
            chars['trend_strength'] = chars['adx']
        else:
            chars['trend_strength'] = 0
        
        # 9. Volatility Score
        if chars['atr_percentage'] > 4:
            chars['volatility'] = 100  # High
        elif chars['atr_percentage'] > 2:
            chars['volatility'] = 50   # Medium
        else:
            chars['volatility'] = 0    # Low
        
        return chars
    
    def _classify_regime(self, chars: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """
        Classify regime based on characteristics
        
        Returns:
            (regime, confidence_percentage)
        """
        # Decision tree for regime classification
        
        # Check for CHOPPY first (high priority)
        if chars['vix'] > 25 or chars['atr_percentage'] > 5:
            if chars['adx'] < 20:
                return MarketRegime.CHOPPY, 85
        
        # Check for TRENDING regimes
        if chars['adx'] > 25:
            # Strong trend detected
            if chars['directional_bias'] > 10:
                # Bullish
                if chars['ma_alignment'] >= 80 and chars['hh_count'] >= 6:
                    return MarketRegime.BULLISH, 95
                else:
                    return MarketRegime.BULLISH, 75
            elif chars['directional_bias'] < -10:
                # Bearish
                if chars['ma_alignment'] <= 20 and chars['ll_count'] >= 6:
                    return MarketRegime.BEARISH, 95
                else:
                    return MarketRegime.BEARISH, 75
        
        # Check for RANGE
        if chars['adx'] < 20 and chars['range_percentage'] < 15:
            return MarketRegime.RANGE, 80
        
        # Default: Mixed signals = CHOPPY
        return MarketRegime.CHOPPY, 60
    
    def _default_regime(self) -> RegimeDetection:
        """Return default regime when insufficient data"""
        return RegimeDetection(
            regime=MarketRegime.RANGE,
            confidence=50,
            characteristics={},
            trend_strength=0,
            volatility=50,
            directional_bias=0
        )


# ============================================================================
# CONFIDENCE DECAY SYSTEM
# ============================================================================

class ConfidenceDecaySystem:
    """
    Track and apply confidence decay to signals over time
    
    Logic:
    - Reduce confidence by 5% per day
    - Exit if confidence drops below 65%
    - Account for market changes since signal generation
    """
    
    def __init__(self, decay_rate_per_day: float = 5.0, exit_threshold: float = 65.0):
        """
        Initialize decay system
        
        Args:
            decay_rate_per_day: % to reduce per day
            exit_threshold: Confidence level to exit
        """
        self.decay_rate = decay_rate_per_day
        self.exit_threshold = exit_threshold
    
    def apply_decay(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        initial_confidence: float,
        created_at: datetime
    ) -> SignalWithDecay:
        """
        Apply time-based confidence decay
        
        Args:
            signal_id: Unique signal identifier
            symbol: Stock symbol
            direction: LONG or SHORT
            initial_confidence: Starting confidence (0-100)
            created_at: When signal was generated
            
        Returns:
            SignalWithDecay with current confidence
        """
        # Calculate days held
        now = datetime.now()
        days_held = (now - created_at).days
        
        # Apply decay
        decay_applied = days_held * self.decay_rate
        current_confidence = max(0, initial_confidence - decay_applied)
        
        # Check if should exit
        should_exit = current_confidence < self.exit_threshold
        exit_reason = None
        
        if should_exit:
            exit_reason = f"Confidence decayed to {current_confidence:.1f}% (below {self.exit_threshold}%)"
        
        return SignalWithDecay(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            initial_confidence=initial_confidence,
            created_at=created_at,
            current_confidence=round(current_confidence, 2),
            days_held=days_held,
            decay_applied=round(decay_applied, 2),
            should_exit=should_exit,
            exit_reason=exit_reason
        )
    
    def check_all_positions(self, positions: List[Dict]) -> List[SignalWithDecay]:
        """
        Check confidence decay for all positions
        
        Args:
            positions: List of position dicts with signal info
            
        Returns:
            List of signals with decay applied
        """
        results = []
        
        for pos in positions:
            result = self.apply_decay(
                signal_id=pos.get('signal_id', 'unknown'),
                symbol=pos['symbol'],
                direction=pos['direction'],
                initial_confidence=pos.get('initial_confidence', 80),
                created_at=pos.get('created_at', datetime.now())
            )
            results.append(result)
        
        return results


# ============================================================================
# PREMIUM SIGNAL FILTER (8-POINT VALIDATION)
# ============================================================================

class PremiumSignalFilter:
    """
    8-point validation system for premium signals
    
    Validation Points:
    1. AI Confidence ≥ 78%
    2. Strategy Confluence ≥ 75%
    3. SMC Confirmation ≥ 70%
    4. Price Action Score ≥ 75%
    5. Technical Alignment ≥ 75%
    6. Regime Fit ≥ 80%
    7. Volume Confirmation = YES
    8. Entry Precision ≥ 80%
    """
    
    def __init__(self):
        self.thresholds = {
            'ai_confidence': 78,
            'strategy_confluence': 75,
            'smc_confirmation': 70,
            'price_action_score': 75,
            'technical_alignment': 75,
            'regime_fit': 80,
            'volume_confirmation': True,
            'entry_precision': 80
        }
    
    def validate_signal(
        self,
        ai_confidence: float,
        strategy_confluence: float,
        smc_features: Dict[str, float],
        price_action_features: Dict[str, float],
        technical_features: Dict[str, float],
        regime: MarketRegime,
        signal_direction: str,
        volume_features: Dict[str, float]
    ) -> ValidationResult:
        """
        Perform 8-point validation
        
        Returns:
            ValidationResult with score and grade
        """
        # 1. AI Confidence (direct)
        check_1 = ai_confidence
        
        # 2. Strategy Confluence (will be provided by strategy layer)
        check_2 = strategy_confluence
        
        # 3. SMC Confirmation
        check_3 = self._calculate_smc_confirmation(smc_features, signal_direction)
        
        # 4. Price Action Score
        check_4 = self._calculate_price_action_score(price_action_features, signal_direction)
        
        # 5. Technical Alignment
        check_5 = self._calculate_technical_alignment(technical_features, signal_direction)
        
        # 6. Regime Fit
        check_6 = self._calculate_regime_fit(regime, signal_direction)
        
        # 7. Volume Confirmation
        check_7_bool = self._check_volume_confirmation(volume_features)
        check_7 = 100 if check_7_bool else 0
        
        # 8. Entry Precision
        check_8 = self._calculate_entry_precision(price_action_features, smc_features)
        
        # Calculate Reliability Score (weighted average)
        weights = {
            'ai': 0.20,
            'strategy': 0.20,
            'smc': 0.15,
            'price_action': 0.15,
            'technical': 0.10,
            'regime': 0.10,
            'volume': 0.05,
            'entry': 0.05
        }
        
        reliability_score = (
            check_1 * weights['ai'] +
            check_2 * weights['strategy'] +
            check_3 * weights['smc'] +
            check_4 * weights['price_action'] +
            check_5 * weights['technical'] +
            check_6 * weights['regime'] +
            check_7 * weights['volume'] +
            check_8 * weights['entry']
        )
        
        # Determine grade
        if reliability_score >= 95:
            grade = SignalGrade.PREMIUM
        elif reliability_score >= 88:
            grade = SignalGrade.EXCELLENT
        elif reliability_score >= 80:
            grade = SignalGrade.GOOD
        else:
            grade = SignalGrade.SKIP
        
        # Check if passed all minimum thresholds
        passed = (
            check_1 >= self.thresholds['ai_confidence'] and
            check_2 >= self.thresholds['strategy_confluence'] and
            check_3 >= self.thresholds['smc_confirmation'] and
            check_4 >= self.thresholds['price_action_score'] and
            check_5 >= self.thresholds['technical_alignment'] and
            check_6 >= self.thresholds['regime_fit'] and
            check_7_bool and
            check_8 >= self.thresholds['entry_precision']
        )
        
        return ValidationResult(
            ai_confidence=round(check_1, 2),
            strategy_confluence=round(check_2, 2),
            smc_confirmation=round(check_3, 2),
            price_action_score=round(check_4, 2),
            technical_alignment=round(check_5, 2),
            regime_fit=round(check_6, 2),
            volume_confirmation=check_7_bool,
            entry_precision=round(check_8, 2),
            reliability_score=round(reliability_score, 2),
            grade=grade,
            passed=passed
        )
    
    def _calculate_smc_confirmation(self, smc: Dict[str, float], direction: str) -> float:
        """Calculate SMC confirmation score (0-100)"""
        score = 0
        
        # Order Block alignment
        if smc.get('order_block_strength', 0) > 60:
            score += 20
        if smc.get('order_block_distance', 100) < 5:
            score += 15
        
        # FVG alignment
        if smc.get('fvg_distance', 100) < 5:
            score += 15
        
        # Institutional activity
        if smc.get('institutional_activity', 0) > 70:
            score += 20
        
        # Accumulation/Distribution
        if direction == "LONG" and smc.get('accumulation_phase', 0) > 60:
            score += 15
        elif direction == "SHORT" and smc.get('distribution_phase', 0) > 60:
            score += 15
        
        # Liquidity
        if smc.get('liquidity_level', 0) > 70:
            score += 15
        
        return min(score, 100)
    
    def _calculate_price_action_score(self, pa: Dict[str, float], direction: str) -> float:
        """Calculate price action score (0-100)"""
        score = 0
        
        # Trend alignment
        if direction == "LONG" and pa.get('trend_direction', 50) >= 70:
            score += 25
        elif direction == "SHORT" and pa.get('trend_direction', 50) <= 30:
            score += 25
        
        # Support/Resistance proximity
        if direction == "LONG" and pa.get('support_distance', 10) < 3:
            score += 25
        elif direction == "SHORT" and pa.get('resistance_distance', 10) < 3:
            score += 25
        
        # Momentum alignment
        if direction == "LONG" and pa.get('momentum_10d', 0) > 2:
            score += 25
        elif direction == "SHORT" and pa.get('momentum_10d', 0) < -2:
            score += 25
        
        # Candle strength
        if pa.get('candle_strength', 0) > 60:
            score += 25
        
        return min(score, 100)
    
    def _calculate_technical_alignment(self, tech: Dict[str, float], direction: str) -> float:
        """Calculate technical alignment score (0-100)"""
        score = 0
        
        # RSI
        if direction == "LONG" and 30 <= tech.get('rsi_14', 50) <= 50:
            score += 20
        elif direction == "SHORT" and 50 <= tech.get('rsi_14', 50) <= 70:
            score += 20
        
        # MACD
        if direction == "LONG" and tech.get('macd_histogram', 0) > 0:
            score += 20
        elif direction == "SHORT" and tech.get('macd_histogram', 0) < 0:
            score += 20
        
        # ADX (trend strength)
        if tech.get('adx', 0) > 25:
            score += 20
        
        # Bollinger Band position
        bb_pct = tech.get('bb_percentage', 0.5)
        if direction == "LONG" and bb_pct < 0.2:
            score += 20
        elif direction == "SHORT" and bb_pct > 0.8:
            score += 20
        
        # Stochastic
        stoch = tech.get('stoch_k', 50)
        if direction == "LONG" and stoch < 30:
            score += 20
        elif direction == "SHORT" and stoch > 70:
            score += 20
        
        return min(score, 100)
    
    def _calculate_regime_fit(self, regime: MarketRegime, direction: str) -> float:
        """Calculate regime fit score (0-100)"""
        # Check if signal direction matches regime
        if regime == MarketRegime.BULLISH and direction == "LONG":
            return 100
        elif regime == MarketRegime.BEARISH and direction == "SHORT":
            return 100
        elif regime == MarketRegime.RANGE:
            return 70  # Range can work for both
        elif regime == MarketRegime.CHOPPY:
            return 40  # Not ideal for any direction
        else:
            return 50  # Neutral
    
    def _check_volume_confirmation(self, vol: Dict[str, float]) -> bool:
        """Check if volume confirms the signal"""
        # Volume should be above average
        volume_ratio = vol.get('volume_ma_ratio', 1.0)
        
        # MFI should be healthy
        mfi = vol.get('mfi', 50)
        
        return volume_ratio > 1.2 and 20 < mfi < 80
    
    def _calculate_entry_precision(self, pa: Dict[str, float], smc: Dict[str, float]) -> float:
        """Calculate entry precision score (0-100)"""
        score = 0
        
        # Close to key levels
        if pa.get('fib_distance', 10) < 2:
            score += 30
        
        # Close to order block
        if smc.get('order_block_distance', 100) < 3:
            score += 30
        
        # Range position (middle = bad, edges = good)
        range_pos = pa.get('range_position', 50)
        if range_pos < 30 or range_pos > 70:
            score += 40
        
        return min(score, 100)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ADVANCED FILTERING SYSTEM TEST")
    print("="*80)
    
    # Test Regime Detector
    print("\n1. MARKET REGIME DETECTION")
    print("-" * 80)
    
    import yfinance as yf
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="3mo", interval="1d")
    
    detector = MarketRegimeDetector()
    regime_result = detector.detect_regime(df, vix=14.5)
    
    print(f"Detected Regime: {regime_result.regime.value}")
    print(f"Confidence: {regime_result.confidence:.2f}%")
    print(f"Trend Strength: {regime_result.trend_strength:.2f}")
    print(f"Volatility: {regime_result.volatility:.2f}")
    print(f"Directional Bias: {regime_result.directional_bias:.2f}")
    
    # Test Confidence Decay
    print("\n2. CONFIDENCE DECAY")
    print("-" * 80)
    
    decay_system = ConfidenceDecaySystem()
    signal_decay = decay_system.apply_decay(
        signal_id="SIG001",
        symbol="RELIANCE",
        direction="LONG",
        initial_confidence=85.0,
        created_at=datetime.now() - timedelta(days=3)
    )
    
    print(f"Initial Confidence: {signal_decay.initial_confidence}%")
    print(f"Days Held: {signal_decay.days_held}")
    print(f"Decay Applied: {signal_decay.decay_applied}%")
    print(f"Current Confidence: {signal_decay.current_confidence}%")
    print(f"Should Exit: {signal_decay.should_exit}")
    
    # Test Premium Filter
    print("\n3. PREMIUM SIGNAL FILTER")
    print("-" * 80)
    
    filter_system = PremiumSignalFilter()
    
    # Mock features
    smc_features = {
        'order_block_strength': 75,
        'order_block_distance': 2.5,
        'fvg_distance': 3.0,
        'institutional_activity': 80,
        'accumulation_phase': 70,
        'liquidity_level': 75
    }
    
    pa_features = {
        'trend_direction': 85,
        'support_distance': 2.0,
        'momentum_10d': 3.5,
        'candle_strength': 70,
        'fib_distance': 1.5,
        'range_position': 75
    }
    
    tech_features = {
        'rsi_14': 45,
        'macd_histogram': 2.5,
        'adx': 30,
        'bb_percentage': 0.3,
        'stoch_k': 35
    }
    
    vol_features = {
        'volume_ma_ratio': 1.5,
        'mfi': 60
    }
    
    validation = filter_system.validate_signal(
        ai_confidence=85.0,
        strategy_confluence=82.0,
        smc_features=smc_features,
        price_action_features=pa_features,
        technical_features=tech_features,
        regime=MarketRegime.BULLISH,
        signal_direction="LONG",
        volume_features=vol_features
    )
    
    print(f"\nValidation Results:")
    print(f"  AI Confidence: {validation.ai_confidence}%")
    print(f"  Strategy Confluence: {validation.strategy_confluence}%")
    print(f"  SMC Confirmation: {validation.smc_confirmation}%")
    print(f"  Price Action Score: {validation.price_action_score}%")
    print(f"  Technical Alignment: {validation.technical_alignment}%")
    print(f"  Regime Fit: {validation.regime_fit}%")
    print(f"  Volume Confirmation: {validation.volume_confirmation}")
    print(f"  Entry Precision: {validation.entry_precision}%")
    print(f"\n  Reliability Score: {validation.reliability_score}%")
    print(f"  Grade: {validation.grade.value}")
    print(f"  Passed: {validation.passed}")
    
    print("\n" + "="*80)
