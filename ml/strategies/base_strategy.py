"""
================================================================================
SWINGAI - BASE STRATEGY CLASS
================================================================================
Base class for all 20 trading strategies
All strategies inherit from this and implement check_confluence()
================================================================================
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    
    All 20 strategies must inherit from this and implement:
    - check_confluence() method
    
    Provides common utilities:
    - Entry/stop/target calculation
    - Risk/reward calculation
    - Strategy information
    """
    
    def __init__(self, name: str, tier: str, win_rate_range: str):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name (e.g., "S1_7Element")
            tier: Performance tier (S+, S, A, B, C)
            win_rate_range: Expected win rate (e.g., "85-90%")
        """
        self.name = name
        self.tier = tier
        self.win_rate_range = win_rate_range
        self.confluence_elements = 7  # Default 7 elements to check
    
    @abstractmethod
    def check_confluence(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        Check confluence score (0-1.0)
        
        Must check at least 7 elements:
        1. Trend alignment
        2. Support/resistance
        3. Volume confirmation
        4. Technical indicator
        5. Price action pattern
        6. Risk/reward ratio
        7. Entry precision
        
        Args:
            df: DataFrame with OHLCV + 70 features
            current_idx: Index of current candle
            
        Returns:
            Float between 0.0 and 1.0 (confluence score)
        """
        pass
    
    def calculate_entry_stop_targets(
        self, 
        df: pd.DataFrame, 
        current_idx: int
    ) -> Dict[str, float]:
        """
        Calculate entry, stop loss, and 3 targets
        
        Args:
            df: DataFrame with OHLCV + features
            current_idx: Index of current candle
            
        Returns:
            Dictionary with:
            - entry: Entry price
            - stop: Stop loss price
            - target1: First target (2R)
            - target2: Second target (4R)
            - target3: Third target (6R)
            - risk: Risk amount
            - reward: Reward amount (to target1)
            - rr_ratio: Risk/reward ratio
        """
        current_price = df['close'].iloc[current_idx]
        atr = df['ATR'].iloc[current_idx]
        support = df['Support_Level'].iloc[current_idx]
        resistance = df['Resistance_Level'].iloc[current_idx]
        
        # Entry = current price
        entry = current_price
        
        # Stop loss = below support or 1.5 ATR
        stop_loss = max(support * 0.99, current_price - (1.5 * atr))
        
        # Ensure stop loss is below entry
        if stop_loss >= entry:
            stop_loss = entry - (1.5 * atr)
        
        # Calculate risk
        risk = entry - stop_loss
        
        # Targets (2R, 4R, 6R)
        target1 = entry + (2 * risk)  # 2R
        target2 = entry + (4 * risk)  # 4R
        target3 = entry + (6 * risk)  # 6R
        
        # Cap target3 at resistance
        if target3 > resistance:
            target3 = resistance * 0.99
        
        # Calculate reward (to first target)
        reward = target1 - entry
        
        # Calculate R:R ratio
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'entry': round(entry, 2),
            'stop': round(stop_loss, 2),
            'target1': round(target1, 2),
            'target2': round(target2, 2),
            'target3': round(target3, 2),
            'risk': round(risk, 2),
            'reward': round(reward, 2),
            'rr_ratio': round(rr_ratio, 2)
        }
    
    def get_info(self) -> Dict[str, str]:
        """
        Return strategy information
        
        Returns:
            Dictionary with name, tier, win_rate
        """
        return {
            'name': self.name,
            'tier': self.tier,
            'win_rate': self.win_rate_range
        }
    
    def __repr__(self):
        return f"<{self.name} (Tier {self.tier}, WR: {self.win_rate_range})>"
    
    def __str__(self):
        return self.name


# ============================================================================
# HELPER FUNCTIONS FOR STRATEGIES
# ============================================================================

def check_trend_alignment(df: pd.DataFrame, idx: int) -> float:
    """
    Check multi-timeframe trend alignment (0-1)
    
    Returns 1.0 if all timeframes aligned, 0.0 if none
    """
    weekly = df['Weekly_Trend'].iloc[idx]
    daily = df['Daily_Trend'].iloc[idx]
    ma_alignment = df['MA_Alignment'].iloc[idx] / 100.0
    
    # Average alignment
    alignment = (weekly + daily + ma_alignment) / 3.0
    return alignment


def check_support_resistance(df: pd.DataFrame, idx: int, threshold: float = 0.02) -> float:
    """
    Check proximity to support/resistance (0-1)
    
    Returns 1.0 if near support, 0.0 if far away
    
    Args:
        threshold: Distance threshold as % (default 2%)
    """
    current_price = df['close'].iloc[idx]
    support = df['Support_Level'].iloc[idx]
    resistance = df['Resistance_Level'].iloc[idx]
    
    # Distance to support (%)
    dist_to_support = abs(current_price - support) / current_price
    
    # Distance to resistance (%)
    dist_to_resistance = abs(current_price - resistance) / current_price
    
    # Return score based on proximity
    min_dist = min(dist_to_support, dist_to_resistance)
    
    if min_dist < threshold:
        return 1.0 - (min_dist / threshold)
    else:
        return 0.0


def check_volume_confirmation(df: pd.DataFrame, idx: int) -> float:
    """
    Check volume confirmation (0-1)
    
    Returns 1.0 if volume confirms, 0.0 if not
    """
    volume_spike = df['Volume_Spike'].iloc[idx]
    volume_ma_ratio = df['volume'].iloc[idx] / df['Volume_MA'].iloc[idx]
    mfi = df['MFI'].iloc[idx]
    
    score = 0.0
    
    # Volume spike
    if volume_spike == 1:
        score += 0.4
    
    # Above average volume
    if volume_ma_ratio > 1.2:
        score += 0.3
    
    # MFI in healthy range (20-80)
    if 20 < mfi < 80:
        score += 0.3
    
    return min(score, 1.0)


def check_technical_indicators(df: pd.DataFrame, idx: int, direction: str = 'LONG') -> float:
    """
    Check technical indicators alignment (0-1)
    
    Args:
        direction: 'LONG' or 'SHORT'
    
    Returns score based on indicator alignment
    """
    rsi = df['RSI'].iloc[idx]
    macd_hist = df['MACD'].iloc[idx] - df['MACD_Signal'].iloc[idx]
    adx = df['ADX'].iloc[idx]
    
    score = 0.0
    
    if direction == 'LONG':
        # RSI oversold or neutral
        if rsi < 50:
            score += 0.3
        
        # MACD positive
        if macd_hist > 0:
            score += 0.3
        
        # ADX shows trend
        if adx > 25:
            score += 0.4
    
    else:  # SHORT
        # RSI overbought or neutral
        if rsi > 50:
            score += 0.3
        
        # MACD negative
        if macd_hist < 0:
            score += 0.3
        
        # ADX shows trend
        if adx > 25:
            score += 0.4
    
    return min(score, 1.0)


def check_price_action_pattern(df: pd.DataFrame, idx: int) -> float:
    """
    Check price action pattern quality (0-1)
    
    Looks for:
    - Candle strength
    - Pattern formation
    - Momentum
    """
    consolidation = df['Consolidation_Score'].iloc[idx] / 100.0
    momentum = abs(df['ROC'].iloc[idx]) / 10.0  # Normalize
    
    # Candle body vs range
    candle_open = df['open'].iloc[idx]
    candle_close = df['close'].iloc[idx]
    candle_high = df['high'].iloc[idx]
    candle_low = df['low'].iloc[idx]
    
    body = abs(candle_close - candle_open)
    total_range = candle_high - candle_low
    
    candle_strength = body / total_range if total_range > 0 else 0
    
    # Average score
    score = (consolidation + min(momentum, 1.0) + candle_strength) / 3.0
    
    return min(score, 1.0)


def check_risk_reward_ratio(df: pd.DataFrame, idx: int, min_rr: float = 2.0) -> float:
    """
    Check if risk/reward ratio is acceptable (0-1)
    
    Args:
        min_rr: Minimum R:R ratio (default 2.0)
    
    Returns 1.0 if RR >= min_rr, scaled down if lower
    """
    current_price = df['close'].iloc[idx]
    support = df['Support_Level'].iloc[idx]
    resistance = df['Resistance_Level'].iloc[idx]
    
    # Calculate potential risk/reward
    risk = current_price - support
    reward = resistance - current_price
    
    if risk <= 0:
        return 0.0
    
    rr_ratio = reward / risk
    
    # Score based on RR ratio
    if rr_ratio >= min_rr:
        return 1.0
    else:
        return rr_ratio / min_rr


def check_entry_precision(df: pd.DataFrame, idx: int) -> float:
    """
    Check entry precision (0-1)
    
    Better entry when:
    - Near fibonacci levels
    - Near support
    - At order blocks
    """
    current_price = df['close'].iloc[idx]
    
    # Check fibonacci proximity
    fib_382 = df['Fib_382'].iloc[idx]
    fib_500 = df['Fib_500'].iloc[idx]
    fib_618 = df['Fib_618'].iloc[idx]
    
    fib_distances = [
        abs(current_price - fib_382) / current_price,
        abs(current_price - fib_500) / current_price,
        abs(current_price - fib_618) / current_price
    ]
    
    min_fib_dist = min(fib_distances)
    
    # Check support proximity
    support = df['Support_Level'].iloc[idx]
    support_dist = abs(current_price - support) / current_price
    
    # Check order block proximity
    ob_distance = df['OB_Distance_Pct'].iloc[idx] / 100.0
    
    # Score (closer = better)
    fib_score = max(0, 1.0 - (min_fib_dist / 0.02))  # Within 2%
    support_score = max(0, 1.0 - (support_dist / 0.02))
    ob_score = max(0, 1.0 - (ob_distance / 0.02))
    
    # Average
    score = (fib_score + support_score + ob_score) / 3.0
    
    return min(score, 1.0)


# ============================================================================
# EXAMPLE STRATEGY (for testing)
# ============================================================================

class ExampleStrategy(BaseStrategy):
    """Example strategy showing how to implement"""
    
    def __init__(self):
        super().__init__(
            name="Example_Strategy",
            tier="C",
            win_rate_range="60-65%"
        )
    
    def check_confluence(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        Check 7-element confluence
        """
        if current_idx < 50:
            return 0.0
        
        # 1. Trend alignment
        trend_score = check_trend_alignment(df, current_idx)
        
        # 2. Support/resistance
        sr_score = check_support_resistance(df, current_idx)
        
        # 3. Volume confirmation
        volume_score = check_volume_confirmation(df, current_idx)
        
        # 4. Technical indicators
        tech_score = check_technical_indicators(df, current_idx, 'LONG')
        
        # 5. Price action pattern
        pa_score = check_price_action_pattern(df, current_idx)
        
        # 6. Risk/reward ratio
        rr_score = check_risk_reward_ratio(df, current_idx)
        
        # 7. Entry precision
        entry_score = check_entry_precision(df, current_idx)
        
        # Average all 7 scores
        confluence = (
            trend_score + sr_score + volume_score + tech_score +
            pa_score + rr_score + entry_score
        ) / 7.0
        
        return confluence


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("BASE STRATEGY CLASS - READY")
    print("="*80)
    
    # Test example strategy
    strategy = ExampleStrategy()
    print(f"\nExample Strategy: {strategy}")
    print(f"Info: {strategy.get_info()}")
    
    print("\n✅ Base strategy class is ready!")
    print("📌 Now implement your 20 strategies by inheriting from BaseStrategy")
    print("="*80)
