"""
================================================================================
SWINGAI - SMART MONEY CONCEPTS (SMC) FEATURES
================================================================================
10 SMC features that AI learns from:
1. Order Block Strength
2. Order Block Distance
3. Fair Value Gap (FVG) Distance
4. FVG Volume Ratio
5. Sweep Detection
6. Post-Sweep Reversal Probability
7. Institutional Activity Score
8. Accumulation Phase
9. Distribution Phase
10. Liquidity Level
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderBlock:
    """Order Block data structure"""
    price_high: float
    price_low: float
    volume: float
    strength: float
    timestamp: pd.Timestamp
    block_type: str  # 'bullish' or 'bearish'


@dataclass
class FairValueGap:
    """Fair Value Gap data structure"""
    top: float
    bottom: float
    volume_ratio: float
    timestamp: pd.Timestamp
    gap_type: str  # 'bullish' or 'bearish'


class SMCFeatureCalculator:
    """
    Smart Money Concepts Feature Calculator
    Extracts institutional trading patterns
    """
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
    
    def calculate_all_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all 10 SMC features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with 10 SMC features
        """
        if len(df) < self.lookback_periods:
            return self._get_default_features()
        
        features = {}
        
        # 1. Order Block features
        order_blocks = self._detect_order_blocks(df)
        features['order_block_strength'] = self._calculate_order_block_strength(order_blocks)
        features['order_block_distance'] = self._calculate_order_block_distance(df, order_blocks)
        
        # 2. Fair Value Gap features
        fvgs = self._detect_fair_value_gaps(df)
        features['fvg_distance'] = self._calculate_fvg_distance(df, fvgs)
        features['fvg_volume_ratio'] = self._calculate_fvg_volume_ratio(fvgs, df)
        
        # 3. Liquidity Sweep features
        features['sweep_detection'] = self._detect_sweep(df)
        features['post_sweep_reversal_prob'] = self._calculate_reversal_probability(df)
        
        # 4. Institutional Activity
        features['institutional_activity'] = self._calculate_institutional_activity(df)
        
        # 5. Accumulation/Distribution
        features['accumulation_phase'] = self._calculate_accumulation(df)
        features['distribution_phase'] = self._calculate_distribution(df)
        
        # 6. Liquidity Level
        features['liquidity_level'] = self._calculate_liquidity_level(df)
        
        return features
    
    def _detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect Order Blocks (OB)
        
        Order Block = Last candle before strong move
        - Bullish OB: Last down candle before strong up move
        - Bearish OB: Last up candle before strong down move
        """
        order_blocks = []
        
        if len(df) < 5:
            return order_blocks
        
        # Calculate price moves
        df['move'] = df['close'].pct_change()
        df['strong_move'] = abs(df['move']) > df['move'].rolling(20).std() * 2
        
        for i in range(4, len(df) - 1):
            # Check for strong move after current candle
            if df['strong_move'].iloc[i + 1]:
                move_direction = df['move'].iloc[i + 1]
                
                # Bullish Order Block (before up move)
                if move_direction > 0 and df['close'].iloc[i] < df['open'].iloc[i]:
                    strength = abs(df['move'].iloc[i + 1]) * 100
                    ob = OrderBlock(
                        price_high=df['high'].iloc[i],
                        price_low=df['low'].iloc[i],
                        volume=df['volume'].iloc[i],
                        strength=min(strength, 100),
                        timestamp=df.index[i],
                        block_type='bullish'
                    )
                    order_blocks.append(ob)
                
                # Bearish Order Block (before down move)
                elif move_direction < 0 and df['close'].iloc[i] > df['open'].iloc[i]:
                    strength = abs(df['move'].iloc[i + 1]) * 100
                    ob = OrderBlock(
                        price_high=df['high'].iloc[i],
                        price_low=df['low'].iloc[i],
                        volume=df['volume'].iloc[i],
                        strength=min(strength, 100),
                        timestamp=df.index[i],
                        block_type='bearish'
                    )
                    order_blocks.append(ob)
        
        # Keep only recent order blocks
        return order_blocks[-10:] if order_blocks else []
    
    def _calculate_order_block_strength(self, order_blocks: List[OrderBlock]) -> float:
        """
        Calculate average strength of recent order blocks (0-100)
        Higher = Stronger institutional presence
        """
        if not order_blocks:
            return 0.0
        
        # Weight recent blocks more
        weights = np.linspace(0.5, 1.0, len(order_blocks))
        weighted_strength = sum(ob.strength * w for ob, w in zip(order_blocks, weights))
        total_weight = sum(weights)
        
        return round(weighted_strength / total_weight, 2)
    
    def _calculate_order_block_distance(self, df: pd.DataFrame, order_blocks: List[OrderBlock]) -> float:
        """
        Calculate distance to nearest order block (%)
        Closer = Higher probability of bounce/rejection
        """
        if not order_blocks or df.empty:
            return 100.0  # Far away
        
        current_price = df['close'].iloc[-1]
        
        # Find nearest order block
        min_distance = float('inf')
        for ob in order_blocks:
            # Distance to OB range
            if current_price > ob.price_high:
                distance = (current_price - ob.price_high) / current_price * 100
            elif current_price < ob.price_low:
                distance = (ob.price_low - current_price) / current_price * 100
            else:
                distance = 0  # Inside OB
            
            min_distance = min(min_distance, distance)
        
        return round(min_distance, 2)
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps (FVG)
        
        FVG = Gap between candles that shows imbalance
        - Bullish FVG: Current low > Previous high (2 candles ago)
        - Bearish FVG: Current high < Previous low (2 candles ago)
        """
        fvgs = []
        
        if len(df) < 3:
            return fvgs
        
        avg_volume = df['volume'].rolling(20).mean()
        
        for i in range(2, len(df)):
            # Bullish FVG
            if df['low'].iloc[i] > df['high'].iloc[i - 2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i - 2]
                volume_ratio = df['volume'].iloc[i - 1] / avg_volume.iloc[i - 1] if avg_volume.iloc[i - 1] > 0 else 1.0
                
                fvg = FairValueGap(
                    top=df['low'].iloc[i],
                    bottom=df['high'].iloc[i - 2],
                    volume_ratio=volume_ratio,
                    timestamp=df.index[i],
                    gap_type='bullish'
                )
                fvgs.append(fvg)
            
            # Bearish FVG
            elif df['high'].iloc[i] < df['low'].iloc[i - 2]:
                gap_size = df['low'].iloc[i - 2] - df['high'].iloc[i]
                volume_ratio = df['volume'].iloc[i - 1] / avg_volume.iloc[i - 1] if avg_volume.iloc[i - 1] > 0 else 1.0
                
                fvg = FairValueGap(
                    top=df['low'].iloc[i - 2],
                    bottom=df['high'].iloc[i],
                    volume_ratio=volume_ratio,
                    timestamp=df.index[i],
                    gap_type='bearish'
                )
                fvgs.append(fvg)
        
        # Keep only recent unfilled FVGs
        return fvgs[-10:] if fvgs else []
    
    def _calculate_fvg_distance(self, df: pd.DataFrame, fvgs: List[FairValueGap]) -> float:
        """Calculate distance to nearest FVG (%)"""
        if not fvgs or df.empty:
            return 100.0
        
        current_price = df['close'].iloc[-1]
        
        min_distance = float('inf')
        for fvg in fvgs:
            if current_price > fvg.top:
                distance = (current_price - fvg.top) / current_price * 100
            elif current_price < fvg.bottom:
                distance = (fvg.bottom - current_price) / current_price * 100
            else:
                distance = 0  # Inside FVG
            
            min_distance = min(min_distance, distance)
        
        return round(min_distance, 2)
    
    def _calculate_fvg_volume_ratio(self, fvgs: List[FairValueGap], df: pd.DataFrame) -> float:
        """
        Calculate average volume ratio of FVGs
        Higher ratio = Stronger institutional imbalance
        """
        if not fvgs:
            return 1.0
        
        avg_ratio = sum(fvg.volume_ratio for fvg in fvgs) / len(fvgs)
        return round(avg_ratio, 2)
    
    def _detect_sweep(self, df: pd.DataFrame) -> float:
        """
        Detect Liquidity Sweep (0 or 1)
        
        Sweep = Price briefly breaks key level then reverses
        - Stop hunt pattern
        - False breakout
        """
        if len(df) < 20:
            return 0.0
        
        # Find recent swing highs/lows
        swing_high = df['high'].rolling(10, center=True).max()
        swing_low = df['low'].rolling(10, center=True).min()
        
        recent_high = swing_high.iloc[-20:-5].max()
        recent_low = swing_low.iloc[-20:-5].min()
        
        # Check last 5 candles for sweep
        last_5 = df.iloc[-5:]
        
        # Bullish sweep (swept low then reversed up)
        if last_5['low'].min() < recent_low * 0.995:  # Broke below by 0.5%
            if df['close'].iloc[-1] > last_5['low'].min() * 1.005:  # Reversed 0.5%
                return 1.0
        
        # Bearish sweep (swept high then reversed down)
        if last_5['high'].max() > recent_high * 1.005:  # Broke above by 0.5%
            if df['close'].iloc[-1] < last_5['high'].max() * 0.995:  # Reversed 0.5%
                return 1.0
        
        return 0.0
    
    def _calculate_reversal_probability(self, df: pd.DataFrame) -> float:
        """
        Calculate probability of reversal after sweep (0-100)
        Based on momentum and volume
        """
        if len(df) < 10:
            return 50.0
        
        # Check if recent sweep detected
        sweep = self._detect_sweep(df)
        if sweep == 0:
            return 50.0  # No sweep, neutral
        
        # Calculate reversal indicators
        score = 50.0
        
        # 1. Volume confirmation
        recent_volume = df['volume'].iloc[-3:].mean()
        avg_volume = df['volume'].iloc[-20:-3].mean()
        if recent_volume > avg_volume * 1.3:
            score += 15
        
        # 2. Momentum shift
        rsi = self._calculate_rsi(df['close'], 14)
        if rsi.iloc[-1] < 35:  # Oversold after sweep
            score += 15
        elif rsi.iloc[-1] > 65:  # Overbought after sweep
            score += 15
        
        # 3. Price action
        last_3 = df.iloc[-3:]
        if last_3['close'].iloc[-1] > last_3['open'].iloc[-1]:  # Bullish candle
            score += 10
        
        return round(min(score, 100), 2)
    
    def _calculate_institutional_activity(self, df: pd.DataFrame) -> float:
        """
        Calculate Institutional Activity Score (0-100)
        
        Based on:
        - Large volume candles
        - Price rejection at levels
        - Absorption patterns
        """
        if len(df) < 20:
            return 50.0
        
        score = 0.0
        
        # 1. Volume spikes (30 points)
        avg_volume = df['volume'].rolling(20).mean()
        volume_spikes = (df['volume'].iloc[-10:] > avg_volume.iloc[-10:] * 2).sum()
        score += min(volume_spikes / 10 * 30, 30)
        
        # 2. Wick rejections (30 points)
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body'] = abs(df['close'] - df['open'])
        
        rejection_candles = 0
        for i in range(-10, 0):
            if df['upper_wick'].iloc[i] > df['body'].iloc[i] * 2:
                rejection_candles += 1
            if df['lower_wick'].iloc[i] > df['body'].iloc[i] * 2:
                rejection_candles += 1
        
        score += min(rejection_candles / 10 * 30, 30)
        
        # 3. Price consolidation after move (40 points)
        recent_volatility = df['high'].iloc[-5:].max() - df['low'].iloc[-5:].min()
        prev_volatility = df['high'].iloc[-15:-5].max() - df['low'].iloc[-15:-5].min()
        
        if prev_volatility > 0:
            consolidation_ratio = 1 - (recent_volatility / prev_volatility)
            score += max(0, consolidation_ratio * 40)
        
        return round(min(score, 100), 2)
    
    def _calculate_accumulation(self, df: pd.DataFrame) -> float:
        """
        Calculate Accumulation Phase Score (0-100)
        
        Accumulation = Smart money buying quietly
        - Price consolidating
        - Volume increasing
        - Higher lows forming
        """
        if len(df) < 30:
            return 0.0
        
        score = 0.0
        
        # 1. Price range contraction (30 points)
        recent_range = (df['high'].iloc[-10:] - df['low'].iloc[-10:]).mean()
        prev_range = (df['high'].iloc[-30:-10] - df['low'].iloc[-30:-10]).mean()
        
        if prev_range > 0:
            contraction = 1 - (recent_range / prev_range)
            score += max(0, contraction * 30)
        
        # 2. Volume trend (30 points)
        recent_volume = df['volume'].iloc[-10:].mean()
        prev_volume = df['volume'].iloc[-30:-10].mean()
        
        if prev_volume > 0:
            volume_increase = (recent_volume / prev_volume - 1)
            score += max(0, min(volume_increase * 30, 30))
        
        # 3. Higher lows pattern (40 points)
        lows = df['low'].iloc[-10:].values
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        score += (higher_lows / len(lows)) * 40
        
        return round(min(score, 100), 2)
    
    def _calculate_distribution(self, df: pd.DataFrame) -> float:
        """
        Calculate Distribution Phase Score (0-100)
        
        Distribution = Smart money selling quietly
        - Price consolidating at top
        - Volume increasing
        - Lower highs forming
        """
        if len(df) < 30:
            return 0.0
        
        score = 0.0
        
        # 1. Price at high with range contraction (30 points)
        recent_high = df['high'].iloc[-10:].max()
        period_high = df['high'].iloc[-30:].max()
        
        if recent_high >= period_high * 0.98:  # Within 2% of high
            recent_range = (df['high'].iloc[-10:] - df['low'].iloc[-10:]).mean()
            prev_range = (df['high'].iloc[-30:-10] - df['low'].iloc[-30:-10]).mean()
            
            if prev_range > 0:
                contraction = 1 - (recent_range / prev_range)
                score += max(0, contraction * 30)
        
        # 2. Volume trend (30 points)
        recent_volume = df['volume'].iloc[-10:].mean()
        prev_volume = df['volume'].iloc[-30:-10].mean()
        
        if prev_volume > 0:
            volume_increase = (recent_volume / prev_volume - 1)
            score += max(0, min(volume_increase * 30, 30))
        
        # 3. Lower highs pattern (40 points)
        highs = df['high'].iloc[-10:].values
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        score += (lower_highs / len(highs)) * 40
        
        return round(min(score, 100), 2)
    
    def _calculate_liquidity_level(self, df: pd.DataFrame) -> float:
        """
        Calculate Liquidity Level (0-100)
        
        High liquidity = Many stop losses clustered
        - Round numbers
        - Recent swing highs/lows
        - Previous day high/low
        """
        if len(df) < 20:
            return 50.0
        
        current_price = df['close'].iloc[-1]
        score = 0.0
        
        # 1. Proximity to round numbers (30 points)
        # Check if near 100s, 500s, 1000s
        for multiplier in [100, 500, 1000]:
            nearest_round = round(current_price / multiplier) * multiplier
            distance_pct = abs(current_price - nearest_round) / current_price * 100
            
            if distance_pct < 1:  # Within 1%
                score += 30
                break
        
        # 2. Proximity to recent swing levels (40 points)
        swing_high = df['high'].rolling(10, center=True).max()
        swing_low = df['low'].rolling(10, center=True).min()
        
        recent_swing_high = swing_high.iloc[-20:-1].max()
        recent_swing_low = swing_low.iloc[-20:-1].min()
        
        dist_to_high = abs(current_price - recent_swing_high) / current_price * 100
        dist_to_low = abs(current_price - recent_swing_low) / current_price * 100
        
        min_dist = min(dist_to_high, dist_to_low)
        if min_dist < 2:  # Within 2%
            score += 40
        
        # 3. Proximity to previous day levels (30 points)
        if len(df) >= 2:
            prev_high = df['high'].iloc[-2]
            prev_low = df['low'].iloc[-2]
            
            dist_to_prev_high = abs(current_price - prev_high) / current_price * 100
            dist_to_prev_low = abs(current_price - prev_low) / current_price * 100
            
            min_prev_dist = min(dist_to_prev_high, dist_to_prev_low)
            if min_prev_dist < 1:  # Within 1%
                score += 30
        
        return round(min(score, 100), 2)
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when insufficient data"""
        return {
            'order_block_strength': 0.0,
            'order_block_distance': 100.0,
            'fvg_distance': 100.0,
            'fvg_volume_ratio': 1.0,
            'sweep_detection': 0.0,
            'post_sweep_reversal_prob': 50.0,
            'institutional_activity': 50.0,
            'accumulation_phase': 0.0,
            'distribution_phase': 0.0,
            'liquidity_level': 50.0
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Fetch data
    ticker = yf.Ticker("RELIANCE.NS")
    df = ticker.history(period="3mo", interval="1d")
    
    # Calculate SMC features
    smc_calc = SMCFeatureCalculator()
    features = smc_calc.calculate_all_features(df)
    
    print("\n" + "="*60)
    print("SMC FEATURES FOR RELIANCE")
    print("="*60)
    for feature, value in features.items():
        print(f"{feature:30s}: {value:8.2f}")
    print("="*60)
