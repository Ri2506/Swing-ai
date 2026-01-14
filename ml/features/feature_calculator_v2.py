"""
================================================================================
SWINGAI - EXACT 70 FEATURE CALCULATOR (V2)
================================================================================
Calculates all 70 features exactly as specified
Uses ta library from Python
Feature names match exact specification
================================================================================
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS FOR SMC FEATURES (Category 4)
# ============================================================================

def calculate_ob_strength(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Order Block Strength (0-100)
    
    Order Block = Last candle before strong move
    Strength = magnitude of the move after the OB
    """
    strength = pd.Series(0.0, index=df.index)
    
    # Calculate price moves
    price_change = df['close'].pct_change()
    
    # Strong move = 2x standard deviation
    std_dev = price_change.rolling(20).std()
    strong_move = abs(price_change) > (std_dev * 2)
    
    # For each strong move, assign strength to previous candle
    for i in range(1, len(df)):
        if strong_move.iloc[i]:
            move_magnitude = abs(price_change.iloc[i]) * 100
            strength.iloc[i-1] = min(move_magnitude * 10, 100)  # Scale to 0-100
    
    return strength.fillna(0)


def calculate_fvg_distance(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Fair Value Gap Distance (in points)
    
    FVG = Gap between candles (imbalance)
    """
    fvg_distance = pd.Series(0.0, index=df.index)
    
    for i in range(2, len(df)):
        # Bullish FVG: current low > high from 2 candles ago
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvg_distance.iloc[i] = df['low'].iloc[i] - df['high'].iloc[i-2]
        
        # Bearish FVG: current high < low from 2 candles ago
        elif df['high'].iloc[i] < df['low'].iloc[i-2]:
            fvg_distance.iloc[i] = df['low'].iloc[i-2] - df['high'].iloc[i]
    
    return fvg_distance


def detect_sweep(df: pd.DataFrame) -> pd.Series:
    """
    Detect Liquidity Sweep (0 or 1)
    
    Sweep = Price briefly breaks key level then reverses
    """
    sweep = pd.Series(0, index=df.index)
    
    # Find swing highs/lows
    swing_high = df['high'].rolling(10, center=True).max()
    swing_low = df['low'].rolling(10, center=True).min()
    
    for i in range(20, len(df)-5):
        recent_high = swing_high.iloc[i-20:i-5].max()
        recent_low = swing_low.iloc[i-20:i-5].min()
        
        # Check last 5 candles for sweep
        last_5_low = df['low'].iloc[i-5:i].min()
        last_5_high = df['high'].iloc[i-5:i].max()
        current_close = df['close'].iloc[i]
        
        # Bullish sweep (swept low then reversed up)
        if last_5_low < recent_low * 0.995:  # Broke below
            if current_close > last_5_low * 1.005:  # Reversed up
                sweep.iloc[i] = 1
        
        # Bearish sweep (swept high then reversed down)
        if last_5_high > recent_high * 1.005:  # Broke above
            if current_close < last_5_high * 0.995:  # Reversed down
                sweep.iloc[i] = 1
    
    return sweep


def calculate_reversal_prob(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Post-Sweep Reversal Probability (0-1)
    
    Based on momentum and volume after sweep
    """
    reversal_prob = pd.Series(0.5, index=df.index)
    
    sweep = detect_sweep(df)
    rsi = ta.momentum.rsi(df['close'], window=14)
    volume_ma = df['volume'].rolling(20).mean()
    
    for i in range(20, len(df)):
        if sweep.iloc[i] == 1:
            prob = 0.5
            
            # Volume confirmation
            if df['volume'].iloc[i-3:i].mean() > volume_ma.iloc[i] * 1.3:
                prob += 0.15
            
            # RSI extreme
            if rsi.iloc[i] < 35 or rsi.iloc[i] > 65:
                prob += 0.15
            
            # Price action (strong reversal candle)
            if abs(df['close'].iloc[i] - df['open'].iloc[i]) > abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]):
                prob += 0.10
            
            reversal_prob.iloc[i] = min(prob, 1.0)
    
    return reversal_prob


def calculate_institutional_activity(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Institutional Activity Score (0-100)
    
    Based on volume spikes, rejection wicks, consolidation
    """
    score = pd.Series(0.0, index=df.index)
    
    volume_ma = df['volume'].rolling(20).mean()
    
    for i in range(20, len(df)):
        inst_score = 0.0
        
        # 1. Volume spikes (30 points)
        recent_volume = df['volume'].iloc[i-10:i]
        volume_spikes = (recent_volume > volume_ma.iloc[i-10:i] * 2).sum()
        inst_score += min(volume_spikes / 10 * 30, 30)
        
        # 2. Rejection wicks (30 points)
        for j in range(i-10, i):
            upper_wick = df['high'].iloc[j] - max(df['open'].iloc[j], df['close'].iloc[j])
            lower_wick = min(df['open'].iloc[j], df['close'].iloc[j]) - df['low'].iloc[j]
            body = abs(df['close'].iloc[j] - df['open'].iloc[j])
            
            if body > 0 and (upper_wick > body * 2 or lower_wick > body * 2):
                inst_score += 3
        
        inst_score = min(inst_score, 60)
        
        # 3. Consolidation after move (40 points)
        recent_range = df['high'].iloc[i-5:i].max() - df['low'].iloc[i-5:i].min()
        prev_range = df['high'].iloc[i-15:i-5].max() - df['low'].iloc[i-15:i-5].min()
        
        if prev_range > 0:
            consolidation = 1 - (recent_range / prev_range)
            inst_score += max(0, consolidation * 40)
        
        score.iloc[i] = min(inst_score, 100)
    
    return score.fillna(0)


def calculate_accumulation(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Accumulation Phase (0-100)
    
    Smart money buying quietly
    """
    accumulation = pd.Series(0.0, index=df.index)
    
    volume_ma = df['volume'].rolling(30).mean()
    
    for i in range(30, len(df)):
        score = 0.0
        
        # 1. Price range contraction (30 points)
        recent_range = (df['high'].iloc[i-10:i] - df['low'].iloc[i-10:i]).mean()
        prev_range = (df['high'].iloc[i-30:i-10] - df['low'].iloc[i-30:i-10]).mean()
        
        if prev_range > 0:
            contraction = 1 - (recent_range / prev_range)
            score += max(0, contraction * 30)
        
        # 2. Volume increase (30 points)
        recent_vol = df['volume'].iloc[i-10:i].mean()
        prev_vol = df['volume'].iloc[i-30:i-10].mean()
        
        if prev_vol > 0:
            vol_increase = (recent_vol / prev_vol - 1)
            score += max(0, min(vol_increase * 30, 30))
        
        # 3. Higher lows (40 points)
        lows = df['low'].iloc[i-10:i].values
        higher_lows = sum(1 for j in range(1, len(lows)) if lows[j] > lows[j-1])
        score += (higher_lows / len(lows)) * 40
        
        accumulation.iloc[i] = min(score, 100)
    
    return accumulation.fillna(0)


def calculate_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Distribution Phase (0-100)
    
    Smart money selling quietly
    """
    distribution = pd.Series(0.0, index=df.index)
    
    for i in range(30, len(df)):
        score = 0.0
        
        # 1. Price at high with contraction (30 points)
        recent_high = df['high'].iloc[i-10:i].max()
        period_high = df['high'].iloc[i-30:i].max()
        
        if recent_high >= period_high * 0.98:  # Within 2% of high
            recent_range = (df['high'].iloc[i-10:i] - df['low'].iloc[i-10:i]).mean()
            prev_range = (df['high'].iloc[i-30:i-10] - df['low'].iloc[i-30:i-10]).mean()
            
            if prev_range > 0:
                contraction = 1 - (recent_range / prev_range)
                score += max(0, contraction * 30)
        
        # 2. Volume increase (30 points)
        recent_vol = df['volume'].iloc[i-10:i].mean()
        prev_vol = df['volume'].iloc[i-30:i-10].mean()
        
        if prev_vol > 0:
            vol_increase = (recent_vol / prev_vol - 1)
            score += max(0, min(vol_increase * 30, 30))
        
        # 3. Lower highs (40 points)
        highs = df['high'].iloc[i-10:i].values
        lower_highs = sum(1 for j in range(1, len(highs)) if highs[j] < highs[j-1])
        score += (lower_highs / len(highs)) * 40
        
        distribution.iloc[i] = min(score, 100)
    
    return distribution.fillna(0)


def calculate_liquidity(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Liquidity Level (0-100)
    
    Based on proximity to key levels
    """
    liquidity = pd.Series(50.0, index=df.index)
    
    for i in range(20, len(df)):
        score = 0.0
        current_price = df['close'].iloc[i]
        
        # 1. Round numbers (30 points)
        for multiplier in [100, 500, 1000]:
            nearest_round = round(current_price / multiplier) * multiplier
            distance_pct = abs(current_price - nearest_round) / current_price * 100
            
            if distance_pct < 1:  # Within 1%
                score += 30
                break
        
        # 2. Swing levels (40 points)
        swing_high = df['high'].iloc[i-20:i].max()
        swing_low = df['low'].iloc[i-20:i].min()
        
        dist_to_high = abs(current_price - swing_high) / current_price * 100
        dist_to_low = abs(current_price - swing_low) / current_price * 100
        
        if min(dist_to_high, dist_to_low) < 2:
            score += 40
        
        # 3. Previous day levels (30 points)
        if i >= 1:
            prev_high = df['high'].iloc[i-1]
            prev_low = df['low'].iloc[i-1]
            
            dist_prev = min(
                abs(current_price - prev_high) / current_price * 100,
                abs(current_price - prev_low) / current_price * 100
            )
            
            if dist_prev < 1:
                score += 30
        
        liquidity.iloc[i] = min(score, 100)
    
    return liquidity


# ============================================================================
# MAIN FEATURE CALCULATOR
# ============================================================================

def calculate_all_70_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all 70 features from OHLCV data
    
    Input: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
    Output: DataFrame with 70 additional feature columns
    
    Returns:
        DataFrame with original data + 70 feature columns
    """
    
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # ========================================
    # CATEGORY 1: TECHNICAL ANALYSIS (10)
    # ========================================
    
    # 1. RSI
    result_df['RSI'] = ta.momentum.rsi(result_df['close'], window=14)
    
    # 2-3. MACD
    macd = ta.trend.MACD(result_df['close'])
    result_df['MACD'] = macd.macd()
    result_df['MACD_Signal'] = macd.macd_signal()
    
    # 4-6. Bollinger Bands
    bb = ta.volatility.BollingerBands(result_df['close'], window=20, window_dev=2)
    result_df['BB_Upper'] = bb.bollinger_hband()
    result_df['BB_Middle'] = bb.bollinger_mavg()
    result_df['BB_Lower'] = bb.bollinger_lband()
    
    # 7-8. Stochastic
    stoch = ta.momentum.StochasticOscillator(result_df['high'], result_df['low'], result_df['close'])
    result_df['Stochastic_K'] = stoch.stoch()
    result_df['Stochastic_D'] = stoch.stoch_signal()
    
    # 9. ATR
    result_df['ATR'] = ta.volatility.average_true_range(result_df['high'], result_df['low'], result_df['close'], window=14)
    
    # 10. CCI
    result_df['CCI'] = ta.trend.cci(result_df['high'], result_df['low'], result_df['close'], window=20)
    
    # ========================================
    # CATEGORY 2: PRICE ACTION (10)
    # ========================================
    
    # 11-12. Support/Resistance
    result_df['Support_Level'] = result_df['low'].rolling(window=20).min()
    result_df['Resistance_Level'] = result_df['high'].rolling(window=20).max()
    
    # 13-15. Fibonacci Levels
    swing_high = result_df['high'].rolling(window=20).max()
    swing_low = result_df['low'].rolling(window=20).min()
    diff = swing_high - swing_low
    result_df['Fib_382'] = swing_high - (diff * 0.382)
    result_df['Fib_500'] = swing_high - (diff * 0.500)
    result_df['Fib_618'] = swing_high - (diff * 0.618)
    
    # 16. Pivot Point
    result_df['Pivot_Point'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3
    
    # 17-18. Higher High / Lower Low
    result_df['Higher_High'] = (result_df['high'] > result_df['high'].shift(1)).astype(int)
    result_df['Lower_Low'] = (result_df['low'] < result_df['low'].shift(1)).astype(int)
    
    # 19. Consolidation Score
    result_df['Consolidation_Score'] = (1 - (result_df['ATR'] / result_df['close'])) * 100
    result_df['Consolidation_Score'] = result_df['Consolidation_Score'].clip(0, 100)
    
    # 20. Gap Detection
    result_df['Gap_Detected'] = ((result_df['open'] > result_df['high'].shift(1)) | 
                                  (result_df['open'] < result_df['low'].shift(1))).astype(int)
    
    # ========================================
    # CATEGORY 3: VOLUME & MOMENTUM (10)
    # ========================================
    
    # 21-22. Volume
    result_df['Volume_MA'] = result_df['volume'].rolling(window=20).mean()
    result_df['Volume_Spike'] = (result_df['volume'] > result_df['Volume_MA'] * 1.5).astype(int)
    
    # 23. OBV
    result_df['OBV'] = ta.volume.on_balance_volume(result_df['close'], result_df['volume'])
    
    # 24. AD Line
    result_df['AD_Line'] = ta.volume.acc_dist_index(result_df['high'], result_df['low'], result_df['close'], result_df['volume'])
    
    # 25. MFI
    result_df['MFI'] = ta.volume.money_flow_index(result_df['high'], result_df['low'], result_df['close'], result_df['volume'], window=14)
    
    # 26. CMF
    result_df['CMF'] = ta.volume.chaikin_money_flow(result_df['high'], result_df['low'], result_df['close'], result_df['volume'], window=20)
    
    # 27. ATR Ratio
    result_df['ATR_Ratio'] = result_df['ATR'] / result_df['close']
    
    # 28. ADX
    result_df['ADX'] = ta.trend.adx(result_df['high'], result_df['low'], result_df['close'], window=14)
    
    # 29-30. DI
    result_df['Plus_DI'] = ta.trend.adx_pos(result_df['high'], result_df['low'], result_df['close'])
    result_df['Minus_DI'] = ta.trend.adx_neg(result_df['high'], result_df['low'], result_df['close'])
    
    # ========================================
    # CATEGORY 4: SMART MONEY CONCEPTS (10)
    # ========================================
    
    # 31. Order Block Strength
    result_df['OB_Strength'] = calculate_ob_strength(result_df)
    
    # 32. Order Block Distance %
    result_df['OB_Distance_Pct'] = ((result_df['close'] - result_df['Support_Level']) / result_df['close']) * 100
    result_df['OB_Distance_Pct'] = result_df['OB_Distance_Pct'].fillna(0).clip(0, 100)
    
    # 33. FVG Distance
    result_df['FVG_Distance'] = calculate_fvg_distance(result_df)
    
    # 34. FVG Volume Ratio
    result_df['FVG_Volume_Ratio'] = result_df['volume'] / result_df['Volume_MA']
    result_df['FVG_Volume_Ratio'] = result_df['FVG_Volume_Ratio'].fillna(1.0)
    
    # 35. Sweep Detection
    result_df['Sweep_Detected'] = detect_sweep(result_df)
    
    # 36. Post-Sweep Reversal Probability
    result_df['Post_Sweep_Reversal_Prob'] = calculate_reversal_prob(result_df)
    
    # 37. Institutional Activity Score
    result_df['Inst_Activity_Score'] = calculate_institutional_activity(result_df)
    
    # 38. Accumulation Phase
    result_df['Accumulation_Phase'] = calculate_accumulation(result_df)
    
    # 39. Distribution Phase
    result_df['Distribution_Phase'] = calculate_distribution(result_df)
    
    # 40. Liquidity Level
    result_df['Liquidity_Level'] = calculate_liquidity(result_df)
    
    # ========================================
    # CATEGORY 5: MULTI-TIMEFRAME (10)
    # ========================================
    
    # 41-42. Trends
    result_df['Weekly_Trend'] = (result_df['close'].rolling(5).mean() > result_df['close'].rolling(10).mean()).astype(int)
    result_df['Daily_Trend'] = (result_df['close'] > result_df['close'].rolling(5).mean()).astype(int)
    
    # 43. Hourly Momentum (using RSI as proxy)
    result_df['Hourly_Momentum'] = result_df['RSI']
    
    # 44. MTF Confluence
    result_df['MTF_Confluence'] = ((result_df['Weekly_Trend'] + result_df['Daily_Trend']) / 2) * 100
    
    # 45. Volume Confirmation
    result_df['Volume_Conf_Weekly'] = (result_df['volume'] > result_df['volume'].rolling(5).mean()).astype(int)
    
    # 46. RSI Divergence
    result_df['RSI_Divergence'] = ((result_df['RSI'] < 30) & (result_df['close'] < result_df['close'].shift(1))).astype(int)
    
    # 47. MA Alignment
    ma_50 = result_df['close'].rolling(50).mean()
    ma_200 = result_df['close'].rolling(200).mean()
    result_df['MA_Alignment'] = (ma_50 > ma_200).astype(int) * 100
    
    # 48. Volatility Regime
    result_df['Volatility_Regime'] = pd.cut(result_df['ATR_Ratio'], bins=[0, 0.01, 0.02, 0.03, 1.0], labels=[0, 1, 2, 3])
    result_df['Volatility_Regime'] = result_df['Volatility_Regime'].astype(float).fillna(1)
    
    # 49-50. MTF Scores
    result_df['Inst_Activity_MTF'] = result_df['Inst_Activity_Score'].rolling(5).mean().fillna(result_df['Inst_Activity_Score'])
    result_df['Reversal_Prob_MTF'] = result_df['Post_Sweep_Reversal_Prob'].rolling(5).mean().fillna(result_df['Post_Sweep_Reversal_Prob'])
    
    # ========================================
    # CATEGORY 6: MARKET MICROSTRUCTURE (10)
    # ========================================
    
    # 51. Order Flow Imbalance
    result_df['Order_Flow_Imbalance'] = ((result_df['close'] - result_df['open']) / 
                                          (result_df['high'] - result_df['low'])).fillna(0) * 100
    result_df['Order_Flow_Imbalance'] = result_df['Order_Flow_Imbalance'].clip(-100, 100)
    
    # 52. Momentum Composite
    result_df['Momentum_Composite'] = (result_df['RSI'] + result_df['ADX'] + (result_df['MACD'] * 10)) / 3
    result_df['Momentum_Composite'] = result_df['Momentum_Composite'].clip(0, 100)
    
    # 53. Reversal Probability
    result_df['Reversal_Probability'] = (
        ((result_df['RSI'] < 30) | (result_df['RSI'] > 70)).astype(int) * 50 +
        (result_df['Stochastic_K'] < 20).astype(int) * 30 +
        (result_df['BB_Lower'] > result_df['close']).astype(int) * 20
    )
    result_df['Reversal_Probability'] = result_df['Reversal_Probability'].clip(0, 100)
    
    # 54. Trend Continuation Probability
    result_df['Trend_Continuation_Prob'] = (
        (result_df['ADX'] > 25).astype(int) * 40 +
        (result_df['Daily_Trend'] == 1).astype(int) * 30 +
        (result_df['MA_Alignment'] == 100).astype(int) * 30
    )
    
    # 55. Support/Resistance Strength
    support_touches = (result_df['low'] <= result_df['Support_Level'] * 1.01).rolling(20).sum()
    result_df['Support_Resistance_Strength'] = (support_touches * 5).clip(0, 100)
    
    # 56. Volatility Squeeze
    result_df['Volatility_Squeeze'] = ((result_df['BB_Upper'] - result_df['BB_Lower']) / result_df['BB_Middle']) * 100
    result_df['Volatility_Squeeze'] = result_df['Volatility_Squeeze'].clip(0, 100)
    
    # 57. Mean Reversion Signal
    result_df['Mean_Reversion_Signal'] = (
        ((result_df['close'] < result_df['BB_Lower']).astype(int) * 50) +
        ((result_df['RSI'] < 30).astype(int) * 50)
    )
    
    # 58-59. Institutional Presence & Liquidity
    result_df['Institutional_Presence'] = result_df['Inst_Activity_Score']
    result_df['Liquidity_Assessment'] = result_df['Liquidity_Level']
    
    # 60. Market Regime
    def categorize_regime(row):
        if row['Weekly_Trend'] == 1 and row['Daily_Trend'] == 1 and row['RSI'] > 50:
            return 0  # BULLISH
        elif row['Weekly_Trend'] == 0 and row['Daily_Trend'] == 0 and row['RSI'] < 50:
            return 1  # BEARISH
        elif row['ATR_Ratio'] > 0.03:
            return 3  # CHOPPY
        else:
            return 2  # RANGE
    
    result_df['Market_Regime'] = result_df.apply(categorize_regime, axis=1)
    
    # ========================================
    # CATEGORY 7: ADVANCED TECHNICAL (10)
    # ========================================
    
    # 61. ROC
    result_df['ROC'] = ta.momentum.roc(result_df['close'], window=12)
    
    # 62. Williams %R
    result_df['Williams_R'] = ta.momentum.williams_r(result_df['high'], result_df['low'], result_df['close'], lbp=14)
    
    # 63-64. EMAs
    result_df['EMA_50'] = ta.trend.ema_indicator(result_df['close'], window=50)
    result_df['EMA_200'] = ta.trend.ema_indicator(result_df['close'], window=200)
    
    # 65. VWAP
    result_df['VWAP'] = (result_df['volume'] * (result_df['high'] + result_df['low'] + result_df['close']) / 3).cumsum() / result_df['volume'].cumsum()
    
    # 66. TSI
    result_df['TSI'] = ta.momentum.tsi(result_df['close'])
    
    # 67. Awesome Oscillator
    result_df['Awesome_Oscillator'] = ta.momentum.awesome_oscillator(result_df['high'], result_df['low'])
    
    # 68. Parabolic SAR
    psar = ta.trend.PSARIndicator(result_df['high'], result_df['low'], result_df['close'])
    result_df['Parabolic_SAR'] = psar.psar()
    
    # 69-70. Keltner Channels
    kc = ta.volatility.KeltnerChannel(result_df['high'], result_df['low'], result_df['close'])
    result_df['Keltner_Upper'] = kc.keltner_channel_hband()
    result_df['Keltner_Lower'] = kc.keltner_channel_lband()
    
    # Fill NaN values with forward fill, then backward fill, then 0
    result_df = result_df.ffill().bfill().fillna(0)
    
    return result_df


def get_feature_vector(df: pd.DataFrame, index: int = -1) -> Dict[str, float]:
    """
    Extract a single feature vector (70 features) from a row
    
    Args:
        df: DataFrame with calculated features
        index: Row index (default: -1 for last row)
        
    Returns:
        Dictionary with 70 features
    """
    feature_names = [
        # Technical (10)
        'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'Stochastic_K', 'Stochastic_D', 'ATR', 'CCI',
        
        # Price Action (10)
        'Support_Level', 'Resistance_Level', 'Fib_382', 'Fib_500', 'Fib_618',
        'Pivot_Point', 'Higher_High', 'Lower_Low', 'Consolidation_Score', 'Gap_Detected',
        
        # Volume & Momentum (10)
        'Volume_MA', 'Volume_Spike', 'OBV', 'AD_Line', 'MFI', 'CMF',
        'ATR_Ratio', 'ADX', 'Plus_DI', 'Minus_DI',
        
        # SMC (10)
        'OB_Strength', 'OB_Distance_Pct', 'FVG_Distance', 'FVG_Volume_Ratio',
        'Sweep_Detected', 'Post_Sweep_Reversal_Prob', 'Inst_Activity_Score',
        'Accumulation_Phase', 'Distribution_Phase', 'Liquidity_Level',
        
        # MTF (10)
        'Weekly_Trend', 'Daily_Trend', 'Hourly_Momentum', 'MTF_Confluence',
        'Volume_Conf_Weekly', 'RSI_Divergence', 'MA_Alignment', 'Volatility_Regime',
        'Inst_Activity_MTF', 'Reversal_Prob_MTF',
        
        # Microstructure (10)
        'Order_Flow_Imbalance', 'Momentum_Composite', 'Reversal_Probability',
        'Trend_Continuation_Prob', 'Support_Resistance_Strength', 'Volatility_Squeeze',
        'Mean_Reversion_Signal', 'Institutional_Presence', 'Liquidity_Assessment',
        'Market_Regime',
        
        # Advanced (10)
        'ROC', 'Williams_R', 'EMA_50', 'EMA_200', 'VWAP', 'TSI',
        'Awesome_Oscillator', 'Parabolic_SAR', 'Keltner_Upper', 'Keltner_Lower'
    ]
    
    feature_vector = {}
    for feature in feature_names:
        feature_vector[feature] = float(df[feature].iloc[index])
    
    return feature_vector


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import yfinance as yf
    
    print("\n" + "="*80)
    print("70-FEATURE CALCULATOR TEST (EXACT SPECIFICATION)")
    print("="*80)
    
    # Fetch data
    ticker = yf.Ticker("RELIANCE.NS")
    df = ticker.history(period="6mo", interval="1d")
    
    # Reset index to have 'date' column
    df = df.reset_index()
    
    # Rename columns to lowercase
    df.columns = df.columns.str.lower()
    
    print(f"\nFetched {len(df)} days of data for RELIANCE")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Calculate all 70 features
    print("\nCalculating 70 features...")
    df_with_features = calculate_all_70_features(df)
    
    # Get feature vector for latest candle
    feature_vector = get_feature_vector(df_with_features, index=-1)
    
    print(f"\n✅ Successfully calculated {len(feature_vector)} features!")
    print("\nLatest Feature Vector:")
    print("-" * 80)
    
    # Display by category
    categories = {
        'Technical Analysis': list(feature_vector.keys())[0:10],
        'Price Action': list(feature_vector.keys())[10:20],
        'Volume & Momentum': list(feature_vector.keys())[20:30],
        'Smart Money Concepts': list(feature_vector.keys())[30:40],
        'Multi-Timeframe': list(feature_vector.keys())[40:50],
        'Market Microstructure': list(feature_vector.keys())[50:60],
        'Advanced Technical': list(feature_vector.keys())[60:70]
    }
    
    for category, features in categories.items():
        print(f"\n{category}:")
        for feature in features:
            value = feature_vector[feature]
            print(f"  {feature:30s}: {value:12.4f}")
    
    print("\n" + "="*80)
    print(f"TOTAL FEATURES: {len(feature_vector)}")
    print("="*80)
    
    # Verify all features present
    assert len(feature_vector) == 70, f"Expected 70 features, got {len(feature_vector)}"
    print("\n✅ All 70 features verified!")
