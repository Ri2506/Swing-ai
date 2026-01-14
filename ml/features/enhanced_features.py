"""
================================================================================
SWINGAI - ENHANCED FEATURE ENGINEERING (70 FEATURES)
================================================================================
Complete feature set for AI models:
- Technical Analysis (10)
- Price Action (10)
- Volume & Momentum (10)
- SMC Features (10) ⭐ NEW
- Multi-Timeframe (10)
- Market Microstructure (10)
- Market Context (10)
================================================================================
"""

import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Tuple, Optional
import logging
from .smc_features import SMCFeatureCalculator

logger = logging.getLogger(__name__)


class EnhancedFeatureEngine:
    """
    Complete 70-feature calculation engine
    """
    
    def __init__(self):
        self.smc_calculator = SMCFeatureCalculator()
    
    def calculate_all_features(
        self, 
        df_daily: pd.DataFrame,
        df_hourly: Optional[pd.DataFrame] = None,
        df_weekly: Optional[pd.DataFrame] = None,
        market_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Calculate all 70 features
        
        Args:
            df_daily: Daily OHLCV data
            df_hourly: Hourly data for MTF analysis
            df_weekly: Weekly data for MTF analysis
            market_data: Market-wide data (Nifty, VIX, FII/DII)
            
        Returns:
            Dictionary with 70 features
        """
        features = {}
        
        # Validate input
        if df_daily is None or len(df_daily) < 50:
            logger.warning("Insufficient data for feature calculation")
            return self._get_default_features()
        
        # 1. Technical Analysis (10 features)
        features.update(self._calculate_technical_features(df_daily))
        
        # 2. Price Action (10 features)
        features.update(self._calculate_price_action_features(df_daily))
        
        # 3. Volume & Momentum (10 features)
        features.update(self._calculate_volume_momentum_features(df_daily))
        
        # 4. ⭐ SMC Features (10 features) - NEW!
        features.update(self.smc_calculator.calculate_all_features(df_daily))
        
        # 5. Multi-Timeframe (10 features)
        features.update(self._calculate_mtf_features(df_daily, df_hourly, df_weekly))
        
        # 6. Market Microstructure (10 features)
        features.update(self._calculate_microstructure_features(df_daily))
        
        # 7. Market Context (10 features)
        features.update(self._calculate_market_context_features(df_daily, market_data))
        
        return features
    
    def _calculate_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate 10 technical analysis features
        """
        features = {}
        
        try:
            # 1. RSI (14)
            rsi = ta.momentum.RSIIndicator(df['close'], window=14)
            features['rsi_14'] = rsi.rsi().iloc[-1]
            
            # 2. MACD
            macd = ta.trend.MACD(df['close'])
            features['macd_value'] = macd.macd().iloc[-1]
            features['macd_signal'] = macd.macd_signal().iloc[-1]
            features['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # 3. Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            features['bb_percentage'] = bb.bollinger_pband().iloc[-1]  # Position in bands
            
            # 4. Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            features['stoch_k'] = stoch.stoch().iloc[-1]
            features['stoch_d'] = stoch.stoch_signal().iloc[-1]
            
            # 5. ATR (14) - Normalized
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            features['atr_percentage'] = (atr.average_true_range().iloc[-1] / df['close'].iloc[-1]) * 100
            
            # 6. ADX (Trend Strength)
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            features['adx'] = adx.adx().iloc[-1]
            
            # 7. CCI (Commodity Channel Index)
            cci = ta.trend.CCIIndicator(df['high'], df['low'], df['close'])
            features['cci'] = cci.cci().iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating technical features: {e}")
            features = {
                'rsi_14': 50, 'macd_value': 0, 'macd_signal': 0, 'macd_histogram': 0,
                'bb_percentage': 0.5, 'stoch_k': 50, 'stoch_d': 50, 
                'atr_percentage': 2, 'adx': 25, 'cci': 0
            }
        
        return features
    
    def _calculate_price_action_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate 10 price action features
        """
        features = {}
        
        try:
            current_price = df['close'].iloc[-1]
            
            # 1-2. Support & Resistance (pivot points)
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-2]  # Previous close
            
            pivot = (high + low + close) / 3
            resistance_1 = 2 * pivot - low
            support_1 = 2 * pivot - high
            
            features['support_distance'] = ((current_price - support_1) / current_price) * 100
            features['resistance_distance'] = ((resistance_1 - current_price) / current_price) * 100
            
            # 3-5. Fibonacci Retracement
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            price_range = recent_high - recent_low
            
            if price_range > 0:
                fib_382 = recent_high - (price_range * 0.382)
                fib_500 = recent_high - (price_range * 0.500)
                fib_618 = recent_high - (price_range * 0.618)
                
                # Distance to nearest fib level
                distances = [
                    abs(current_price - fib_382),
                    abs(current_price - fib_500),
                    abs(current_price - fib_618)
                ]
                features['fib_distance'] = (min(distances) / current_price) * 100
                
                # Position relative to range
                features['range_position'] = ((current_price - recent_low) / price_range) * 100
            else:
                features['fib_distance'] = 50
                features['range_position'] = 50
            
            # 6. Trend Direction
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            
            if current_price > sma_20 > sma_50:
                features['trend_direction'] = 100  # Strong uptrend
            elif current_price < sma_20 < sma_50:
                features['trend_direction'] = 0    # Strong downtrend
            else:
                features['trend_direction'] = 50   # Sideways
            
            # 7. Price Momentum (Rate of Change)
            roc_10 = ((current_price - df['close'].iloc[-10]) / df['close'].iloc[-10]) * 100
            features['momentum_10d'] = roc_10
            
            # 8. Candle Pattern Strength
            body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            total_range = df['high'].iloc[-1] - df['low'].iloc[-1]
            features['candle_strength'] = (body / total_range * 100) if total_range > 0 else 50
            
            # 9. Gap from previous close
            prev_close = df['close'].iloc[-2]
            features['gap_percentage'] = ((current_price - prev_close) / prev_close) * 100
            
            # 10. Higher Highs / Lower Lows pattern
            highs = df['high'].iloc[-10:].values
            lows = df['low'].iloc[-10:].values
            
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            
            features['hh_ll_score'] = ((higher_highs - lower_lows) / len(highs)) * 100
            
        except Exception as e:
            logger.error(f"Error calculating price action features: {e}")
            features = {
                'support_distance': 2, 'resistance_distance': 2, 'fib_distance': 2,
                'range_position': 50, 'trend_direction': 50, 'momentum_10d': 0,
                'candle_strength': 50, 'gap_percentage': 0, 'hh_ll_score': 0
            }
        
        return features
    
    def _calculate_volume_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate 10 volume & momentum features
        """
        features = {}
        
        try:
            # 1. Volume MA Ratio
            volume_ma_20 = df['volume'].rolling(20).mean()
            features['volume_ma_ratio'] = df['volume'].iloc[-1] / volume_ma_20.iloc[-1] if volume_ma_20.iloc[-1] > 0 else 1.0
            
            # 2. On Balance Volume (OBV)
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            obv_values = obv.on_balance_volume()
            features['obv_slope'] = (obv_values.iloc[-1] - obv_values.iloc[-10]) / 10  # Trend
            
            # 3. Money Flow Index (MFI)
            mfi = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'])
            features['mfi'] = mfi.money_flow_index().iloc[-1]
            
            # 4. Force Index
            force = ta.volume.ForceIndexIndicator(df['close'], df['volume'], window=13)
            features['force_index'] = force.force_index().iloc[-1]
            
            # 5. Volume Price Trend (VPT)
            vpt = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume'])
            features['vpt'] = vpt.volume_price_trend().iloc[-1]
            
            # 6. Accumulation/Distribution Index
            ad = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume'])
            ad_values = ad.acc_dist_index()
            features['ad_slope'] = (ad_values.iloc[-1] - ad_values.iloc[-10]) / 10
            
            # 7. Chaikin Money Flow
            cmf = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'])
            features['cmf'] = cmf.chaikin_money_flow().iloc[-1]
            
            # 8. Volume Weighted Average Price (VWAP) Distance
            vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            features['vwap_distance'] = ((df['close'].iloc[-1] - vwap.iloc[-1]) / df['close'].iloc[-1]) * 100
            
            # 9. Ease of Movement
            eom = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume'])
            features['eom'] = eom.ease_of_movement().iloc[-1]
            
            # 10. Volume Oscillator
            vol_fast = df['volume'].rolling(5).mean()
            vol_slow = df['volume'].rolling(20).mean()
            features['volume_oscillator'] = ((vol_fast.iloc[-1] - vol_slow.iloc[-1]) / vol_slow.iloc[-1]) * 100 if vol_slow.iloc[-1] > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating volume/momentum features: {e}")
            features = {
                'volume_ma_ratio': 1.0, 'obv_slope': 0, 'mfi': 50, 'force_index': 0,
                'vpt': 0, 'ad_slope': 0, 'cmf': 0, 'vwap_distance': 0,
                'eom': 0, 'volume_oscillator': 0
            }
        
        return features
    
    def _calculate_mtf_features(
        self, 
        df_daily: pd.DataFrame,
        df_hourly: Optional[pd.DataFrame],
        df_weekly: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calculate 10 multi-timeframe features
        """
        features = {}
        
        try:
            # Daily timeframe
            daily_rsi = ta.momentum.RSIIndicator(df_daily['close']).rsi().iloc[-1]
            daily_macd = ta.trend.MACD(df_daily['close']).macd_diff().iloc[-1]
            
            features['daily_rsi'] = daily_rsi
            features['daily_macd'] = 1 if daily_macd > 0 else 0
            
            # Hourly timeframe (if available)
            if df_hourly is not None and len(df_hourly) >= 50:
                hourly_rsi = ta.momentum.RSIIndicator(df_hourly['close']).rsi().iloc[-1]
                hourly_macd = ta.trend.MACD(df_hourly['close']).macd_diff().iloc[-1]
                
                features['hourly_rsi'] = hourly_rsi
                features['hourly_macd'] = 1 if hourly_macd > 0 else 0
                
                # Alignment score
                features['hourly_daily_alignment'] = 100 if (daily_macd > 0) == (hourly_macd > 0) else 0
            else:
                features['hourly_rsi'] = daily_rsi
                features['hourly_macd'] = features['daily_macd']
                features['hourly_daily_alignment'] = 100
            
            # Weekly timeframe (if available)
            if df_weekly is not None and len(df_weekly) >= 50:
                weekly_rsi = ta.momentum.RSIIndicator(df_weekly['close']).rsi().iloc[-1]
                weekly_macd = ta.trend.MACD(df_weekly['close']).macd_diff().iloc[-1]
                
                features['weekly_rsi'] = weekly_rsi
                features['weekly_macd'] = 1 if weekly_macd > 0 else 0
                
                # Weekly trend
                weekly_sma_20 = df_weekly['close'].rolling(20).mean().iloc[-1]
                features['weekly_trend'] = 100 if df_weekly['close'].iloc[-1] > weekly_sma_20 else 0
            else:
                features['weekly_rsi'] = daily_rsi
                features['weekly_macd'] = features['daily_macd']
                features['weekly_trend'] = 50
            
            # MTF Confluence Score (0-100)
            mtf_signals = [
                features['daily_macd'],
                features['hourly_macd'],
                features['weekly_macd']
            ]
            features['mtf_confluence'] = (sum(mtf_signals) / 3) * 100
            
            # Trend strength across timeframes
            rsi_avg = (daily_rsi + features['hourly_rsi'] + features['weekly_rsi']) / 3
            features['mtf_momentum'] = rsi_avg
            
        except Exception as e:
            logger.error(f"Error calculating MTF features: {e}")
            features = {
                'daily_rsi': 50, 'daily_macd': 0, 'hourly_rsi': 50, 'hourly_macd': 0,
                'hourly_daily_alignment': 50, 'weekly_rsi': 50, 'weekly_macd': 0,
                'weekly_trend': 50, 'mtf_confluence': 50, 'mtf_momentum': 50
            }
        
        return features
    
    def _calculate_microstructure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate 10 market microstructure features
        """
        features = {}
        
        try:
            # 1. Bid-Ask Spread Proxy (High-Low range as %)
            spread = ((df['high'] - df['low']) / df['close']) * 100
            features['avg_spread'] = spread.iloc[-10:].mean()
            
            # 2. Price Impact (Volume vs Price change correlation)
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            features['price_impact'] = price_change.iloc[-20:].corr(volume_change.iloc[-20:])
            
            # 3. Order Flow Imbalance Proxy
            up_volume = df.loc[df['close'] > df['open'], 'volume'].sum()
            down_volume = df.loc[df['close'] < df['open'], 'volume'].sum()
            total_volume = up_volume + down_volume
            features['order_flow_imbalance'] = ((up_volume - down_volume) / total_volume) * 100 if total_volume > 0 else 0
            
            # 4. Tick Direction (last 10 candles)
            up_ticks = (df['close'].iloc[-10:] > df['close'].shift(1).iloc[-10:]).sum()
            features['tick_direction'] = (up_ticks / 10) * 100
            
            # 5. Volatility Clustering
            returns = df['close'].pct_change()
            volatility = returns.rolling(10).std()
            features['volatility_clustering'] = volatility.iloc[-1] / volatility.iloc[-30:].mean() if volatility.iloc[-30:].mean() > 0 else 1.0
            
            # 6. Liquidity Depth Proxy (Volume stability)
            volume_cv = df['volume'].iloc[-20:].std() / df['volume'].iloc[-20:].mean() if df['volume'].iloc[-20:].mean() > 0 else 1.0
            features['liquidity_depth'] = 1 / volume_cv if volume_cv > 0 else 1.0
            
            # 7. Momentum Acceleration
            mom_5 = df['close'].iloc[-5:].pct_change().mean()
            mom_10 = df['close'].iloc[-10:-5].pct_change().mean()
            features['momentum_acceleration'] = ((mom_5 - mom_10) / abs(mom_10)) * 100 if mom_10 != 0 else 0
            
            # 8. Microstructure Noise (Price reversals)
            reversals = ((df['close'] > df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))).sum()
            features['microstructure_noise'] = (reversals / len(df)) * 100
            
            # 9. Effective Spread (Intraday volatility)
            intraday_range = ((df['high'] - df['low']) / df['open']) * 100
            features['effective_spread'] = intraday_range.iloc[-10:].mean()
            
            # 10. Trade Intensity (Volume per candle trend)
            volume_trend = np.polyfit(range(20), df['volume'].iloc[-20:].values, 1)[0]
            features['trade_intensity'] = volume_trend / df['volume'].iloc[-20:].mean() if df['volume'].iloc[-20:].mean() > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {e}")
            features = {
                'avg_spread': 2, 'price_impact': 0, 'order_flow_imbalance': 0,
                'tick_direction': 50, 'volatility_clustering': 1, 'liquidity_depth': 1,
                'momentum_acceleration': 0, 'microstructure_noise': 10,
                'effective_spread': 2, 'trade_intensity': 0
            }
        
        return features
    
    def _calculate_market_context_features(
        self, 
        df: pd.DataFrame,
        market_data: Optional[Dict]
    ) -> Dict[str, float]:
        """
        Calculate 10 market context features
        """
        features = {}
        
        try:
            # Use market_data if provided, otherwise use defaults
            if market_data:
                features['nifty_change'] = market_data.get('nifty_change_percent', 0)
                features['vix_level'] = market_data.get('vix_close', 15)
                features['fii_flow'] = market_data.get('fii_cash', 0)
                features['dii_flow'] = market_data.get('dii_cash', 0)
                features['advance_decline'] = market_data.get('advances', 0) - market_data.get('declines', 0)
            else:
                features['nifty_change'] = 0
                features['vix_level'] = 15
                features['fii_flow'] = 0
                features['dii_flow'] = 0
                features['advance_decline'] = 0
            
            # Stock-specific context
            # 6. Beta approximation (correlation with market)
            if 'nifty_close' in df.columns:
                features['beta'] = df['close'].pct_change().iloc[-50:].corr(
                    df['nifty_close'].pct_change().iloc[-50:]
                )
            else:
                features['beta'] = 1.0
            
            # 7. Relative Strength vs Market
            stock_return = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
            features['relative_strength'] = stock_return  # vs market is implicit
            
            # 8. Sector Momentum Proxy (volume-weighted performance)
            features['sector_momentum'] = stock_return * features['volume_ma_ratio']
            
            # 9. Market Regime (trending vs ranging)
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            range_size = ((df['high_20'].iloc[-1] - df['low_20'].iloc[-1]) / df['close'].iloc[-1]) * 100
            features['market_regime'] = 100 if range_size > 10 else 0  # 100=trending, 0=ranging
            
            # 10. Volatility Regime
            returns_vol = df['close'].pct_change().iloc[-20:].std() * 100
            features['volatility_regime'] = min(returns_vol * 10, 100)  # Normalize to 0-100
            
        except Exception as e:
            logger.error(f"Error calculating market context features: {e}")
            features = {
                'nifty_change': 0, 'vix_level': 15, 'fii_flow': 0, 'dii_flow': 0,
                'advance_decline': 0, 'beta': 1.0, 'relative_strength': 0,
                'sector_momentum': 0, 'market_regime': 50, 'volatility_regime': 50
            }
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when insufficient data"""
        features = {}
        
        # Technical (10)
        features.update({
            'rsi_14': 50, 'macd_value': 0, 'macd_signal': 0, 'macd_histogram': 0,
            'bb_percentage': 0.5, 'stoch_k': 50, 'stoch_d': 50,
            'atr_percentage': 2, 'adx': 25, 'cci': 0
        })
        
        # Price Action (10)
        features.update({
            'support_distance': 2, 'resistance_distance': 2, 'fib_distance': 2,
            'range_position': 50, 'trend_direction': 50, 'momentum_10d': 0,
            'candle_strength': 50, 'gap_percentage': 0, 'hh_ll_score': 0
        })
        
        # Volume & Momentum (10)
        features.update({
            'volume_ma_ratio': 1.0, 'obv_slope': 0, 'mfi': 50, 'force_index': 0,
            'vpt': 0, 'ad_slope': 0, 'cmf': 0, 'vwap_distance': 0,
            'eom': 0, 'volume_oscillator': 0
        })
        
        # SMC (10)
        features.update({
            'order_block_strength': 0, 'order_block_distance': 100, 'fvg_distance': 100,
            'fvg_volume_ratio': 1.0, 'sweep_detection': 0, 'post_sweep_reversal_prob': 50,
            'institutional_activity': 50, 'accumulation_phase': 0,
            'distribution_phase': 0, 'liquidity_level': 50
        })
        
        # MTF (10)
        features.update({
            'daily_rsi': 50, 'daily_macd': 0, 'hourly_rsi': 50, 'hourly_macd': 0,
            'hourly_daily_alignment': 50, 'weekly_rsi': 50, 'weekly_macd': 0,
            'weekly_trend': 50, 'mtf_confluence': 50, 'mtf_momentum': 50
        })
        
        # Microstructure (10)
        features.update({
            'avg_spread': 2, 'price_impact': 0, 'order_flow_imbalance': 0,
            'tick_direction': 50, 'volatility_clustering': 1, 'liquidity_depth': 1,
            'momentum_acceleration': 0, 'microstructure_noise': 10,
            'effective_spread': 2, 'trade_intensity': 0
        })
        
        # Market Context (10)
        features.update({
            'nifty_change': 0, 'vix_level': 15, 'fii_flow': 0, 'dii_flow': 0,
            'advance_decline': 0, 'beta': 1.0, 'relative_strength': 0,
            'sector_momentum': 0, 'market_regime': 50, 'volatility_regime': 50
        })
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of all 70 feature names"""
        return list(self._get_default_features().keys())


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import yfinance as yf
    
    # Fetch data
    ticker = yf.Ticker("RELIANCE.NS")
    df_daily = ticker.history(period="6mo", interval="1d")
    df_hourly = ticker.history(period="1mo", interval="1h")
    df_weekly = ticker.history(period="2y", interval="1wk")
    
    # Mock market data
    market_data = {
        'nifty_change_percent': 0.5,
        'vix_close': 14.5,
        'fii_cash': 1500,
        'dii_cash': 800,
        'advances': 1200,
        'declines': 800
    }
    
    # Calculate features
    engine = EnhancedFeatureEngine()
    features = engine.calculate_all_features(df_daily, df_hourly, df_weekly, market_data)
    
    print("\n" + "="*80)
    print("70 ENHANCED FEATURES FOR RELIANCE")
    print("="*80)
    
    # Group features by category
    categories = {
        'Technical': ['rsi', 'macd', 'bb', 'stoch', 'atr', 'adx', 'cci'],
        'Price Action': ['support', 'resistance', 'fib', 'range', 'trend', 'momentum', 'candle', 'gap', 'hh_ll'],
        'Volume': ['volume', 'obv', 'mfi', 'force', 'vpt', 'ad', 'cmf', 'vwap', 'eom'],
        'SMC': ['order_block', 'fvg', 'sweep', 'institutional', 'accumulation', 'distribution', 'liquidity'],
        'MTF': ['daily', 'hourly', 'weekly', 'mtf'],
        'Microstructure': ['spread', 'impact', 'flow', 'tick', 'volatility', 'liquidity', 'noise', 'intensity'],
        'Market Context': ['nifty', 'vix', 'fii', 'dii', 'advance', 'beta', 'relative', 'sector', 'regime']
    }
    
    for category, keywords in categories.items():
        print(f"\n{category}:")
        for key, value in features.items():
            if any(keyword in key.lower() for keyword in keywords):
                print(f"  {key:35s}: {value:10.2f}")
    
    print("\n" + "="*80)
    print(f"Total Features: {len(features)}")
    print("="*80)
