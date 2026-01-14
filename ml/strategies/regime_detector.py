"""
================================================================================
SWINGAI - MARKET REGIME DETECTOR
================================================================================
Detects 4 market regimes: BULLISH, BEARISH, RANGE, CHOPPY
Selects optimal strategies for each regime
================================================================================
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detect current market regime
    
    Regimes:
    - BULLISH: Strong uptrend (ADX >25, price above MAs)
    - BEARISH: Strong downtrend (ADX >25, price below MAs)
    - RANGE: Consolidation (ADX <20, tight range)
    - CHOPPY: High volatility, no direction
    """
    
    def __init__(self):
        self.regimes = ['BULLISH', 'BEARISH', 'RANGE', 'CHOPPY']
        
        # Strategy mapping for each regime
        self.regime_strategies = {
            'BULLISH': [
                'S1_7Element',
                'S3_BollingerRSI',
                'S4_GoldenPocket',
                'S5_CupHandle',
                'S7_MTFOrderFlow',
                'S8_WyckoffSMC',
                'S10_TripleMAVolume',
                'S14_MomentumBreakout'
            ],
            'BEARISH': [
                'S2_TripleRSI',
                'S3_BollingerRSI',
                'S6_DoubleBottomDiv',
                'S8_WyckoffSMC',
                'S9_LiquiditySweep',
                'S12_ICTKillzone',
                'S13_ThreeDrive',
                'S14_MomentumBreakout'
            ],
            'RANGE': [
                'S2_TripleRSI',
                'S3_BollingerRSI',
                'S4_GoldenPocket',
                'S11_SupplyDemand',
                'S13_ThreeDrive',
                'S15_TrendPullback',
                'S16_GapFill',
                'S18_PurePrice'
            ],
            'CHOPPY': [
                'S1_7Element',
                'S5_CupHandle',
                'S7_MTFOrderFlow',
                'S8_WyckoffSMC'
            ]
        }
    
    def detect_regime(
        self, 
        df: pd.DataFrame, 
        current_idx: int
    ) -> Tuple[str, float]:
        """
        Detect market regime using multiple factors
        
        Args:
            df: DataFrame with OHLCV + 70 features
            current_idx: Index of current candle
            
        Returns:
            Tuple of (regime_name, confidence_score)
            - regime_name: 'BULLISH', 'BEARISH', 'RANGE', or 'CHOPPY'
            - confidence_score: 0-100
        """
        # Need at least 50 candles for reliable detection
        if current_idx < 50:
            return 'RANGE', 50.0
        
        try:
            # Get current data
            weekly_trend = df['Weekly_Trend'].iloc[current_idx]
            daily_trend = df['Daily_Trend'].iloc[current_idx]
            rsi = df['RSI'].iloc[current_idx]
            adx = df['ADX'].iloc[current_idx]
            atr_ratio = df['ATR_Ratio'].iloc[current_idx]
            plus_di = df['Plus_DI'].iloc[current_idx]
            minus_di = df['Minus_DI'].iloc[current_idx]
            
            # Get price position vs moving averages
            current_price = df['close'].iloc[current_idx]
            
            # Calculate 200 SMA if we have enough data
            if current_idx >= 200:
                sma_200 = df['close'].iloc[current_idx-200:current_idx].mean()
            else:
                sma_200 = df['close'].iloc[:current_idx].mean()
            
            # Get 50 SMA
            if current_idx >= 50:
                sma_50 = df['close'].iloc[current_idx-50:current_idx].mean()
            else:
                sma_50 = df['close'].iloc[:current_idx].mean()
            
            # Decision logic with confidence
            
            # 1. Check for CHOPPY first (high priority)
            if atr_ratio > 0.03:
                confidence = min(85 + (atr_ratio - 0.03) * 1000, 95)
                return 'CHOPPY', confidence
            
            # 2. Check for BULLISH trend
            if (weekly_trend == 1 and 
                daily_trend == 1 and 
                current_price > sma_200 and 
                adx > 25 and
                plus_di > minus_di):
                
                # Calculate confidence
                confidence = 70.0
                if current_price > sma_50 > sma_200:
                    confidence += 10
                if adx > 30:
                    confidence += 10
                if rsi > 50:
                    confidence += 5
                
                return 'BULLISH', min(confidence, 95)
            
            # 3. Check for BEARISH trend
            if (weekly_trend == 0 and 
                daily_trend == 0 and 
                current_price < sma_200 and 
                adx > 25 and
                minus_di > plus_di):
                
                # Calculate confidence
                confidence = 70.0
                if current_price < sma_50 < sma_200:
                    confidence += 10
                if adx > 30:
                    confidence += 10
                if rsi < 50:
                    confidence += 5
                
                return 'BEARISH', min(confidence, 95)
            
            # 4. Check for RANGE (weak trend)
            if adx < 20:
                # Calculate range size
                high_20 = df['high'].iloc[current_idx-20:current_idx].max()
                low_20 = df['low'].iloc[current_idx-20:current_idx].min()
                range_size = ((high_20 - low_20) / current_price) * 100
                
                if range_size < 15:
                    confidence = 70 + (15 - range_size)
                    return 'RANGE', min(confidence, 90)
            
            # 5. Default to RANGE with moderate confidence
            return 'RANGE', 60.0
        
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return 'RANGE', 50.0
    
    def get_regime_strategies(self, regime: str) -> List[str]:
        """
        Return optimal strategy list for each regime
        
        Args:
            regime: Market regime ('BULLISH', 'BEARISH', 'RANGE', 'CHOPPY')
            
        Returns:
            List of strategy names suitable for the regime
        """
        return self.regime_strategies.get(regime, self.regime_strategies['RANGE'])
    
    def get_regime_info(self, regime: str) -> dict:
        """
        Get detailed information about a regime
        
        Args:
            regime: Market regime name
            
        Returns:
            Dictionary with regime characteristics
        """
        regime_info = {
            'BULLISH': {
                'description': 'Strong uptrend with momentum',
                'characteristics': [
                    'ADX > 25',
                    'Price above 200 SMA',
                    'Weekly + Daily trends aligned',
                    'Higher highs and higher lows'
                ],
                'strategy_count': 8,
                'risk_level': 'MODERATE'
            },
            'BEARISH': {
                'description': 'Strong downtrend with momentum',
                'characteristics': [
                    'ADX > 25',
                    'Price below 200 SMA',
                    'Weekly + Daily trends aligned down',
                    'Lower lows and lower highs'
                ],
                'strategy_count': 8,
                'risk_level': 'MODERATE'
            },
            'RANGE': {
                'description': 'Sideways consolidation',
                'characteristics': [
                    'ADX < 20',
                    'Price range < 15%',
                    'No clear trend',
                    'Support/resistance bouncing'
                ],
                'strategy_count': 8,
                'risk_level': 'LOW'
            },
            'CHOPPY': {
                'description': 'High volatility, no direction',
                'characteristics': [
                    'ATR Ratio > 3%',
                    'Erratic price movement',
                    'High uncertainty',
                    'Frequent reversals'
                ],
                'strategy_count': 4,
                'risk_level': 'HIGH'
            }
        }
        
        return regime_info.get(regime, regime_info['RANGE'])
    
    def should_trade_in_regime(self, regime: str, min_confidence: float = 70.0) -> bool:
        """
        Determine if we should trade in this regime
        
        Args:
            regime: Current regime
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if safe to trade, False otherwise
        """
        # Always avoid CHOPPY markets
        if regime == 'CHOPPY':
            return False
        
        # Other regimes are tradeable
        return True


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MARKET REGIME DETECTOR - READY")
    print("="*80)
    
    # Initialize detector
    detector = MarketRegimeDetector()
    
    print("\n📊 Regime Strategy Mapping:")
    print("-" * 80)
    
    for regime in detector.regimes:
        strategies = detector.get_regime_strategies(regime)
        info = detector.get_regime_info(regime)
        
        print(f"\n{regime}:")
        print(f"  Description: {info['description']}")
        print(f"  Strategies: {len(strategies)}")
        print(f"  Risk Level: {info['risk_level']}")
        print(f"  Active Strategies: {', '.join(strategies[:3])}...")
    
    print("\n" + "="*80)
    print("✅ Regime detector is ready!")
    print("📌 Next: Implement your 20 strategies")
    print("="*80)
