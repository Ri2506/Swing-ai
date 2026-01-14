"""
================================================================================
STRATEGY SELECTOR - Chooses Best Strategy for Current Market Conditions
================================================================================
Selects the most suitable strategy based on:
1. Market Regime (BULLISH, BEARISH, RANGE, CHOPPY)
2. Strategy Confluence Score (0-100%)
3. Strategy Tier (S+, S, A, B, C)
================================================================================
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from .all_strategies import (
    S1_7Element, S2_TripleRSI, S3_BollingerRSI,
    S4_GoldenPocket, S5_CupHandle, S6_DoubleBottomDiv,
    S7_MTFOrderFlow, S8_WyckoffSMC, S9_LiquiditySweep,
    S10_TripleMAVolume, S11_SupplyDemand, S12_ICTKillzone,
    S13_ThreeDrive, S14_MomentumBreakout, S15_TrendPullback,
    S16_GapFill, S17_BreakRetest, S18_PurePrice,
    S19_ATRBreakout, S20_PivotMTF, get_all_strategies
)
from .regime_detector import MarketRegimeDetector
from .base_strategy import BaseStrategy


class StrategySelector:
    """
    Select best strategy based on market regime and confluence
    
    Features:
    - Market Regime Detection (BULLISH, BEARISH, RANGE, CHOPPY)
    - Regime-Appropriate Strategy Filtering
    - Confluence Scoring (0-100%)
    - Best Strategy Selection
    - Multi-Strategy Analysis
    """
    
    def __init__(self):
        # Initialize all 20 strategies
        self.strategies = get_all_strategies()
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector()
        
        print(f"✅ StrategySelector initialized with {len(self.strategies)} strategies")
    
    def select_best_strategy(
        self, 
        df: pd.DataFrame, 
        current_idx: int
    ) -> Tuple[Optional[BaseStrategy], float, str]:
        """
        Select best strategy for current market conditions
        
        Args:
            df: DataFrame with 70 features + OHLCV
            current_idx: Index of current candle
        
        Returns:
            (best_strategy, confluence_score, market_regime)
        """
        # Detect current market regime
        regime = self.regime_detector.detect_regime(df, current_idx)
        
        # Get regime-appropriate strategies
        suitable_strategy_names = self.regime_detector.get_regime_strategies(regime)
        
        # Calculate confluence for suitable strategies
        strategy_scores = {}
        
        for strategy in self.strategies:
            if strategy.name in suitable_strategy_names:
                try:
                    score = strategy.check_confluence(df, current_idx)
                    strategy_scores[strategy] = score
                except Exception as e:
                    print(f"⚠️ Error in {strategy.name}: {e}")
                    strategy_scores[strategy] = 0.0
        
        # Return best strategy
        if not strategy_scores:
            return None, 0.0, regime
        
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy]
        
        return best_strategy, best_score, regime
    
    def get_all_strategy_scores(
        self, 
        df: pd.DataFrame, 
        current_idx: int
    ) -> Tuple[str, Dict]:
        """
        Get scores for ALL strategies (for analysis)
        
        Args:
            df: DataFrame with 70 features + OHLCV
            current_idx: Index of current candle
        
        Returns:
            (regime, {strategy_name: {score, tier, win_rate}})
        """
        regime = self.regime_detector.detect_regime(df, current_idx)
        
        scores = {}
        for strategy in self.strategies:
            try:
                score = strategy.check_confluence(df, current_idx)
                scores[strategy.name] = {
                    'score': score * 100,  # Convert to percentage
                    'tier': strategy.tier,
                    'win_rate': strategy.win_rate_range
                }
            except Exception as e:
                scores[strategy.name] = {
                    'score': 0.0,
                    'tier': strategy.tier,
                    'win_rate': strategy.win_rate_range,
                    'error': str(e)
                }
        
        return regime, scores
    
    def get_top_strategies(
        self, 
        df: pd.DataFrame, 
        current_idx: int,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get top N strategies ranked by confluence score
        
        Args:
            df: DataFrame with 70 features + OHLCV
            current_idx: Index of current candle
            top_n: Number of top strategies to return
        
        Returns:
            List of {name, tier, win_rate, confluence, entry, stop, targets}
        """
        regime = self.regime_detector.detect_regime(df, current_idx)
        suitable_strategy_names = self.regime_detector.get_regime_strategies(regime)
        
        strategy_data = []
        
        for strategy in self.strategies:
            if strategy.name in suitable_strategy_names:
                try:
                    confluence = strategy.check_confluence(df, current_idx)
                    levels = strategy.calculate_entry_stop_targets(df, current_idx)
                    
                    strategy_data.append({
                        'name': strategy.name,
                        'tier': strategy.tier,
                        'win_rate': strategy.win_rate_range,
                        'confluence': confluence * 100,  # Convert to percentage
                        'entry': levels['entry'],
                        'stop': levels['stop'],
                        'target1': levels['target1'],
                        'target2': levels['target2'],
                        'target3': levels['target3'],
                        'rr_ratio': levels['rr_ratio']
                    })
                except Exception as e:
                    pass
        
        # Sort by confluence score descending
        strategy_data.sort(key=lambda x: x['confluence'], reverse=True)
        
        return strategy_data[:top_n]
    
    def validate_strategy_requirements(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, bool]:
        """
        Validate that DataFrame has all required features for strategies
        
        Args:
            df: DataFrame to validate
        
        Returns:
            {feature_category: has_all_features}
        """
        required_features = {
            'OHLCV': ['open', 'high', 'low', 'close', 'volume'],
            'Technical': ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'BB_Middle', 
                         'Stochastic_K', 'ATR', 'CCI'],
            'Price_Action': ['Support_Level', 'Resistance_Level', 'Fib_618', 'Pivot_Point'],
            'Volume': ['Volume_MA', 'Volume_Spike'],
            'SMC': ['OB_Strength', 'OB_Distance_Pct', 'FVG_Distance', 'FVG_Volume_Ratio',
                   'Sweep_Detected', 'Post_Sweep_Reversal_Prob', 'Inst_Activity_Score',
                   'Accumulation_Phase', 'Distribution_Phase', 'Liquidity_Level'],
            'MTF': ['Weekly_Trend', 'Daily_Trend', 'MTF_Confluence', 'Volume_Conf_Weekly'],
            'Microstructure': ['Order_Flow_Imbalance', 'Momentum_Composite', 'Reversal_Probability',
                              'Support_Resistance_Strength', 'Volatility_Squeeze'],
            'Advanced': ['EMA_50', 'EMA_200', 'VWAP', 'ADX']
        }
        
        validation = {}
        
        for category, features in required_features.items():
            has_all = all(f in df.columns for f in features)
            validation[category] = has_all
        
        return validation


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STRATEGY SELECTOR - USAGE EXAMPLE")
    print("="*80)
    
    # Example: Load your data with 70 features
    # df = pd.read_csv('stock_data_with_features.csv')
    
    # For demo, create mock data
    print("\n📊 Creating mock data with 70 features...")
    
    # Mock data (in production, use real data)
    df = pd.DataFrame({
        'open': np.random.rand(200) * 100 + 100,
        'high': np.random.rand(200) * 100 + 105,
        'low': np.random.rand(200) * 100 + 95,
        'close': np.random.rand(200) * 100 + 100,
        'volume': np.random.randint(1000000, 10000000, 200),
        
        # Technical
        'RSI': np.random.rand(200) * 100,
        'MACD': np.random.randn(200),
        'MACD_Signal': np.random.randn(200),
        'BB_Upper': np.random.rand(200) * 100 + 110,
        'BB_Middle': np.random.rand(200) * 100 + 100,
        'BB_Lower': np.random.rand(200) * 100 + 90,
        'Stochastic_K': np.random.rand(200) * 100,
        'ATR': np.random.rand(200) * 5,
        'CCI': np.random.randn(200) * 100,
        'ADX': np.random.rand(200) * 50,
        
        # Price Action
        'Support_Level': np.random.rand(200) * 100 + 95,
        'Resistance_Level': np.random.rand(200) * 100 + 105,
        'Fib_618': np.random.rand(200) * 100 + 100,
        'Pivot_Point': np.random.rand(200) * 100 + 100,
        'Higher_High': np.random.randint(0, 2, 200),
        'Lower_Low': np.random.randint(0, 2, 200),
        'Consolidation_Score': np.random.rand(200) * 100,
        'Gap_Detected': np.random.randint(0, 2, 200),
        
        # Volume
        'Volume_MA': np.random.randint(1000000, 10000000, 200),
        'Volume_Spike': np.random.randint(0, 2, 200),
        
        # SMC
        'OB_Strength': np.random.rand(200) * 100,
        'OB_Distance_Pct': np.random.rand(200) * 5,
        'FVG_Distance': np.random.rand(200) * 10,
        'FVG_Volume_Ratio': np.random.rand(200) * 2,
        'Sweep_Detected': np.random.randint(0, 2, 200),
        'Post_Sweep_Reversal_Prob': np.random.rand(200),
        'Inst_Activity_Score': np.random.rand(200) * 100,
        'Accumulation_Phase': np.random.rand(200) * 100,
        'Distribution_Phase': np.random.rand(200) * 100,
        'Liquidity_Level': np.random.rand(200) * 100,
        
        # MTF
        'Weekly_Trend': np.random.choice([-1, 0, 1], 200),
        'Daily_Trend': np.random.choice([-1, 0, 1], 200),
        'Hourly_Trend': np.random.choice([-1, 0, 1], 200),
        'MTF_Confluence': np.random.rand(200) * 100,
        'Volume_Conf_Weekly': np.random.randint(0, 2, 200),
        'RSI_Divergence': np.random.randint(0, 2, 200),
        'MA_Alignment': np.random.rand(200) * 100,
        'Inst_Activity_MTF': np.random.rand(200) * 100,
        
        # Microstructure
        'Order_Flow_Imbalance': np.random.rand(200) * 100,
        'Momentum_Composite': np.random.rand(200) * 100,
        'Reversal_Probability': np.random.rand(200) * 100,
        'Support_Resistance_Strength': np.random.rand(200) * 100,
        'Volatility_Squeeze': np.random.rand(200) * 100,
        'Mean_Reversion_Signal': np.random.rand(200) * 100,
        'Institutional_Presence': np.random.rand(200) * 100,
        
        # Advanced
        'EMA_50': np.random.rand(200) * 100 + 100,
        'EMA_200': np.random.rand(200) * 100 + 100,
        'VWAP': np.random.rand(200) * 100 + 100,
    })
    
    # Initialize selector
    selector = StrategySelector()
    
    # Select best strategy for current day
    current_idx = len(df) - 1
    
    print("\n🔍 Selecting best strategy...")
    best_strategy, confluence, regime = selector.select_best_strategy(df, current_idx)
    
    if best_strategy:
        print(f"\n✅ BEST STRATEGY SELECTED:")
        print("-" * 80)
        print(f"   Strategy: {best_strategy.name}")
        print(f"   Tier: {best_strategy.tier}")
        print(f"   Win Rate: {best_strategy.win_rate_range}")
        print(f"   Confluence: {confluence:.2%}")
        print(f"   Market Regime: {regime}")
        
        # Get entry/stop/targets
        levels = best_strategy.calculate_entry_stop_targets(df, current_idx)
        print(f"\n📊 TRADE LEVELS:")
        print("-" * 80)
        print(f"   Entry: ₹{levels['entry']:.2f}")
        print(f"   Stop: ₹{levels['stop']:.2f}")
        print(f"   Target 1: ₹{levels['target1']:.2f} (2R)")
        print(f"   Target 2: ₹{levels['target2']:.2f} (4R)")
        print(f"   Target 3: ₹{levels['target3']:.2f} (6R)")
        print(f"   R:R Ratio: {levels['rr_ratio']}")
    else:
        print(f"\n❌ No suitable strategy found for {regime} market")
    
    # Get top 5 strategies
    print(f"\n📈 TOP 5 STRATEGIES (Ranked by Confluence):")
    print("-" * 80)
    top_strategies = selector.get_top_strategies(df, current_idx, top_n=5)
    
    for i, strat in enumerate(top_strategies, 1):
        print(f"{i}. {strat['name']:20s} | Tier: {strat['tier']:2s} | "
              f"Confluence: {strat['confluence']:.1f}% | "
              f"Entry: ₹{strat['entry']:.2f}")
    
    print("\n" + "="*80)
    print("✅ STRATEGY SELECTOR TEST COMPLETE!")
    print("="*80)
