"""
SwingAI Strategies Module

Contains:
- Base strategy class
- Market regime detector  
- Strategy selector
- All 20 trading strategies
"""

from .base_strategy import BaseStrategy
from .regime_detector import MarketRegimeDetector
from .strategy_selector import StrategySelector
from .all_strategies import get_all_strategies

__all__ = [
    'BaseStrategy',
    'MarketRegimeDetector',
    'StrategySelector',
    'get_all_strategies'
]
