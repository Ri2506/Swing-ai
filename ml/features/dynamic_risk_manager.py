"""
================================================================================
SWINGAI - DYNAMIC RISK MANAGEMENT (5 MULTIPLIERS)
================================================================================
Dynamic position sizing based on 5 risk multipliers:
1. Confidence Multiplier (0.5x - 1.5x)
2. Volatility Multiplier (0.5x - 1.5x)
3. Correlation Multiplier (0.6x - 1.2x)
4. Portfolio Load Multiplier (0.6x - 1.2x)
5. Recent Performance Multiplier (0.7x - 1.1x)

Base Risk: 2% → Final Risk: 0.5% - 3%
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskMultipliers:
    """Individual risk multipliers"""
    confidence: float      # 0.5 - 1.5
    volatility: float      # 0.5 - 1.5
    correlation: float     # 0.6 - 1.2
    portfolio_load: float  # 0.6 - 1.2
    performance: float     # 0.7 - 1.1
    
    # Combined
    total_multiplier: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'confidence': self.confidence,
            'volatility': self.volatility,
            'correlation': self.correlation,
            'portfolio_load': self.portfolio_load,
            'performance': self.performance,
            'total': self.total_multiplier
        }


@dataclass
class DynamicRiskAllocation:
    """Final risk allocation for a signal"""
    symbol: str
    base_risk_percent: float
    final_risk_percent: float
    multipliers: RiskMultipliers
    
    # Position sizing
    account_value: float
    risk_amount: float
    position_size_shares: int
    entry_price: float
    stop_loss_price: float
    
    # Metadata
    justification: str


class DynamicRiskManager:
    """
    Dynamic Risk Manager with 5 Multipliers
    
    Adjusts position size based on signal quality, market conditions,
    portfolio state, and recent performance.
    """
    
    def __init__(
        self, 
        base_risk_percent: float = 2.0,
        min_risk_percent: float = 0.5,
        max_risk_percent: float = 3.0
    ):
        """
        Initialize risk manager
        
        Args:
            base_risk_percent: Base risk per trade (default 2%)
            min_risk_percent: Minimum risk floor (default 0.5%)
            max_risk_percent: Maximum risk ceiling (default 3%)
        """
        self.base_risk = base_risk_percent
        self.min_risk = min_risk_percent
        self.max_risk = max_risk_percent
    
    def calculate_risk_allocation(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        ai_confidence: float,
        volatility_atr_percent: float,
        portfolio_positions: List[Dict],
        recent_trades: List[Dict],
        account_value: float,
        market_correlation: Optional[float] = None
    ) -> DynamicRiskAllocation:
        """
        Calculate dynamic risk allocation
        
        Args:
            symbol: Stock symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            ai_confidence: AI model confidence (0-100)
            volatility_atr_percent: Current ATR as % of price
            portfolio_positions: Current open positions
            recent_trades: Recent closed trades (for performance tracking)
            account_value: Current account value
            market_correlation: Correlation with existing positions (optional)
            
        Returns:
            DynamicRiskAllocation with final position size
        """
        # Calculate 5 multipliers
        multipliers = self._calculate_multipliers(
            ai_confidence=ai_confidence,
            volatility_atr_percent=volatility_atr_percent,
            portfolio_positions=portfolio_positions,
            recent_trades=recent_trades,
            market_correlation=market_correlation
        )
        
        # Apply multipliers to base risk
        final_risk_percent = self.base_risk * multipliers.total_multiplier
        
        # Clamp to min/max bounds
        final_risk_percent = max(self.min_risk, min(final_risk_percent, self.max_risk))
        
        # Calculate position size
        risk_amount = account_value * (final_risk_percent / 100)
        
        # Calculate shares based on stop loss distance
        stop_loss_distance = abs(entry_price - stop_loss_price)
        if stop_loss_distance > 0:
            position_size_shares = int(risk_amount / stop_loss_distance)
        else:
            position_size_shares = 0
        
        # Generate justification
        justification = self._generate_justification(multipliers)
        
        return DynamicRiskAllocation(
            symbol=symbol,
            base_risk_percent=self.base_risk,
            final_risk_percent=round(final_risk_percent, 2),
            multipliers=multipliers,
            account_value=account_value,
            risk_amount=round(risk_amount, 2),
            position_size_shares=position_size_shares,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            justification=justification
        )
    
    def _calculate_multipliers(
        self,
        ai_confidence: float,
        volatility_atr_percent: float,
        portfolio_positions: List[Dict],
        recent_trades: List[Dict],
        market_correlation: Optional[float]
    ) -> RiskMultipliers:
        """Calculate all 5 multipliers"""
        
        # 1. Confidence Multiplier (0.5 - 1.5)
        confidence_mult = self._calculate_confidence_multiplier(ai_confidence)
        
        # 2. Volatility Multiplier (0.5 - 1.5)
        volatility_mult = self._calculate_volatility_multiplier(volatility_atr_percent)
        
        # 3. Correlation Multiplier (0.6 - 1.2)
        correlation_mult = self._calculate_correlation_multiplier(
            portfolio_positions,
            market_correlation
        )
        
        # 4. Portfolio Load Multiplier (0.6 - 1.2)
        portfolio_mult = self._calculate_portfolio_load_multiplier(portfolio_positions)
        
        # 5. Performance Multiplier (0.7 - 1.1)
        performance_mult = self._calculate_performance_multiplier(recent_trades)
        
        # Total multiplier (product of all)
        total_mult = (
            confidence_mult *
            volatility_mult *
            correlation_mult *
            portfolio_mult *
            performance_mult
        )
        
        return RiskMultipliers(
            confidence=round(confidence_mult, 3),
            volatility=round(volatility_mult, 3),
            correlation=round(correlation_mult, 3),
            portfolio_load=round(portfolio_mult, 3),
            performance=round(performance_mult, 3),
            total_multiplier=round(total_mult, 3)
        )
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """
        Confidence Multiplier (0.5x - 1.5x)
        
        Logic:
        - 95%+ confidence → 1.5x (max)
        - 80% confidence → 1.0x (base)
        - 65% confidence → 0.5x (min)
        """
        if confidence >= 95:
            return 1.5
        elif confidence >= 80:
            # Linear interpolation between 1.0 and 1.5
            return 1.0 + ((confidence - 80) / 15) * 0.5
        elif confidence >= 65:
            # Linear interpolation between 0.5 and 1.0
            return 0.5 + ((confidence - 65) / 15) * 0.5
        else:
            return 0.5
    
    def _calculate_volatility_multiplier(self, atr_percent: float) -> float:
        """
        Volatility Multiplier (0.5x - 1.5x)
        
        Logic:
        - Low volatility (< 1.5%) → 1.5x (increase size)
        - Normal volatility (2-3%) → 1.0x (base)
        - High volatility (> 5%) → 0.5x (reduce size)
        """
        if atr_percent < 1.5:
            return 1.5
        elif atr_percent < 2.0:
            # Linear from 1.5 to 1.2
            return 1.5 - ((atr_percent - 1.5) / 0.5) * 0.3
        elif atr_percent < 3.0:
            # Linear from 1.2 to 1.0
            return 1.2 - ((atr_percent - 2.0) / 1.0) * 0.2
        elif atr_percent < 5.0:
            # Linear from 1.0 to 0.5
            return 1.0 - ((atr_percent - 3.0) / 2.0) * 0.5
        else:
            return 0.5
    
    def _calculate_correlation_multiplier(
        self,
        portfolio_positions: List[Dict],
        market_correlation: Optional[float]
    ) -> float:
        """
        Correlation Multiplier (0.6x - 1.2x)
        
        Logic:
        - Low correlation with existing positions → 1.2x
        - High correlation → 0.6x (avoid concentration)
        """
        if not portfolio_positions or market_correlation is None:
            return 1.0  # Neutral if no positions
        
        # Use provided correlation
        corr = abs(market_correlation)
        
        if corr < 0.3:
            return 1.2  # Low correlation, diversifying
        elif corr < 0.6:
            # Linear from 1.2 to 0.9
            return 1.2 - ((corr - 0.3) / 0.3) * 0.3
        elif corr < 0.8:
            # Linear from 0.9 to 0.7
            return 0.9 - ((corr - 0.6) / 0.2) * 0.2
        else:
            return 0.6  # High correlation, too concentrated
    
    def _calculate_portfolio_load_multiplier(
        self,
        portfolio_positions: List[Dict]
    ) -> float:
        """
        Portfolio Load Multiplier (0.6x - 1.2x)
        
        Logic:
        - 0-2 positions → 1.2x (room to add)
        - 3-5 positions → 1.0x (base)
        - 6-8 positions → 0.8x (getting full)
        - 9+ positions → 0.6x (too many)
        """
        position_count = len(portfolio_positions)
        
        if position_count <= 2:
            return 1.2
        elif position_count <= 5:
            # Linear from 1.2 to 1.0
            return 1.2 - ((position_count - 2) / 3) * 0.2
        elif position_count <= 8:
            # Linear from 1.0 to 0.8
            return 1.0 - ((position_count - 5) / 3) * 0.2
        else:
            return 0.6
    
    def _calculate_performance_multiplier(
        self,
        recent_trades: List[Dict]
    ) -> float:
        """
        Recent Performance Multiplier (0.7x - 1.1x)
        
        Logic:
        - Last 10 trades win rate > 70% → 1.1x
        - Last 10 trades win rate 50-70% → 1.0x
        - Last 10 trades win rate < 30% → 0.7x
        """
        if not recent_trades:
            return 1.0  # Neutral if no history
        
        # Get last 10 trades
        last_10 = recent_trades[-10:]
        
        # Calculate win rate
        wins = sum(1 for trade in last_10 if trade.get('pnl', 0) > 0)
        win_rate = (wins / len(last_10)) * 100
        
        if win_rate >= 70:
            return 1.1
        elif win_rate >= 50:
            # Linear from 1.0 to 1.1
            return 1.0 + ((win_rate - 50) / 20) * 0.1
        elif win_rate >= 30:
            # Linear from 0.85 to 1.0
            return 0.85 + ((win_rate - 30) / 20) * 0.15
        else:
            return 0.7
    
    def _generate_justification(self, multipliers: RiskMultipliers) -> str:
        """Generate human-readable justification"""
        parts = []
        
        # Confidence
        if multipliers.confidence >= 1.3:
            parts.append("High confidence (boost)")
        elif multipliers.confidence <= 0.7:
            parts.append("Lower confidence (reduce)")
        
        # Volatility
        if multipliers.volatility >= 1.3:
            parts.append("Low volatility (boost)")
        elif multipliers.volatility <= 0.7:
            parts.append("High volatility (reduce)")
        
        # Correlation
        if multipliers.correlation >= 1.1:
            parts.append("Low correlation (diversifying)")
        elif multipliers.correlation <= 0.8:
            parts.append("High correlation (concentrated)")
        
        # Portfolio load
        if multipliers.portfolio_load >= 1.1:
            parts.append("Room in portfolio")
        elif multipliers.portfolio_load <= 0.8:
            parts.append("Portfolio getting full")
        
        # Performance
        if multipliers.performance >= 1.05:
            parts.append("Good recent performance")
        elif multipliers.performance <= 0.85:
            parts.append("Recent underperformance (reduce)")
        
        if not parts:
            return "Standard risk allocation"
        
        return "; ".join(parts)
    
    def get_position_size_summary(self, allocation: DynamicRiskAllocation) -> str:
        """Generate summary string for position size"""
        return f"""
Dynamic Risk Allocation for {allocation.symbol}
{'='*60}
Base Risk: {allocation.base_risk_percent}%
Final Risk: {allocation.final_risk_percent}% (multiplier: {allocation.multipliers.total_multiplier}x)

Risk Multipliers:
  Confidence:     {allocation.multipliers.confidence}x
  Volatility:     {allocation.multipliers.volatility}x
  Correlation:    {allocation.multipliers.correlation}x
  Portfolio Load: {allocation.multipliers.portfolio_load}x
  Performance:    {allocation.multipliers.performance}x

Position Sizing:
  Account Value:  ₹{allocation.account_value:,.2f}
  Risk Amount:    ₹{allocation.risk_amount:,.2f}
  Entry Price:    ₹{allocation.entry_price:,.2f}
  Stop Loss:      ₹{allocation.stop_loss_price:,.2f}
  Position Size:  {allocation.position_size_shares:,} shares

Justification: {allocation.justification}
{'='*60}
"""


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DYNAMIC RISK MANAGEMENT TEST")
    print("="*80)
    
    # Initialize risk manager
    risk_mgr = DynamicRiskManager(base_risk_percent=2.0)
    
    # Test Case 1: High confidence, low volatility, empty portfolio
    print("\nTEST 1: Ideal Conditions (Should BOOST risk)")
    print("-" * 80)
    
    allocation1 = risk_mgr.calculate_risk_allocation(
        symbol="RELIANCE",
        entry_price=2500.0,
        stop_loss_price=2450.0,
        ai_confidence=92.0,
        volatility_atr_percent=1.2,
        portfolio_positions=[],
        recent_trades=[
            {'pnl': 500}, {'pnl': 300}, {'pnl': 400},
            {'pnl': 200}, {'pnl': 100}, {'pnl': 150},
            {'pnl': 250}, {'pnl': 180}
        ],
        account_value=1000000.0,
        market_correlation=0.2
    )
    
    print(risk_mgr.get_position_size_summary(allocation1))
    
    # Test Case 2: Low confidence, high volatility, full portfolio
    print("\nTEST 2: Poor Conditions (Should REDUCE risk)")
    print("-" * 80)
    
    allocation2 = risk_mgr.calculate_risk_allocation(
        symbol="VOLATILE_STOCK",
        entry_price=100.0,
        stop_loss_price=95.0,
        ai_confidence=68.0,
        volatility_atr_percent=6.5,
        portfolio_positions=[
            {'symbol': 'S1'}, {'symbol': 'S2'}, {'symbol': 'S3'},
            {'symbol': 'S4'}, {'symbol': 'S5'}, {'symbol': 'S6'},
            {'symbol': 'S7'}, {'symbol': 'S8'}, {'symbol': 'S9'}
        ],
        recent_trades=[
            {'pnl': -200}, {'pnl': -150}, {'pnl': -100},
            {'pnl': 50}, {'pnl': -80}, {'pnl': -120},
            {'pnl': -90}, {'pnl': -60}
        ],
        account_value=1000000.0,
        market_correlation=0.85
    )
    
    print(risk_mgr.get_position_size_summary(allocation2))
    
    # Test Case 3: Normal conditions
    print("\nTEST 3: Normal Conditions (Should be close to base risk)")
    print("-" * 80)
    
    allocation3 = risk_mgr.calculate_risk_allocation(
        symbol="INFOSYS",
        entry_price=1500.0,
        stop_loss_price=1470.0,
        ai_confidence=80.0,
        volatility_atr_percent=2.5,
        portfolio_positions=[
            {'symbol': 'S1'}, {'symbol': 'S2'}, {'symbol': 'S3'}
        ],
        recent_trades=[
            {'pnl': 100}, {'pnl': -50}, {'pnl': 150},
            {'pnl': -80}, {'pnl': 200}, {'pnl': 120}
        ],
        account_value=1000000.0,
        market_correlation=0.5
    )
    
    print(risk_mgr.get_position_size_summary(allocation3))
    
    print("\n" + "="*80)
    print("SUMMARY: Dynamic risk adjusts from 0.5% to 3% based on conditions")
    print("="*80)
