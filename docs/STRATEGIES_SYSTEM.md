# SwingAI - 20 Strategy System Documentation

## 🎯 Overview

The strategy system implements a **regime-aware, multi-strategy approach** where:
1. **Market regime** is detected (BULLISH, BEARISH, RANGE, CHOPPY)
2. **8 optimal strategies** are activated for that regime
3. Each strategy checks **7-element confluence**
4. **Overall confluence** is calculated across all active strategies
5. Trade is taken only if **75%+ confluence** and **5+ strategies agree**

---

## 📂 File Structure

```
ml/strategies/
├── __init__.py              # Module exports
├── base_strategy.py         # ✅ Base class (COMPLETE)
├── regime_detector.py       # ✅ Regime detection (COMPLETE)
├── strategy_selector.py     # ✅ Strategy selector (COMPLETE)
└── all_strategies.py        # ⏳ 20 strategies (PLACEHOLDERS - waiting for you)
```

---

## 🏗️ Architecture

### **1. Base Strategy Class**
**File:** `base_strategy.py`

All 20 strategies inherit from `BaseStrategy` and must implement:

```python
class BaseStrategy(ABC):
    def __init__(self, name, tier, win_rate_range):
        # Initialize strategy
        pass
    
    @abstractmethod
    def check_confluence(self, df, current_idx) -> float:
        """
        Check 7-element confluence
        
        Must check:
        1. Trend alignment
        2. Support/resistance
        3. Volume confirmation
        4. Technical indicators
        5. Price action pattern
        6. Risk/reward ratio
        7. Entry precision
        
        Returns: 0.0 to 1.0
        """
        pass
    
    def calculate_entry_stop_targets(self, df, current_idx):
        # Auto-calculates levels
        pass
```

**Helper Functions Provided:**
- `check_trend_alignment(df, idx)` - MTF trend check
- `check_support_resistance(df, idx)` - S/R proximity
- `check_volume_confirmation(df, idx)` - Volume checks
- `check_technical_indicators(df, idx, direction)` - RSI, MACD, ADX
- `check_price_action_pattern(df, idx)` - Candle patterns
- `check_risk_reward_ratio(df, idx)` - R:R calculation
- `check_entry_precision(df, idx)` - Fib/OB proximity

---

### **2. Market Regime Detector**
**File:** `regime_detector.py`

Detects 4 market regimes:

```python
class MarketRegimeDetector:
    def detect_regime(self, df, current_idx):
        """
        Returns: (regime, confidence)
        
        Regimes:
        - BULLISH: ADX >25, price above MAs, uptrend
        - BEARISH: ADX >25, price below MAs, downtrend
        - RANGE: ADX <20, consolidation
        - CHOPPY: ATR >3%, high volatility
        """
        pass
    
    def get_regime_strategies(self, regime):
        """Returns 8 strategies for regime"""
        pass
```

**Regime → Strategy Mapping:**

| Regime | Active Strategies (8) | Risk Level |
|--------|-----------------------|------------|
| BULLISH | S1, S3, S4, S5, S7, S8, S10, S14 | MODERATE |
| BEARISH | S2, S3, S6, S8, S9, S12, S13, S14 | MODERATE |
| RANGE | S2, S3, S4, S11, S13, S15, S16, S18 | LOW |
| CHOPPY | S1, S5, S7, S8 (only 4) | HIGH |

---

### **3. Strategy Selector**
**File:** `strategy_selector.py`

Orchestrates strategy evaluation:

```python
class StrategySelector:
    def evaluate_strategies(self, df, current_idx):
        """
        1. Detect regime
        2. Get active strategies for regime
        3. Calculate confluence for each
        4. Calculate overall confluence
        5. Return evaluation
        """
        pass
    
    def should_take_trade(self, evaluation):
        """
        Checks:
        - Regime not CHOPPY
        - Overall confluence ≥ 75%
        - At least 5 strategies passed (≥70%)
        - Best strategy ≥ 80%
        """
        pass
    
    def get_trade_recommendation(self, df, current_idx):
        """Complete recommendation with levels"""
        pass
```

---

### **4. All 20 Strategies**
**File:** `all_strategies.py`

**Strategy Tiers:**
- **S+ Tier (2)**: S1_7Element, S2_TripleRSI (85-90% WR)
- **S Tier (3)**: S3_BollingerRSI, S4_GoldenPocket, S8_WyckoffSMC (75-83% WR)
- **A Tier (4)**: S5_CupHandle, S6_DoubleBottomDiv, S7_MTFOrderFlow, S12_ICTKillzone (68-77% WR)
- **B Tier (4)**: S9_LiquiditySweep, S10_TripleMAVolume, S11_SupplyDemand, S14_MomentumBreakout (62-70% WR)
- **C Tier (7)**: S13-S20 (55-66% WR)

---

## 📝 How to Implement Your Strategies

### **Step 1: Choose a Strategy**

Pick one of the 20 strategies in `all_strategies.py`, e.g., `S1_7Element`

### **Step 2: Implement `check_confluence()` Method**

```python
class S1_7Element(BaseStrategy):
    def check_confluence(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        7-Element confluence check
        """
        if current_idx < 50:
            return 0.0
        
        # 1. Trend alignment (0-1)
        trend_score = check_trend_alignment(df, current_idx)
        
        # 2. Support/resistance proximity (0-1)
        sr_score = check_support_resistance(df, current_idx)
        
        # 3. Volume confirmation (0-1)
        volume_score = check_volume_confirmation(df, current_idx)
        
        # 4. Technical indicators (0-1)
        tech_score = check_technical_indicators(df, current_idx, 'LONG')
        
        # 5. Price action pattern (0-1)
        pa_score = check_price_action_pattern(df, current_idx)
        
        # 6. Risk/reward ratio (0-1)
        rr_score = check_risk_reward_ratio(df, current_idx)
        
        # 7. Entry precision (0-1)
        entry_score = check_entry_precision(df, current_idx)
        
        # Average all 7 scores
        confluence = (
            trend_score + sr_score + volume_score + tech_score +
            pa_score + rr_score + entry_score
        ) / 7.0
        
        return confluence
```

### **Step 3: Add Custom Logic**

You can add custom checks using the 70 features:

```python
def check_confluence(self, df, current_idx):
    # Use any of the 70 features
    rsi = df['RSI'].iloc[current_idx]
    ob_strength = df['OB_Strength'].iloc[current_idx]
    accumulation = df['Accumulation_Phase'].iloc[current_idx]
    
    # Your custom logic
    if rsi < 35 and ob_strength > 80 and accumulation > 70:
        return 0.95  # Very high confluence
    
    # Or use helper functions
    return (check_trend_alignment(df, current_idx) + 
            check_volume_confirmation(df, current_idx)) / 2.0
```

### **Step 4: Test Your Strategy**

```python
from ml.strategies import get_all_strategies

# Load all strategies
strategies = get_all_strategies()

# Test one strategy
strategy = strategies[0]  # S1_7Element

# Mock data
import pandas as pd
df = pd.DataFrame({...})  # With 70 features

# Check confluence
score = strategy.check_confluence(df, -1)
print(f"Confluence: {score:.2%}")

# Get levels
levels = strategy.calculate_entry_stop_targets(df, -1)
print(f"Entry: {levels['entry']}, Stop: {levels['stop']}")
```

---

## 🚀 Usage Example

### **Complete Flow:**

```python
from ml.strategies import StrategySelector, get_all_strategies
from ml.features.feature_calculator_v2 import calculate_all_70_features
import yfinance as yf

# Step 1: Get data
ticker = yf.Ticker("RELIANCE.NS")
df = ticker.history(period="6mo", interval="1d")
df.columns = df.columns.str.lower()

# Step 2: Calculate 70 features
df_with_features = calculate_all_70_features(df)

# Step 3: Load all 20 strategies
all_strategies = get_all_strategies()

# Step 4: Create selector
selector = StrategySelector(all_strategies)

# Step 5: Get recommendation
recommendation = selector.get_trade_recommendation(df_with_features, -1)

# Step 6: Print report
report = selector.get_strategy_report(recommendation)
print(report)
```

**Example Output:**

```
================================================================================
STRATEGY EVALUATION REPORT
================================================================================

Market Regime: BULLISH (87.5% confidence)
Active Strategies: 8
Passed Strategies: 6/8
Overall Confluence: 78.3%

Best Strategy: S1_7Element (85.2%)

Strategy Scores:
--------------------------------------------------------------------------------
  S1_7Element              : 85.2%  ✅ PASS
  S8_WyckoffSMC            : 82.1%  ✅ PASS
  S3_BollingerRSI          : 78.5%  ✅ PASS
  S4_GoldenPocket          : 75.3%  ✅ PASS
  S7_MTFOrderFlow          : 73.8%  ✅ PASS
  S10_TripleMAVolume       : 71.2%  ✅ PASS
  S5_CupHandle             : 68.7%  ❌ FAIL
  S14_MomentumBreakout     : 65.4%  ❌ FAIL
--------------------------------------------------------------------------------

✅ TRADE SIGNAL: 6/8 strategies agree (78.3%)

Entry Levels:
  Entry:   ₹2456.75
  Stop:    ₹2412.30
  Target1: ₹2545.65 (2R)
  Target2: ₹2634.55 (4R)
  Target3: ₹2680.00 (6R)
  R:R Ratio: 1:2.00

================================================================================
```

---

## 🔗 Integration with Enhanced AI Core

The strategies integrate seamlessly with the AI core:

```python
# In enhanced_signal_generator.py

from ml.strategies import StrategySelector, get_all_strategies

class EnhancedSignalGenerator:
    def __init__(self):
        # ... existing code ...
        
        # Add strategy selector
        all_strategies = get_all_strategies()
        self.strategy_selector = StrategySelector(all_strategies)
    
    async def generate_signal(self, symbol, ...):
        # ... calculate 70 features ...
        
        # Get AI prediction (5-model ensemble)
        ensemble_pred = self.ensemble.predict(...)
        
        # Get strategy recommendation
        strategy_eval = self.strategy_selector.evaluate_strategies(
            df_with_features, -1
        )
        
        # NOW YOU HAVE BOTH:
        # 1. AI confidence from ensemble
        # 2. Strategy confluence from 20 strategies
        
        # Use BOTH in premium filter
        validation = self.premium_filter.validate_signal(
            ai_confidence=ensemble_pred.confidence,
            strategy_confluence=strategy_eval['overall_confluence'],  # ✅ Real!
            ...
        )
```

---

## ✅ Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Base Strategy | ✅ **COMPLETE** | Fully functional |
| Regime Detector | ✅ **COMPLETE** | All 4 regimes |
| Strategy Selector | ✅ **COMPLETE** | Full orchestration |
| Helper Functions | ✅ **COMPLETE** | 7 helper functions |
| S1-S20 Structure | ✅ **COMPLETE** | Placeholder implementations |
| **Your Part** | ⏳ **PENDING** | Replace placeholders with actual logic |

---

## 📌 Next Steps

1. **Review** the base system (base_strategy.py, regime_detector.py)
2. **Understand** how strategies work (see example)
3. **Implement** your 20 strategies one by one:
   - Start with S+ tier (S1, S2)
   - Then S tier (S3, S4, S8)
   - Then A tier, B tier, C tier
4. **Test** each strategy individually
5. **Integrate** with AI core once complete

---

## 🎯 Example Strategy Template

Use this template for each of your 20 strategies:

```python
class SX_YourStrategy(BaseStrategy):
    """Strategy X: Description (Tier ?)"""
    
    def __init__(self):
        super().__init__(
            name="SX_YourStrategy",
            tier="?",  # S+, S, A, B, or C
            win_rate_range="XX-YY%"
        )
    
    def check_confluence(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        Check confluence for your strategy
        
        Returns: 0.0 to 1.0
        """
        # Minimum data check
        if current_idx < 50:
            return 0.0
        
        # YOUR CUSTOM LOGIC HERE
        # Access any of the 70 features:
        # df['RSI'], df['MACD'], df['OB_Strength'], etc.
        
        score = 0.0
        
        # Check 1: ...
        if condition1:
            score += 0.15
        
        # Check 2: ...
        if condition2:
            score += 0.15
        
        # ... up to check 7
        
        # Or use helper functions:
        score += check_trend_alignment(df, current_idx) * 0.3
        score += check_volume_confirmation(df, current_idx) * 0.2
        
        return min(score, 1.0)
```

---

## 🎉 Summary

✅ **Strategy infrastructure is COMPLETE**  
✅ **Base classes ready**  
✅ **Regime detection working**  
✅ **Strategy selector ready**  
✅ **Integration points prepared**  

**Ready for you to implement your 20 strategies!** 🚀

When you provide the actual logic for each strategy, simply replace the `check_confluence()` method in each class. Everything else is handled automatically!
