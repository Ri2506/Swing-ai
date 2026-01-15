# SwingAI Strategy System

The strategy system lives under `ml/strategies/` and contains 20 fully implemented strategies with regime-aware selection.

## Files

```
ml/strategies/
├── base_strategy.py         # BaseStrategy + helper checks
├── regime_detector.py       # 4-regime detection + strategy mapping
├── strategy_selector.py     # Selects best strategy for a regime
└── all_strategies.py        # 20 concrete strategies
```

## Core concepts

### BaseStrategy
Each strategy implements `check_confluence(df, current_idx)` and returns a 0.0-1.0 confluence score. `BaseStrategy` also provides `calculate_entry_stop_targets` for basic 1R/2R/3R levels.

### Regime detection
`MarketRegimeDetector` classifies the market into BULLISH, BEARISH, RANGE, or CHOPPY using ADX, ATR, RSI, and MA alignment. It then selects a regime-specific subset of strategies.

Current regime-to-strategy mapping:

- BULLISH: S1, S3, S4, S5, S7, S8, S10, S14
- BEARISH: S2, S3, S6, S8, S9, S12, S13, S14
- RANGE: S2, S3, S4, S11, S13, S15, S16, S18
- CHOPPY: S1, S5, S7, S8

### Strategy selection
`StrategySelector` provides:
- `select_best_strategy(df, current_idx)`
- `get_all_strategy_scores(df, current_idx)`
- `get_top_strategies(df, current_idx, top_n=5)`

## Data requirements
Strategies expect a DataFrame with OHLCV plus feature columns such as RSI, MACD, support/resistance, volume spike, and SMC fields. `StrategySelector.validate_strategy_requirements` can be used to verify the inputs.

## Full strategy details
See `ALL_20_STRATEGIES_COMPLETE.md` for the per-strategy descriptions and confluence checks.

## Integration status
The strategy system is currently standalone. `EnhancedSignalGenerator` uses a placeholder strategy confluence score; wiring the selector into that pipeline is still pending.
