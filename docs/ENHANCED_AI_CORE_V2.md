# SwingAI Enhanced AI Core (V2)

This doc describes the code under `ml/` that powers the enhanced signal generation pipeline.

## Overview

The primary orchestrator is `ml/inference/enhanced_signal_generator.py`. It combines:
- 70-feature engineering (`ml/features/enhanced_features.py`)
- 5-model hierarchical ensemble (`ml/models/hierarchical_ensemble.py`)
- Market regime detection + premium filters (`ml/filters/advanced_filters.py`)
- Dynamic risk sizing (`ml/features/dynamic_risk_manager.py`)

## Feature engineering (70 features)

`EnhancedFeatureEngine` generates seven 10-feature blocks:
- Technical analysis (RSI, MACD, BB, Stoch, ATR, ADX, CCI)
- Price action (support/resistance, fib distance, trend/momentum)
- Volume and momentum (volume ratios, OBV/MFI/CMF-like signals)
- SMC features (order blocks, FVG, sweeps, institutional activity)
- Multi-timeframe alignment (daily/hourly/weekly)
- Market microstructure (order flow proxies, volatility, squeeze)
- Market context (VIX, market breadth, FII/DII)

## Hierarchical ensemble (5 models)

`HierarchicalEnsemble` uses base weights:
- TFT: 0.40
- LSTM: 0.25
- XGBoost: 0.20
- RandomForest: 0.10
- SVM: 0.05

Weights are adjusted per prediction based on model agreement and regime context. The output includes agreement score, uncertainty, and a final LONG/SHORT/NEUTRAL direction.

## Market regime detection

`MarketRegimeDetector` (in `ml/filters/advanced_filters.py`) classifies four regimes: BULLISH, BEARISH, RANGE, CHOPPY. It uses ADX, MA alignment, ATR %, range size, and optional VIX.

## Premium signal filter (8-point validation)

`PremiumSignalFilter` produces a reliability score and grade (PREMIUM/EXCELLENT/GOOD/SKIP) based on:
- AI confidence
- Strategy confluence
- SMC confirmation
- Price action score
- Technical alignment
- Regime fit
- Volume confirmation
- Entry precision

## Dynamic risk management

`DynamicRiskManager` applies five multipliers to a base risk percentage:
- Confidence multiplier
- Volatility multiplier
- Correlation multiplier
- Portfolio load multiplier
- Performance multiplier

Final risk is clamped between 0.5% and 3%.

## Confidence decay

`ConfidenceDecaySystem` tracks signal confidence over time and can flag when a signal should be exited based on decay thresholds.

## Strategy layer

The strategy system is implemented under `ml/strategies/` (20 strategies + selector + regime mapping). In `EnhancedSignalGenerator`, strategy confluence is currently a placeholder value; wiring the strategy selector into this pipeline is still pending.

## Output structure

`EnhancedSignal` includes:
- Model predictions and weights
- Feature set and regime metadata
- Validation grade and confidence
- Risk allocation and position sizing inputs
- Entry/stop/targets

## Integration note

The enhanced AI core can be enabled in the backend by setting `ENABLE_ENHANCED_AI=true`. When enabled, `src/backend/services/signal_generator.py` uses `EnhancedSignalGenerator` to produce signals and maps model outputs into the existing `catboost_score`, `tft_score`, and `stockformer_score` fields (XGBoost, TFT, and RandomForest respectively). The standard pipeline is used as a fallback if the enhanced pipeline fails.
