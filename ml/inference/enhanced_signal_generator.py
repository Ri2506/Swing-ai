"""
================================================================================
SWINGAI - ENHANCED SIGNAL GENERATOR (AI CORE ORCHESTRATOR)
================================================================================
Complete signal generation pipeline with:
- 70 Feature Engineering
- Hierarchical Ensemble (5 models with adaptive weighting)
- Market Regime Detection
- Premium Signal Filter (8-point validation)
- Dynamic Risk Management
- Confidence Decay Tracking
================================================================================
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import httpx

# Import our new modules
from ml.features.enhanced_features import EnhancedFeatureEngine
from ml.models.hierarchical_ensemble import HierarchicalEnsemble, ModelPrediction
from ml.filters.advanced_filters import (
    MarketRegimeDetector, 
    PremiumSignalFilter, 
    ConfidenceDecaySystem,
    MarketRegime,
    SignalGrade
)
from ml.features.dynamic_risk_manager import DynamicRiskManager

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSignal:
    """Complete signal with all metadata"""
    # Basic info
    symbol: str
    direction: str  # LONG/SHORT/NEUTRAL
    timestamp: datetime
    
    # AI predictions
    ai_confidence: float
    model_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    agreement_score: float
    uncertainty: float
    
    # Features
    features: Dict[str, float]
    
    # Market context
    regime: str
    regime_confidence: float
    
    # Validation
    validation_score: float
    signal_grade: str
    passed_validation: bool
    validation_details: Dict
    
    # Strategy (placeholder - will be filled by strategy layer)
    strategy_confluence: float
    active_strategies: List[str]
    
    # Risk management
    base_risk_percent: float
    final_risk_percent: float
    risk_multipliers: Dict[str, float]
    
    # Entry/Exit levels
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward_ratio: float
    
    # Metadata
    reliability_score: float
    execution_priority: str  # PREMIUM/EXCELLENT/GOOD/SKIP


class EnhancedSignalGenerator:
    """
    Enhanced Signal Generator - Complete AI Core
    
    Orchestrates the entire signal generation pipeline:
    1. Fetch data (multi-timeframe)
    2. Calculate 70 features
    3. Get AI predictions from 5 models
    4. Apply hierarchical ensemble
    5. Detect market regime
    6. Validate with 8-point filter
    7. Calculate dynamic risk
    8. Generate final signal
    """
    
    def __init__(
        self,
        modal_endpoint: Optional[str] = None,
        use_adaptive_weighting: bool = True
    ):
        """
        Initialize Enhanced Signal Generator
        
        Args:
            modal_endpoint: Modal.com ML inference endpoint URL
            use_adaptive_weighting: Use adaptive ensemble weighting
        """
        self.modal_endpoint = modal_endpoint
        
        # Initialize components
        self.feature_engine = EnhancedFeatureEngine()
        self.ensemble = HierarchicalEnsemble(use_adaptive_weights=use_adaptive_weighting)
        self.regime_detector = MarketRegimeDetector()
        self.premium_filter = PremiumSignalFilter()
        self.risk_manager = DynamicRiskManager()
        self.decay_system = ConfidenceDecaySystem()
        
        logger.info("Enhanced Signal Generator initialized with all components")
    
    async def generate_signal(
        self,
        symbol: str,
        account_value: float,
        portfolio_positions: List[Dict],
        recent_trades: List[Dict],
        market_data: Optional[Dict] = None
    ) -> Optional[EnhancedSignal]:
        """
        Generate complete enhanced signal
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            account_value: Current account value
            portfolio_positions: Current open positions
            recent_trades: Recent trade history
            market_data: Market-wide data (Nifty, VIX, FII/DII)
            
        Returns:
            EnhancedSignal if passed all filters, else None
        """
        logger.info(f"Generating enhanced signal for {symbol}")
        
        try:
            # Step 1: Fetch multi-timeframe data
            df_daily, df_hourly, df_weekly = self._fetch_data(symbol)
            
            if df_daily is None or len(df_daily) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Step 2: Calculate 70 features
            features = self.feature_engine.calculate_all_features(
                df_daily, df_hourly, df_weekly, market_data
            )
            
            logger.info(f"Calculated {len(features)} features for {symbol}")
            
            # Step 3: Detect market regime
            vix = market_data.get('vix_close', 15) if market_data else 15
            regime_result = self.regime_detector.detect_regime(df_daily, vix)
            
            logger.info(f"Detected regime: {regime_result.regime.value} (confidence: {regime_result.confidence}%)")
            
            # Step 4: Get AI model predictions
            model_predictions = await self._get_model_predictions(features, regime_result.regime)
            
            # Step 5: Apply hierarchical ensemble
            ensemble_pred = self.ensemble.predict(model_predictions, regime_result.regime.value)
            
            logger.info(
                f"Ensemble prediction: {ensemble_pred.prediction:.2f} "
                f"({ensemble_pred.direction}, confidence: {ensemble_pred.confidence:.2f}%)"
            )
            
            # Step 6: Strategy confluence - wire in actual strategy selector
            strategy_confluence, active_strategies = self._get_strategy_confluence(df_daily, features)
            
            # Step 7: Validate with premium filter
            validation = self.premium_filter.validate_signal(
                ai_confidence=ensemble_pred.confidence,
                strategy_confluence=strategy_confluence,
                smc_features={k: v for k, v in features.items() if 'order_block' in k or 'fvg' in k or 'sweep' in k or 'institutional' in k or 'accumulation' in k or 'distribution' in k or 'liquidity' in k},
                price_action_features={k: v for k, v in features.items() if 'support' in k or 'resistance' in k or 'fib' in k or 'range' in k or 'trend' in k or 'momentum' in k or 'candle' in k or 'gap' in k or 'hh_ll' in k},
                technical_features={k: v for k, v in features.items() if 'rsi' in k or 'macd' in k or 'bb' in k or 'stoch' in k or 'atr' in k or 'adx' in k or 'cci' in k},
                regime=regime_result.regime,
                signal_direction=ensemble_pred.direction,
                volume_features={k: v for k, v in features.items() if 'volume' in k or 'obv' in k or 'mfi' in k or 'force' in k or 'vpt' in k or 'ad' in k or 'cmf' in k or 'vwap' in k or 'eom' in k}
            )
            
            logger.info(f"Validation: {validation.grade.value} (score: {validation.reliability_score:.2f}%)")
            
            # Skip if didn't pass validation
            if not validation.passed or validation.grade == SignalGrade.SKIP:
                logger.info(f"Signal for {symbol} did not pass premium filter")
                return None
            
            # Step 8: Calculate entry/exit levels
            current_price = df_daily['close'].iloc[-1]
            entry_price, stop_loss, target_1, target_2 = self._calculate_levels(
                current_price,
                ensemble_pred.direction,
                features['atr_percentage'],
                validation.reliability_score
            )
            
            risk_reward = abs(target_1 - entry_price) / abs(entry_price - stop_loss)
            
            # Step 9: Calculate dynamic risk
            risk_allocation = self.risk_manager.calculate_risk_allocation(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                ai_confidence=ensemble_pred.confidence,
                volatility_atr_percent=features['atr_percentage'],
                portfolio_positions=portfolio_positions,
                recent_trades=recent_trades,
                account_value=account_value,
                market_correlation=features.get('beta', 1.0)
            )
            
            logger.info(
                f"Dynamic risk: {risk_allocation.final_risk_percent}% "
                f"(base: {risk_allocation.base_risk_percent}%, "
                f"multiplier: {risk_allocation.multipliers.total_multiplier}x)"
            )
            
            # Step 10: Create enhanced signal
            signal = EnhancedSignal(
                symbol=symbol,
                direction=ensemble_pred.direction,
                timestamp=datetime.now(),
                
                # AI predictions
                ai_confidence=ensemble_pred.confidence,
                model_predictions=ensemble_pred.model_predictions,
                model_weights=ensemble_pred.final_weights,
                agreement_score=ensemble_pred.agreement_score,
                uncertainty=ensemble_pred.uncertainty,
                
                # Features
                features=features,
                
                # Market context
                regime=regime_result.regime.value,
                regime_confidence=regime_result.confidence,
                
                # Validation
                validation_score=validation.reliability_score,
                signal_grade=validation.grade.value,
                passed_validation=validation.passed,
                validation_details=asdict(validation),
                
                # Strategy (placeholder)
                strategy_confluence=strategy_confluence,
                active_strategies=active_strategies,
                
                # Risk management
                base_risk_percent=risk_allocation.base_risk_percent,
                final_risk_percent=risk_allocation.final_risk_percent,
                risk_multipliers=risk_allocation.multipliers.to_dict(),
                
                # Entry/Exit levels
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                risk_reward_ratio=risk_reward,
                
                # Metadata
                reliability_score=validation.reliability_score,
                execution_priority=validation.grade.value
            )
            
            logger.info(f"✅ Enhanced signal generated for {symbol}: {signal.direction} @ {entry_price:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None
    
    def _fetch_data(
        self, 
        symbol: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch multi-timeframe data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch different timeframes
            df_daily = ticker.history(period="6mo", interval="1d")
            df_hourly = ticker.history(period="1mo", interval="1h")
            df_weekly = ticker.history(period="2y", interval="1wk")
            
            return df_daily, df_hourly, df_weekly
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None, None
    
    async def _get_model_predictions(
        self, 
        features: Dict[str, float],
        regime: MarketRegime
    ) -> List[ModelPrediction]:
        """
        Get predictions from all 5 models
        
        In production, this would call the Modal.com endpoint.
        For now, we'll generate mock predictions based on features.
        """
        # If Modal endpoint is configured, use it
        if self.modal_endpoint:
            try:
                return await self._call_modal_endpoint(features)
            except Exception as e:
                logger.warning(f"Modal endpoint failed: {e}, using fallback")
        
        # Fallback: Generate predictions based on features
        return self._generate_fallback_predictions(features, regime)
    
    async def _call_modal_endpoint(
        self, 
        features: Dict[str, float]
    ) -> List[ModelPrediction]:
        """Call Modal.com ML inference endpoint"""
        endpoint = self.modal_endpoint.rstrip("/")
        if not endpoint.endswith("/predict"):
            endpoint = f"{endpoint}/predict"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, json={"features": features})
            response.raise_for_status()
            data = response.json()

            payload = data.get("result") or data
            predictions = []

            # Legacy payload: {"predictions": {model: {prediction, confidence}}}
            if "predictions" in payload:
                for model_name, pred_data in payload["predictions"].items():
                    predictions.append(
                        ModelPrediction(
                            model_name=model_name,
                            prediction=pred_data["prediction"],
                            confidence=pred_data.get("confidence", 70.0),
                            features_used=len(features),
                        )
                    )
                return predictions

            # V2 payload: {"model_predictions": {...}, "model_confidences": {...}}
            if "model_predictions" in payload:
                confidences = payload.get("model_confidences", {})
                for model_name, pred in payload["model_predictions"].items():
                    predictions.append(
                        ModelPrediction(
                            model_name=model_name,
                            prediction=pred,
                            confidence=confidences.get(model_name, payload.get("confidence", 70.0)),
                            features_used=len(features),
                        )
                    )
                return predictions

            raise ValueError("Unsupported inference response format")
    
    def _generate_fallback_predictions(
        self,
        features: Dict[str, float],
        regime: MarketRegime
    ) -> List[ModelPrediction]:
        """
        Generate rule-based predictions as fallback
        
        This simulates what the 5 AI models would return.
        """
        # Calculate base prediction from features
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Technical indicators
        if features.get('rsi_14', 50) < 40:
            bullish_signals += 2
            total_signals += 2
        elif features.get('rsi_14', 50) > 60:
            bearish_signals += 2
            total_signals += 2
        
        if features.get('macd_histogram', 0) > 0:
            bullish_signals += 2
            total_signals += 2
        elif features.get('macd_histogram', 0) < 0:
            bearish_signals += 2
            total_signals += 2
        
        # Price action
        if features.get('trend_direction', 50) > 60:
            bullish_signals += 2
            total_signals += 2
        elif features.get('trend_direction', 50) < 40:
            bearish_signals += 2
            total_signals += 2
        
        # SMC
        if features.get('accumulation_phase', 0) > 60:
            bullish_signals += 2
            total_signals += 2
        elif features.get('distribution_phase', 0) > 60:
            bearish_signals += 2
            total_signals += 2
        
        # MTF
        if features.get('mtf_confluence', 50) > 60:
            bullish_signals += 1
            total_signals += 1
        elif features.get('mtf_confluence', 50) < 40:
            bearish_signals += 1
            total_signals += 1
        
        # Calculate base prediction (0-100 scale)
        if total_signals > 0:
            base_prediction = (bullish_signals / total_signals) * 100
        else:
            base_prediction = 50
        
        # Generate predictions for each model with variation
        predictions = []
        
        # TFT (Temporal Fusion Transformer) - Best for sequential patterns
        tft_pred = base_prediction + np.random.normal(0, 5)
        tft_pred = np.clip(tft_pred, 0, 100)
        predictions.append(ModelPrediction(
            model_name="TFT",
            prediction=round(tft_pred, 2),
            confidence=round(75 + np.random.uniform(0, 15), 2),
            features_used=70
        ))
        
        # LSTM - Good for time series
        lstm_pred = base_prediction + np.random.normal(0, 8)
        lstm_pred = np.clip(lstm_pred, 0, 100)
        predictions.append(ModelPrediction(
            model_name="LSTM",
            prediction=round(lstm_pred, 2),
            confidence=round(70 + np.random.uniform(0, 15), 2),
            features_used=70
        ))
        
        # XGBoost - Rule-based trees
        xgb_pred = base_prediction + np.random.normal(0, 7)
        xgb_pred = np.clip(xgb_pred, 0, 100)
        predictions.append(ModelPrediction(
            model_name="XGBoost",
            prediction=round(xgb_pred, 2),
            confidence=round(72 + np.random.uniform(0, 15), 2),
            features_used=70
        ))
        
        # Random Forest - Ensemble of trees
        rf_pred = base_prediction + np.random.normal(0, 10)
        rf_pred = np.clip(rf_pred, 0, 100)
        predictions.append(ModelPrediction(
            model_name="RandomForest",
            prediction=round(rf_pred, 2),
            confidence=round(68 + np.random.uniform(0, 15), 2),
            features_used=70
        ))
        
        # SVM - Boundary detection
        svm_pred = base_prediction + np.random.normal(0, 12)
        svm_pred = np.clip(svm_pred, 0, 100)
        predictions.append(ModelPrediction(
            model_name="SVM",
            prediction=round(svm_pred, 2),
            confidence=round(65 + np.random.uniform(0, 15), 2),
            features_used=70
        ))
        
        return predictions
    
    def _calculate_levels(
        self,
        current_price: float,
        direction: str,
        atr_percent: float,
        confidence: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate entry, stop loss, and target levels
        
        Returns:
            (entry_price, stop_loss, target_1, target_2)
        """
        # ATR in price terms
        atr = current_price * (atr_percent / 100)
        
        # Adjust levels based on confidence
        confidence_factor = confidence / 100
        
        if direction == "LONG":
            entry = current_price * 1.002  # Slight premium
            stop_loss = entry - (atr * 2.0)
            target_1 = entry + (atr * 3.0 * confidence_factor)
            target_2 = entry + (atr * 5.0 * confidence_factor)
        elif direction == "SHORT":
            entry = current_price * 0.998  # Slight discount
            stop_loss = entry + (atr * 2.0)
            target_1 = entry - (atr * 3.0 * confidence_factor)
            target_2 = entry - (atr * 5.0 * confidence_factor)
        else:  # NEUTRAL
            entry = current_price
            stop_loss = current_price
            target_1 = current_price
            target_2 = current_price
        
        return (
            round(entry, 2),
            round(stop_loss, 2),
            round(target_1, 2),
            round(target_2, 2)
        )
    
    def _get_strategy_confluence(
        self,
        df: pd.DataFrame,
        features: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """
        Get strategy confluence score from StrategySelector.
        
        Args:
            df: Daily DataFrame with OHLCV
            features: Calculated features dict
            
        Returns:
            (confluence_score, list of active strategy names)
        """
        try:
            from ml.strategies.strategy_selector import StrategySelector
            
            # Prepare DataFrame with required columns for strategy selector
            strategy_df = df.copy()
            
            # Add features to DataFrame for strategy analysis
            # Map our features to what strategies expect
            feature_mapping = {
                'RSI': features.get('rsi_14', 50),
                'MACD': features.get('macd_line', 0),
                'MACD_Signal': features.get('macd_signal', 0),
                'BB_Upper': features.get('bb_upper', 0),
                'BB_Middle': features.get('bb_middle', 0),
                'BB_Lower': features.get('bb_lower', 0),
                'Stochastic_K': features.get('stoch_k', 50),
                'ATR': features.get('atr', 0),
                'CCI': features.get('cci', 0),
                'ADX': features.get('adx', 25),
                'EMA_50': features.get('ema_50', 0),
                'EMA_200': features.get('ema_200', 0),
                'VWAP': features.get('vwap', 0),
                'Support_Level': features.get('support_level', 0),
                'Resistance_Level': features.get('resistance_level', 0),
                'Fib_618': features.get('fib_618', 0),
                'Pivot_Point': features.get('pivot_point', 0),
                'Volume_MA': features.get('volume_ma_20', 0),
                'Volume_Spike': 1 if features.get('volume_ratio', 1) > 1.5 else 0,
                'OB_Strength': features.get('order_block_strength', 50),
                'OB_Distance_Pct': features.get('ob_distance_pct', 0),
                'FVG_Distance': features.get('fvg_distance', 0),
                'FVG_Volume_Ratio': features.get('fvg_volume_ratio', 1),
                'Sweep_Detected': 1 if features.get('liquidity_sweep', 0) > 0.5 else 0,
                'Post_Sweep_Reversal_Prob': features.get('post_sweep_reversal_prob', 0.5),
                'Inst_Activity_Score': features.get('institutional_activity', 50),
                'Accumulation_Phase': features.get('accumulation_phase', 0),
                'Distribution_Phase': features.get('distribution_phase', 0),
                'Liquidity_Level': features.get('liquidity_level', 50),
                'Weekly_Trend': features.get('weekly_trend', 0),
                'Daily_Trend': features.get('daily_trend', 0),
                'Hourly_Trend': features.get('hourly_trend', 0),
                'MTF_Confluence': features.get('mtf_confluence', 50),
                'Volume_Conf_Weekly': 1 if features.get('weekly_volume_conf', 0) > 0 else 0,
                'RSI_Divergence': 1 if features.get('rsi_divergence', 0) > 0 else 0,
                'MA_Alignment': features.get('ma_alignment', 50),
                'Order_Flow_Imbalance': features.get('order_flow_imbalance', 0),
                'Momentum_Composite': features.get('momentum_composite', 50),
                'Reversal_Probability': features.get('reversal_probability', 50),
                'Support_Resistance_Strength': features.get('sr_strength', 50),
                'Volatility_Squeeze': features.get('volatility_squeeze', 50),
            }
            
            # Add features to last row of DataFrame
            for col, val in feature_mapping.items():
                if col not in strategy_df.columns:
                    strategy_df[col] = val
            
            # Initialize strategy selector and get top strategies
            selector = StrategySelector()
            current_idx = len(strategy_df) - 1
            
            top_strategies = selector.get_top_strategies(strategy_df, current_idx, top_n=5)
            
            if top_strategies:
                # Calculate weighted average confluence
                total_weight = 0
                weighted_confluence = 0
                active_names = []
                
                for i, strat in enumerate(top_strategies):
                    # Weight by position (first strategy gets more weight)
                    weight = 1.0 / (i + 1)
                    weighted_confluence += strat['confluence'] * weight
                    total_weight += weight
                    
                    if strat['confluence'] > 60:  # Only include high-confluence strategies
                        active_names.append(f"{strat['name']} ({strat['tier']}, {strat['confluence']:.0f}%)")
                
                final_confluence = weighted_confluence / total_weight if total_weight > 0 else 50.0
                
                logger.info(f"Strategy confluence: {final_confluence:.1f}% from {len(active_names)} strategies")
                return final_confluence, active_names
            
            return 50.0, ["No strategies matched"]
            
        except ImportError as e:
            logger.warning(f"Strategy selector not available: {e}")
            return 50.0, ["Strategy selector not available"]
        except Exception as e:
            logger.warning(f"Strategy confluence calculation failed: {e}")
            # Fallback to feature-based confluence
            confluence = self._calculate_fallback_confluence(features)
            return confluence, ["Fallback confluence calculation"]
    
    def _calculate_fallback_confluence(self, features: Dict[str, float]) -> float:
        """
        Calculate a basic confluence score from features when strategy selector fails.
        """
        score = 50.0  # Start neutral
        
        # RSI contribution
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            score += 15  # Oversold = bullish
        elif rsi > 70:
            score -= 15  # Overbought = bearish
        
        # MACD contribution
        if features.get('macd_histogram', 0) > 0:
            score += 10
        else:
            score -= 10
        
        # Trend contribution
        trend = features.get('trend_direction', 50)
        score += (trend - 50) * 0.3
        
        # Volume contribution
        if features.get('volume_ratio', 1) > 1.5:
            score += 5
        
        # MTF alignment
        mtf = features.get('mtf_confluence', 50)
        score += (mtf - 50) * 0.2
        
        return max(0, min(100, score))


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage"""
    print("\n" + "="*80)
    print("ENHANCED SIGNAL GENERATOR TEST")
    print("="*80)
    
    # Initialize generator
    generator = EnhancedSignalGenerator(
        modal_endpoint=None,  # Will use fallback
        use_adaptive_weighting=True
    )
    
    # Mock portfolio data
    portfolio_positions = [
        {'symbol': 'TCS.NS', 'direction': 'LONG'},
        {'symbol': 'INFY.NS', 'direction': 'LONG'}
    ]
    
    recent_trades = [
        {'pnl': 500}, {'pnl': -200}, {'pnl': 300},
        {'pnl': 150}, {'pnl': 400}, {'pnl': -100}
    ]
    
    market_data = {
        'nifty_change_percent': 0.5,
        'vix_close': 14.5,
        'fii_cash': 1200,
        'dii_cash': 800,
        'advances': 1300,
        'declines': 700
    }
    
    # Generate signal
    signal = await generator.generate_signal(
        symbol="RELIANCE.NS",
        account_value=1000000.0,
        portfolio_positions=portfolio_positions,
        recent_trades=recent_trades,
        market_data=market_data
    )
    
    if signal:
        print(f"\n✅ SIGNAL GENERATED")
        print("-" * 80)
        print(f"Symbol: {signal.symbol}")
        print(f"Direction: {signal.direction}")
        print(f"Grade: {signal.signal_grade}")
        print(f"\nAI Confidence: {signal.ai_confidence:.2f}%")
        print(f"Agreement Score: {signal.agreement_score:.2f}%")
        print(f"Uncertainty: {signal.uncertainty:.2f}%")
        print(f"\nRegime: {signal.regime} ({signal.regime_confidence:.2f}%)")
        print(f"\nEntry: ₹{signal.entry_price:.2f}")
        print(f"Stop Loss: ₹{signal.stop_loss:.2f}")
        print(f"Target 1: ₹{signal.target_1:.2f}")
        print(f"Target 2: ₹{signal.target_2:.2f}")
        print(f"Risk:Reward: 1:{signal.risk_reward_ratio:.2f}")
        print(f"\nRisk: {signal.final_risk_percent}% (base: {signal.base_risk_percent}%)")
        print(f"\nReliability Score: {signal.reliability_score:.2f}%")
        print("-" * 80)
    else:
        print("\n❌ No signal generated (did not pass filters)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
