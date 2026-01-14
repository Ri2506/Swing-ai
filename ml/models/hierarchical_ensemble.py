"""
================================================================================
SWINGAI - HIERARCHICAL ENSEMBLE WITH ADAPTIVE WEIGHTING
================================================================================
Adaptive ensemble that adjusts model weights based on agreement:
- Base models: TFT (40%), LSTM (25%), XGBoost (20%), Random Forest (10%), SVM (5%)
- Agreement detection: Compare each model to ensemble mean
- Dynamic weighting: Models that agree get MORE weight
- Uncertainty quantification: Track prediction variance
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Individual model prediction"""
    model_name: str
    prediction: float  # 0-100 (0=SHORT, 100=LONG)
    confidence: float  # 0-100
    features_used: int


@dataclass
class EnsemblePrediction:
    """Final ensemble prediction with metadata"""
    prediction: float  # 0-100 (0=SHORT, 100=LONG)
    confidence: float  # 0-100
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    
    # Model contributions
    model_predictions: Dict[str, float]
    final_weights: Dict[str, float]
    
    # Agreement metrics
    agreement_score: float  # 0-100
    disagreement_penalty: float  # 0-1
    uncertainty: float  # 0-100
    
    # Metadata
    models_count: int
    consensus_strength: float  # 0-100


class HierarchicalEnsemble:
    """
    Hierarchical Ensemble with Adaptive Weighting
    
    Combines predictions from multiple models with dynamic weight adjustment
    based on model agreement and historical performance.
    """
    
    def __init__(self, use_adaptive_weights: bool = True):
        """
        Initialize ensemble
        
        Args:
            use_adaptive_weights: If True, adjust weights based on agreement
        """
        self.use_adaptive_weights = use_adaptive_weights
        
        # Base weights (sum to 1.0)
        self.base_weights = {
            'TFT': 0.40,
            'LSTM': 0.25,
            'XGBoost': 0.20,
            'RandomForest': 0.10,
            'SVM': 0.05
        }
        
        # Performance tracking for adaptive weighting
        self.model_performance_history = {model: [] for model in self.base_weights.keys()}
        self.prediction_history = []
    
    def predict(
        self, 
        model_predictions: List[ModelPrediction],
        market_regime: Optional[str] = None
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction with adaptive weighting
        
        Args:
            model_predictions: List of predictions from individual models
            market_regime: Current market regime (BULLISH/BEARISH/RANGE/CHOPPY)
            
        Returns:
            EnsemblePrediction with final prediction and metadata
        """
        if not model_predictions:
            raise ValueError("No model predictions provided")
        
        # Step 1: Extract predictions and confidences
        pred_dict = {mp.model_name: mp.prediction for mp in model_predictions}
        conf_dict = {mp.model_name: mp.confidence for mp in model_predictions}
        
        # Step 2: Calculate agreement metrics
        agreement_score, std_dev = self._calculate_agreement(list(pred_dict.values()))
        
        # Step 3: Calculate adaptive weights
        if self.use_adaptive_weights:
            final_weights = self._calculate_adaptive_weights(
                pred_dict, 
                agreement_score,
                market_regime
            )
        else:
            final_weights = self.base_weights.copy()
        
        # Step 4: Calculate weighted prediction
        weighted_pred = sum(
            pred_dict.get(model, 50) * final_weights.get(model, 0)
            for model in final_weights.keys()
        )
        
        # Step 5: Calculate weighted confidence
        weighted_conf = sum(
            conf_dict.get(model, 50) * final_weights.get(model, 0)
            for model in final_weights.keys()
        )
        
        # Step 6: Apply disagreement penalty to confidence
        disagreement_penalty = self._calculate_disagreement_penalty(std_dev)
        final_confidence = weighted_conf * (1 - disagreement_penalty)
        
        # Step 7: Calculate uncertainty
        uncertainty = self._calculate_uncertainty(list(pred_dict.values()))
        
        # Step 8: Determine direction
        if weighted_pred >= 60:
            direction = "LONG"
        elif weighted_pred <= 40:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"
        
        # Step 9: Calculate consensus strength
        consensus_strength = self._calculate_consensus_strength(
            list(pred_dict.values()),
            direction
        )
        
        # Create ensemble prediction
        ensemble_pred = EnsemblePrediction(
            prediction=round(weighted_pred, 2),
            confidence=round(final_confidence, 2),
            direction=direction,
            model_predictions=pred_dict,
            final_weights=final_weights,
            agreement_score=round(agreement_score, 2),
            disagreement_penalty=round(disagreement_penalty, 3),
            uncertainty=round(uncertainty, 2),
            models_count=len(model_predictions),
            consensus_strength=round(consensus_strength, 2)
        )
        
        # Track prediction
        self.prediction_history.append(ensemble_pred)
        
        return ensemble_pred
    
    def _calculate_agreement(self, predictions: List[float]) -> Tuple[float, float]:
        """
        Calculate agreement score among models
        
        Args:
            predictions: List of model predictions (0-100)
            
        Returns:
            (agreement_score, standard_deviation)
        """
        predictions_array = np.array(predictions)
        
        # Calculate standard deviation
        std_dev = np.std(predictions_array)
        
        # Convert std_dev to agreement score (0-100)
        # Lower std_dev = Higher agreement
        # Assume max std_dev of 50 for normalization
        agreement_score = max(0, 100 - (std_dev * 2))
        
        return agreement_score, std_dev
    
    def _calculate_adaptive_weights(
        self,
        predictions: Dict[str, float],
        agreement_score: float,
        market_regime: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on model agreement and regime
        
        Logic:
        - Models closer to ensemble mean get boosted
        - Models far from mean get reduced
        - Weights adjusted by market regime
        """
        # Calculate ensemble mean
        ensemble_mean = sum(
            pred * self.base_weights.get(model, 0)
            for model, pred in predictions.items()
        )
        
        # Calculate distance of each model from mean
        distances = {
            model: abs(pred - ensemble_mean)
            for model, pred in predictions.items()
        }
        
        # Calculate adjustment factors based on distance
        # Closer = higher factor (0.8 to 1.2)
        max_distance = max(distances.values()) if distances.values() else 1
        
        adjustment_factors = {}
        for model, distance in distances.items():
            if max_distance > 0:
                normalized_distance = distance / max_distance
                # Inverse relationship: close = 1.2x, far = 0.8x
                adjustment_factors[model] = 1.2 - (normalized_distance * 0.4)
            else:
                adjustment_factors[model] = 1.0
        
        # Apply adjustments to base weights
        adjusted_weights = {
            model: self.base_weights.get(model, 0) * adjustment_factors.get(model, 1.0)
            for model in self.base_weights.keys()
        }
        
        # Apply regime-specific adjustments
        if market_regime:
            adjusted_weights = self._apply_regime_adjustments(
                adjusted_weights,
                market_regime,
                predictions
            )
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                model: weight / total_weight
                for model, weight in adjusted_weights.items()
            }
        
        return adjusted_weights
    
    def _apply_regime_adjustments(
        self,
        weights: Dict[str, float],
        regime: str,
        predictions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply market regime-specific weight adjustments
        
        Different models perform better in different regimes:
        - TRENDING: Boost TFT and LSTM (sequential models)
        - RANGE: Boost tree models (XGBoost, RF)
        - CHOPPY: Equal weighting (high uncertainty)
        """
        regime_multipliers = {
            'BULLISH': {
                'TFT': 1.2,
                'LSTM': 1.1,
                'XGBoost': 1.0,
                'RandomForest': 0.9,
                'SVM': 0.8
            },
            'BEARISH': {
                'TFT': 1.2,
                'LSTM': 1.1,
                'XGBoost': 1.0,
                'RandomForest': 0.9,
                'SVM': 0.8
            },
            'RANGE': {
                'TFT': 0.9,
                'LSTM': 0.9,
                'XGBoost': 1.2,
                'RandomForest': 1.1,
                'SVM': 1.0
            },
            'CHOPPY': {
                'TFT': 1.0,
                'LSTM': 1.0,
                'XGBoost': 1.0,
                'RandomForest': 1.0,
                'SVM': 1.0
            }
        }
        
        multipliers = regime_multipliers.get(regime, {})
        
        adjusted = {
            model: weight * multipliers.get(model, 1.0)
            for model, weight in weights.items()
        }
        
        return adjusted
    
    def _calculate_disagreement_penalty(self, std_dev: float) -> float:
        """
        Calculate confidence penalty based on model disagreement
        
        Args:
            std_dev: Standard deviation of predictions
            
        Returns:
            Penalty factor (0-1), where higher = more penalty
        """
        # Low std_dev (< 10) = No penalty
        # High std_dev (> 30) = Max penalty (30%)
        
        if std_dev < 10:
            return 0.0
        elif std_dev > 30:
            return 0.30
        else:
            # Linear interpolation between 0 and 0.30
            return (std_dev - 10) / 20 * 0.30
    
    def _calculate_uncertainty(self, predictions: List[float]) -> float:
        """
        Calculate prediction uncertainty (0-100)
        
        Uses entropy and variance to measure uncertainty
        """
        predictions_array = np.array(predictions)
        
        # Variance-based uncertainty
        variance = np.var(predictions_array)
        variance_uncertainty = min(variance / 10, 100)  # Normalize
        
        # Entropy-based uncertainty (binned predictions)
        # Bin predictions into 10 buckets
        bins = np.linspace(0, 100, 11)
        hist, _ = np.histogram(predictions_array, bins=bins)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        
        # Calculate entropy
        pred_entropy = entropy(hist + 1e-10)  # Add small value to avoid log(0)
        max_entropy = np.log(10)  # Max entropy for 10 bins
        entropy_uncertainty = (pred_entropy / max_entropy) * 100
        
        # Combined uncertainty
        uncertainty = (variance_uncertainty + entropy_uncertainty) / 2
        
        return min(uncertainty, 100)
    
    def _calculate_consensus_strength(
        self,
        predictions: List[float],
        direction: str
    ) -> float:
        """
        Calculate how strongly models agree on the direction
        
        Args:
            predictions: List of model predictions
            direction: Final ensemble direction
            
        Returns:
            Consensus strength (0-100)
        """
        if direction == "NEUTRAL":
            return 50.0
        
        # Count models aligned with direction
        aligned_count = 0
        threshold_long = 60
        threshold_short = 40
        
        for pred in predictions:
            if direction == "LONG" and pred >= threshold_long:
                aligned_count += 1
            elif direction == "SHORT" and pred <= threshold_short:
                aligned_count += 1
        
        consensus = (aligned_count / len(predictions)) * 100
        
        return consensus
    
    def update_performance(
        self,
        model_name: str,
        actual_outcome: float,
        predicted: float
    ):
        """
        Update model performance history
        
        Args:
            model_name: Name of the model
            actual_outcome: Actual result (0-100)
            predicted: Model's prediction (0-100)
        """
        error = abs(actual_outcome - predicted)
        accuracy = max(0, 100 - error)
        
        if model_name in self.model_performance_history:
            self.model_performance_history[model_name].append(accuracy)
            
            # Keep only recent 100 predictions
            if len(self.model_performance_history[model_name]) > 100:
                self.model_performance_history[model_name].pop(0)
    
    def get_model_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance summary for all models
        
        Returns:
            Dictionary with model statistics
        """
        summary = {}
        
        for model, history in self.model_performance_history.items():
            if history:
                summary[model] = {
                    'mean_accuracy': np.mean(history),
                    'std_accuracy': np.std(history),
                    'min_accuracy': np.min(history),
                    'max_accuracy': np.max(history),
                    'predictions_count': len(history)
                }
            else:
                summary[model] = {
                    'mean_accuracy': 0,
                    'std_accuracy': 0,
                    'min_accuracy': 0,
                    'max_accuracy': 0,
                    'predictions_count': 0
                }
        
        return summary
    
    def get_prediction_statistics(self) -> Dict[str, float]:
        """
        Get statistics about recent ensemble predictions
        
        Returns:
            Dictionary with ensemble statistics
        """
        if not self.prediction_history:
            return {}
        
        recent = self.prediction_history[-50:]  # Last 50 predictions
        
        return {
            'avg_confidence': np.mean([p.confidence for p in recent]),
            'avg_agreement': np.mean([p.agreement_score for p in recent]),
            'avg_uncertainty': np.mean([p.uncertainty for p in recent]),
            'avg_consensus': np.mean([p.consensus_strength for p in recent]),
            'predictions_count': len(recent),
            'long_signals': sum(1 for p in recent if p.direction == "LONG"),
            'short_signals': sum(1 for p in recent if p.direction == "SHORT"),
            'neutral_signals': sum(1 for p in recent if p.direction == "NEUTRAL")
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create ensemble
    ensemble = HierarchicalEnsemble(use_adaptive_weights=True)
    
    # Example: 5 model predictions
    model_preds = [
        ModelPrediction("TFT", 75.0, 82.0, 70),
        ModelPrediction("LSTM", 68.0, 75.0, 70),
        ModelPrediction("XGBoost", 78.0, 85.0, 70),
        ModelPrediction("RandomForest", 72.0, 80.0, 70),
        ModelPrediction("SVM", 65.0, 70.0, 70)
    ]
    
    # Get ensemble prediction
    result = ensemble.predict(model_preds, market_regime="BULLISH")
    
    print("\n" + "="*80)
    print("HIERARCHICAL ENSEMBLE PREDICTION")
    print("="*80)
    print(f"\nFinal Prediction: {result.prediction:.2f} ({result.direction})")
    print(f"Confidence: {result.confidence:.2f}%")
    print(f"\nAgreement Score: {result.agreement_score:.2f}%")
    print(f"Consensus Strength: {result.consensus_strength:.2f}%")
    print(f"Uncertainty: {result.uncertainty:.2f}%")
    print(f"Disagreement Penalty: {result.disagreement_penalty:.3f}")
    
    print(f"\n{'Model':<15} {'Prediction':<12} {'Base Weight':<12} {'Final Weight':<12}")
    print("-" * 80)
    for model in ensemble.base_weights.keys():
        pred = result.model_predictions.get(model, 0)
        base_w = ensemble.base_weights.get(model, 0)
        final_w = result.final_weights.get(model, 0)
        print(f"{model:<15} {pred:>8.2f}     {base_w:>8.2%}     {final_w:>8.2%}")
    
    print("\n" + "="*80)
    
    # Test with disagreement
    print("\n\nTEST 2: High Disagreement Scenario")
    print("="*80)
    
    conflicting_preds = [
        ModelPrediction("TFT", 85.0, 90.0, 70),
        ModelPrediction("LSTM", 80.0, 85.0, 70),
        ModelPrediction("XGBoost", 30.0, 75.0, 70),  # Disagrees!
        ModelPrediction("RandomForest", 75.0, 80.0, 70),
        ModelPrediction("SVM", 25.0, 70.0, 70)  # Disagrees!
    ]
    
    result2 = ensemble.predict(conflicting_preds, market_regime="CHOPPY")
    
    print(f"\nFinal Prediction: {result2.prediction:.2f} ({result2.direction})")
    print(f"Confidence: {result2.confidence:.2f}% (reduced due to disagreement)")
    print(f"Agreement Score: {result2.agreement_score:.2f}% (LOW!)")
    print(f"Uncertainty: {result2.uncertainty:.2f}% (HIGH!)")
    print(f"Disagreement Penalty: {result2.disagreement_penalty:.3f} (penalty applied)")
    
    print("\n" + "="*80)
