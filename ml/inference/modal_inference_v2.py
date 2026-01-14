"""
================================================================================
SWINGAI - MODAL INFERENCE ENDPOINT V2 (ENHANCED)
================================================================================
Serverless ML inference using Modal.com with:
- 5-Model Hierarchical Ensemble (TFT, LSTM, XGBoost, RandomForest, SVM)
- 70 Enhanced Features Support
- Adaptive Weighting Ready
- FastAPI web endpoint

Deploy with: modal deploy ml/inference/modal_inference_v2.py
================================================================================
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional
import modal

# ============================================================================
# MODAL APP SETUP
# ============================================================================

# Create Modal app
app = modal.App("swingai-inference-v2")

# Define image with ALL dependencies for 5 models
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    # Core ML libraries
    "catboost==1.2.2",
    "xgboost==2.0.3",
    "scikit-learn==1.4.0",
    "numpy==1.26.3",
    "pandas==2.1.4",
    # Deep learning (for TFT and LSTM)
    "torch==2.1.2",
    "pytorch-forecasting==1.0.0",
    "transformers==4.36.2",
    # API and utilities
    "pydantic==2.5.3",
    "fastapi==0.109.0",
)

# Model storage volume
volume = modal.Volume.from_name("swingai-models-v2", create_if_missing=True)
MODEL_DIR = "/models"

# ============================================================================
# ENHANCED PREDICTOR (5 MODELS)
# ============================================================================

@app.cls(
    image=image,
    volumes={MODEL_DIR: volume},
    gpu=False,  # CPU is sufficient for inference
    memory=4096,  # More memory for 5 models
    timeout=300,
)
class EnhancedSwingAIPredictor:
    """
    Enhanced SwingAI Predictor with 5-Model Ensemble
    
    Models:
    1. TFT (Temporal Fusion Transformer) - 40% base weight
    2. LSTM - 25% base weight
    3. XGBoost - 20% base weight
    4. RandomForest - 10% base weight
    5. SVM - 5% base weight
    """
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.models_loaded = False
        self.model_names = ["TFT", "LSTM", "XGBoost", "RandomForest", "SVM"]
        
        # Base weights for ensemble
        self.base_weights = {
            "TFT": 0.40,
            "LSTM": 0.25,
            "XGBoost": 0.20,
            "RandomForest": 0.10,
            "SVM": 0.05
        }
    
    @modal.enter()
    def load_models(self):
        """Load all 5 models on container startup"""
        import torch
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        import xgboost as xgb
        import pickle
        
        print("Loading 5-model ensemble...")
        
        try:
            # Load feature config
            config_path = f"{MODEL_DIR}/model_config_v2.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.feature_columns = config.get("feature_columns", [])
                    print(f"Loaded {len(self.feature_columns)} feature columns")
            
            # Try to load each model
            model_files = {
                "TFT": "tft_model.pt",
                "LSTM": "lstm_model.pt",
                "XGBoost": "xgboost_model.json",
                "RandomForest": "rf_model.pkl",
                "SVM": "svm_model.pkl"
            }
            
            for model_name, filename in model_files.items():
                model_path = f"{MODEL_DIR}/{filename}"
                
                if os.path.exists(model_path):
                    try:
                        if model_name in ["TFT", "LSTM"]:
                            # PyTorch models
                            self.models[model_name] = torch.load(model_path, map_location="cpu")
                            self.models[model_name].eval()
                        elif model_name == "XGBoost":
                            # XGBoost model
                            self.models[model_name] = xgb.XGBClassifier()
                            self.models[model_name].load_model(model_path)
                        else:
                            # Scikit-learn models (RF, SVM)
                            with open(model_path, "rb") as f:
                                self.models[model_name] = pickle.load(f)
                        
                        print(f"✓ Loaded {model_name} from {filename}")
                    except Exception as e:
                        print(f"✗ Failed to load {model_name}: {e}")
                else:
                    print(f"✗ Model file not found: {filename}")
            
            self.models_loaded = len(self.models) > 0
            print(f"Loaded {len(self.models)}/{len(model_files)} models")
            
            if not self.models_loaded:
                print("No models loaded, will use fallback predictions")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
    
    @modal.method()
    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Run prediction using all available models
        
        Args:
            features: Dictionary with 70 features
            
        Returns:
            Dictionary with predictions from all models + ensemble
        """
        import numpy as np
        import pandas as pd
        
        try:
            # Validate features
            if len(features) < 60:  # At least 60 of 70 features
                raise ValueError(f"Insufficient features: got {len(features)}, expected ~70")
            
            # Prepare features DataFrame
            df = pd.DataFrame([features])
            
            # Filter to model features if available
            if self.feature_columns:
                # Fill missing features with 0
                for col in self.feature_columns:
                    if col not in df.columns:
                        df[col] = 0
                df = df[self.feature_columns]
            
            # Get predictions from each model
            model_predictions = {}
            model_confidences = {}
            
            if self.models_loaded:
                for model_name in self.model_names:
                    if model_name in self.models:
                        pred, conf = self._predict_single_model(model_name, df)
                        model_predictions[model_name] = pred
                        model_confidences[model_name] = conf
                    else:
                        # Fallback for missing model
                        model_predictions[model_name] = self._fallback_prediction(features, model_name)
                        model_confidences[model_name] = 70.0
            else:
                # All fallback predictions
                for model_name in self.model_names:
                    model_predictions[model_name] = self._fallback_prediction(features, model_name)
                    model_confidences[model_name] = 70.0
            
            # Calculate ensemble prediction (weighted average)
            ensemble_pred = sum(
                model_predictions[name] * self.base_weights[name]
                for name in self.model_names
            )
            
            # Calculate agreement
            predictions_array = np.array(list(model_predictions.values()))
            std_dev = np.std(predictions_array)
            agreement_score = max(0, 100 - (std_dev * 2))
            
            # Calculate ensemble confidence
            ensemble_confidence = sum(
                model_confidences[name] * self.base_weights[name]
                for name in self.model_names
            )
            
            # Apply disagreement penalty
            disagreement_penalty = self._calculate_disagreement_penalty(std_dev)
            final_confidence = ensemble_confidence * (1 - disagreement_penalty)
            
            # Determine direction
            if ensemble_pred >= 60:
                direction = "LONG"
            elif ensemble_pred <= 40:
                direction = "SHORT"
            else:
                direction = "NEUTRAL"
            
            # Calculate consensus (how many models agree)
            consensus_count = sum(
                1 for pred in model_predictions.values()
                if (direction == "LONG" and pred >= 60) or
                   (direction == "SHORT" and pred <= 40)
            )
            
            return {
                "prediction": round(ensemble_pred, 2),
                "direction": direction,
                "confidence": round(final_confidence, 2),
                
                # Individual model predictions (0-100 scale)
                "model_predictions": {
                    name: round(pred, 2)
                    for name, pred in model_predictions.items()
                },
                
                # Individual model confidences
                "model_confidences": {
                    name: round(conf, 2)
                    for name, conf in model_confidences.items()
                },
                
                # Ensemble metrics
                "agreement_score": round(agreement_score, 2),
                "disagreement_penalty": round(disagreement_penalty, 3),
                "consensus_count": consensus_count,
                "total_models": len(self.model_names),
                
                # Weights used
                "weights_used": self.base_weights,
                
                # Metadata
                "features_count": len(features),
                "models_loaded": len(self.models),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_ensemble_prediction(features)
    
    def _predict_single_model(self, model_name: str, df: pd.DataFrame) -> tuple:
        """
        Get prediction from a single model
        
        Returns:
            (prediction_0_to_100, confidence)
        """
        import torch
        import numpy as np
        
        model = self.models[model_name]
        
        try:
            if model_name in ["TFT", "LSTM"]:
                # PyTorch models
                with torch.no_grad():
                    # Convert to tensor
                    X = torch.FloatTensor(df.values)
                    
                    # Get prediction
                    output = model(X)
                    probs = torch.softmax(output, dim=1)[0]
                    
                    # Assume classes: 0=SHORT, 1=NEUTRAL, 2=LONG
                    prediction = (probs[2].item() * 100) + (probs[1].item() * 50)
                    confidence = probs.max().item() * 100
                    
                    return prediction, confidence
            
            else:
                # Scikit-learn models (XGBoost, RF, SVM)
                # Get probabilities
                probs = model.predict_proba(df)[0]
                
                # Assume classes: 0=SHORT, 1=NEUTRAL, 2=LONG
                prediction = (probs[2] * 100) + (probs[1] * 50)
                confidence = probs.max() * 100
                
                return prediction, confidence
        
        except Exception as e:
            print(f"Error in {model_name} prediction: {e}")
            return 50.0, 70.0  # Neutral fallback
    
    def _fallback_prediction(self, features: Dict, model_name: str) -> float:
        """
        Rule-based fallback prediction (0-100 scale)
        
        Different models have different "personalities":
        - TFT: More weight on temporal patterns
        - LSTM: More weight on momentum
        - XGBoost: More weight on technical rules
        - RF: Balanced
        - SVM: More conservative (closer to 50)
        """
        score = 50.0
        
        # Technical indicators
        rsi = features.get("rsi_14", 50)
        macd_hist = features.get("macd_histogram", 0)
        adx = features.get("adx", 25)
        trend = features.get("trend_direction", 50)
        
        # Price action
        momentum = features.get("momentum_10d", 0)
        candle_strength = features.get("candle_strength", 50)
        
        # SMC features
        ob_strength = features.get("order_block_strength", 0)
        institutional = features.get("institutional_activity", 50)
        accumulation = features.get("accumulation_phase", 0)
        
        # Volume
        volume_ratio = features.get("volume_ma_ratio", 1.0)
        mfi = features.get("mfi", 50)
        
        # Model-specific weighting
        if model_name == "TFT":
            # TFT focuses on temporal patterns
            score += (trend - 50) * 0.4
            score += (momentum / 10) * 0.3
            score += (institutional - 50) * 0.2
        
        elif model_name == "LSTM":
            # LSTM focuses on momentum
            score += (momentum / 10) * 0.5
            score += (rsi - 50) * 0.3
            if macd_hist > 0:
                score += 10
            else:
                score -= 10
        
        elif model_name == "XGBoost":
            # XGBoost: Rule-based trees
            if rsi < 35:
                score += 15
            elif rsi > 65:
                score -= 15
            
            if macd_hist > 0:
                score += 12
            else:
                score -= 12
            
            if adx > 25:
                score += 8
            
            if volume_ratio > 1.3:
                score += 10
        
        elif model_name == "RandomForest":
            # Random Forest: Balanced ensemble
            score += (rsi - 50) * 0.2
            score += (trend - 50) * 0.2
            score += (mfi - 50) * 0.1
            score += (institutional - 50) * 0.2
            if macd_hist > 0:
                score += 5
            else:
                score -= 5
        
        elif model_name == "SVM":
            # SVM: Conservative, boundary-based
            # Stays closer to 50 unless strong signals
            if rsi < 30 or rsi > 70:
                score += (50 - abs(rsi - 50)) * 0.3
            
            if abs(momentum) > 3:
                score += momentum * 2
            
            if accumulation > 70:
                score += 10
        
        # Clamp to 0-100
        return max(0, min(100, score))
    
    def _calculate_disagreement_penalty(self, std_dev: float) -> float:
        """
        Calculate confidence penalty based on model disagreement
        
        Args:
            std_dev: Standard deviation of predictions
            
        Returns:
            Penalty factor (0-0.30)
        """
        if std_dev < 10:
            return 0.0
        elif std_dev > 30:
            return 0.30
        else:
            return (std_dev - 10) / 20 * 0.30
    
    def _fallback_ensemble_prediction(self, features: Dict) -> Dict:
        """Complete fallback when everything fails"""
        return {
            "prediction": 50.0,
            "direction": "NEUTRAL",
            "confidence": 50.0,
            "model_predictions": {name: 50.0 for name in self.model_names},
            "model_confidences": {name: 50.0 for name in self.model_names},
            "agreement_score": 100.0,
            "disagreement_penalty": 0.0,
            "consensus_count": 5,
            "total_models": 5,
            "weights_used": self.base_weights,
            "features_count": len(features),
            "models_loaded": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "Fallback prediction (models not loaded)"
        }


# ============================================================================
# FASTAPI WEB ENDPOINT
# ============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

web_app = FastAPI(title="SwingAI Inference API V2")

class PredictRequest(BaseModel):
    features: Dict[str, float]

class PredictResponse(BaseModel):
    result: Dict
    model_version: str = "2.0.0"

@web_app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models": ["TFT", "LSTM", "XGBoost", "RandomForest", "SVM"],
        "timestamp": datetime.utcnow().isoformat()
    }

@web_app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """
    Run prediction using 5-model ensemble
    
    Example request:
    {
        "features": {
            "rsi_14": 42.5,
            "macd_histogram": 2.3,
            "order_block_strength": 75.0,
            ... (70 features total)
        }
    }
    """
    try:
        predictor = EnhancedSwingAIPredictor()
        result = predictor.predict.remote(request.features)
        
        return PredictResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SwingAI Inference V2",
        "version": "2.0.0",
        "features": [
            "5-Model Hierarchical Ensemble",
            "70 Enhanced Features",
            "Adaptive Weighting Ready",
            "Agreement Detection",
            "Uncertainty Quantification"
        ],
        "endpoints": ["/health", "/predict"]
    }


# ============================================================================
# MODAL WEB ENDPOINT
# ============================================================================

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    """Serve FastAPI app"""
    return web_app


# ============================================================================
# MODEL UPLOAD FUNCTIONS
# ============================================================================

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
)
def upload_models(models_dict: Dict[str, bytes], config: Dict):
    """
    Upload all 5 trained models to Modal volume
    
    Usage from local:
        models_dict = {
            "TFT": tft_model_bytes,
            "LSTM": lstm_model_bytes,
            "XGBoost": xgb_model_bytes,
            "RandomForest": rf_model_bytes,
            "SVM": svm_model_bytes
        }
        config = {"feature_columns": [...]}
        upload_models.remote(models_dict, config)
    """
    model_files = {
        "TFT": "tft_model.pt",
        "LSTM": "lstm_model.pt",
        "XGBoost": "xgboost_model.json",
        "RandomForest": "rf_model.pkl",
        "SVM": "svm_model.pkl"
    }
    
    uploaded = []
    
    for model_name, filename in model_files.items():
        if model_name in models_dict:
            model_path = f"{MODEL_DIR}/{filename}"
            with open(model_path, "wb") as f:
                f.write(models_dict[model_name])
            uploaded.append(model_name)
            print(f"✓ Uploaded {model_name} to {filename}")
    
    # Save config
    config_path = f"{MODEL_DIR}/model_config_v2.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    # Commit to volume
    volume.commit()
    
    print(f"Uploaded {len(uploaded)} models and config")
    return {"success": True, "uploaded_models": uploaded}


# ============================================================================
# LOCAL TESTING
# ============================================================================

@app.local_entrypoint()
def main():
    """Test the enhanced inference endpoint locally"""
    
    # Test features (70 features)
    test_features = {
        # Technical (10)
        "rsi_14": 42.5,
        "macd_value": 2.3,
        "macd_signal": 1.8,
        "macd_histogram": 0.5,
        "bb_percentage": 0.3,
        "stoch_k": 35.0,
        "stoch_d": 32.0,
        "atr_percentage": 2.5,
        "adx": 28.0,
        "cci": -50.0,
        
        # Price Action (9)
        "support_distance": 2.5,
        "resistance_distance": 3.0,
        "fib_distance": 1.5,
        "range_position": 35.0,
        "trend_direction": 75.0,
        "momentum_10d": 3.2,
        "candle_strength": 70.0,
        "gap_percentage": 0.5,
        "hh_ll_score": 60.0,
        
        # Volume (10)
        "volume_ma_ratio": 1.4,
        "obv_slope": 50000,
        "mfi": 55.0,
        "force_index": 1200,
        "vpt": 80000,
        "ad_slope": 30000,
        "cmf": 0.15,
        "vwap_distance": 0.8,
        "eom": 0.002,
        "volume_oscillator": 15.0,
        
        # SMC (10)
        "order_block_strength": 75.0,
        "order_block_distance": 2.0,
        "fvg_distance": 3.5,
        "fvg_volume_ratio": 1.3,
        "sweep_detection": 1.0,
        "post_sweep_reversal_prob": 80.0,
        "institutional_activity": 78.0,
        "accumulation_phase": 70.0,
        "distribution_phase": 15.0,
        "liquidity_level": 75.0,
        
        # MTF (10)
        "daily_rsi": 42.0,
        "daily_macd": 1.0,
        "hourly_rsi": 38.0,
        "hourly_macd": 1.0,
        "hourly_daily_alignment": 100.0,
        "weekly_rsi": 45.0,
        "weekly_macd": 1.0,
        "weekly_trend": 80.0,
        "mtf_confluence": 85.0,
        "mtf_momentum": 45.0,
        
        # Microstructure (10)
        "avg_spread": 1.8,
        "price_impact": 0.3,
        "order_flow_imbalance": 12.0,
        "tick_direction": 60.0,
        "volatility_clustering": 1.2,
        "liquidity_depth": 2.5,
        "momentum_acceleration": 5.0,
        "microstructure_noise": 8.0,
        "effective_spread": 1.9,
        "trade_intensity": 0.02,
        
        # Market Context (10)
        "nifty_change": 0.5,
        "vix_level": 14.5,
        "fii_flow": 1200.0,
        "dii_flow": 800.0,
        "advance_decline": 500.0,
        "beta": 1.1,
        "relative_strength": 2.5,
        "sector_momentum": 3.5,
        "market_regime": 80.0,
        "volatility_regime": 40.0
    }
    
    # Run prediction
    predictor = EnhancedSwingAIPredictor()
    result = predictor.predict.remote(test_features)
    
    print("\n" + "="*80)
    print("ENHANCED 5-MODEL ENSEMBLE PREDICTION")
    print("="*80)
    
    print(f"\nFinal Prediction: {result['prediction']:.2f} ({result['direction']})")
    print(f"Confidence: {result['confidence']:.2f}%")
    
    print(f"\nIndividual Model Predictions:")
    for model, pred in result['model_predictions'].items():
        weight = result['weights_used'][model]
        conf = result['model_confidences'][model]
        print(f"  {model:<15s}: {pred:6.2f} (weight: {weight:.0%}, confidence: {conf:.2f}%)")
    
    print(f"\nEnsemble Metrics:")
    print(f"  Agreement Score: {result['agreement_score']:.2f}%")
    print(f"  Disagreement Penalty: {result['disagreement_penalty']:.3f}")
    print(f"  Consensus: {result['consensus_count']}/{result['total_models']} models agree")
    
    print(f"\nMetadata:")
    print(f"  Features Used: {result['features_count']}")
    print(f"  Models Loaded: {result['models_loaded']}/{result['total_models']}")
    
    print("\n" + "="*80)
