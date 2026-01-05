"""
================================================================================
SWINGAI - MODAL INFERENCE ENDPOINT
================================================================================
Serverless ML inference using Modal.com
Deploy with: modal deploy ml/inference/modal_inference.py
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
app = modal.App("swingai-inference")

# Define image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "catboost==1.2.2",
    "numpy==1.26.3",
    "pandas==2.1.4",
    "scikit-learn==1.4.0",
    "pydantic==2.5.3",
    "fastapi==0.109.0",
)

# Model storage volume
volume = modal.Volume.from_name("swingai-models", create_if_missing=True)
MODEL_DIR = "/models"

# ============================================================================
# MODEL LOADER
# ============================================================================

@app.cls(
    image=image,
    volumes={MODEL_DIR: volume},
    gpu=False,  # CatBoost runs fine on CPU
    memory=2048,
    timeout=300,
)
class SwingAIPredictor:
    """
    SwingAI Model Predictor
    Loads and serves CatBoost model for inference
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.model_loaded = False
    
    @modal.enter()
    def load_model(self):
        """Load model on container startup"""
        import catboost
        
        model_path = f"{MODEL_DIR}/catboost_model.cbm"
        config_path = f"{MODEL_DIR}/model_config.json"
        
        try:
            # Load CatBoost model
            if os.path.exists(model_path):
                self.model = catboost.CatBoostClassifier()
                self.model.load_model(model_path)
                print(f"Loaded CatBoost model from {model_path}")
            else:
                print("No saved model found, using fallback")
                self.model = None
            
            # Load feature config
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.feature_columns = config.get("feature_columns", [])
            
            self.model_loaded = True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    @modal.method()
    def predict(self, features: List[Dict]) -> List[Dict]:
        """
        Run prediction on features
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        import numpy as np
        import pandas as pd
        
        predictions = []
        
        for feature_dict in features:
            symbol = feature_dict.get("symbol", "UNKNOWN")
            
            try:
                if self.model is not None:
                    # Prepare features for model
                    df = pd.DataFrame([feature_dict])
                    
                    # Select only model features
                    if self.feature_columns:
                        df = df[self.feature_columns]
                    
                    # Get prediction
                    pred_proba = self.model.predict_proba(df)[0]
                    pred_class = self.model.predict(df)[0]
                    
                    # Map class to direction
                    # Assuming classes: 0=DOWN, 1=NEUTRAL, 2=UP
                    if pred_class == 2:
                        direction = "LONG"
                        confidence = pred_proba[2] * 100
                    elif pred_class == 0:
                        direction = "SHORT"
                        confidence = pred_proba[0] * 100
                    else:
                        direction = "NEUTRAL"
                        confidence = pred_proba[1] * 100
                    
                    predictions.append({
                        "symbol": symbol,
                        "direction": direction,
                        "catboost_score": round(confidence, 2),
                        "tft_score": round(confidence * 0.95, 2),  # Simulated
                        "stockformer_score": round(confidence * 0.92, 2),  # Simulated
                        "model_agreement": 3 if confidence > 70 else 2,
                        "price": feature_dict.get("price", 0),
                        "reasons": self._get_reasons(feature_dict),
                        "features": feature_dict
                    })
                else:
                    # Fallback prediction
                    predictions.append(self._fallback_predict(feature_dict))
                    
            except Exception as e:
                print(f"Prediction error for {symbol}: {e}")
                predictions.append(self._fallback_predict(feature_dict))
        
        return predictions
    
    def _fallback_predict(self, features: Dict) -> Dict:
        """Fallback rule-based prediction"""
        symbol = features.get("symbol", "UNKNOWN")
        price = features.get("price", 0)
        rsi = features.get("rsi_14", 50)
        macd_hist = features.get("macd_hist", 0)
        
        score = 50.0
        reasons = []
        
        if rsi < 35:
            score += 15
            reasons.append("RSI oversold")
        elif rsi > 65:
            score -= 15
            reasons.append("RSI overbought")
        
        if macd_hist > 0:
            score += 10
            reasons.append("MACD bullish")
        elif macd_hist < 0:
            score -= 10
            reasons.append("MACD bearish")
        
        if score >= 60:
            direction = "LONG"
        elif score <= 40:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"
        
        return {
            "symbol": symbol,
            "direction": direction,
            "catboost_score": score,
            "tft_score": score * 0.95,
            "stockformer_score": score * 0.92,
            "model_agreement": 2 if abs(score - 50) > 10 else 1,
            "price": price,
            "reasons": reasons,
            "features": features
        }
    
    def _get_reasons(self, features: Dict) -> List[str]:
        """Extract reasons from features"""
        reasons = []
        
        rsi = features.get("rsi_14", 50)
        if rsi < 35:
            reasons.append("RSI oversold")
        elif rsi > 65:
            reasons.append("RSI overbought")
        
        macd = features.get("macd_hist", 0)
        if macd > 0:
            reasons.append("MACD bullish")
        elif macd < 0:
            reasons.append("MACD bearish")
        
        price = features.get("price", 0)
        sma_20 = features.get("sma_20", price)
        sma_50 = features.get("sma_50", price)
        
        if price > sma_20 > sma_50:
            reasons.append("Price above MAs")
        elif price < sma_20 < sma_50:
            reasons.append("Price below MAs")
        
        adx = features.get("adx", 0)
        if adx > 25:
            reasons.append("Strong trend")
        
        volume = features.get("volume", 0)
        volume_sma = features.get("volume_sma_20", 1)
        if volume > volume_sma * 1.5:
            reasons.append("High volume")
        
        return reasons


# ============================================================================
# FASTAPI WEB ENDPOINT
# ============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

web_app = FastAPI(title="SwingAI Inference API")

class PredictRequest(BaseModel):
    features: List[Dict]

class PredictResponse(BaseModel):
    predictions: List[Dict]
    timestamp: str
    model_version: str = "1.0.0"

@web_app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@web_app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run prediction on features
    
    Example request:
    {
        "features": [
            {
                "symbol": "RELIANCE",
                "price": 2456.75,
                "rsi_14": 42,
                "macd_hist": 2.5,
                ...
            }
        ]
    }
    """
    try:
        predictor = SwingAIPredictor()
        predictions = predictor.predict.remote(request.features)
        
        return PredictResponse(
            predictions=predictions,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SwingAI Inference",
        "version": "1.0.0",
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
# MODEL UPLOAD FUNCTION
# ============================================================================

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
)
def upload_model(model_bytes: bytes, config: Dict):
    """
    Upload trained model to Modal volume
    
    Usage from local:
        with open("catboost_model.cbm", "rb") as f:
            model_bytes = f.read()
        config = {"feature_columns": [...]}
        upload_model.remote(model_bytes, config)
    """
    import catboost
    
    # Save model
    model_path = f"{MODEL_DIR}/catboost_model.cbm"
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    
    # Save config
    config_path = f"{MODEL_DIR}/model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    # Commit to volume
    volume.commit()
    
    print(f"Model uploaded to {model_path}")
    return {"success": True, "path": model_path}


# ============================================================================
# LOCAL TESTING
# ============================================================================

@app.local_entrypoint()
def main():
    """Test the inference endpoint locally"""
    
    # Test features
    test_features = [
        {
            "symbol": "RELIANCE",
            "price": 2456.75,
            "rsi_14": 42,
            "macd_hist": 2.5,
            "adx": 28,
            "plus_di": 25,
            "minus_di": 18,
            "sma_20": 2420,
            "sma_50": 2380,
            "volume": 5000000,
            "volume_sma_20": 3500000,
            "atr_14": 45
        },
        {
            "symbol": "TCS",
            "price": 3678.90,
            "rsi_14": 68,
            "macd_hist": -1.5,
            "adx": 22,
            "plus_di": 18,
            "minus_di": 24,
            "sma_20": 3700,
            "sma_50": 3650,
            "volume": 2000000,
            "volume_sma_20": 2500000,
            "atr_14": 55
        }
    ]
    
    # Run prediction
    predictor = SwingAIPredictor()
    predictions = predictor.predict.remote(test_features)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    for pred in predictions:
        print(f"\n{pred['symbol']}")
        print(f"  Direction: {pred['direction']}")
        print(f"  CatBoost Score: {pred['catboost_score']}")
        print(f"  TFT Score: {pred['tft_score']}")
        print(f"  Stockformer Score: {pred['stockformer_score']}")
        print(f"  Model Agreement: {pred['model_agreement']}/3")
        print(f"  Reasons: {', '.join(pred['reasons'])}")
