# 🧠 SwingAI Model Deployment Guide

## Complete Guide to Deploy, Monitor & Optimize Your Trading Models

---

## 📋 Table of Contents

1. [Model Training Recap](#model-training-recap)
2. [Where to Deploy Models](#where-to-deploy-models)
3. [Performance Monitoring](#performance-monitoring)
4. [Model Accuracy Tracking](#model-accuracy-tracking)
5. [Retraining Schedule](#retraining-schedule)
6. [Production Inference Pipeline](#production-inference-pipeline)

---

## 🎯 Model Training Recap

After training in Google Colab, you have these model files:

```
models/
├── catboost_model.cbm       # CatBoost (35% weight)
├── tft_model.pt             # TFT Transformer (35% weight)
├── stockformer_model.pt     # Stockformer (30% weight)
├── feature_names.json       # 60 feature names
├── ensemble_config.json     # Model weights config
└── scaler.pkl               # Feature scaler (if used)
```

**Expected Metrics from Training:**

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Direction Accuracy | 55% | 58% | 62%+ |
| Win Rate | 52% | 55% | 60%+ |
| Precision (UP) | 55% | 60% | 65%+ |
| Precision (DOWN) | 50% | 55% | 60%+ |

---

## 🚀 Where to Deploy Models

### Option 1: Modal (Recommended - Serverless GPU)

**Best for**: Production inference, auto-scaling, cost-effective

```python
# modal_inference.py
import modal

app = modal.App("swingai-inference")

# Create image with dependencies
image = modal.Image.debian_slim().pip_install([
    "catboost",
    "torch",
    "pytorch-forecasting",
    "pandas",
    "numpy"
])

# Mount model files
volume = modal.Volume.from_name("swingai-models")

@app.function(
    image=image,
    volumes={"/models": volume},
    gpu="T4",  # Use GPU for inference
    timeout=300
)
def predict(features: dict):
    """Run inference on GPU"""
    import torch
    from catboost import CatBoostClassifier
    import numpy as np
    
    # Load models
    catboost = CatBoostClassifier()
    catboost.load_model("/models/catboost_model.cbm")
    
    tft = torch.load("/models/tft_model.pt")
    stockformer = torch.load("/models/stockformer_model.pt")
    
    # Prepare features
    X = np.array(features['data'])
    
    # Get predictions
    cat_probs = catboost.predict_proba(X[:, -1, :])
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        tft_probs = torch.softmax(tft(X_tensor), dim=1).numpy()
        sf_probs = torch.softmax(stockformer(X_tensor), dim=1).numpy()
    
    # Ensemble
    ensemble_probs = 0.35 * cat_probs + 0.35 * tft_probs + 0.30 * sf_probs
    predictions = np.argmax(ensemble_probs, axis=1)
    confidences = np.max(ensemble_probs, axis=1)
    
    return {
        'predictions': predictions.tolist(),
        'confidences': confidences.tolist(),
        'probabilities': ensemble_probs.tolist()
    }

@app.local_entrypoint()
def main():
    # Test inference
    import numpy as np
    test_features = {'data': np.random.randn(5, 60, 60).tolist()}
    result = predict.remote(test_features)
    print(result)
```

**Deploy:**
```bash
pip install modal
modal token new
modal deploy modal_inference.py
```

**Call from backend:**
```python
import modal
predict = modal.Function.lookup("swingai-inference", "predict")
result = predict.remote({"data": features.tolist()})
```

---

### Option 2: AWS SageMaker (Enterprise)

**Best for**: Large scale, enterprise features

```python
# sagemaker_deploy.py
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Create model
model = PyTorchModel(
    model_data='s3://swingai-models/model.tar.gz',
    role='arn:aws:iam::xxx:role/SageMakerRole',
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py'
)

# Deploy endpoint
predictor = model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1,
    endpoint_name='swingai-inference'
)

# Call
result = predictor.predict(features)
```

---

### Option 3: Replicate (Simple API)

**Best for**: Quick deployment, pay-per-prediction

```python
# cog.yaml
build:
  python_version: "3.10"
  python_packages:
    - catboost
    - torch
    - pytorch-forecasting

predict: "predict.py:Predictor"

# predict.py
from cog import BasePredictor, Input
import torch
import numpy as np

class Predictor(BasePredictor):
    def setup(self):
        self.catboost = CatBoostClassifier()
        self.catboost.load_model("models/catboost_model.cbm")
        self.tft = torch.load("models/tft_model.pt")
        self.stockformer = torch.load("models/stockformer_model.pt")
    
    def predict(self, features: str = Input(description="JSON features")) -> dict:
        X = np.array(json.loads(features))
        # ... inference logic
        return {"prediction": prediction, "confidence": confidence}
```

**Deploy:**
```bash
cog push r8.im/yourusername/swingai
```

---

### Option 4: Self-Hosted (Railway/Render with GPU)

**Best for**: Full control, consistent costs

```dockerfile
# Dockerfile.inference
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install catboost torch pytorch-forecasting fastapi uvicorn

COPY models/ /app/models/
COPY inference_server.py /app/

WORKDIR /app
CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 📊 Performance Monitoring

### Daily Accuracy Tracking

```python
# Track in database after market closes
async def track_daily_performance():
    supabase = get_supabase_admin()
    today = date.today().isoformat()
    
    # Get today's signals
    signals = supabase.table("signals").select("*").eq("date", today).execute()
    
    # Compare predictions vs actual results
    correct = 0
    total = 0
    
    for signal in signals.data:
        if signal['status'] in ['target_hit', 'sl_hit']:
            total += 1
            if signal['result'] == 'win':
                correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Save to model_performance table
    supabase.table("model_performance").insert({
        "date": today,
        "total_signals": len(signals.data),
        "correct_signals": correct,
        "accuracy": accuracy,
        "ensemble_accuracy": accuracy
    }).execute()
    
    return accuracy
```

### Real-time Monitoring Dashboard

```python
# Backend endpoint
@app.get("/api/models/performance")
async def get_model_performance(days: int = 30):
    supabase = get_supabase_admin()
    
    start_date = (date.today() - timedelta(days=days)).isoformat()
    
    result = supabase.table("model_performance").select("*").gte("date", start_date).order("date").execute()
    
    data = result.data
    
    # Calculate averages
    avg_accuracy = sum(d['accuracy'] for d in data) / len(data) if data else 0
    
    return {
        "daily_performance": data,
        "avg_accuracy": round(avg_accuracy, 2),
        "total_signals": sum(d['total_signals'] for d in data),
        "trend": "improving" if data[-1]['accuracy'] > data[0]['accuracy'] else "declining"
    }
```

---

## 📈 Model Accuracy Tracking

### Backtesting New Models

```python
# backtest.py
def backtest_model(model, test_data, start_date, end_date):
    """
    Backtest model on historical data
    """
    results = {
        'total_signals': 0,
        'correct': 0,
        'wrong': 0,
        'total_return': 0,
        'max_drawdown': 0,
        'sharpe_ratio': 0,
        'trades': []
    }
    
    equity = 100000
    peak_equity = equity
    daily_returns = []
    
    for date in date_range(start_date, end_date):
        # Get features for this date
        features = get_features_for_date(test_data, date)
        
        # Get prediction
        pred = model.predict(features)
        confidence = model.predict_proba(features).max()
        
        if confidence < 0.7:
            continue
        
        # Simulate trade
        entry_price = get_price(date, 'open')
        
        # Check next 5 days for target/SL
        for i in range(1, 6):
            future_date = date + timedelta(days=i)
            high = get_price(future_date, 'high')
            low = get_price(future_date, 'low')
            
            if pred == 'LONG':
                target = entry_price * 1.06
                sl = entry_price * 0.97
                
                if high >= target:
                    pnl = 0.06 * equity * 0.03  # 3% position
                    results['correct'] += 1
                    break
                elif low <= sl:
                    pnl = -0.03 * equity * 0.03
                    results['wrong'] += 1
                    break
        
        equity += pnl
        daily_returns.append(pnl / equity)
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity
        results['max_drawdown'] = max(results['max_drawdown'], drawdown)
        
        results['trades'].append({
            'date': date,
            'prediction': pred,
            'confidence': confidence,
            'pnl': pnl
        })
    
    results['total_return'] = (equity - 100000) / 100000 * 100
    results['sharpe_ratio'] = calculate_sharpe(daily_returns)
    results['accuracy'] = results['correct'] / (results['correct'] + results['wrong']) * 100
    
    return results
```

### A/B Testing Models

```python
# Compare old vs new model
def ab_test_models(old_model, new_model, test_period_days=30):
    """
    Run both models in parallel, compare results
    """
    old_results = backtest_model(old_model, test_data, start, end)
    new_results = backtest_model(new_model, test_data, start, end)
    
    comparison = {
        'accuracy_diff': new_results['accuracy'] - old_results['accuracy'],
        'return_diff': new_results['total_return'] - old_results['total_return'],
        'sharpe_diff': new_results['sharpe_ratio'] - old_results['sharpe_ratio'],
        'recommendation': 'deploy_new' if new_results['sharpe_ratio'] > old_results['sharpe_ratio'] else 'keep_old'
    }
    
    return comparison
```

---

## 🔄 Retraining Schedule

### Weekly Retraining Pipeline

```python
# retrain_pipeline.py
from datetime import datetime, timedelta
import schedule

def weekly_retrain():
    """
    Run every Sunday to retrain models with latest data
    """
    print(f"Starting weekly retrain: {datetime.now()}")
    
    # 1. Fetch latest data
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365*3)  # 3 years
    
    data = download_all_stocks(start_date, end_date)
    
    # 2. Calculate features
    features, labels = prepare_features(data)
    
    # 3. Split data
    train_end = end_date - timedelta(days=60)
    val_end = end_date - timedelta(days=30)
    
    train_data = filter_by_date(features, labels, None, train_end)
    val_data = filter_by_date(features, labels, train_end, val_end)
    test_data = filter_by_date(features, labels, val_end, None)
    
    # 4. Retrain each model
    catboost = train_catboost(train_data, val_data)
    tft = train_tft(train_data, val_data)
    stockformer = train_stockformer(train_data, val_data)
    
    # 5. Evaluate on test set
    results = evaluate_ensemble(catboost, tft, stockformer, test_data)
    
    print(f"New model accuracy: {results['accuracy']}%")
    
    # 6. Compare with current production model
    current_accuracy = get_current_model_accuracy()
    
    if results['accuracy'] > current_accuracy - 2:  # Allow 2% margin
        # 7. Deploy new model
        save_models(catboost, tft, stockformer)
        deploy_to_production()
        print("New model deployed!")
    else:
        print("New model not better, keeping current")
    
    # 8. Log results
    log_retrain_results(results)

# Schedule weekly
schedule.every().sunday.at("06:00").do(weekly_retrain)
```

---

## ⚡ Production Inference Pipeline

### Daily Signal Generation

```python
# signal_generator.py
async def generate_daily_signals():
    """
    Run at 8:30 AM IST daily
    """
    logger.info("Starting daily signal generation...")
    
    # 1. Get stock candidates from PKScreener
    candidates = pkscreener.get_swing_candidates(max_stocks=50)
    logger.info(f"Got {len(candidates)} candidates")
    
    # 2. Fetch latest data
    data = fetch_stock_data(candidates, days=60)
    
    # 3. Calculate features
    features = calculate_features(data)
    
    # 4. Get market context
    market_data = fetch_market_data()
    vix = market_data['vix']
    nifty_trend = market_data['nifty_trend']
    
    # 5. Run inference
    predictions = await run_inference(features)
    
    # 6. Filter by confidence and generate signals
    signals = []
    
    for symbol, pred in predictions.items():
        if pred['confidence'] < 70:
            continue
        
        if pred['agreement'] < 2:  # At least 2/3 models agree
            continue
        
        # Calculate entry, SL, target
        current_price = data[symbol]['close'].iloc[-1]
        atr = calculate_atr(data[symbol])
        
        if pred['direction'] == 'LONG':
            entry = current_price
            sl = entry - (1.5 * atr)
            target = entry + (2.5 * atr)
        else:  # SHORT
            entry = current_price
            sl = entry + (1.5 * atr)
            target = entry - (2.5 * atr)
        
        risk_reward = abs(target - entry) / abs(entry - sl)
        
        if risk_reward < 1.5:
            continue
        
        signal = {
            'symbol': symbol,
            'direction': pred['direction'],
            'confidence': pred['confidence'],
            'entry_price': round(entry, 2),
            'stop_loss': round(sl, 2),
            'target_1': round(target, 2),
            'risk_reward': round(risk_reward, 2),
            'model_agreement': f"{pred['agreement']}/3",
            'catboost_score': pred['catboost'],
            'tft_score': pred['tft'],
            'stockformer_score': pred['stockformer'],
            'nifty_level': market_data['nifty_close'],
            'vix_level': vix,
            'is_premium': pred['confidence'] >= 80
        }
        
        signals.append(signal)
    
    # 7. Save to database
    supabase = get_supabase_admin()
    
    for signal in signals:
        supabase.table("signals").insert({
            **signal,
            'date': date.today().isoformat(),
            'status': 'active'
        }).execute()
    
    logger.info(f"Generated {len(signals)} signals")
    
    # 8. Send notifications
    await send_signal_notifications(signals)
    
    return signals
```

### Inference Service Health Checks

```python
# Health monitoring
@app.get("/api/models/health")
async def model_health():
    """Check model service health"""
    try:
        # Test inference with dummy data
        test_input = np.random.randn(1, 60, 60)
        
        start_time = time.time()
        result = await run_inference({'test': test_input})
        latency = time.time() - start_time
        
        return {
            "status": "healthy",
            "latency_ms": round(latency * 1000, 2),
            "models_loaded": True,
            "last_check": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat()
        }
```

---

## 📊 Accuracy Metrics to Track

| Metric | Formula | Target |
|--------|---------|--------|
| **Direction Accuracy** | Correct / Total | 58%+ |
| **Win Rate** | Profitable Trades / Total Trades | 55%+ |
| **Profit Factor** | Gross Profit / Gross Loss | 1.5+ |
| **Sharpe Ratio** | (Returns - Rf) / Std(Returns) | 1.5+ |
| **Max Drawdown** | (Peak - Trough) / Peak | < 15% |
| **Calmar Ratio** | Annual Return / Max Drawdown | 1.0+ |

---

## 🎯 Summary

1. **Train models** in Google Colab (~2 hours)
2. **Deploy to Modal** for serverless GPU inference
3. **Track accuracy daily** in model_performance table
4. **Retrain weekly** with latest data
5. **A/B test** before deploying new models
6. **Monitor latency** and health continuously

**Key Commands:**

```bash
# Deploy to Modal
modal deploy modal_inference.py

# Test inference
curl -X POST https://swingai--predict.modal.run \
  -H "Content-Type: application/json" \
  -d '{"data": [[...features...]]}'

# Check model health
curl https://api.swingai.com/api/models/health
```
