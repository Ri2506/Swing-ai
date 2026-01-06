"""
# 🚀 SwingAI Training V3 - State of the Art
# PhD-Level AI Trading Model for Indian Markets

## V3 Improvements Over V2
| Issue in V2 | V3 Solution |
|-------------|-------------|
| Class imbalance (LONG only 16%) | **Focal Loss + Balanced Sampling** |
| Feature dominance (2 features = 63%) | **Outlier capping + Normalization** |
| Overfitting (val-test gap 6-7%) | **Deeper regularization + Dropout** |
| Poor LONG/SHORT recall (24-29%) | **Class-weighted training** |
| Static ensemble weights | **Dynamic confidence-based weighting** |

## Target Metrics
- **Overall Accuracy**: 65%+ (up from 58%)
- **High-Confidence (>70%)**: 85%+ accuracy
- **LONG/SHORT Recall**: 50%+ (up from 24-29%)

Copy this entire file into a Colab notebook and run!
"""

# ============================================================
# CELL 1: Install Dependencies (Run once, restart runtime)
# ============================================================
# %pip uninstall -y -q numpy pandas scipy scikit-learn catboost yfinance
# %pip install -q "numpy<2.0" "pandas==2.2.2" "scipy<1.13" "scikit-learn==1.5.2" \
#     "catboost==1.2.7" "yfinance>=0.2.54" "curl_cffi>=0.7.4"

# ============================================================
# CELL 2: Imports & GPU Check
# ============================================================
import os, json, time, random, warnings
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings('ignore')

print(f"✅ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# CELL 3: Configuration V3
# ============================================================
@dataclass
class ConfigV3:
    LOOKBACK_DAYS: int = 60
    PREDICTION_HORIZON: int = 5
    MIN_HISTORY_DAYS: int = 252
    
    # BALANCED thresholds (critical fix!)
    UP_THRESHOLD: float = 0.025     # Lowered from 0.03
    DOWN_THRESHOLD: float = -0.015  # Raised from -0.02
    
    TRAIN_END: str = "2024-06-30"
    VAL_END: str = "2024-09-30"
    
    CATBOOST_WEIGHT: float = 0.30
    TFT_WEIGHT: float = 0.40
    STOCKFORMER_WEIGHT: float = 0.30
    
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 5e-4
    MAX_EPOCHS: int = 100
    PATIENCE: int = 15
    
    # CatBoost V3
    CATBOOST_ITERATIONS: int = 2000
    CATBOOST_DEPTH: int = 8
    CATBOOST_LR: float = 0.03
    CATBOOST_L2_REG: float = 3.0
    
    # TFT V3
    TFT_HIDDEN_SIZE: int = 128
    TFT_ATTENTION_HEADS: int = 8
    TFT_DROPOUT: float = 0.2
    TFT_NUM_LAYERS: int = 3
    
    # Stockformer V3
    STOCKFORMER_D_MODEL: int = 128
    STOCKFORMER_N_HEADS: int = 8
    STOCKFORMER_N_LAYERS: int = 3
    STOCKFORMER_DROPOUT: float = 0.15
    
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTHING: float = 0.1
    
    NUM_FEATURES: int = 40
    MODEL_SAVE_PATH: str = "/content/drive/MyDrive/SwingAI/models_v3/"

config = ConfigV3()

FEATURE_NAMES = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d', 'volatility_20d',
    'close_to_sma_20', 'close_to_sma_50', 'rsi_14_norm', 'macd_histogram_norm', 'bb_position',
    'structure_score', 'range_position', 'dist_to_swing_high', 'dist_to_swing_low',
    'in_discount', 'in_deep_discount', 'in_premium', 'near_bullish_ob', 'near_bearish_ob',
    'bullish_fvg', 'bearish_fvg', 'sweep_high', 'sweep_low', 'bos_bullish', 'bos_bearish',
    'volume_ratio', 'volume_trend', 'obv_slope', 'close_to_vwap',
    'buying_pressure', 'accumulation_score', 'big_volume_day', 'higher_high',
    'daily_trend', 'weekly_trend', 'monthly_trend', 'mtf_alignment',
    'weekly_range_pos', 'monthly_range_pos', 'trend_strength',
]

print(f"✅ Config V3 loaded: {len(FEATURE_NAMES)} features")
print(f"   Thresholds: UP={config.UP_THRESHOLD}, DOWN={config.DOWN_THRESHOLD}")

# ============================================================
# CELL 4: Download Stock Data
# ============================================================
FO_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "NTPC.NS",
    "WIPRO.NS", "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "TATASTEEL.NS", "ONGC.NS",
    "TECHM.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "BAJAJFINSV.NS", "GRASIM.NS",
    "DRREDDY.NS", "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
    "COALINDIA.NS", "SBILIFE.NS", "BPCL.NS", "INDUSINDBK.NS", "TATACONSUM.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "LTIM.NS", "SHRIRAMFIN.NS",
    "TRENT.NS", "POLYCAB.NS", "PERSISTENT.NS", "DIXON.NS", "TATAELXSI.NS",
    "ABB.NS", "SIEMENS.NS", "HAL.NS", "BEL.NS", "IRCTC.NS",
    "COFORGE.NS", "MUTHOOTFIN.NS", "INDHOTEL.NS", "BANKBARODA.NS",
    "PNB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "CHOLAFIN.NS", "VEDL.NS",
]

def yf_download_safe(tickers, start, end, chunk_size=5, max_retries=5, base_sleep=2.0):
    out = {}
    total = len(tickers)
    print(f"📥 Downloading {total} stocks...")
    
    for i in range(0, total, chunk_size):
        chunk = tickers[i:i+chunk_size]
        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(" ".join(chunk), start=start, end=end, 
                                group_by="ticker", auto_adjust=True, 
                                threads=False, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    for t in chunk:
                        if t in df.columns.get_level_values(0):
                            tdf = df[t].dropna(how="all")
                            if len(tdf) >= config.MIN_HISTORY_DAYS:
                                out[t] = tdf
                else:
                    t = chunk[0]
                    if len(df) >= config.MIN_HISTORY_DAYS:
                        out[t] = df.dropna(how="all")
                break
            except:
                time.sleep(base_sleep * (2 ** (attempt - 1)) + random.random())
        
        done = min(i + chunk_size, total)
        if done % 20 == 0 or done == total:
            print(f"   Progress: {done}/{total} | downloaded={len(out)}")
        time.sleep(base_sleep + random.random())
    
    print(f"✅ Downloaded {len(out)} stocks")
    return out

stock_data = yf_download_safe(FO_STOCKS, "2019-01-01", "2024-12-31")

# ============================================================
# CELL 5: Feature Engineering V3
# ============================================================
def calculate_features_v3(df):
    data = df.copy()
    
    # Price Action
    data['return_1d'] = data['Close'].pct_change(1)
    data['return_5d'] = data['Close'].pct_change(5)
    data['return_10d'] = data['Close'].pct_change(10)
    data['return_20d'] = data['Close'].pct_change(20)
    data['volatility_20d'] = data['return_1d'].rolling(20).std() * np.sqrt(252)
    
    data['sma_20'] = data['Close'].rolling(20).mean()
    data['sma_50'] = data['Close'].rolling(50).mean()
    data['close_to_sma_20'] = (data['Close'] - data['sma_20']) / data['sma_20']
    data['close_to_sma_50'] = (data['Close'] - data['sma_50']) / data['sma_50']
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data['rsi_14_norm'] = (100 - (100 / (1 + rs)) - 50) / 50
    
    ema_12 = data['Close'].ewm(span=12).mean()
    ema_26 = data['Close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    data['macd_histogram_norm'] = (macd - macd_signal) / data['Close']
    
    bb_mid = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['bb_position'] = (data['Close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    # SMC/ICT (reduced lookback)
    data['swing_high'] = data['High'].rolling(5, center=True).max()
    data['swing_low'] = data['Low'].rolling(5, center=True).min()
    data['prev_swing_high'] = data['swing_high'].shift(5)
    data['prev_swing_low'] = data['swing_low'].shift(5)
    data['higher_high'] = (data['swing_high'] > data['prev_swing_high']).astype(int)
    data['higher_low'] = (data['swing_low'] > data['prev_swing_low']).astype(int)
    data['lower_high'] = (data['swing_high'] < data['prev_swing_high']).astype(int)
    data['lower_low'] = (data['swing_low'] < data['prev_swing_low']).astype(int)
    
    data['structure_score'] = (data['higher_high'].rolling(5).sum() + 
                               data['higher_low'].rolling(5).sum() -
                               data['lower_high'].rolling(5).sum() - 
                               data['lower_low'].rolling(5).sum()) / 10
    
    range_high = data['High'].rolling(20).max()
    range_low = data['Low'].rolling(20).min()
    data['range_position'] = (data['Close'] - range_low) / (range_high - range_low + 1e-10)
    
    # CAPPED swing distances (critical fix!)
    data['dist_to_swing_high'] = np.clip((data['swing_high'] - data['Close']) / data['Close'], -0.15, 0.15)
    data['dist_to_swing_low'] = np.clip((data['Close'] - data['swing_low']) / data['Close'], -0.15, 0.15)
    
    data['in_discount'] = (data['range_position'] < 0.5).astype(int)
    data['in_deep_discount'] = (data['range_position'] < 0.3).astype(int)
    data['in_premium'] = (data['range_position'] > 0.7).astype(int)
    
    vol_threshold = data['return_1d'].rolling(20).std() * 2
    data['near_bullish_ob'] = (data['return_1d'] > vol_threshold).rolling(10).sum() / 10
    data['near_bearish_ob'] = (data['return_1d'] < -vol_threshold).rolling(10).sum() / 10
    
    data['gap_up'] = ((data['Low'] > data['High'].shift(1)) & (data['return_1d'] > 0.01)).astype(int)
    data['gap_down'] = ((data['High'] < data['Low'].shift(1)) & (data['return_1d'] < -0.01)).astype(int)
    data['bullish_fvg'] = data['gap_up'].rolling(5).sum() / 5
    data['bearish_fvg'] = data['gap_down'].rolling(5).sum() / 5
    
    data['sweep_high'] = ((data['High'] > data['swing_high'].shift(1)) & 
                          (data['Close'] < data['swing_high'].shift(1))).astype(int)
    data['sweep_low'] = ((data['Low'] < data['swing_low'].shift(1)) & 
                         (data['Close'] > data['swing_low'].shift(1))).astype(int)
    
    data['bos_bullish'] = ((data['Close'] > data['swing_high'].shift(1)) & 
                           (data['higher_high'] == 1)).astype(int)
    data['bos_bearish'] = ((data['Close'] < data['swing_low'].shift(1)) & 
                           (data['lower_low'] == 1)).astype(int)
    
    # Volume (capped)
    data['volume_ma_20'] = data['Volume'].rolling(20).mean()
    data['volume_ratio'] = np.clip(data['Volume'] / (data['volume_ma_20'] + 1e-10), 0, 5)
    data['volume_trend'] = data['Volume'].rolling(5).mean() / (data['Volume'].rolling(20).mean() + 1e-10)
    
    obv = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
    obv_std = obv.rolling(20).std().replace(0, 1)
    data['obv_slope'] = np.clip(obv.diff(5) / obv_std, -3, 3)
    
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['close_to_vwap'] = np.clip((data['Close'] - data['vwap']) / (data['vwap'] + 1e-10), -0.2, 0.2)
    
    data['buying_pressure'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-10)
    data['accumulation_score'] = (data['buying_pressure'] * data['volume_ratio']).rolling(5).mean()
    data['big_volume_day'] = (data['volume_ratio'] > 2).astype(int)
    
    # Multi-timeframe
    data['daily_trend'] = (data['Close'] > data['sma_20']).astype(int)
    data['weekly_close'] = data['Close'].rolling(5).mean()
    data['weekly_high'] = data['High'].rolling(5).max()
    data['weekly_low'] = data['Low'].rolling(5).min()
    data['weekly_trend'] = (data['weekly_close'] > data['weekly_close'].shift(5)).astype(int)
    data['monthly_close'] = data['Close'].rolling(21).mean()
    data['monthly_trend'] = (data['monthly_close'] > data['monthly_close'].shift(21)).astype(int)
    data['mtf_alignment'] = (data['daily_trend'] + data['weekly_trend'] + data['monthly_trend']) / 3
    data['weekly_range_pos'] = (data['Close'] - data['weekly_low']) / (data['weekly_high'] - data['weekly_low'] + 1e-10)
    data['monthly_range_pos'] = (data['Close'] - data['Low'].rolling(21).min()) / \
                                (data['High'].rolling(21).max() - data['Low'].rolling(21).min() + 1e-10)
    data['trend_strength'] = abs(data['close_to_sma_20']) + abs(data['close_to_sma_50'])
    
    features_df = data[FEATURE_NAMES].copy()
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features_df

def create_labels_v3(df):
    forward_return = df['Close'].shift(-config.PREDICTION_HORIZON) / df['Close'] - 1
    labels = pd.Series(1, index=df.index)
    labels[forward_return >= config.UP_THRESHOLD] = 2
    labels[forward_return <= config.DOWN_THRESHOLD] = 0
    return labels

print("🔧 Calculating features...")
feature_data, labels_data = {}, {}
for symbol, data in stock_data.items():
    try:
        features = calculate_features_v3(data)
        labels = create_labels_v3(data)
        if len(features) > config.LOOKBACK_DAYS + config.PREDICTION_HORIZON:
            feature_data[symbol] = features
            labels_data[symbol] = labels
    except: pass
print(f"✅ Processed {len(feature_data)} stocks")

# ============================================================
# CELL 6: Dataset with Balanced Sampling
# ============================================================
class SwingDatasetV3(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        class_counts = Counter(labels)
        total = len(labels)
        self.class_weights = {c: total / (3 * count) for c, count in class_counts.items()}
        self.sample_weights = torch.FloatTensor([self.class_weights[int(l)] for l in labels])
    
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return {'features': self.features[idx], 'labels': self.labels[idx]}
    def get_sampler(self): return WeightedRandomSampler(self.sample_weights, len(self.sample_weights), True)

print("📊 Creating sequences...")
all_sequences, all_labels, all_dates = [], [], []
for symbol in feature_data.keys():
    features = feature_data[symbol].values
    labels = labels_data[symbol].values
    dates = feature_data[symbol].index
    for i in range(config.LOOKBACK_DAYS, len(features) - config.PREDICTION_HORIZON):
        seq = features[i-config.LOOKBACK_DAYS:i]
        label = labels[i]
        if not np.isnan(label) and not np.any(np.isnan(seq)):
            all_sequences.append(seq)
            all_labels.append(int(label))
            all_dates.append(dates[i])

sequences = np.array(all_sequences)
labels_arr = np.array(all_labels)
dates_arr = np.array(all_dates)

print(f"✅ Total sequences: {len(sequences):,}")

train_end = pd.Timestamp(config.TRAIN_END)
val_end = pd.Timestamp(config.VAL_END)
train_mask = dates_arr <= train_end
val_mask = (dates_arr > train_end) & (dates_arr <= val_end)
test_mask = dates_arr > val_end

train_dataset = SwingDatasetV3(sequences[train_mask], labels_arr[train_mask])
val_dataset = SwingDatasetV3(sequences[val_mask], labels_arr[val_mask])
test_dataset = SwingDatasetV3(sequences[test_mask], labels_arr[test_mask])

print(f"📊 Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
print(f"📊 Test Class Distribution: {Counter(labels_arr[test_mask])}")

# ============================================================
# CELL 7: Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma
        loss = -focal_weight * smooth_targets * log_probs
        return loss.sum(dim=-1).mean()

criterion = FocalLoss(config.FOCAL_GAMMA, config.LABEL_SMOOTHING)
print("✅ Focal Loss initialized")

# ============================================================
# CELL 8: Train CatBoost V3
# ============================================================
print("="*60)
print("🚀 TRAINING CATBOOST V3")
print("="*60)

X_train = train_dataset.features[:, -1, :].numpy()
y_train = train_dataset.labels.numpy()
X_val = val_dataset.features[:, -1, :].numpy()
y_val = val_dataset.labels.numpy()

class_counts = Counter(y_train)
total = len(y_train)
class_weights = {c: total / (3 * count) for c, count in class_counts.items()}
print(f"Class weights: {class_weights}")

catboost_model = CatBoostClassifier(
    iterations=config.CATBOOST_ITERATIONS,
    depth=config.CATBOOST_DEPTH,
    learning_rate=config.CATBOOST_LR,
    l2_leaf_reg=config.CATBOOST_L2_REG,
    loss_function='MultiClass',
    eval_metric='TotalF1:average=Macro',
    random_seed=42,
    verbose=200,
    early_stopping_rounds=150,
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    class_weights=class_weights,
    bootstrap_type='Bayesian',
)

catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
cat_preds = catboost_model.predict(X_val)
cat_f1 = f1_score(y_val, cat_preds, average='macro')
print(f"✅ CatBoost Val Macro-F1: {cat_f1:.4f}")

# ============================================================
# CELL 9: TFT Model V3
# ============================================================
class TFTModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        h, heads, drop = config.TFT_HIDDEN_SIZE, config.TFT_ATTENTION_HEADS, config.TFT_DROPOUT
        self.var_sel = nn.Sequential(nn.Linear(40, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, 40), nn.Softmax(dim=-1))
        self.embed = nn.Linear(40, h)
        self.lstm = nn.LSTM(h, h, config.TFT_NUM_LAYERS, batch_first=True, dropout=drop)
        self.attn1 = nn.MultiheadAttention(h, heads, dropout=drop, batch_first=True)
        self.attn2 = nn.MultiheadAttention(h, heads, dropout=drop, batch_first=True)
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)
        self.grn_fc1 = nn.Linear(h, h*2)
        self.grn_fc2 = nn.Linear(h*2, h)
        self.grn_gate = nn.Linear(h*2, h)
        self.grn_norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(drop)
        self.output = nn.Linear(h, 3)
    
    def forward(self, x):
        w = self.var_sel(x.mean(1))
        x = self.embed(x * w.unsqueeze(1))
        o, _ = self.lstm(x)
        a1, _ = self.attn1(o, o, o)
        o = self.norm1(o + a1)
        a2, _ = self.attn2(o, o, o)
        o = self.norm2(o + a2)
        final = o[:, -1, :]
        g = F.gelu(self.grn_fc1(final))
        out = self.grn_norm(final + torch.sigmoid(self.grn_gate(g)) * self.grn_fc2(g))
        return self.output(self.dropout(out))

print("="*60)
print("🚀 TRAINING TFT V3")
print("="*60)

tft_model = TFTModelV3().to(DEVICE)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_dataset.get_sampler())
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
optimizer = torch.optim.AdamW(tft_model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

best_f1, patience_count, best_state = 0, 0, None
for epoch in range(config.MAX_EPOCHS):
    tft_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = tft_model(batch['features'].to(DEVICE))
        loss = criterion(out, batch['labels'].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tft_model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    
    tft_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in val_loader:
            preds.extend(tft_model(b['features'].to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(b['labels'].numpy())
    
    macro_f1 = f1_score(labels, preds, average='macro')
    print(f"Epoch {epoch+1}: Macro-F1={macro_f1:.4f}")
    
    if macro_f1 > best_f1:
        best_f1, patience_count = macro_f1, 0
        best_state = {k: v.cpu().clone() for k, v in tft_model.state_dict().items()}
    else:
        patience_count += 1
        if patience_count >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

tft_model.load_state_dict(best_state)
print(f"✅ TFT V3 Best Macro-F1: {best_f1:.4f}")

# ============================================================
# CELL 10: Stockformer V3
# ============================================================
class StockformerV3(nn.Module):
    def __init__(self):
        super().__init__()
        d, h, n, drop = config.STOCKFORMER_D_MODEL, config.STOCKFORMER_N_HEADS, config.STOCKFORMER_N_LAYERS, config.STOCKFORMER_DROPOUT
        self.input_proj = nn.Linear(40, d)
        self.pos = nn.Parameter(torch.randn(1, 60, d) * 0.02)
        self.trend_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, d*4, drop, batch_first=True, activation='gelu'), n)
        self.seasonal_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, d*4, drop, batch_first=True, activation='gelu'), n)
        self.residual_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, d*4, drop, batch_first=True, activation='gelu'), n)
        self.fusion = nn.Sequential(nn.Linear(d*3, d*2), nn.LayerNorm(d*2), nn.GELU(), nn.Dropout(drop), nn.Linear(d*2, d))
        self.output = nn.Linear(d, 3)
    
    def forward(self, x):
        b, s, _ = x.shape
        x = self.input_proj(x) + self.pos[:, :s]
        trend = x.cumsum(1) / torch.arange(1, s+1, device=x.device).view(1, -1, 1)
        seasonal = x - trend
        t = self.trend_enc(trend)[:, -1]
        se = self.seasonal_enc(seasonal)[:, -1]
        r = self.residual_enc(x)[:, -1]
        return self.output(self.fusion(torch.cat([t, se, r], -1)))

print("="*60)
print("🚀 TRAINING STOCKFORMER V3")
print("="*60)

sf_model = StockformerV3().to(DEVICE)
optimizer = torch.optim.AdamW(sf_model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

best_f1, patience_count, best_state = 0, 0, None
for epoch in range(config.MAX_EPOCHS):
    sf_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = sf_model(batch['features'].to(DEVICE))
        loss = criterion(out, batch['labels'].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sf_model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    
    sf_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in val_loader:
            preds.extend(sf_model(b['features'].to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(b['labels'].numpy())
    
    macro_f1 = f1_score(labels, preds, average='macro')
    print(f"Epoch {epoch+1}: Macro-F1={macro_f1:.4f}")
    
    if macro_f1 > best_f1:
        best_f1, patience_count = macro_f1, 0
        best_state = {k: v.cpu().clone() for k, v in sf_model.state_dict().items()}
    else:
        patience_count += 1
        if patience_count >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

sf_model.load_state_dict(best_state)
print(f"✅ Stockformer V3 Best Macro-F1: {best_f1:.4f}")

# ============================================================
# CELL 11: Ensemble Evaluation
# ============================================================
print("="*60)
print("📊 ENSEMBLE EVALUATION V3")
print("="*60)

X_test = test_dataset.features.numpy()
y_test = test_dataset.labels.numpy()

cat_probs = catboost_model.predict_proba(X_test[:, -1, :])
tft_model.eval()
sf_model.eval()
with torch.no_grad():
    tft_probs = F.softmax(tft_model(torch.FloatTensor(X_test).to(DEVICE)), 1).cpu().numpy()
    sf_probs = F.softmax(sf_model(torch.FloatTensor(X_test).to(DEVICE)), 1).cpu().numpy()

# Dynamic weighting
cat_conf = cat_probs.max(axis=1)
tft_conf = tft_probs.max(axis=1)
sf_conf = sf_probs.max(axis=1)
conf_sum = cat_conf + tft_conf + sf_conf + 1e-6

w_cat = cat_conf / conf_sum * 0.5 + config.CATBOOST_WEIGHT * 0.5
w_tft = tft_conf / conf_sum * 0.5 + config.TFT_WEIGHT * 0.5
w_sf = sf_conf / conf_sum * 0.5 + config.STOCKFORMER_WEIGHT * 0.5
w_sum = w_cat + w_tft + w_sf

ensemble_probs = (w_cat[:, None]/w_sum[:, None] * cat_probs + 
                  w_tft[:, None]/w_sum[:, None] * tft_probs + 
                  w_sf[:, None]/w_sum[:, None] * sf_probs)

ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_conf = np.max(ensemble_probs, axis=1)

ensemble_acc = (ensemble_preds == y_test).mean()
ensemble_f1 = f1_score(y_test, ensemble_preds, average='macro')

print(f"\n🎯 ENSEMBLE TEST ACCURACY: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
print(f"🎯 ENSEMBLE MACRO-F1: {ensemble_f1:.4f}")

print(f"\n📈 High-Confidence Performance:")
for thresh in [0.5, 0.6, 0.7, 0.8]:
    mask = ensemble_conf >= thresh
    if mask.sum() > 0:
        acc = (ensemble_preds[mask] == y_test[mask]).mean()
        print(f"   Conf >= {int(thresh*100)}%: {mask.sum():,} ({mask.mean()*100:.1f}%), Acc={acc:.4f}")

print(f"\n📊 Per-Class Recall:")
for c, name in [(0, 'SHORT'), (1, 'NEUTRAL'), (2, 'LONG')]:
    mask = y_test == c
    if mask.sum() > 0:
        recall = (ensemble_preds[mask] == c).mean()
        print(f"   {name}: {recall:.4f}")

print("\n" + "="*60)
print(classification_report(y_test, ensemble_preds, target_names=['SHORT', 'NEUTRAL', 'LONG']))

# ============================================================
# CELL 12: Save Models
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

print("💾 Saving V3 models...")
catboost_model.save_model(f"{config.MODEL_SAVE_PATH}catboost_v3.cbm")
torch.save(tft_model.state_dict(), f"{config.MODEL_SAVE_PATH}tft_v3.pt")
torch.save(sf_model.state_dict(), f"{config.MODEL_SAVE_PATH}stockformer_v3.pt")

model_config = {
    "version": "V3_SOTA",
    "feature_columns": FEATURE_NAMES,
    "thresholds": {"up": config.UP_THRESHOLD, "down": config.DOWN_THRESHOLD},
    "test_accuracy": float(ensemble_acc),
    "test_macro_f1": float(ensemble_f1),
    "trained_at": timestamp
}
with open(f"{config.MODEL_SAVE_PATH}model_config_v3.json", "w") as f:
    json.dump(model_config, f, indent=2)

print(f"✅ V3 Models saved to: {config.MODEL_SAVE_PATH}")
print(f"🎉 Training complete! Accuracy: {ensemble_acc*100:.1f}%, Macro-F1: {ensemble_f1:.4f}")
