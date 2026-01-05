"""
================================================================================
                    SWINGAI COMPLETE TRAINING NOTEBOOK
                    ==================================
                    
    AI Engine: CatBoost (35%) + TFT (35%) + Stockformer (30%)
    Features: 60 (Stock + Market + Institutional)
    Market: Indian NSE/BSE
    
    Run on: Google Colab with T4 GPU
    Training Time: ~2 hours
    
================================================================================
"""

# ==============================================================================
# SECTION 1: INSTALLATION & SETUP
# ==============================================================================

"""
Run this cell first in Colab:

!pip install -q catboost pytorch-forecasting pytorch-lightning
!pip install -q yfinance pandas numpy scikit-learn
!pip install -q statsmodels  # For STL decomposition
!pip install -q ta  # Technical analysis library
!pip install -q torch torchvision torchaudio
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ==============================================================================
# SECTION 2: CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Central configuration for SwingAI"""
    
    # Data settings
    LOOKBACK_DAYS: int = 60  # Days of history for each prediction
    PREDICTION_HORIZON: int = 5  # Predict 5 days ahead (swing trading)
    MIN_HISTORY_DAYS: int = 252  # Minimum 1 year history required
    
    # Label thresholds
    UP_THRESHOLD: float = 0.03  # +3% = UP
    DOWN_THRESHOLD: float = -0.02  # -2% = DOWN
    
    # Training settings
    TRAIN_START: str = "2019-01-01"
    TRAIN_END: str = "2024-06-30"
    VAL_END: str = "2024-09-30"
    TEST_END: str = "2024-12-31"
    
    # Model weights
    CATBOOST_WEIGHT: float = 0.35
    TFT_WEIGHT: float = 0.35
    STOCKFORMER_WEIGHT: float = 0.30
    
    # Training hyperparameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    MAX_EPOCHS: int = 50
    PATIENCE: int = 5
    
    # CatBoost settings
    CATBOOST_ITERATIONS: int = 1000
    CATBOOST_DEPTH: int = 6
    CATBOOST_LR: float = 0.05
    
    # TFT settings
    TFT_HIDDEN_SIZE: int = 64
    TFT_ATTENTION_HEADS: int = 4
    TFT_DROPOUT: float = 0.1
    
    # Stockformer settings
    STOCKFORMER_D_MODEL: int = 64
    STOCKFORMER_N_HEADS: int = 4
    STOCKFORMER_N_LAYERS: int = 2
    STL_PERIOD: int = 5  # Weekly seasonality (5 trading days)
    
    # Paths
    MODEL_SAVE_PATH: str = "./models/"
    DATA_CACHE_PATH: str = "./data_cache/"

config = Config()

# ==============================================================================
# SECTION 3: STOCK UNIVERSE
# ==============================================================================

# Nifty 200 stocks (F&O eligible for shorting)
NIFTY_200 = [
    # Nifty 50
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "NTPC.NS",
    "WIPRO.NS", "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "TATASTEEL.NS", "ONGC.NS",
    "TECHM.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "BAJAJFINSV.NS", "GRASIM.NS",
    "DRREDDY.NS", "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
    "COALINDIA.NS", "SBILIFE.NS", "BPCL.NS", "INDUSINDBK.NS", "TATACONSUM.NS",
    "HEROMOTOCO.NS", "UPL.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "LTIM.NS",
    
    # Nifty Next 50
    "ADANIGREEN.NS", "ADANITRANS.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BANDHANBNK.NS", "BANKBARODA.NS", "BERGEPAINT.NS", "BIOCON.NS",
    "BOSCHLTD.NS", "CADILAHC.NS", "COLPAL.NS", "CONCOR.NS", "DABUR.NS",
    "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "ICICIPRULI.NS",
    "IGL.NS", "INDUSTOWER.NS", "JUBLFOOD.NS", "LUPIN.NS", "MARICO.NS",
    "MCDOWELL-N.NS", "MOTHERSON.NS", "MUTHOOTFIN.NS", "NAUKRI.NS",
    "NMDC.NS", "PAGEIND.NS", "PEL.NS", "PETRONET.NS", "PIDILITIND.NS",
    "PNB.NS", "SIEMENS.NS", "SRF.NS", "TATAPOWER.NS", "TORNTPHARM.NS",
    "TRENT.NS", "VEDL.NS", "VOLTAS.NS", "ZEEL.NS", "ZOMATO.NS",
    
    # Additional F&O stocks
    "AARTIIND.NS", "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS",
    "ALKEM.NS", "ASHOKLEY.NS", "ASTRAL.NS", "ATUL.NS", "AUBANK.NS",
    "BALKRISIND.NS", "BALRAMCHIN.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS",
    "CANFINHOME.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS", "COFORGE.NS", "COROMANDEL.NS",
    "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS", "DEEPAKNTR.NS", "DELTACORP.NS",
    "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "FSL.NS",
    "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS", "GODREJPROP.NS", "GRANULES.NS",
    "GSPL.NS", "GUJGASLTD.NS", "HAL.NS", "HDFCAMC.NS", "HINDCOPPER.NS",
    "HINDPETRO.NS", "IBREALEST.NS", "IDFCFIRSTB.NS", "IEX.NS", "INDIANB.NS",
    "INDHOTEL.NS", "INDIACEM.NS", "INDIGO.NS", "IOC.NS", "IRCTC.NS",
    "JINDALSTEL.NS", "JKCEMENT.NS", "JSWENERGY.NS", "L&TFH.NS", "LALPATHLAB.NS",
    "LAURUSLABS.NS", "LICHSGFIN.NS", "LTI.NS", "M&MFIN.NS", "MANAPPURAM.NS",
    "MFSL.NS", "MGL.NS", "MINDTREE.NS", "MRF.NS", "NAM-INDIA.NS",
    "NATIONALUM.NS", "NAVINFLUOR.NS", "OBEROIRLTY.NS", "OFSS.NS", "PERSISTENT.NS",
    "PFIZER.NS", "PIIND.NS", "POLYCAB.NS", "PVR.NS", "RAIN.NS",
    "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", "SAIL.NS", "SBICARD.NS",
    "SHREECEM.NS", "STAR.NS", "SUNTV.NS", "SYNGENE.NS", "TATACHEM.NS",
    "TATACOMM.NS", "TATAELXSI.NS", "TATAMTRDVR.NS", "TECHM.NS", "TORNTPOWER.NS",
    "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", "WHIRLPOOL.NS"
]

# Market indices
MARKET_INDICES = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK", 
    "VIX": "^INDIAVIX"
}

print(f"Stock Universe: {len(NIFTY_200)} stocks")

# ==============================================================================
# SECTION 4: DATA COLLECTION
# ==============================================================================

import yfinance as yf

class DataCollector:
    """Collects and caches stock data"""
    
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.DATA_CACHE_PATH, exist_ok=True)
    
    def download_stock(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Download single stock data"""
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if len(df) < self.config.MIN_HISTORY_DAYS:
                return None
            df['Symbol'] = symbol.replace('.NS', '')
            return df
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
    
    def download_all_stocks(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Download all stocks in parallel"""
        data = {}
        
        print(f"Downloading {len(symbols)} stocks...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.download_stock, sym, start, end): sym 
                for sym in symbols
            }
            
            for i, future in enumerate(as_completed(futures)):
                symbol = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        data[symbol] = df
                except Exception as e:
                    print(f"Error with {symbol}: {e}")
                
                if (i + 1) % 50 == 0:
                    print(f"  Downloaded {i+1}/{len(symbols)} stocks")
        
        print(f"Successfully downloaded {len(data)} stocks")
        return data
    
    def download_market_data(self, start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Download market indices"""
        market_data = {}
        
        for name, symbol in MARKET_INDICES.items():
            try:
                df = yf.download(symbol, start=start, end=end, progress=False)
                if len(df) > 0:
                    market_data[name] = df
                    print(f"  Downloaded {name}")
            except Exception as e:
                print(f"Error downloading {name}: {e}")
        
        return market_data
    
    def get_fii_dii_proxy(self, nifty_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate FII/DII proxy from market data.
        In production, use actual NSE FII/DII data API.
        """
        df = pd.DataFrame(index=nifty_data.index)
        
        # Proxy based on market movements and volume
        # Positive days with high volume = FII buying
        # Negative days with high volume = FII selling
        returns = nifty_data['Close'].pct_change()
        volume = nifty_data['Volume']
        vol_ma = volume.rolling(20).mean()
        vol_ratio = volume / vol_ma
        
        # FII proxy (in crores, synthetic)
        df['FII_Net'] = (returns * vol_ratio * 1000).fillna(0)  # Synthetic FII
        df['DII_Net'] = (-returns * vol_ratio * 500).fillna(0)  # DII often contra
        
        # Smooth
        df['FII_Net_5d'] = df['FII_Net'].rolling(5).mean()
        df['DII_Net_5d'] = df['DII_Net'].rolling(5).mean()
        
        return df

# ==============================================================================
# SECTION 5: FEATURE ENGINEERING
# ==============================================================================

class FeatureEngineer:
    """Calculates all 60 features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_names = []
    
    def calculate_all_features(
        self, 
        stock_data: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        fii_dii_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate all 60 features for a stock"""
        
        df = stock_data.copy()
        
        # ==================== PRICE ACTION FEATURES (10) ====================
        
        # Returns
        df['return_1d'] = df['Close'].pct_change(1)
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)
        
        # Volatility
        df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
        
        # Moving average distance
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['close_to_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
        df['close_to_sma_50'] = (df['Close'] - df['sma_50']) / df['sma_50']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14_norm'] = (df['rsi_14'] - 50) / 50
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram_norm'] = (df['macd'] - df['macd_signal']) / df['Close']
        
        # Bollinger Bands position
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ==================== SMC/ICT FEATURES (15) ====================
        
        # Market structure
        df['swing_high'] = df['High'].rolling(10, center=True).max()
        df['swing_low'] = df['Low'].rolling(10, center=True).min()
        
        # Higher highs, higher lows
        df['prev_swing_high'] = df['swing_high'].shift(10)
        df['prev_swing_low'] = df['swing_low'].shift(10)
        df['higher_high'] = (df['swing_high'] > df['prev_swing_high']).astype(int)
        df['higher_low'] = (df['swing_low'] > df['prev_swing_low']).astype(int)
        df['lower_high'] = (df['swing_high'] < df['prev_swing_high']).astype(int)
        df['lower_low'] = (df['swing_low'] < df['prev_swing_low']).astype(int)
        
        # Structure score
        df['structure_score'] = (df['higher_high'].rolling(5).sum() + 
                                 df['higher_low'].rolling(5).sum() -
                                 df['lower_high'].rolling(5).sum() - 
                                 df['lower_low'].rolling(5).sum()) / 10
        
        # Range position (Premium/Discount)
        range_high = df['High'].rolling(50).max()
        range_low = df['Low'].rolling(50).min()
        df['range_position'] = (df['Close'] - range_low) / (range_high - range_low + 1e-10)
        
        # Distance to swing points
        df['dist_to_swing_high'] = (df['swing_high'] - df['Close']) / df['Close']
        df['dist_to_swing_low'] = (df['Close'] - df['swing_low']) / df['Close']
        
        # Discount/Premium zones
        df['in_discount'] = (df['range_position'] < 0.5).astype(int)
        df['in_deep_discount'] = (df['range_position'] < 0.3).astype(int)
        df['in_premium'] = (df['range_position'] > 0.7).astype(int)
        
        # Order Blocks (simplified)
        df['big_move_up'] = (df['return_1d'] > df['return_1d'].rolling(20).std() * 2).astype(int)
        df['big_move_down'] = (df['return_1d'] < -df['return_1d'].rolling(20).std() * 2).astype(int)
        df['near_bullish_ob'] = df['big_move_up'].rolling(10).sum()
        df['near_bearish_ob'] = df['big_move_down'].rolling(10).sum()
        
        # Fair Value Gap (simplified)
        df['gap_up'] = ((df['Low'] > df['High'].shift(1)) & (df['return_1d'] > 0.01)).astype(int)
        df['gap_down'] = ((df['High'] < df['Low'].shift(1)) & (df['return_1d'] < -0.01)).astype(int)
        df['bullish_fvg'] = df['gap_up'].rolling(5).sum()
        df['bearish_fvg'] = df['gap_down'].rolling(5).sum()
        
        # Liquidity sweeps
        df['sweep_high'] = ((df['High'] > df['swing_high'].shift(1)) & 
                           (df['Close'] < df['swing_high'].shift(1))).astype(int)
        df['sweep_low'] = ((df['Low'] < df['swing_low'].shift(1)) & 
                          (df['Close'] > df['swing_low'].shift(1))).astype(int)
        
        # Break of Structure
        df['bos_bullish'] = ((df['Close'] > df['swing_high'].shift(1)) & 
                            (df['higher_high'] == 1)).astype(int)
        df['bos_bearish'] = ((df['Close'] < df['swing_low'].shift(1)) & 
                            (df['lower_low'] == 1)).astype(int)
        
        # ==================== VOLUME FEATURES (8) ====================
        
        df['volume_ma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_ma_20'] + 1e-10)
        df['volume_trend'] = df['Volume'].rolling(5).mean() / (df['Volume'].rolling(20).mean() + 1e-10)
        
        # On-Balance Volume
        obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['obv_slope'] = obv.diff(5) / (obv.rolling(20).std() + 1e-10)
        
        # VWAP distance
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['close_to_vwap'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-10)
        
        # Buying pressure
        df['buying_pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # Accumulation
        df['accumulation_score'] = (df['buying_pressure'] * df['volume_ratio']).rolling(5).mean()
        
        # Big volume day
        df['big_volume_day'] = (df['volume_ratio'] > 2).astype(int)
        
        # ==================== MULTI-TIMEFRAME FEATURES (7) ====================
        
        # Weekly trend
        df['weekly_close'] = df['Close'].rolling(5).mean()
        df['weekly_high'] = df['High'].rolling(5).max()
        df['weekly_low'] = df['Low'].rolling(5).min()
        df['weekly_trend'] = (df['weekly_close'] > df['weekly_close'].shift(5)).astype(int)
        
        # Monthly trend
        df['monthly_close'] = df['Close'].rolling(21).mean()
        df['monthly_trend'] = (df['monthly_close'] > df['monthly_close'].shift(21)).astype(int)
        
        # Daily trend
        df['daily_trend'] = (df['Close'] > df['sma_20']).astype(int)
        
        # MTF alignment
        df['mtf_alignment'] = (df['daily_trend'] + df['weekly_trend'] + df['monthly_trend']) / 3
        
        # Range positions
        df['weekly_range_pos'] = (df['Close'] - df['weekly_low']) / (df['weekly_high'] - df['weekly_low'] + 1e-10)
        df['monthly_range_pos'] = (df['Close'] - df['Low'].rolling(21).min()) / \
                                  (df['High'].rolling(21).max() - df['Low'].rolling(21).min() + 1e-10)
        
        # Trend strength
        df['trend_strength'] = abs(df['close_to_sma_20']) + abs(df['close_to_sma_50'])
        
        # ==================== MARKET CONTEXT FEATURES (15) ====================
        
        if 'NIFTY50' in market_data:
            nifty = market_data['NIFTY50']['Close'].reindex(df.index, method='ffill')
            
            # Nifty features
            df['nifty_return_1d'] = nifty.pct_change(1)
            df['nifty_return_5d'] = nifty.pct_change(5)
            
            # Nifty trend
            nifty_sma_20 = nifty.rolling(20).mean()
            df['nifty_trend'] = (nifty > nifty_sma_20).astype(int)
            
            # Nifty RSI
            nifty_delta = nifty.diff()
            nifty_gain = nifty_delta.where(nifty_delta > 0, 0).rolling(14).mean()
            nifty_loss = (-nifty_delta.where(nifty_delta < 0, 0)).rolling(14).mean()
            nifty_rs = nifty_gain / (nifty_loss + 1e-10)
            df['nifty_rsi'] = (100 - (100 / (1 + nifty_rs)) - 50) / 50
            
            # Relative strength vs Nifty
            df['relative_strength'] = df['return_5d'] - df['nifty_return_5d']
        else:
            df['nifty_return_1d'] = 0
            df['nifty_return_5d'] = 0
            df['nifty_trend'] = 0.5
            df['nifty_rsi'] = 0
            df['relative_strength'] = 0
        
        # VIX features
        if 'VIX' in market_data:
            vix = market_data['VIX']['Close'].reindex(df.index, method='ffill')
            df['vix_level'] = vix / 20  # Normalize around 20
            df['vix_percentile'] = vix.rolling(252).rank(pct=True)
            df['vix_regime'] = pd.cut(vix, bins=[0, 15, 20, 25, 100], 
                                      labels=[0, 1, 2, 3]).astype(float)
            df['vix_trend'] = (vix > vix.rolling(5).mean()).astype(int)
        else:
            df['vix_level'] = 1
            df['vix_percentile'] = 0.5
            df['vix_regime'] = 1
            df['vix_trend'] = 0
        
        # Market breadth proxy (from Nifty movement)
        if 'NIFTY50' in market_data:
            nifty = market_data['NIFTY50']['Close']
            df['advance_decline'] = (nifty.pct_change() > 0).rolling(5).mean()
            df['pct_above_sma20'] = (nifty > nifty.rolling(20).mean()).rolling(20).mean()
            df['pct_above_sma50'] = (nifty > nifty.rolling(50).mean()).rolling(50).mean()
        else:
            df['advance_decline'] = 0.5
            df['pct_above_sma20'] = 0.5
            df['pct_above_sma50'] = 0.5
        
        # Sector features (simplified - use stock's own performance as proxy)
        df['sector_performance'] = df['return_5d']
        df['sector_rank'] = df['return_5d'].rolling(20).rank(pct=True)
        df['sector_momentum'] = df['return_20d']
        
        # ==================== INSTITUTIONAL FEATURES (5) ====================
        
        if fii_dii_data is not None:
            fii_dii = fii_dii_data.reindex(df.index, method='ffill')
            df['fii_net_1d'] = fii_dii['FII_Net'] / 1000  # Normalize
            df['fii_net_5d'] = fii_dii['FII_Net_5d'] / 1000
            df['dii_net_1d'] = fii_dii['DII_Net'] / 1000
            df['fii_dii_divergence'] = (df['fii_net_1d'] - df['dii_net_1d'])
            df['institutional_flow'] = df['fii_net_5d'] + fii_dii['DII_Net_5d'] / 1000
        else:
            df['fii_net_1d'] = 0
            df['fii_net_5d'] = 0
            df['dii_net_1d'] = 0
            df['fii_dii_divergence'] = 0
            df['institutional_flow'] = 0
        
        # ==================== SELECT FINAL 60 FEATURES ====================
        
        self.feature_names = [
            # Price Action (10)
            'return_1d', 'return_5d', 'return_10d', 'return_20d', 'volatility_20d',
            'close_to_sma_20', 'close_to_sma_50', 'rsi_14_norm', 'macd_histogram_norm', 'bb_position',
            
            # SMC/ICT (15)
            'structure_score', 'range_position', 'dist_to_swing_high', 'dist_to_swing_low',
            'in_discount', 'in_deep_discount', 'in_premium', 'near_bullish_ob', 'near_bearish_ob',
            'bullish_fvg', 'bearish_fvg', 'sweep_high', 'sweep_low', 'bos_bullish', 'bos_bearish',
            
            # Volume (8)
            'volume_ratio', 'volume_trend', 'obv_slope', 'close_to_vwap',
            'buying_pressure', 'accumulation_score', 'big_volume_day', 'higher_high',
            
            # Multi-timeframe (7)
            'daily_trend', 'weekly_trend', 'monthly_trend', 'mtf_alignment',
            'weekly_range_pos', 'monthly_range_pos', 'trend_strength',
            
            # Market Context (15)
            'nifty_return_1d', 'nifty_return_5d', 'nifty_trend', 'nifty_rsi', 'relative_strength',
            'vix_level', 'vix_percentile', 'vix_regime', 'vix_trend',
            'advance_decline', 'pct_above_sma20', 'pct_above_sma50',
            'sector_performance', 'sector_rank', 'sector_momentum',
            
            # Institutional (5)
            'fii_net_1d', 'fii_net_5d', 'dii_net_1d', 'fii_dii_divergence', 'institutional_flow'
        ]
        
        # Select and clean
        df_features = df[self.feature_names].copy()
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(0)
        
        return df_features
    
    def create_labels(self, stock_data: pd.DataFrame) -> pd.Series:
        """Create labels: 0=DOWN, 1=SIDEWAYS, 2=UP"""
        
        # Forward return over prediction horizon
        forward_return = stock_data['Close'].shift(-self.config.PREDICTION_HORIZON) / stock_data['Close'] - 1
        
        labels = pd.Series(1, index=stock_data.index)  # Default SIDEWAYS
        labels[forward_return >= self.config.UP_THRESHOLD] = 2  # UP
        labels[forward_return <= self.config.DOWN_THRESHOLD] = 0  # DOWN
        
        return labels

print("Feature Engineering module loaded")
print(f"Total features: 60")

# ==============================================================================
# SECTION 6: DATASET PREPARATION
# ==============================================================================

import torch
from torch.utils.data import Dataset, DataLoader

class SwingDataset(Dataset):
    """PyTorch Dataset for SwingAI"""
    
    def __init__(
        self, 
        features: np.ndarray,  # Shape: (samples, lookback, features)
        labels: np.ndarray,    # Shape: (samples,)
        symbols: List[str] = None
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.symbols = symbols
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }


def prepare_sequences(
    stock_data: Dict[str, pd.DataFrame],
    feature_data: Dict[str, pd.DataFrame],
    labels_data: Dict[str, pd.Series],
    config: Config
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare sequences for training"""
    
    all_sequences = []
    all_labels = []
    all_symbols = []
    
    lookback = config.LOOKBACK_DAYS
    
    for symbol in feature_data.keys():
        features = feature_data[symbol].values
        labels = labels_data[symbol].values
        
        # Create sequences
        for i in range(lookback, len(features) - config.PREDICTION_HORIZON):
            seq = features[i-lookback:i]
            label = labels[i]
            
            if not np.isnan(label) and not np.any(np.isnan(seq)):
                all_sequences.append(seq)
                all_labels.append(int(label))
                all_symbols.append(symbol)
    
    return np.array(all_sequences), np.array(all_labels), all_symbols


def create_data_splits(
    sequences: np.ndarray,
    labels: np.ndarray,
    symbols: List[str],
    config: Config
) -> Dict[str, SwingDataset]:
    """Create train/val/test splits"""
    
    # For simplicity, use percentage splits
    # In production, use time-based splits
    n = len(labels)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    datasets = {
        'train': SwingDataset(sequences[:train_end], labels[:train_end]),
        'val': SwingDataset(sequences[train_end:val_end], labels[train_end:val_end]),
        'test': SwingDataset(sequences[val_end:], labels[val_end:])
    }
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(datasets['train'])} samples")
    print(f"  Val: {len(datasets['val'])} samples")
    print(f"  Test: {len(datasets['test'])} samples")
    
    return datasets

# ==============================================================================
# SECTION 7: MODEL 1 - CATBOOST
# ==============================================================================

from catboost import CatBoostClassifier

class CatBoostModel:
    """CatBoost model for tabular features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.name = "CatBoost"
    
    def train(self, train_dataset: SwingDataset, val_dataset: SwingDataset):
        """Train CatBoost model"""
        
        print(f"\n{'='*60}")
        print("Training CatBoost Model")
        print(f"{'='*60}")
        
        # Flatten sequences to use only last day features
        X_train = train_dataset.features[:, -1, :].numpy()  # Last day only
        y_train = train_dataset.labels.numpy()
        
        X_val = val_dataset.features[:, -1, :].numpy()
        y_val = val_dataset.labels.numpy()
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        self.model = CatBoostClassifier(
            iterations=self.config.CATBOOST_ITERATIONS,
            depth=self.config.CATBOOST_DEPTH,
            learning_rate=self.config.CATBOOST_LR,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            task_type='GPU' if torch.cuda.is_available() else 'CPU'
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # Evaluate
        train_acc = (self.model.predict(X_train) == y_train).mean()
        val_acc = (self.model.predict(X_val) == y_val).mean()
        
        print(f"\nCatBoost Results:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        
        return {'train_acc': train_acc, 'val_acc': val_acc}
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        if features.ndim == 3:
            features = features[:, -1, :]  # Use last day
        
        probs = self.model.predict_proba(features)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def save(self, path: str):
        """Save model"""
        self.model.save_model(f"{path}/catboost_model.cbm")
    
    def load(self, path: str):
        """Load model"""
        self.model = CatBoostClassifier()
        self.model.load_model(f"{path}/catboost_model.cbm")

# ==============================================================================
# SECTION 8: MODEL 2 - TFT (Temporal Fusion Transformer)
# ==============================================================================

import torch.nn as nn

class TFTModel(nn.Module):
    """
    Simplified TFT implementation for SwingAI.
    For production, use pytorch-forecasting library.
    """
    
    def __init__(self, config: Config, num_features: int = 60):
        super().__init__()
        
        self.config = config
        self.name = "TFT"
        
        hidden_size = config.TFT_HIDDEN_SIZE
        num_heads = config.TFT_ATTENTION_HEADS
        dropout = config.TFT_DROPOUT
        
        # Variable selection network (simplified)
        self.variable_selection = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_features),
            nn.Softmax(dim=-1)
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated residual network
        self.grn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.grn_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.grn_norm = nn.LayerNorm(hidden_size)
        
        # Output heads for multi-horizon
        self.output_layer = nn.Linear(hidden_size, 3)  # 3 classes
        
    def forward(self, x):
        """
        x: (batch, seq_len, features)
        """
        batch_size, seq_len, num_features = x.shape
        
        # Variable selection
        var_weights = self.variable_selection(x.mean(dim=1))  # (batch, features)
        x = x * var_weights.unsqueeze(1)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Gated residual
        grn_out = self.grn(attn_out)
        gate = self.grn_gate(attn_out)
        grn_out = self.grn_norm(attn_out + gate * grn_out)
        
        # Take last hidden state
        final = grn_out[:, -1, :]  # (batch, hidden)
        
        # Output
        logits = self.output_layer(final)  # (batch, 3)
        
        return logits
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability"""
        with torch.no_grad():
            var_weights = self.variable_selection(x.mean(dim=1))
            lstm_out, _ = self.lstm(x * var_weights.unsqueeze(1))
            _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        return var_weights, attn_weights


class TFTTrainer:
    """Trainer for TFT model"""
    
    def __init__(self, config: Config, num_features: int = 60):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TFTModel(config, num_features).to(self.device)
        self.name = "TFT"
        
    def train(self, train_dataset: SwingDataset, val_dataset: SwingDataset):
        """Train TFT model"""
        
        print(f"\n{'='*60}")
        print("Training TFT Model")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config.MAX_EPOCHS):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = val_correct / val_total
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{self.config.MAX_EPOCHS} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_state)
        
        print(f"\nTFT Results:")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        
        return {'train_acc': train_acc, 'val_acc': best_val_acc}
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def save(self, path: str):
        """Save model"""
        torch.save(self.model.state_dict(), f"{path}/tft_model.pt")
    
    def load(self, path: str):
        """Load model"""
        self.model.load_state_dict(torch.load(f"{path}/tft_model.pt"))

# ==============================================================================
# SECTION 9: MODEL 3 - STOCKFORMER
# ==============================================================================

from statsmodels.tsa.seasonal import STL

class StockformerModel(nn.Module):
    """
    Stockformer: STL Decomposition + Self-Attention
    Based on: "Stockformer: A Swing Trading Strategy Based on 
              STL Decomposition and Self-Attention Networks"
    """
    
    def __init__(self, config: Config, num_features: int = 60):
        super().__init__()
        
        self.config = config
        self.name = "Stockformer"
        
        d_model = config.STOCKFORMER_D_MODEL
        n_heads = config.STOCKFORMER_N_HEADS
        n_layers = config.STOCKFORMER_N_LAYERS
        
        # Separate encoders for each STL component
        # Trend encoder
        self.trend_encoder = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Seasonal encoder  
        self.seasonal_encoder = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual encoder
        self.residual_encoder = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.LOOKBACK_DAYS, d_model) * 0.02
        )
        
        # Self-attention for each component
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.trend_attention = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.seasonal_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, 0.1, batch_first=True),
            num_layers=n_layers
        )
        self.residual_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, 0.1, batch_first=True),
            num_layers=n_layers
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Output
        self.output = nn.Linear(d_model, 3)
    
    def forward(self, x, trend=None, seasonal=None, residual=None):
        """
        x: (batch, seq_len, features) - raw features
        If STL components not provided, use raw features for all
        """
        batch_size, seq_len, num_features = x.shape
        
        # If STL components not provided, use approximations
        if trend is None:
            # Approximate trend with moving average
            trend = x.cumsum(dim=1) / torch.arange(1, seq_len+1, device=x.device).view(1, -1, 1)
            seasonal = x - trend
            residual = torch.zeros_like(x)
        
        # Encode each component
        trend_enc = self.trend_encoder(trend) + self.pos_encoding[:, :seq_len, :]
        seasonal_enc = self.seasonal_encoder(seasonal) + self.pos_encoding[:, :seq_len, :]
        residual_enc = self.residual_encoder(residual) + self.pos_encoding[:, :seq_len, :]
        
        # Self-attention
        trend_out = self.trend_attention(trend_enc)
        seasonal_out = self.seasonal_attention(seasonal_enc)
        residual_out = self.residual_attention(residual_enc)
        
        # Take last time step
        trend_final = trend_out[:, -1, :]
        seasonal_final = seasonal_out[:, -1, :]
        residual_final = residual_out[:, -1, :]
        
        # Fuse
        combined = torch.cat([trend_final, seasonal_final, residual_final], dim=-1)
        fused = self.fusion(combined)
        
        # Output
        logits = self.output(fused)
        
        return logits


class STLDecomposer:
    """STL Decomposition for each feature"""
    
    def __init__(self, period: int = 5):
        self.period = period
    
    def decompose_batch(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose batch of sequences
        x: (batch, seq_len, features)
        Returns: trend, seasonal, residual with same shape
        """
        batch_size, seq_len, num_features = x.shape
        
        trend = np.zeros_like(x)
        seasonal = np.zeros_like(x)
        residual = np.zeros_like(x)
        
        for b in range(batch_size):
            for f in range(num_features):
                try:
                    series = pd.Series(x[b, :, f])
                    # Only decompose if enough data and variance
                    if len(series) >= self.period * 2 and series.std() > 1e-6:
                        result = STL(series, period=self.period, robust=True).fit()
                        trend[b, :, f] = result.trend
                        seasonal[b, :, f] = result.seasonal
                        residual[b, :, f] = result.resid
                    else:
                        trend[b, :, f] = x[b, :, f]
                except:
                    trend[b, :, f] = x[b, :, f]
        
        return trend, seasonal, residual


class StockformerTrainer:
    """Trainer for Stockformer"""
    
    def __init__(self, config: Config, num_features: int = 60):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StockformerModel(config, num_features).to(self.device)
        self.decomposer = STLDecomposer(config.STL_PERIOD)
        self.name = "Stockformer"
    
    def train(self, train_dataset: SwingDataset, val_dataset: SwingDataset):
        """Train Stockformer"""
        
        print(f"\n{'='*60}")
        print("Training Stockformer Model")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print("Note: STL decomposition may take time...")
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.MAX_EPOCHS)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config.MAX_EPOCHS):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                features = batch['features'].numpy()
                labels = batch['labels'].to(self.device)
                
                # STL decomposition (skip for speed in training, use approximation)
                # In production, pre-compute decomposition
                features_tensor = torch.FloatTensor(features).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features_tensor)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = train_correct / train_total
            scheduler.step()
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = torch.FloatTensor(batch['features'].numpy()).to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(features)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = val_correct / val_total
            
            print(f"Epoch {epoch+1}/{self.config.MAX_EPOCHS} - "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(self.best_state)
        
        print(f"\nStockformer Results:")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        
        return {'train_acc': train_acc, 'val_acc': best_val_acc}
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with probabilities"""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def save(self, path: str):
        """Save model"""
        torch.save(self.model.state_dict(), f"{path}/stockformer_model.pt")
    
    def load(self, path: str):
        """Load model"""
        self.model.load_state_dict(torch.load(f"{path}/stockformer_model.pt"))

# ==============================================================================
# SECTION 10: ENSEMBLE
# ==============================================================================

class SwingAIEnsemble:
    """
    Ensemble of CatBoost + TFT + Stockformer
    Weights: 35% + 35% + 30%
    """
    
    def __init__(self, config: Config, num_features: int = 60):
        self.config = config
        self.num_features = num_features
        
        # Initialize models
        self.catboost = CatBoostModel(config)
        self.tft = TFTTrainer(config, num_features)
        self.stockformer = StockformerTrainer(config, num_features)
        
        self.models = {
            'catboost': self.catboost,
            'tft': self.tft,
            'stockformer': self.stockformer
        }
        
        self.weights = {
            'catboost': config.CATBOOST_WEIGHT,
            'tft': config.TFT_WEIGHT,
            'stockformer': config.STOCKFORMER_WEIGHT
        }
        
        self.trained = False
    
    def train(self, train_dataset: SwingDataset, val_dataset: SwingDataset):
        """Train all models"""
        
        print(f"\n{'='*60}")
        print("TRAINING SWINGAI ENSEMBLE")
        print(f"{'='*60}")
        print(f"Models: CatBoost ({self.weights['catboost']*100}%) + "
              f"TFT ({self.weights['tft']*100}%) + "
              f"Stockformer ({self.weights['stockformer']*100}%)")
        print(f"{'='*60}\n")
        
        results = {}
        
        # Train CatBoost
        results['catboost'] = self.catboost.train(train_dataset, val_dataset)
        
        # Train TFT
        results['tft'] = self.tft.train(train_dataset, val_dataset)
        
        # Train Stockformer
        results['stockformer'] = self.stockformer.train(train_dataset, val_dataset)
        
        self.trained = True
        
        # Summary
        print(f"\n{'='*60}")
        print("ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*60}")
        print("\nValidation Accuracies:")
        for name, res in results.items():
            print(f"  {name}: {res['val_acc']:.4f}")
        
        return results
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Ensemble prediction with weighted average
        
        Returns:
            Dict with:
            - direction: 0=DOWN, 1=SIDEWAYS, 2=UP
            - confidence: 0-100%
            - probabilities: [P(DOWN), P(SIDEWAYS), P(UP)]
            - individual_predictions: per-model predictions
            - agreement: number of models agreeing
        """
        
        if not self.trained:
            raise ValueError("Models not trained yet!")
        
        # Get predictions from each model
        cat_preds, cat_probs = self.catboost.predict(features)
        tft_preds, tft_probs = self.tft.predict(features)
        sf_preds, sf_probs = self.stockformer.predict(features)
        
        # Weighted average of probabilities
        ensemble_probs = (
            self.weights['catboost'] * cat_probs +
            self.weights['tft'] * tft_probs +
            self.weights['stockformer'] * sf_probs
        )
        
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        confidences = np.max(ensemble_probs, axis=1)
        
        # Check agreement
        all_preds = np.stack([cat_preds, tft_preds, sf_preds], axis=1)
        agreements = (all_preds == ensemble_preds[:, None]).sum(axis=1)
        
        # Agreement boost (if all 3 agree, boost confidence)
        agreement_boost = np.where(agreements == 3, 0.1, 0)
        confidences = np.minimum(confidences + agreement_boost, 1.0)
        
        results = {
            'direction': ensemble_preds,
            'confidence': confidences * 100,  # Convert to percentage
            'probabilities': ensemble_probs,
            'individual': {
                'catboost': {'pred': cat_preds, 'prob': cat_probs},
                'tft': {'pred': tft_preds, 'prob': tft_probs},
                'stockformer': {'pred': sf_preds, 'prob': sf_probs}
            },
            'agreement': agreements
        }
        
        return results
    
    def predict_single(self, features: np.ndarray) -> Dict:
        """Predict for single sample"""
        if features.ndim == 2:
            features = features[np.newaxis, :]
        
        result = self.predict(features)
        
        direction_map = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
        
        return {
            'direction': direction_map[result['direction'][0]],
            'confidence': result['confidence'][0],
            'probabilities': {
                'DOWN': result['probabilities'][0, 0],
                'SIDEWAYS': result['probabilities'][0, 1],
                'UP': result['probabilities'][0, 2]
            },
            'agreement': f"{result['agreement'][0]}/3 models agree",
            'signal': 'LONG' if result['direction'][0] == 2 else ('SHORT' if result['direction'][0] == 0 else 'NEUTRAL')
        }
    
    def save(self, path: str):
        """Save all models"""
        os.makedirs(path, exist_ok=True)
        
        self.catboost.save(path)
        self.tft.save(path)
        self.stockformer.save(path)
        
        # Save config
        config_dict = {
            'weights': self.weights,
            'num_features': self.num_features
        }
        with open(f"{path}/ensemble_config.json", 'w') as f:
            json.dump(config_dict, f)
        
        print(f"Models saved to {path}")
    
    def load(self, path: str):
        """Load all models"""
        self.catboost.load(path)
        self.tft.load(path)
        self.stockformer.load(path)
        
        with open(f"{path}/ensemble_config.json", 'r') as f:
            config_dict = json.load(f)
            self.weights = config_dict['weights']
        
        self.trained = True
        print(f"Models loaded from {path}")

# ==============================================================================
# SECTION 11: EVALUATION
# ==============================================================================

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_ensemble(ensemble: SwingAIEnsemble, test_dataset: SwingDataset) -> Dict:
    """Comprehensive evaluation of ensemble"""
    
    print(f"\n{'='*60}")
    print("EVALUATING ENSEMBLE ON TEST SET")
    print(f"{'='*60}")
    
    features = test_dataset.features.numpy()
    labels = test_dataset.labels.numpy()
    
    # Get predictions
    results = ensemble.predict(features)
    
    # Overall accuracy
    accuracy = (results['direction'] == labels).mean()
    
    # Per-class metrics
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        labels, 
        results['direction'],
        target_names=['DOWN', 'SIDEWAYS', 'UP']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(labels, results['direction'])
    print(f"\nConfusion Matrix:")
    print(f"           Predicted")
    print(f"           DOWN  SIDE  UP")
    print(f"Actual DOWN  {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    print(f"       SIDE  {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    print(f"       UP    {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    # Agreement analysis
    print(f"\nModel Agreement Analysis:")
    for i in [1, 2, 3]:
        mask = results['agreement'] == i
        if mask.sum() > 0:
            acc = (results['direction'][mask] == labels[mask]).mean()
            print(f"  {i}/3 models agree: {mask.sum()} samples, accuracy: {acc:.4f}")
    
    # Confidence analysis
    print(f"\nConfidence Analysis:")
    for threshold in [50, 60, 70, 80]:
        mask = results['confidence'] >= threshold
        if mask.sum() > 0:
            acc = (results['direction'][mask] == labels[mask]).mean()
            print(f"  Confidence >= {threshold}%: {mask.sum()} samples, accuracy: {acc:.4f}")
    
    # Per-model accuracy
    print(f"\nPer-Model Test Accuracy:")
    for name in ['catboost', 'tft', 'stockformer']:
        acc = (results['individual'][name]['pred'] == labels).mean()
        print(f"  {name}: {acc:.4f}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': results
    }

# ==============================================================================
# SECTION 12: MAIN TRAINING PIPELINE
# ==============================================================================

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("        SWINGAI TRAINING PIPELINE")
    print("        CatBoost + TFT + Stockformer")
    print("="*60)
    print(f"\nStart time: {datetime.now()}")
    
    # Initialize
    config = Config()
    collector = DataCollector(config)
    engineer = FeatureEngineer(config)
    
    # Download data
    print("\n" + "="*60)
    print("STEP 1: DOWNLOADING DATA")
    print("="*60)
    
    start_date = "2018-01-01"  # Extra data for feature calculation
    end_date = "2024-12-31"
    
    # Download stocks
    stock_data = collector.download_all_stocks(NIFTY_200[:100], start_date, end_date)  # First 100 for demo
    
    # Download market data
    print("\nDownloading market data...")
    market_data = collector.download_market_data(start_date, end_date)
    
    # Get FII/DII proxy
    fii_dii_data = None
    if 'NIFTY50' in market_data:
        fii_dii_data = collector.get_fii_dii_proxy(market_data['NIFTY50'])
    
    # Calculate features
    print("\n" + "="*60)
    print("STEP 2: CALCULATING FEATURES")
    print("="*60)
    
    feature_data = {}
    labels_data = {}
    
    for symbol, data in stock_data.items():
        try:
            features = engineer.calculate_all_features(data, market_data, fii_dii_data)
            labels = engineer.create_labels(data)
            
            if len(features) > config.LOOKBACK_DAYS + config.PREDICTION_HORIZON:
                feature_data[symbol] = features
                labels_data[symbol] = labels
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    print(f"Processed {len(feature_data)} stocks")
    print(f"Features: {len(engineer.feature_names)}")
    
    # Prepare sequences
    print("\n" + "="*60)
    print("STEP 3: PREPARING SEQUENCES")
    print("="*60)
    
    sequences, labels, symbols = prepare_sequences(
        stock_data, feature_data, labels_data, config
    )
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Label distribution: DOWN={sum(labels==0)}, SIDEWAYS={sum(labels==1)}, UP={sum(labels==2)}")
    
    # Create datasets
    datasets = create_data_splits(sequences, labels, symbols, config)
    
    # Train ensemble
    print("\n" + "="*60)
    print("STEP 4: TRAINING ENSEMBLE")
    print("="*60)
    
    ensemble = SwingAIEnsemble(config, num_features=60)
    training_results = ensemble.train(datasets['train'], datasets['val'])
    
    # Evaluate
    print("\n" + "="*60)
    print("STEP 5: EVALUATION")
    print("="*60)
    
    eval_results = evaluate_ensemble(ensemble, datasets['test'])
    
    # Save models
    print("\n" + "="*60)
    print("STEP 6: SAVING MODELS")
    print("="*60)
    
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    ensemble.save(config.MODEL_SAVE_PATH)
    
    # Save feature names
    with open(f"{config.MODEL_SAVE_PATH}/feature_names.json", 'w') as f:
        json.dump(engineer.feature_names, f)
    
    print(f"\nTraining complete!")
    print(f"End time: {datetime.now()}")
    print(f"\nModels saved to: {config.MODEL_SAVE_PATH}")
    print(f"\nTo use in production:")
    print(f"  ensemble = SwingAIEnsemble(config)")
    print(f"  ensemble.load('{config.MODEL_SAVE_PATH}')")
    print(f"  result = ensemble.predict_single(features)")
    
    return ensemble, eval_results


if __name__ == "__main__":
    ensemble, results = main()
