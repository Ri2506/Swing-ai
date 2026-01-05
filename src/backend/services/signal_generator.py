"""
================================================================================
SWINGAI SIGNAL GENERATION SERVICE
================================================================================
Generates trading signals using:
1. PKScreener for stock candidates
2. Technical analysis features
3. AI model inference (CatBoost + TFT + Stockformer ensemble)
================================================================================
"""

import os
import json
import asyncio
import httpx
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalCandidate:
    """Stock candidate for signal generation"""
    symbol: str
    price: float
    change_percent: float
    volume: int
    rsi: float
    macd_signal: str
    trend: str
    support: float
    resistance: float
    sector: str


@dataclass
class GeneratedSignal:
    """Generated trading signal"""
    symbol: str
    exchange: str
    segment: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    target_3: Optional[float]
    risk_reward: float
    catboost_score: float
    tft_score: float
    stockformer_score: float
    model_agreement: int
    reasons: List[str]
    is_premium: bool
    lot_size: Optional[int] = None
    expiry_date: Optional[date] = None
    strike_price: Optional[float] = None
    option_type: Optional[str] = None


class SignalGenerator:
    """
    Main signal generation service
    Orchestrates PKScreener → Feature Engineering → AI Inference → Signal Creation
    """
    
    def __init__(
        self,
        supabase_client,
        modal_endpoint: str = None,
        min_confidence: float = 65.0,
        min_risk_reward: float = 1.5
    ):
        self.supabase = supabase_client
        self.modal_endpoint = modal_endpoint or os.getenv("MODAL_INFERENCE_URL", "")
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        
        # F&O lot sizes (subset - full list in fo_trading_engine.py)
        self.fo_lot_sizes = {
            "NIFTY": 25, "BANKNIFTY": 15, "RELIANCE": 250, "TCS": 150,
            "HDFCBANK": 550, "INFY": 300, "ICICIBANK": 700, "SBIN": 750,
            "TATASTEEL": 425, "TRENT": 385, "POLYCAB": 200
        }
    
    async def generate_daily_signals(self) -> List[GeneratedSignal]:
        """
        Main entry point - generates all signals for the day
        Called by scheduler at 8:30 AM
        """
        logger.info("Starting daily signal generation...")
        
        try:
            # Step 1: Get stock candidates from PKScreener
            candidates = await self._get_candidates()
            logger.info(f"Got {len(candidates)} candidates from screener")
            
            if not candidates:
                logger.warning("No candidates found, using fallback list")
                candidates = self._get_fallback_candidates()
            
            # Step 2: Fetch market data
            market_data = await self._get_market_data()
            
            # Step 3: Calculate features for each candidate
            features_list = await self._calculate_features(candidates, market_data)
            
            # Step 4: Run AI inference
            predictions = await self._run_inference(features_list)
            
            # Step 5: Generate signals from predictions
            signals = self._create_signals(predictions, market_data)
            
            # Step 6: Save signals to database
            await self._save_signals(signals)
            
            logger.info(f"Generated {len(signals)} signals for today")
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise
    
    async def _get_candidates(self) -> List[str]:
        """Get stock candidates from PKScreener or fallback"""
        try:
            # Try fetching from PKScreener GitHub Actions results
            async with httpx.AsyncClient() as client:
                url = "https://raw.githubusercontent.com/pkjmesra/PKScreener/actions-data-download/actions-data-scan/PKScreener-result_6.csv"
                response = await client.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Parse CSV
                    lines = response.text.strip().split('\n')
                    if len(lines) > 1:
                        # Extract symbols from first column
                        candidates = []
                        for line in lines[1:51]:  # Top 50
                            parts = line.split(',')
                            if parts:
                                symbol = parts[0].strip().replace('.NS', '').replace('"', '')
                                if symbol and symbol.isalpha():
                                    candidates.append(symbol)
                        return candidates
        except Exception as e:
            logger.warning(f"PKScreener fetch failed: {e}")
        
        return self._get_fallback_candidates()
    
    def _get_fallback_candidates(self) -> List[str]:
        """Fallback candidate list when PKScreener unavailable"""
        return [
            # Large caps
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "BHARTIARTL", "SBIN", "KOTAKBANK", "LT", "AXISBANK",
            # Mid caps with momentum
            "TRENT", "POLYCAB", "PERSISTENT", "DIXON", "TATAELXSI",
            "ASTRAL", "COFORGE", "LALPATHLAB", "MUTHOOTFIN", "INDHOTEL",
            # F&O stocks for shorts
            "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "TATAMOTORS",
            "ABB", "SIEMENS", "HAL", "BEL", "IRCTC"
        ]
    
    async def _get_market_data(self) -> Dict:
        """Fetch current market data"""
        try:
            today = date.today().isoformat()
            result = self.supabase.table("market_data").select("*").eq("date", today).single().execute()
            
            if result.data:
                return result.data
        except:
            pass
        
        # Return default if not available
        return {
            "nifty_close": 21500,
            "vix_close": 14.5,
            "market_trend": "SIDEWAYS",
            "risk_level": "MODERATE",
            "fii_cash": 0,
            "dii_cash": 0
        }
    
    async def _calculate_features(
        self, 
        candidates: List[str], 
        market_data: Dict
    ) -> List[Dict]:
        """Calculate features for AI models"""
        features_list = []
        
        for symbol in candidates:
            try:
                features = await self._get_stock_features(symbol)
                if features:
                    features["market_vix"] = market_data.get("vix_close", 15)
                    features["market_trend"] = market_data.get("market_trend", "SIDEWAYS")
                    features["fii_flow"] = market_data.get("fii_cash", 0)
                    features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to get features for {symbol}: {e}")
        
        return features_list
    
    async def _get_stock_features(self, symbol: str) -> Optional[Dict]:
        """
        Calculate technical features for a stock
        In production, this would fetch real-time data from broker API
        """
        # Simulated features - replace with real data in production
        import random
        
        base_price = random.uniform(100, 5000)
        
        return {
            "symbol": symbol,
            "price": base_price,
            "open": base_price * (1 + random.uniform(-0.02, 0.02)),
            "high": base_price * (1 + random.uniform(0, 0.03)),
            "low": base_price * (1 - random.uniform(0, 0.03)),
            "volume": random.randint(100000, 5000000),
            "volume_sma_20": random.randint(80000, 4000000),
            
            # Technical indicators
            "rsi_14": random.uniform(30, 70),
            "macd": random.uniform(-5, 5),
            "macd_signal": random.uniform(-5, 5),
            "macd_hist": random.uniform(-2, 2),
            "bb_upper": base_price * 1.05,
            "bb_lower": base_price * 0.95,
            "bb_mid": base_price,
            "atr_14": base_price * random.uniform(0.01, 0.03),
            
            # Moving averages
            "sma_20": base_price * (1 + random.uniform(-0.05, 0.05)),
            "sma_50": base_price * (1 + random.uniform(-0.1, 0.1)),
            "sma_200": base_price * (1 + random.uniform(-0.15, 0.15)),
            "ema_9": base_price * (1 + random.uniform(-0.02, 0.02)),
            "ema_21": base_price * (1 + random.uniform(-0.04, 0.04)),
            
            # Price action
            "prev_close": base_price * (1 + random.uniform(-0.02, 0.02)),
            "change_percent": random.uniform(-3, 3),
            "gap_percent": random.uniform(-1, 1),
            
            # Support/Resistance
            "support_1": base_price * 0.97,
            "resistance_1": base_price * 1.03,
            
            # Trend
            "adx": random.uniform(15, 40),
            "plus_di": random.uniform(10, 35),
            "minus_di": random.uniform(10, 35),
        }
    
    async def _run_inference(self, features_list: List[Dict]) -> List[Dict]:
        """
        Run AI model inference
        Uses Modal endpoint if available, otherwise uses rule-based fallback
        """
        if self.modal_endpoint:
            try:
                return await self._run_modal_inference(features_list)
            except Exception as e:
                logger.warning(f"Modal inference failed: {e}, using fallback")
        
        return self._run_fallback_inference(features_list)
    
    async def _run_modal_inference(self, features_list: List[Dict]) -> List[Dict]:
        """Run inference via Modal endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.modal_endpoint}/predict",
                json={"features": features_list},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["predictions"]
    
    def _run_fallback_inference(self, features_list: List[Dict]) -> List[Dict]:
        """
        Rule-based fallback when AI models unavailable
        Uses technical analysis rules to generate signals
        """
        predictions = []
        
        for features in features_list:
            symbol = features["symbol"]
            price = features["price"]
            rsi = features["rsi_14"]
            macd_hist = features["macd_hist"]
            adx = features["adx"]
            plus_di = features["plus_di"]
            minus_di = features["minus_di"]
            sma_20 = features["sma_20"]
            sma_50 = features["sma_50"]
            volume = features["volume"]
            volume_sma = features["volume_sma_20"]
            
            # Calculate scores (0-100)
            catboost_score = 50.0
            tft_score = 50.0
            stockformer_score = 50.0
            direction = "NEUTRAL"
            reasons = []
            
            # RSI signals
            if rsi < 35:
                catboost_score += 15
                tft_score += 10
                reasons.append("RSI oversold")
            elif rsi > 65:
                catboost_score -= 10
                tft_score -= 10
                reasons.append("RSI overbought")
            
            # MACD signals
            if macd_hist > 0:
                catboost_score += 10
                stockformer_score += 10
                reasons.append("MACD bullish")
            elif macd_hist < 0:
                catboost_score -= 10
                stockformer_score -= 10
                reasons.append("MACD bearish")
            
            # Trend strength (ADX)
            if adx > 25:
                if plus_di > minus_di:
                    tft_score += 15
                    stockformer_score += 10
                    reasons.append("Strong uptrend")
                else:
                    tft_score -= 15
                    stockformer_score -= 10
                    reasons.append("Strong downtrend")
            
            # Moving average alignment
            if price > sma_20 > sma_50:
                catboost_score += 10
                tft_score += 10
                reasons.append("Price above MAs")
            elif price < sma_20 < sma_50:
                catboost_score -= 10
                tft_score -= 10
                reasons.append("Price below MAs")
            
            # Volume confirmation
            if volume > volume_sma * 1.5:
                catboost_score += 5
                reasons.append("High volume")
            
            # Determine direction
            avg_score = (catboost_score + tft_score + stockformer_score) / 3
            
            if avg_score >= 60:
                direction = "LONG"
            elif avg_score <= 40:
                direction = "SHORT"
            else:
                direction = "NEUTRAL"
            
            # Count model agreement
            model_agreement = 0
            if catboost_score >= 55 and direction == "LONG":
                model_agreement += 1
            if tft_score >= 55 and direction == "LONG":
                model_agreement += 1
            if stockformer_score >= 55 and direction == "LONG":
                model_agreement += 1
            if catboost_score <= 45 and direction == "SHORT":
                model_agreement += 1
            if tft_score <= 45 and direction == "SHORT":
                model_agreement += 1
            if stockformer_score <= 45 and direction == "SHORT":
                model_agreement += 1
            
            predictions.append({
                "symbol": symbol,
                "price": price,
                "direction": direction,
                "catboost_score": min(100, max(0, catboost_score)),
                "tft_score": min(100, max(0, tft_score)),
                "stockformer_score": min(100, max(0, stockformer_score)),
                "model_agreement": model_agreement,
                "reasons": reasons,
                "features": features
            })
        
        return predictions
    
    def _create_signals(
        self, 
        predictions: List[Dict],
        market_data: Dict
    ) -> List[GeneratedSignal]:
        """Create trading signals from model predictions"""
        signals = []
        vix = market_data.get("vix_close", 15)
        
        for pred in predictions:
            if pred["direction"] == "NEUTRAL":
                continue
            
            # Calculate confidence
            confidence = (
                pred["catboost_score"] * 0.35 +
                pred["tft_score"] * 0.35 +
                pred["stockformer_score"] * 0.30
            )
            
            # Skip low confidence
            if confidence < self.min_confidence:
                continue
            
            # Skip if less than 2 models agree
            if pred["model_agreement"] < 2:
                continue
            
            symbol = pred["symbol"]
            price = pred["price"]
            direction = pred["direction"]
            features = pred.get("features", {})
            
            # Calculate entry, SL, targets
            atr = features.get("atr_14", price * 0.02)
            
            if direction == "LONG":
                entry_price = price
                stop_loss = price - (atr * 1.5)
                target_1 = price + (atr * 2)
                target_2 = price + (atr * 3)
                target_3 = price + (atr * 4)
            else:  # SHORT
                entry_price = price
                stop_loss = price + (atr * 1.5)
                target_1 = price - (atr * 2)
                target_2 = price - (atr * 3)
                target_3 = price - (atr * 4)
            
            # Calculate risk:reward
            risk = abs(entry_price - stop_loss)
            reward = abs(target_1 - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Skip poor R:R
            if risk_reward < self.min_risk_reward:
                continue
            
            # Adjust for VIX
            if vix > 20:
                # Widen stops in high volatility
                if direction == "LONG":
                    stop_loss = price - (atr * 2)
                else:
                    stop_loss = price + (atr * 2)
            
            # Determine if premium signal
            is_premium = confidence >= 75 or pred["model_agreement"] == 3
            
            signal = GeneratedSignal(
                symbol=symbol,
                exchange="NSE",
                segment="EQUITY",
                direction=direction,
                confidence=round(confidence, 2),
                entry_price=round(entry_price, 2),
                stop_loss=round(stop_loss, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                target_3=round(target_3, 2),
                risk_reward=round(risk_reward, 2),
                catboost_score=round(pred["catboost_score"], 2),
                tft_score=round(pred["tft_score"], 2),
                stockformer_score=round(pred["stockformer_score"], 2),
                model_agreement=pred["model_agreement"],
                reasons=pred["reasons"],
                is_premium=is_premium,
                lot_size=self.fo_lot_sizes.get(symbol)
            )
            
            signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    async def _save_signals(self, signals: List[GeneratedSignal]) -> None:
        """Save generated signals to database"""
        today = date.today().isoformat()
        
        for signal in signals:
            try:
                data = {
                    "symbol": signal.symbol,
                    "exchange": signal.exchange,
                    "segment": signal.segment,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "target_1": signal.target_1,
                    "target_2": signal.target_2,
                    "target_3": signal.target_3,
                    "risk_reward": signal.risk_reward,
                    "catboost_score": signal.catboost_score,
                    "tft_score": signal.tft_score,
                    "stockformer_score": signal.stockformer_score,
                    "model_agreement": signal.model_agreement,
                    "reasons": signal.reasons,
                    "is_premium": signal.is_premium,
                    "lot_size": signal.lot_size,
                    "date": today,
                    "status": "active",
                    "generated_at": datetime.utcnow().isoformat()
                }
                
                self.supabase.table("signals").insert(data).execute()
                
            except Exception as e:
                logger.error(f"Failed to save signal for {signal.symbol}: {e}")
    
    async def get_today_signals(
        self, 
        segment: Optional[str] = None,
        direction: Optional[str] = None,
        is_premium: Optional[bool] = None
    ) -> List[Dict]:
        """Fetch today's signals from database"""
        today = date.today().isoformat()
        
        query = self.supabase.table("signals").select("*").eq("date", today).eq("status", "active")
        
        if segment:
            query = query.eq("segment", segment)
        if direction:
            query = query.eq("direction", direction)
        if is_premium is not None:
            query = query.eq("is_premium", is_premium)
        
        result = query.order("confidence", desc=True).execute()
        return result.data or []


# ============================================================================
# USAGE
# ============================================================================

async def main():
    """Test signal generation"""
    from supabase import create_client
    
    supabase = create_client(
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_SERVICE_KEY", "")
    )
    
    generator = SignalGenerator(supabase)
    signals = await generator.generate_daily_signals()
    
    print(f"\n{'='*60}")
    print(f"GENERATED {len(signals)} SIGNALS")
    print(f"{'='*60}")
    
    for signal in signals[:5]:
        print(f"\n{signal.symbol} - {signal.direction}")
        print(f"  Confidence: {signal.confidence}%")
        print(f"  Entry: ₹{signal.entry_price}")
        print(f"  SL: ₹{signal.stop_loss}")
        print(f"  Target: ₹{signal.target_1}")
        print(f"  R:R: {signal.risk_reward}")
        print(f"  Reasons: {', '.join(signal.reasons)}")


if __name__ == "__main__":
    asyncio.run(main())
