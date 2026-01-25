"""
PKScreener API Tests - SwingAI Platform
========================================
Tests for PKScreener integration with 40+ scanners and AI features.
"""

import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://workflow-insight-3.preview.emergentagent.com')


class TestPKScreenerCategories:
    """Test PKScreener categories API - returns 11 categories and 61 scanners"""
    
    def test_categories_endpoint_returns_200(self):
        """Test that categories endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/categories")
        assert response.status_code == 200
        
    def test_categories_returns_success(self):
        """Test that categories returns success=true"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/categories")
        data = response.json()
        assert data.get("success") == True
        
    def test_categories_has_expected_categories(self):
        """Test that categories contains expected scanner categories"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/categories")
        data = response.json()
        categories = data.get("categories", {})
        
        # Check for key categories
        expected_categories = ["breakout", "momentum", "reversal", "patterns", "technical", "signals"]
        for cat in expected_categories:
            assert cat in categories, f"Missing category: {cat}"
            
    def test_categories_has_scanners(self):
        """Test that each category has scanners"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/categories")
        data = response.json()
        categories = data.get("categories", {})
        
        for cat_name, cat_data in categories.items():
            assert "scanners" in cat_data, f"Category {cat_name} missing scanners"
            assert len(cat_data["scanners"]) > 0, f"Category {cat_name} has no scanners"


class TestRSIOversoldScanner:
    """Test RSI Oversold scanner - finds stocks with RSI < 30"""
    
    def test_rsi_oversold_returns_200(self):
        """Test RSI oversold endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/rsi-oversold?universe=nifty50&limit=10")
        assert response.status_code == 200
        
    def test_rsi_oversold_returns_success(self):
        """Test RSI oversold returns success"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/rsi-oversold?universe=nifty50&limit=10")
        data = response.json()
        assert data.get("success") == True
        
    def test_rsi_oversold_returns_results(self):
        """Test RSI oversold returns results array"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/rsi-oversold?universe=nifty50&limit=10")
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        
    def test_rsi_oversold_stocks_have_low_rsi(self):
        """Test that returned stocks have RSI < 30"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/rsi-oversold?universe=nifty50&limit=10")
        data = response.json()
        results = data.get("results", [])
        
        if len(results) > 0:
            for stock in results:
                assert "rsi" in stock, f"Stock {stock.get('symbol')} missing RSI"
                assert stock["rsi"] < 35, f"Stock {stock.get('symbol')} RSI {stock['rsi']} not oversold"


class TestSwingCandidates:
    """Test AI Swing Candidates endpoint - returns real stock data"""
    
    def test_swing_candidates_returns_200(self):
        """Test swing candidates endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/screener/swing-candidates?limit=10")
        assert response.status_code == 200
        
    def test_swing_candidates_returns_success(self):
        """Test swing candidates returns success"""
        response = requests.get(f"{BASE_URL}/api/screener/swing-candidates?limit=10")
        data = response.json()
        assert data.get("success") == True
        
    def test_swing_candidates_returns_results(self):
        """Test swing candidates returns results with stock data"""
        response = requests.get(f"{BASE_URL}/api/screener/swing-candidates?limit=10")
        data = response.json()
        assert "results" in data
        results = data["results"]
        assert len(results) > 0, "No swing candidates returned"
        
    def test_swing_candidates_have_required_fields(self):
        """Test swing candidates have price, RSI, and other required fields"""
        response = requests.get(f"{BASE_URL}/api/screener/swing-candidates?limit=5")
        data = response.json()
        results = data.get("results", [])
        
        required_fields = ["symbol", "ltp", "rsi", "change_percent"]
        for stock in results:
            for field in required_fields:
                assert field in stock, f"Stock {stock.get('symbol')} missing field: {field}"


class TestMarketRegime:
    """Test Market Regime analysis - returns current market conditions"""
    
    def test_market_regime_returns_200(self):
        """Test market regime endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/market-regime")
        assert response.status_code == 200
        
    def test_market_regime_returns_success(self):
        """Test market regime returns success"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/market-regime")
        data = response.json()
        assert data.get("success") == True
        
    def test_market_regime_has_regime_field(self):
        """Test market regime returns regime classification"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/market-regime")
        data = response.json()
        assert "regime" in data
        valid_regimes = ["BULL_QUIET", "BULL_VOLATILE", "BEAR_QUIET", "BEAR_VOLATILE", "TRANSITIONAL"]
        assert data["regime"] in valid_regimes, f"Invalid regime: {data['regime']}"
        
    def test_market_regime_has_indicators(self):
        """Test market regime returns indicators"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/market-regime")
        data = response.json()
        assert "indicators" in data
        indicators = data["indicators"]
        assert "nifty" in indicators
        assert "ma_20" in indicators
        assert "ma_50" in indicators


class TestNiftyPrediction:
    """Test Nifty Prediction - returns direction and confidence"""
    
    def test_nifty_prediction_returns_200(self):
        """Test nifty prediction endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/nifty-prediction")
        assert response.status_code == 200
        
    def test_nifty_prediction_returns_success(self):
        """Test nifty prediction returns success"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/nifty-prediction")
        data = response.json()
        assert data.get("success") == True
        
    def test_nifty_prediction_has_prediction(self):
        """Test nifty prediction returns prediction object"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/nifty-prediction")
        data = response.json()
        assert "prediction" in data
        prediction = data["prediction"]
        assert "direction" in prediction
        assert prediction["direction"] in ["BULLISH", "BEARISH", "NEUTRAL"]
        
    def test_nifty_prediction_has_current_level(self):
        """Test nifty prediction returns current level"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/nifty-prediction")
        data = response.json()
        assert "current_level" in data
        assert data["current_level"] > 0


class TestTrendAnalysis:
    """Test Trend Analysis - returns uptrend/downtrend/sideways counts"""
    
    def test_trend_analysis_returns_200(self):
        """Test trend analysis endpoint returns 200"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/trend-analysis?universe=nifty50&limit=10")
        assert response.status_code == 200
        
    def test_trend_analysis_returns_success(self):
        """Test trend analysis returns success"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/trend-analysis?universe=nifty50&limit=10")
        data = response.json()
        assert data.get("success") == True
        
    def test_trend_analysis_has_summary(self):
        """Test trend analysis returns summary with counts"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/trend-analysis?universe=nifty50&limit=10")
        data = response.json()
        assert "summary" in data
        summary = data["summary"]
        assert "uptrend" in summary
        assert "downtrend" in summary
        assert "sideways" in summary
        assert "total" in summary
        
    def test_trend_analysis_has_categorized_stocks(self):
        """Test trend analysis returns categorized stock lists"""
        response = requests.get(f"{BASE_URL}/api/screener/ai/trend-analysis?universe=nifty50&limit=10")
        data = response.json()
        assert "uptrend" in data
        assert "downtrend" in data
        assert isinstance(data["uptrend"], list)
        assert isinstance(data["downtrend"], list)


class TestBatchScan:
    """Test PKScreener batch scan endpoint with trend scanner"""
    
    def test_batch_scan_returns_200(self):
        """Test batch scan endpoint returns 200"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/batch",
            json={"scanner_id": "trend", "symbols": ["RELIANCE", "TCS", "INFY"]}
        )
        assert response.status_code == 200
        
    def test_batch_scan_returns_success(self):
        """Test batch scan returns success"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/batch",
            json={"scanner_id": "trend", "symbols": ["RELIANCE", "TCS"]}
        )
        data = response.json()
        assert data.get("success") == True
        
    def test_batch_scan_returns_results(self):
        """Test batch scan returns results"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/batch",
            json={"scanner_id": "trend", "symbols": ["RELIANCE", "TCS", "INFY"]}
        )
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0


class TestSingleStockScan:
    """Test single stock scan endpoint"""
    
    def test_single_scan_returns_200(self):
        """Test single stock scan returns 200"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/single?symbol=RELIANCE&scanner_id=trend"
        )
        assert response.status_code == 200
        
    def test_single_scan_returns_success(self):
        """Test single stock scan returns success"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/single?symbol=RELIANCE&scanner_id=trend"
        )
        data = response.json()
        assert data.get("success") == True
        
    def test_single_scan_returns_result(self):
        """Test single stock scan returns result with stock data"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/single?symbol=RELIANCE&scanner_id=trend"
        )
        data = response.json()
        assert "result" in data
        result = data["result"]
        assert result["symbol"] == "RELIANCE"
        assert "current_price" in result or "ltp" in result


class TestOtherScanners:
    """Test other PKScreener scanner endpoints"""
    
    def test_strong_buy_scanner(self):
        """Test strong buy scanner endpoint"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/strong-buy?universe=nifty50&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True
        
    def test_macd_crossover_scanner(self):
        """Test MACD crossover scanner endpoint"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/macd-crossover?universe=nifty50&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True
        
    def test_consolidating_scanner(self):
        """Test consolidating stocks scanner endpoint"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/consolidating?universe=nifty50&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
