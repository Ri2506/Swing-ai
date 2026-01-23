"""
Test Stock Detail Page APIs
Tests for /api/screener/prices/{symbol} and /api/screener/prices/{symbol}/history endpoints
"""

import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://swingai-trade.preview.emergentagent.com')

class TestStockPriceAPI:
    """Test single stock price endpoint"""
    
    def test_get_reliance_price(self):
        """Test getting RELIANCE price data"""
        response = requests.get(f"{BASE_URL}/api/screener/prices/RELIANCE")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["symbol"] == "RELIANCE"
        assert "price" in data
        assert "change" in data
        assert "change_percent" in data
        assert "open" in data
        assert "high" in data
        assert "low" in data
        assert "volume" in data
        
        # Verify data types
        assert isinstance(data["price"], (int, float))
        assert isinstance(data["volume"], int)
        print(f"✅ RELIANCE price: ₹{data['price']}, change: {data['change_percent']}%")
    
    def test_get_tcs_price(self):
        """Test getting TCS price data"""
        response = requests.get(f"{BASE_URL}/api/screener/prices/TCS")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["symbol"] == "TCS"
        assert "price" in data
        print(f"✅ TCS price: ₹{data['price']}")
    
    def test_get_infy_price(self):
        """Test getting INFY price data"""
        response = requests.get(f"{BASE_URL}/api/screener/prices/INFY")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["symbol"] == "INFY"
        assert "price" in data
        print(f"✅ INFY price: ₹{data['price']}")
    
    def test_invalid_symbol_returns_error(self):
        """Test that invalid symbol returns appropriate error"""
        response = requests.get(f"{BASE_URL}/api/screener/prices/INVALID_SYMBOL_XYZ")
        # Should return 404, 500, or 520 for invalid symbol
        assert response.status_code in [404, 500, 520]


class TestStockHistoryAPI:
    """Test stock price history endpoint"""
    
    def test_reliance_history_1m(self):
        """Test RELIANCE 1 month history"""
        response = requests.get(f"{BASE_URL}/api/screener/prices/RELIANCE/history?period=1m")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["symbol"] == "RELIANCE"
        assert data["period"] == "1m"
        assert "history" in data
        assert len(data["history"]) > 0
        
        # Verify history data structure
        first_entry = data["history"][0]
        assert "date" in first_entry
        assert "open" in first_entry
        assert "high" in first_entry
        assert "low" in first_entry
        assert "close" in first_entry
        assert "volume" in first_entry
        
        print(f"✅ RELIANCE 1M history: {len(data['history'])} data points")
    
    def test_tcs_history_1w(self):
        """Test TCS 1 week history"""
        response = requests.get(f"{BASE_URL}/api/screener/prices/TCS/history?period=1w")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["symbol"] == "TCS"
        assert data["period"] == "1w"
        assert "history" in data
        assert len(data["history"]) > 0
        print(f"✅ TCS 1W history: {len(data['history'])} data points")
    
    def test_infy_history_3m(self):
        """Test INFY 3 month history"""
        response = requests.get(f"{BASE_URL}/api/screener/prices/INFY/history?period=3m")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["symbol"] == "INFY"
        assert data["period"] == "3m"
        assert "history" in data
        assert len(data["history"]) > 0
        print(f"✅ INFY 3M history: {len(data['history'])} data points")
    
    def test_reliance_history_1y(self):
        """Test RELIANCE 1 year history"""
        response = requests.get(f"{BASE_URL}/api/screener/prices/RELIANCE/history?period=1y")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["symbol"] == "RELIANCE"
        assert data["period"] == "1y"
        assert "history" in data
        assert len(data["history"]) > 0
        print(f"✅ RELIANCE 1Y history: {len(data['history'])} data points")


class TestTechnicalAnalysisAPI:
    """Test technical analysis endpoint"""
    
    def test_reliance_technical_scan(self):
        """Test RELIANCE technical analysis scan"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/single?symbol=RELIANCE&scanner_id=trend"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "result" in data
        
        result = data["result"]
        assert "symbol" in result
        assert "rsi" in result
        assert "trend" in result
        print(f"✅ RELIANCE technical: RSI={result.get('rsi')}, Trend={result.get('trend')}")
    
    def test_tcs_technical_scan(self):
        """Test TCS technical analysis scan"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/single?symbol=TCS&scanner_id=trend"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        print(f"✅ TCS technical scan successful")
    
    def test_infy_technical_scan(self):
        """Test INFY technical analysis scan"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/single?symbol=INFY&scanner_id=trend"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        print(f"✅ INFY technical scan successful")


class TestWatchlistAPI:
    """Test watchlist endpoints for stock detail page"""
    
    TEST_USER_ID = "ffb9e2ca-6733-4e84-9286-0aa134e6f57e"
    
    def test_check_watchlist_status(self):
        """Test checking if stock is in watchlist"""
        response = requests.get(
            f"{BASE_URL}/api/watchlist/{self.TEST_USER_ID}/check/RELIANCE"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "in_watchlist" in data
        print(f"✅ RELIANCE in watchlist: {data['in_watchlist']}")
    
    def test_add_to_watchlist(self):
        """Test adding stock to watchlist"""
        response = requests.post(
            f"{BASE_URL}/api/watchlist/add",
            json={"user_id": self.TEST_USER_ID, "symbol": "HDFCBANK"}
        )
        # Should succeed or return conflict if already exists
        assert response.status_code in [200, 201, 409]
        print(f"✅ Add to watchlist: status {response.status_code}")
    
    def test_remove_from_watchlist(self):
        """Test removing stock from watchlist"""
        response = requests.delete(
            f"{BASE_URL}/api/watchlist/{self.TEST_USER_ID}/HDFCBANK"
        )
        # Should succeed or return 404 if not found
        assert response.status_code in [200, 204, 404]
        print(f"✅ Remove from watchlist: status {response.status_code}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
