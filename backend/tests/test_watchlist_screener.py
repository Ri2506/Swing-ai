"""
SwingAI Backend Tests - Watchlist & Screener APIs
Tests for:
- Watchlist CRUD operations (add, get, remove)
- Screener categories and scanners (61 scanners, 11 categories)
- AI Swing Candidates endpoint
"""

import pytest
import requests
import os
import uuid

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

# Test user ID (valid UUID format for Supabase)
TEST_USER_ID = "ffb9e2ca-6733-4e84-9286-0aa134e6f57e"
TEST_SYMBOL = f"TEST_{uuid.uuid4().hex[:6].upper()}"  # Unique test symbol


class TestWatchlistAPI:
    """Watchlist CRUD API tests"""
    
    def test_get_watchlist_success(self):
        """Test getting user's watchlist"""
        response = requests.get(f"{BASE_URL}/api/watchlist/{TEST_USER_ID}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["user_id"] == TEST_USER_ID
        assert "watchlist" in data
        assert "count" in data
        assert isinstance(data["watchlist"], list)
    
    def test_get_watchlist_with_live_prices(self):
        """Test that watchlist items include live price data"""
        response = requests.get(f"{BASE_URL}/api/watchlist/{TEST_USER_ID}")
        assert response.status_code == 200
        
        data = response.json()
        if data["count"] > 0:
            item = data["watchlist"][0]
            # Verify price data fields exist
            assert "symbol" in item
            assert "current_price" in item
            assert "change_percent" in item
            assert "name" in item
    
    def test_add_stock_to_watchlist(self):
        """Test adding a stock to watchlist"""
        payload = {
            "user_id": TEST_USER_ID,
            "symbol": "HDFCBANK"
        }
        response = requests.post(
            f"{BASE_URL}/api/watchlist/add",
            json=payload
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True or data.get("already_exists") == True
        
        if data["success"] and not data.get("already_exists"):
            assert "item" in data
            assert data["item"]["symbol"] == "HDFCBANK"
    
    def test_add_duplicate_stock_returns_already_exists(self):
        """Test that adding duplicate stock returns already_exists flag"""
        # First add
        payload = {"user_id": TEST_USER_ID, "symbol": "ICICIBANK"}
        requests.post(f"{BASE_URL}/api/watchlist/add", json=payload)
        
        # Second add (duplicate)
        response = requests.post(f"{BASE_URL}/api/watchlist/add", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        # Should indicate already exists
        assert data.get("already_exists") == True or "already in your watchlist" in data.get("message", "")
    
    def test_add_invalid_stock_returns_404(self):
        """Test adding non-existent stock returns 404"""
        payload = {
            "user_id": TEST_USER_ID,
            "symbol": "INVALIDXYZ123"
        }
        response = requests.post(f"{BASE_URL}/api/watchlist/add", json=payload)
        # Should return 404 for invalid stock
        assert response.status_code == 404
    
    def test_remove_stock_from_watchlist(self):
        """Test removing a stock from watchlist"""
        # First add a stock
        add_payload = {"user_id": TEST_USER_ID, "symbol": "WIPRO"}
        requests.post(f"{BASE_URL}/api/watchlist/add", json=add_payload)
        
        # Then remove it
        response = requests.delete(
            f"{BASE_URL}/api/watchlist/{TEST_USER_ID}/WIPRO"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "removed" in data["message"].lower()
    
    def test_check_stock_in_watchlist(self):
        """Test checking if stock is in watchlist"""
        # Add a stock first
        requests.post(
            f"{BASE_URL}/api/watchlist/add",
            json={"user_id": TEST_USER_ID, "symbol": "TCS"}
        )
        
        response = requests.get(
            f"{BASE_URL}/api/watchlist/{TEST_USER_ID}/check/TCS"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "in_watchlist" in data
        assert data["symbol"] == "TCS"
    
    def test_invalid_user_id_format_returns_500(self):
        """Test that invalid UUID format returns 500"""
        response = requests.get(f"{BASE_URL}/api/watchlist/invalid-user-id")
        # Supabase expects UUID format
        assert response.status_code == 500


class TestScreenerCategoriesAPI:
    """Screener categories and scanners API tests"""
    
    def test_get_categories_returns_11_categories(self):
        """Test that API returns exactly 11 scanner categories"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/categories")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert len(data["categories"]) == 11
    
    def test_get_categories_returns_61_scanners(self):
        """Test that API returns exactly 61 total scanners"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/categories")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_scanners"] == 61
    
    def test_categories_have_expected_names(self):
        """Test that expected category names exist"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/categories")
        assert response.status_code == 200
        
        data = response.json()
        expected_categories = [
            "breakout", "momentum", "reversal", "patterns", 
            "ma_signals", "technical", "signals", "consolidation",
            "trend", "ml", "short_sell"
        ]
        
        for cat in expected_categories:
            assert cat in data["categories"], f"Missing category: {cat}"
    
    def test_each_category_has_scanners(self):
        """Test that each category has at least one scanner"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/categories")
        assert response.status_code == 200
        
        data = response.json()
        for cat_name, cat_data in data["categories"].items():
            assert "scanners" in cat_data
            assert len(cat_data["scanners"]) > 0, f"Category {cat_name} has no scanners"
    
    def test_get_all_scanners_flat_list(self):
        """Test getting flat list of all scanners"""
        response = requests.get(f"{BASE_URL}/api/screener/pk/scanners")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["count"] == 61
        assert len(data["scanners"]) == 61


class TestAISwingCandidatesAPI:
    """AI Swing Candidates endpoint tests"""
    
    def test_swing_candidates_returns_success(self):
        """Test AI Swing Candidates endpoint returns success"""
        response = requests.get(f"{BASE_URL}/api/screener/swing-candidates?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
    
    def test_swing_candidates_returns_stock_data(self):
        """Test that swing candidates include required stock data"""
        response = requests.get(f"{BASE_URL}/api/screener/swing-candidates?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        if data["count"] > 0:
            stock = data["results"][0]
            # Verify required fields
            assert "symbol" in stock
            assert "current_price" in stock or "price" in stock
            assert "rsi" in stock
            assert "ai_score" in stock
            assert "signal_reason" in stock
    
    def test_swing_candidates_has_price_and_rsi(self):
        """Test that results include price and RSI values"""
        response = requests.get(f"{BASE_URL}/api/screener/swing-candidates?limit=5")
        assert response.status_code == 200
        
        data = response.json()
        for stock in data.get("results", []):
            price = stock.get("current_price") or stock.get("price") or stock.get("ltp")
            assert price is not None and price > 0, f"Invalid price for {stock.get('symbol')}"
            assert "rsi" in stock and stock["rsi"] > 0, f"Invalid RSI for {stock.get('symbol')}"


class TestBatchScanAPI:
    """Batch scan endpoint tests"""
    
    def test_batch_scan_trend_scanner(self):
        """Test batch scan with trend scanner"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/batch?scanner_id=trend&universe=nifty50&limit=10"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "results" in data
    
    def test_batch_scan_breakout_scanner(self):
        """Test batch scan with breakout scanner"""
        response = requests.post(
            f"{BASE_URL}/api/screener/pk/scan/batch?scanner_id=probable_breakout&universe=nifty50&limit=10"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True


class TestBasicScannerAPI:
    """Basic scanner endpoint tests"""
    
    def test_scanner_0_ai_swing(self):
        """Test scanner 0 - AI Swing Candidates"""
        response = requests.get(f"{BASE_URL}/api/screener/scan/0?universe=nifty50&limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert data["scanner_name"] == "AI Swing Candidates"
    
    def test_scanner_2_top_gainers(self):
        """Test scanner 2 - Top Gainers"""
        response = requests.get(f"{BASE_URL}/api/screener/scan/2?universe=nifty50&limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "Top Gainers" in data["scanner_name"]
    
    def test_scanner_5_52w_high(self):
        """Test scanner 5 - 52-Week High"""
        response = requests.get(f"{BASE_URL}/api/screener/scan/5?universe=nifty50&limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "52-Week High" in data["scanner_name"]


# Cleanup fixture
@pytest.fixture(scope="module", autouse=True)
def cleanup_test_data():
    """Cleanup test data after all tests"""
    yield
    # Remove test stocks added during testing
    test_symbols = ["HDFCBANK", "ICICIBANK", "WIPRO"]
    for symbol in test_symbols:
        try:
            requests.delete(f"{BASE_URL}/api/watchlist/{TEST_USER_ID}/{symbol}")
        except:
            pass
