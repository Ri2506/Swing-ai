# 📖 SwingAI API Documentation

Complete reference for all backend API endpoints.

**Base URL**: `https://api.swingai.com`

---

## 🔐 Authentication

All protected endpoints require JWT token in header:
```
Authorization: Bearer <jwt_token>
```

Get token from Supabase auth after signup/login.

---

## 📡 Endpoints

### Authentication

#### POST `/api/auth/signup`
Create new user account

**Request**:
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "full_name": "John Doe",
  "phone": "+91 9876543210"
}
```

**Response** (201):
```json
{
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "full_name": "John Doe"
  },
  "token": "eyJxxx..."
}
```

#### POST `/api/auth/login`
Login existing user

**Request**:
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

#### POST `/api/auth/logout`
Logout user (protected)

---

### User Profile

#### GET `/api/user/profile`
Get user profile (protected)

**Response**:
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "full_name": "John Doe",
  "phone": "+91 9876543210",
  "subscription_plan": "pro",
  "capital": 100000.00,
  "risk_profile": "moderate",
  "trading_mode": "semi_auto",
  "fo_enabled": true,
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### PUT `/api/user/profile`
Update user profile (protected)

**Request**:
```json
{
  "capital": 200000.00,
  "risk_profile": "aggressive",
  "trading_mode": "full_auto",
  "fo_enabled": true
}
```

---

### Subscription Plans

#### GET `/api/plans`
Get all subscription plans

**Response**:
```json
[
  {
    "id": "uuid",
    "name": "pro",
    "display_name": "Pro Plan",
    "price_monthly": 199900,
    "max_signals_per_day": 25,
    "max_positions": 10,
    "features": {
      "full_auto": true,
      "futures_trading": true,
      "options_trading": false
    }
  }
]
```

---

### Payments

#### POST `/api/payments/create-order`
Create Razorpay order (protected)

**Request**:
```json
{
  "plan_id": "uuid",
  "billing_cycle": "monthly"
}
```

**Response**:
```json
{
  "order_id": "order_xxx",
  "amount": 199900,
  "currency": "INR",
  "razorpay_key": "rzp_live_xxx"
}
```

#### POST `/api/payments/verify`
Verify payment signature (protected)

**Request**:
```json
{
  "order_id": "order_xxx",
  "payment_id": "pay_xxx",
  "signature": "xxx"
}
```

---

### Trading Signals

#### GET `/api/signals/today`
Get today's signals (protected)

**Query Params**:
- `limit` (optional): Max signals (default: 50)
- `segment` (optional): equity | futures | options

**Response**:
```json
[
  {
    "id": "uuid",
    "symbol": "TRENT",
    "segment": "equity",
    "action": "BUY",
    "entry_price": 3500.00,
    "target_price": 3700.00,
    "stop_loss": 3400.00,
    "confidence": 0.75,
    "risk_reward_ratio": 2.0,
    "position_size": 28,
    "timeframe": "swing",
    "status": "active",
    "generated_at": "2024-01-01T08:30:00Z",
    "expires_at": "2024-01-01T15:30:00Z"
  }
]
```

#### GET `/api/signals/{signal_id}`
Get specific signal details (protected)

---

### Trade Execution

#### POST `/api/trades/execute`
Execute a trading signal (protected)

**Request**:
```json
{
  "signal_id": "uuid",
  "quantity": 28,
  "order_type": "MARKET"
}
```

**Response**:
```json
{
  "trade_id": "uuid",
  "order_id": "241001000123456",
  "status": "COMPLETE",
  "filled_qty": 28,
  "avg_price": 3505.50,
  "brokerage": 35.50,
  "message": "Trade executed successfully"
}
```

#### GET `/api/trades/active`
Get all active trades (protected)

#### GET `/api/trades/history`
Get trade history (protected)

**Query Params**:
- `from_date`: YYYY-MM-DD
- `to_date`: YYYY-MM-DD
- `status`: all | open | closed
- `limit`: default 100

---

### Portfolio

#### GET `/api/portfolio/summary`
Get portfolio summary (protected)

**Response**:
```json
{
  "total_capital": 200000.00,
  "invested": 150000.00,
  "available": 50000.00,
  "current_value": 165000.00,
  "unrealized_pnl": 15000.00,
  "unrealized_pnl_percent": 10.0,
  "realized_pnl": 5000.00,
  "total_pnl": 20000.00,
  "win_rate": 0.65,
  "total_trades": 20,
  "winning_trades": 13,
  "losing_trades": 7
}
```

#### GET `/api/portfolio/positions`
Get all open positions (protected)

---

### Market Data

#### GET `/api/market/condition`
Get current market condition

**Response**:
```json
{
  "nifty": 21500.50,
  "nifty_change": 1.25,
  "vix": 12.5,
  "vix_status": "low",
  "fii_dii": {
    "fii": -500.00,
    "dii": 750.00
  },
  "market_status": "open",
  "risk_level": "normal",
  "trading_allowed": true
}
```

#### GET `/api/market/quote/{symbol}`
Get live quote for a symbol

---

### Broker Integration

#### POST `/api/broker/connect`
Connect broker account (protected)

**Request**:
```json
{
  "broker": "zerodha",
  "api_key": "xxx",
  "api_secret": "xxx",
  "request_token": "xxx"
}
```

#### GET `/api/broker/status`
Get broker connection status (protected)

---

### Notifications

#### GET `/api/notifications`
Get user notifications (protected)

**Query Params**:
- `unread_only`: boolean

#### PUT `/api/notifications/{id}/read`
Mark notification as read (protected)

---

### Watchlist

#### GET `/api/watchlist`
Get user watchlist (protected)

#### POST `/api/watchlist`
Add stock to watchlist (protected)

**Request**:
```json
{
  "symbol": "TRENT"
}
```

---

## 🔌 WebSocket API

### Connection

```javascript
const ws = new WebSocket('wss://api.swingai.com/ws/<jwt_token>')
```

### Subscribe to Symbol

```json
{
  "type": "subscribe_symbol",
  "symbol": "TRENT"
}
```

### Events Received

**New Signal**:
```json
{
  "type": "new_signal",
  "data": {
    "symbol": "TRENT",
    "action": "BUY",
    "entry_price": 3500.00
  }
}
```

**Price Update**:
```json
{
  "type": "price_update",
  "symbol": "TRENT",
  "price": 3510.00,
  "change": 0.29
}
```

**Stop Loss Hit**:
```json
{
  "type": "sl_hit",
  "trade_id": "uuid",
  "symbol": "TRENT",
  "exit_price": 3400.00,
  "pnl": -2800.00
}
```

---

## 📊 Rate Limits

- **Free**: 10 requests/minute
- **Starter**: 30 requests/minute
- **Pro**: 60 requests/minute
- **Elite**: 120 requests/minute

---

## ❌ Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid/missing token |
| 403 | Forbidden - Plan limit reached |
| 404 | Not Found |
| 429 | Too Many Requests - Rate limit |
| 500 | Internal Server Error |

**Error Response**:
```json
{
  "detail": "Error message here",
  "error_code": "INVALID_SIGNAL"
}
```

---

For full interactive API docs, visit:
- Swagger UI: `https://api.swingai.com/docs`
- ReDoc: `https://api.swingai.com/redoc`
