# SwingAI - Product Requirements Document

## Original Problem Statement
Build a cutting-edge, institutional-grade AI swing trading platform for the Indian stock market (NSE). Features:
1. **Full-Featured Screener** - All 61 scanners from pkscreener library
2. **Paper Trading** - Virtual trading with ₹10 Lakh starting capital
3. **AI Intelligence** - ML-powered market analysis
4. **Watchlist** - Per-user stock tracking
5. **Stock Detail Page** - Full analysis with interactive charts
6. **Real-time Data** - WebSocket for live price updates
7. **Auto-updating Stock List** - 2229+ NSE stocks, refreshes from NSE

## Tech Stack
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Backend**: FastAPI (Python), WebSocket
- **Database**: Supabase (PostgreSQL)
- **Data**: yfinance, nsepython, pkscreener
- **Charts**: TradingView embedded widget (BSE format), Custom recharts AreaChart

---

## What's Been Implemented (January 2026)

### Stock Detail Page ✅ COMPLETED
- `/stock/[symbol]` route for any NSE stock (RELIANCE, TCS, INFY, etc.)
- Real-time price display with change % (green/red)
- OHLCV data (Open, High, Low, Volume)
- Technical indicators panel:
  - Trend (Strong Up/Weak Up/Down)
  - RSI (14) with Oversold/Overbought labels
  - Volume ratio vs 20-day average
- **TradingView Real-time Chart** (NEW - WORKING):
  - Full candlestick chart with volume bars
  - RSI indicator integrated
  - Uses BSE: symbol format (NSE format blocked in embeds)
  - All TradingView tools: timeframes, indicators, drawings
- **Price Overview Chart**:
  - Custom recharts AreaChart with yfinance historical data
  - Timeframe buttons: 1W, 1M, 3M, 1Y
  - Dynamic colors: Green for uptrend, Red for downtrend
- Key Levels section (52W High/Low, SMA 20, SMA 50)
- MACD Analysis section with crossover signals
- Watchlist toggle (Watch/Watching) - WORKING
- Links: Technical Analysis, Financials, Screener.in
- Paper Trade and Back to Screener navigation

### Watchlist Feature ✅ WORKING
- Per-user watchlist with real-time prices
- Add/Remove stocks from stock detail page
- Watchlist page with summary stats (Total, Gainers, Losers, Avg Price)
- Search and add stocks functionality
- Target price and notes support

### WebSocket Real-time Updates ✅
- `/ws/prices/{client_id}` endpoint
- Subscribe/unsubscribe to symbols
- Auto-broadcast every 5 seconds
- Price cache for efficiency
- Frontend auto-reconnect

### Auto-updating Stock Universe ✅
- `GET /api/screener/stocks/universe` - Shows 2229 NSE stocks
- `POST /api/screener/stocks/refresh` - Force refresh from NSE
- `GET /api/screener/stocks/search?query=X` - Search stocks
- Auto-refreshes every 24 hours
- New IPOs automatically added

### PKScreener (61 Scanners)
- 11 categories: Breakout, Momentum, Reversal, Patterns, etc.
- All scanners return stocks with analysis
- RSI, MACD, Trend, Volume metrics

### Watchlist Feature
- Per-user Supabase storage
- Add/remove via API and UI
- Live price updates

---

## API Endpoints

### Real-time
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws/prices/{client_id}` | WS | WebSocket for live prices |
| `/api/screener/prices/live?symbols=X,Y` | GET | Batch live prices |
| `/api/screener/prices/{symbol}` | GET | Single stock price |

### Stock Universe
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/screener/stocks/universe` | GET | All 2229 NSE stocks |
| `/api/screener/stocks/refresh` | POST | Force refresh from NSE |
| `/api/screener/stocks/search` | GET | Search by symbol |

### Screener (61 Scanners)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/screener/pk/categories` | GET | 11 categories |
| `/api/screener/pk/scan/batch` | POST | Run scanner |
| `/api/screener/swing-candidates` | GET | AI-ranked stocks |

---

## Frontend Pages

| Page | Route | Status |
|------|-------|--------|
| Screener | `/screener` | ✅ 61 scanners |
| Stock Detail | `/stock/[symbol]` | ✅ NEW |
| Watchlist | `/watchlist` | ✅ |
| Paper Trading | `/paper-trading` | ✅ |
| AI Intelligence | `/ai-intelligence` | ✅ |
| Dashboard | `/dashboard` | ✅ |

---

## Architecture
```
/app/backend/
├── server.py              # FastAPI + WebSocket
├── websocket_service.py   # Real-time price broadcasting
├── screener.py            # 61 PKScreener scanners
├── pkscreener_service.py  # PKScreener wrapper
├── watchlist.py           # Watchlist CRUD
├── paper_trading.py       # Paper trading
└── database.py            # Supabase

/app/frontend/app/
├── screener/              # AI Market Screener
├── stock/[symbol]/        # Stock Detail Page
├── watchlist/             # Watchlist
├── paper-trading/         # Paper Trading
└── dashboard/             # Dashboard
```

---

## Backlog

### P1 - High Priority
- [ ] **Complete Google Auth** - Connect real user to all features
- [ ] **Real TradingView embed** - Requires CSP configuration

### P2 - Medium Priority  
- [ ] Price alerts for watchlist
- [ ] PDF report generation

### P3 - Future
- [ ] SwingAI Bot (user's custom AI model)
- [ ] Broker API (Zerodha, Upstox)
- [ ] Backtesting module

---

## Test User
- **ID**: `ffb9e2ca-6733-4e84-9286-0aa134e6f57e`
- **Email**: `test@swingai.com`

## Last Updated
January 23, 2025
