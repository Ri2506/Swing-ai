# SwingAI - Product Requirements Document

## Original Problem Statement
Build a cutting-edge, institutional-grade AI swing trading platform for the Indian stock market (NSE). The platform should be inspired by intellectia.ai with real backend data featuring:
1. **Full-Featured Screener** - All 61 scanners from pkscreener library
2. **Paper Trading** - Virtual trading with ₹10 Lakh starting capital
3. **AI Intelligence** - ML-powered market analysis and predictions
4. **Watchlist** - Per-user stock tracking with live prices
5. **TradingView Charts** - Professional charting integration
6. **SwingAI Bot** (Future) - Custom AI model integration for BUY/NO_TRADE signals

## Core Requirements
- Full-stack application with Next.js frontend + FastAPI backend
- Dark theme with professional gradient text effects
- Real-time market data from yfinance and nsepython
- Integration with pkscreener library for all AI/ML scanning features
- Paper trading with virtual capital and realistic brokerage simulations
- Supabase for database and Google Authentication

---

## What's Been Implemented (January 2025)

### PKScreener Full Integration ✅ COMPLETE
- **61 scanners** across **11 categories**:
  - Breakout (6), Momentum (6), Reversal (8), Chart Patterns (8)
  - Moving Average Signals (7), Technical (7), Buy/Sell Signals (6)
  - Consolidation (3), Trend (4), Machine Learning (3), Short Sell (3)

### Watchlist Feature ✅ COMPLETE
- Per-user watchlist stored in Supabase
- Add/remove stocks with real-time price updates
- Bulk add from scan results
- Stats: Gainers/Losers count, Average price
- Target price and stop loss tracking (schema pending)

### TradingView Charts ✅ COMPLETE
- Embedded TradingView widget in modal
- Opens from stock cards in screener
- Full-featured chart with indicators, drawing tools
- Dark theme integrated

### Backend APIs

#### Screener APIs (PKScreener Powered)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/screener/pk/categories` | GET | All 11 categories with 61 scanners |
| `/api/screener/pk/scanners` | GET | Flat list of all scanners |
| `/api/screener/pk/scan/batch` | POST | Run scanner on stock universe |
| `/api/screener/pk/scan/single` | POST | Run scanner on single stock |
| `/api/screener/pk/strong-buy` | GET | Strong buy signals |
| `/api/screener/pk/breakouts` | GET | Breakout detection |
| `/api/screener/pk/momentum` | GET | High momentum stocks |
| `/api/screener/pk/rsi-oversold` | GET | RSI oversold stocks |
| `/api/screener/pk/lorentzian` | GET | ML Lorentzian classifier |

#### Watchlist APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/watchlist/{user_id}` | GET | Get user's watchlist with live prices |
| `/api/watchlist/add` | POST | Add stock to watchlist |
| `/api/watchlist/{user_id}/{symbol}` | DELETE | Remove from watchlist |
| `/api/watchlist/bulk-add` | POST | Add multiple stocks |
| `/api/watchlist/{user_id}/check/{symbol}` | GET | Check if in watchlist |

#### AI Intelligence APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/screener/ai/nifty-prediction` | GET | AI Nifty direction prediction |
| `/api/screener/ai/market-regime` | GET | Market regime analysis |
| `/api/screener/ai/trend-analysis` | GET | Trend categorization |
| `/api/screener/swing-candidates` | GET | AI-ranked swing setups |

### Frontend Pages
- [x] **AI Market Screener** (`/screener`) - 61 scanners, TradingView, Watchlist
- [x] **Watchlist** (`/watchlist`) - Personal stock tracking
- [x] **AI Intelligence** (`/ai-intelligence`) - AI features display
- [x] **Paper Trading** (`/paper-trading`) - Virtual trading
- [x] **Dashboard** (`/dashboard`) - Market overview
- [x] **Login** (`/login`) - Google authentication

### Database (Supabase)
- [x] `users` - User accounts
- [x] `paper_orders` - Paper trade orders
- [x] `paper_holdings` - Paper portfolio holdings
- [x] `watchlist` - User watchlists
- [x] `user_sessions` - Session management

---

## Tech Stack
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS, Framer Motion
- **Backend**: FastAPI (Python), uvicorn
- **Database**: Supabase (PostgreSQL)
- **Data Sources**: yfinance, nsepython, pkscreener
- **Charts**: TradingView embedded widget
- **Auth**: Emergent-managed Google OAuth

## Architecture
```
/app
├── backend/
│   ├── server.py           # FastAPI entry point
│   ├── screener.py         # Screener APIs with PKScreener
│   ├── pkscreener_service.py # PKScreener wrapper (61 scanners)
│   ├── watchlist.py        # Watchlist CRUD API
│   ├── paper_trading.py    # Paper trading APIs
│   ├── auth.py             # Google OAuth APIs
│   ├── users.py            # User management
│   └── database.py         # Supabase connection
├── frontend/
│   ├── app/
│   │   ├── screener/       # AI Market Screener with TradingView
│   │   ├── watchlist/      # Personal watchlist page
│   │   ├── (platform)/
│   │   │   ├── ai-intelligence/
│   │   │   ├── paper-trading/
│   │   │   └── layout.tsx
│   │   └── login/
│   └── .env
└── memory/
    └── PRD.md
```

---

## Prioritized Backlog

### ✅ P0 - Critical (DONE)
- [x] PKScreener full integration (61 scanners, 11 categories)
- [x] Real market data from yfinance
- [x] All NSE stocks via nsepython (2229 stocks)
- [x] Paper trading MVP
- [x] Watchlist feature
- [x] TradingView charts integration

### P1 - High Priority
- [ ] **Complete Google Auth Flow** - Test end-to-end login
- [ ] **User-specific data** - Connect watchlist/paper trading to logged-in user
- [ ] **Stock Detail Page** - Dedicated page with full TradingView and analysis

### P2 - Medium Priority
- [ ] Email notifications for watchlist alerts
- [ ] Real-time WebSocket updates
- [ ] PDF report generation

### P3 - Future Tasks
- [ ] **SwingAI Bot Integration** - User's custom AI model
- [ ] Broker API integration (Zerodha, Upstox)
- [ ] Backtesting module

---

## Test Reports
- `/app/test_reports/iteration_1.json` - PKScreener integration tests (33/33 passed)
- `/app/test_reports/iteration_2.json` - Watchlist & Screener tests (20/21 passed)
- `/app/backend/tests/test_pkscreener_api.py` - Backend API tests
- `/app/backend/tests/test_watchlist_screener.py` - Watchlist tests

## Test User Credentials
- **User ID**: `ffb9e2ca-6733-4e84-9286-0aa134e6f57e`
- **Email**: `test@swingai.com`

## Last Updated
January 23, 2025
