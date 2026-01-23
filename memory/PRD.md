# SwingAI - Product Requirements Document

## Original Problem Statement
Build a cutting-edge, institutional-grade AI swing trading platform for the Indian stock market (NSE). The platform should be inspired by intellectia.ai with real backend data featuring:
1. **Full-Featured Screener** - All 40+ scanners from pkscreener library
2. **Paper Trading** - Virtual trading with ₹10 Lakh starting capital
3. **AI Intelligence** - ML-powered market analysis and predictions
4. **SwingAI Bot** (Future) - Custom AI model integration for BUY/NO_TRADE signals

## Core Requirements
- Full-stack application with Next.js frontend + FastAPI backend
- Dark theme with professional gradient text effects
- Real-time market data from yfinance and nsepython
- Integration with pkscreener library for all AI/ML scanning features
- Paper trading with virtual capital and realistic brokerage simulations
- Supabase for database and Google Authentication

## User Personas
1. **Retail Swing Traders** - Part-time traders looking for AI-powered signals
2. **Professional Traders** - Full-time traders seeking institutional-grade tools
3. **Investment Analysts** - Professionals validating AI recommendations

---

## What's Been Implemented (January 2025)

### PKScreener Full Integration ✅ COMPLETE
- **61 scanners** across **11 categories**:
  - Breakout (6 scanners)
  - Momentum (6 scanners)
  - Reversal (8 scanners)
  - Chart Patterns (8 scanners)
  - Moving Average Signals (7 scanners)
  - Technical (7 scanners)
  - Buy/Sell Signals (6 scanners)
  - Consolidation (3 scanners)
  - Trend (4 scanners)
  - Machine Learning (3 scanners)
  - Short Sell (3 scanners)

### Backend APIs (All Working with Real Data)

#### Screener APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/screener/pk/categories` | GET | All 11 scanner categories with 61 scanners |
| `/api/screener/pk/scanners` | GET | Flat list of all scanners |
| `/api/screener/pk/scan/batch` | POST | Run scanner on stock universe |
| `/api/screener/pk/scan/single` | POST | Run scanner on single stock |
| `/api/screener/pk/strong-buy` | GET | Strong buy signals |
| `/api/screener/pk/strong-sell` | GET | Strong sell signals |
| `/api/screener/pk/breakouts` | GET | Breakout detection |
| `/api/screener/pk/reversals` | GET | Reversal candidates |
| `/api/screener/pk/momentum` | GET | High momentum stocks |
| `/api/screener/pk/patterns/vcp` | GET | VCP pattern detection |
| `/api/screener/pk/patterns/inside-bar` | GET | Inside bar patterns |
| `/api/screener/pk/bullish-tomorrow` | GET | Bullish for tomorrow |
| `/api/screener/pk/macd-crossover` | GET | MACD crossover signals |
| `/api/screener/pk/rsi-oversold` | GET | RSI oversold stocks |
| `/api/screener/pk/consolidating` | GET | Consolidating stocks |
| `/api/screener/pk/higher-highs` | GET | Higher highs/lows pattern |
| `/api/screener/pk/lorentzian` | GET | ML Lorentzian classifier |

#### AI Intelligence APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/screener/ai/nifty-prediction` | GET | AI Nifty direction prediction |
| `/api/screener/ai/market-regime` | GET | Market regime analysis |
| `/api/screener/ai/trend-analysis` | GET | Trend categorization |
| `/api/screener/ai/momentum-radar` | GET | High momentum detection |
| `/api/screener/ai/breakout-scanner` | GET | AI breakout prediction |
| `/api/screener/ai/reversal-scanner` | GET | AI reversal detection |

#### Paper Trading APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/paper/portfolio/{user_id}` | GET | User's paper portfolio |
| `/api/paper/order` | POST | Execute paper trade |
| `/api/paper/orders/{user_id}` | GET | User's order history |
| `/api/paper/price/{symbol}` | GET | Real-time stock price |
| `/api/paper/reset/{user_id}` | POST | Reset paper account |

#### Auth APIs (Pending User DB Setup)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/google/login` | GET | Initiate Google login |
| `/api/auth/google/callback` | GET | Google OAuth callback |

### Frontend Pages (All Working)
- [x] Landing Page (`/`) - Hero, features, pricing
- [x] Dashboard (`/dashboard`) - Market overview
- [x] **AI Market Screener (`/screener`)** - 43+ scanners with real data ✅
- [x] AI Intelligence (`/ai-intelligence`) - AI features display
- [x] Paper Trading (`/paper-trading`) - Virtual trading
- [x] Signals (`/signals`) - AI trading signals
- [x] Stocks (`/stocks`) - Stock list
- [x] Login (`/login`) - Google authentication

### Database (Supabase)
- [x] `users` - User accounts
- [x] `paper_orders` - Paper trade orders
- [x] `paper_holdings` - Paper portfolio holdings
- [x] `watchlist` - User watchlists
- [x] `signals` - AI generated signals
- [ ] `user_sessions` - Session management (PENDING USER ACTION)

---

## Tech Stack
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS, Framer Motion
- **Backend**: FastAPI (Python), uvicorn
- **Database**: Supabase (PostgreSQL)
- **Data Sources**: yfinance, nsepython, pkscreener
- **Auth**: Emergent-managed Google OAuth

## Architecture
```
/app
├── backend/
│   ├── server.py           # FastAPI entry point
│   ├── screener.py         # Screener APIs with PKScreener integration
│   ├── pkscreener_service.py # PKScreener wrapper service
│   ├── paper_trading.py    # Paper trading APIs
│   ├── auth.py             # Google OAuth APIs
│   ├── users.py            # User management
│   ├── market_data.py      # Market data APIs
│   └── database.py         # Supabase connection
├── frontend/
│   ├── app/
│   │   ├── screener/       # AI Market Screener page
│   │   ├── (platform)/     # Dashboard route group
│   │   │   ├── ai-intelligence/
│   │   │   ├── paper-trading/
│   │   │   └── layout.tsx
│   │   ├── login/          # Google login page
│   │   └── auth/callback/  # OAuth callback
│   ├── components/         # React components
│   └── .env                # Frontend config with Supabase keys
└── memory/
    └── PRD.md              # This file
```

---

## Prioritized Backlog

### P0 - Critical ✅ DONE
- [x] PKScreener full integration (61 scanners, 11 categories)
- [x] Real market data from yfinance
- [x] All NSE stocks via nsepython (2229 stocks)
- [x] Paper trading MVP

### P1 - High Priority
- [ ] **Complete Google Auth** - User needs to create `user_sessions` table in Supabase
- [ ] Frontend for all 61 PKScreener scanners - Currently shows 43+ but can be expanded
- [ ] Comprehensive scanner testing

### P2 - Medium Priority
- [ ] TradingView/Lightweight Charts integration
- [ ] Real-time WebSocket updates
- [ ] Email notifications for signals

### P3 - Future Tasks
- [ ] **SwingAI Bot Integration** - User's custom AI model for BUY/NO_TRADE signals
- [ ] Broker API integration (Zerodha, Upstox) for auto-trading
- [ ] PDF report generation
- [ ] Backtesting module

---

## Pending User Actions
1. **Create `user_sessions` table in Supabase** - Required for Google Auth to work
   ```sql
   CREATE TABLE user_sessions (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     user_id UUID REFERENCES users(id),
     session_token TEXT UNIQUE NOT NULL,
     expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );
   ```

---

## Test Reports
- `/app/test_reports/iteration_1.json` - PKScreener integration tests (33/33 passed)
- `/app/backend/tests/test_pkscreener_api.py` - Backend API tests

## Last Updated
January 23, 2025
