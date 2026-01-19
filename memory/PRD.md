# SwingAI - Product Requirements Document

## Original Problem Statement
Build a cutting-edge, institutional-grade homepage for SwingAI, an AI swing trading platform for the Indian stock market (NSE/BSE). The platform should clone the design/layout of intellectia.ai but adapted for SwingAI branding with real backend data.

## Core Requirements
- Full-stack application with Next.js frontend + FastAPI backend
- Dark theme with professional gradient text effects
- AI-focused marketing terminology throughout
- Real-time market data for NSE/BSE stocks
- Risk Management & Position Sizing calculators
- Stock screener and signal generation

## User Personas
1. **Retail Swing Traders** - Part-time traders looking for AI-powered signals
2. **Professional Traders** - Full-time traders seeking institutional-grade tools
3. **Investment Analysts** - Professionals validating AI recommendations

## What's Been Implemented (Dec 2025)

### Frontend Pages (All Working)
- [x] Landing Page (`/`) - Hero, features, pricing, testimonials, FAQ
- [x] Dashboard (`/dashboard`) - Market indices, AI Top Picks, Top Gainers/Losers
- [x] AI Screener (`/screener`) - Stock screening with filters
- [x] Signals (`/signals`) - AI trading signals with entry/target/stop
- [x] Stocks (`/stocks`) - Stock list with prices and AI scores
- [x] Portfolio (`/portfolio`) - Holdings with P&L tracking
- [x] Trades (`/trades`) - Trade history with win/loss tracking
- [x] Analytics (`/analytics`) - Performance metrics and charts
- [x] Settings (`/settings`) - Profile and risk management settings
- [x] Pricing (`/pricing`) - Subscription tiers
- [x] Login/Signup pages

### Backend APIs (Working with Mock Data)
- [x] `/api/health` - Health check
- [x] `/api/market/overview` - Market indices (NIFTY, SENSEX, BANKNIFTY)
- [x] `/api/market/stocks` - Stock list with prices
- [x] `/api/market/stocks/{symbol}` - Individual stock details
- [x] `/api/market/trending` - AI trending picks
- [x] `/api/market/top-movers` - Gainers and losers
- [x] `/api/market/sectors` - Sector performance

### UI/UX Features
- [x] Dark theme enforced (no light mode toggle)
- [x] Animated gradient text effects
- [x] Responsive design
- [x] Position Sizing Calculator modal
- [x] Risk Management Calculator modal
- [x] Quick Access dropdown menu

## Tech Stack
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS, Framer Motion
- **Backend**: FastAPI (Python), uvicorn
- **Database**: MongoDB (configured but not actively used)
- **UI Components**: Shadcn/UI, Radix UI, Lucide React icons

## Known Limitations (MOCKED)
1. **All market data is MOCK data** - Backend returns hardcoded stock prices
2. **No real authentication** - Supabase code exists but is disabled
3. **No real trade execution** - Signals are for display only
4. **No real portfolio tracking** - All positions are mock data

## Prioritized Backlog

### P0 - Critical (Needed for MVP)
1. [ ] Integrate real Indian market data API (Yahoo Finance or NSE APIs)
2. [ ] Connect frontend to real backend APIs (replace mock data)
3. [ ] Implement user authentication (JWT or Supabase)

### P1 - High Priority
1. [ ] Add TradingView/Lightweight Charts for stock analysis
2. [ ] Real-time WebSocket data updates
3. [ ] User portfolio persistence in MongoDB
4. [ ] Trade history storage and retrieval

### P2 - Nice to Have
1. [ ] Email notifications for signals
2. [ ] Mobile-responsive optimization
3. [ ] PDF report generation
4. [ ] Backtesting module

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/market/overview` | GET | Market indices |
| `/api/market/stocks` | GET | All stocks with prices |
| `/api/market/stocks/{symbol}` | GET | Stock details |
| `/api/market/trending` | GET | Trending stocks |
| `/api/market/top-movers` | GET | Top gainers/losers |
| `/api/market/sectors` | GET | Sector performance |

## Architecture
```
/app
├── backend/
│   ├── server.py        # FastAPI entry point
│   ├── market_data.py   # Market data router (MOCK)
│   └── .env             # Backend config
├── frontend/
│   ├── app/             # Next.js pages
│   │   ├── (platform)/  # Dashboard route group
│   │   ├── signals/     # Signals pages
│   │   ├── stocks/      # Stock pages
│   │   └── ...
│   ├── components/      # React components
│   └── .env             # Frontend config
└── memory/
    └── PRD.md           # This file
```

## Environment Variables Required
```bash
# Frontend (.env)
REACT_APP_BACKEND_URL=<preview_url>
NEXT_PUBLIC_API_URL=<preview_url>

# Backend (.env)
MONGO_URL=<mongodb_connection_string>
DB_NAME=swingai
```

## Last Updated
January 2025
