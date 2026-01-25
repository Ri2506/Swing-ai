# SwingAI - Product Requirements Document

## Original Problem Statement
Build a cutting-edge, institutional-grade AI swing trading platform for the Indian stock market (NSE). 

## Latest Update - January 26, 2025
**Removed TradingView Charts - Keeping Only Custom Chart Module**

### What's Been Changed
1. **Stock Detail Page** (`/stock/[symbol]`):
   - Removed TradingView external link
   - Custom recharts AreaChart with yfinance data is now the only chart
   - Added Refresh button for chart data reload

2. **Screener Page** (`/screener`):
   - Replaced `TradingViewChart` component with `StockChartModal`
   - Custom recharts-based chart modal with timeframe selection
   - Full OHLCV data in tooltip

3. **Watchlist Page** (`/watchlist`):
   - Replaced TradingView iframe modal with `StockChartModal`
   - Same custom recharts implementation as screener

### Custom Chart Features
- **Timeframe Buttons**: 1W, 1M, 3M, 1Y
- **Real-time NSE Data**: Powered by yfinance API
- **OHLCV Tooltips**: Open, High, Low, Close, Volume on hover
- **Trend Indicators**: Green gradient for bullish, Red gradient for bearish
- **Period Change Display**: Shows % change for selected timeframe
- **Responsive Design**: Works across all screen sizes

## Tech Stack
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS, recharts
- **Backend**: FastAPI (Python), WebSocket
- **Database**: Supabase (PostgreSQL)
- **Data**: yfinance, nsepython, pkscreener
- **Charts**: recharts (AreaChart)

## What's Working
- 61 PKScreener scanners
- Paper trading with ₹10 Lakh virtual capital
- Watchlist with per-user storage
- WebSocket real-time price updates
- Stock detail page with custom charts
- All chart modals across pages

## Backlog
### P1 - High Priority
- [ ] Complete Google Auth flow
- [ ] Price alerts for watchlist

### P2 - Medium Priority
- [ ] PDF report generation
- [ ] Advanced chart indicators (MACD, RSI overlay)

### P3 - Future
- [ ] Broker API integration (Zerodha, Upstox)
- [ ] Backtesting module
- [ ] SwingAI Bot

## Last Updated
January 26, 2025
