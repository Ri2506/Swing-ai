# SwingAI - Product Requirements Document

## Original Problem Statement
Build a cutting-edge, institutional-grade AI swing trading platform for the Indian stock market (NSE).

## Latest Update - January 26, 2025
**Implemented Cutting-Edge Advanced Stock Chart with Real-time Data**

### New AdvancedStockChart Component Features
1. **Multiple Chart Types**:
   - Area Chart with gradient fill
   - Line Chart with smooth lines
   - Candlestick Chart with OHLC data

2. **Volume Bars**: Blue volume bars below main chart with toggle

3. **Moving Averages**: 
   - MA20 (orange line)
   - MA50 (purple line)
   - Toggle button to show/hide

4. **Timeframe Selection**: 1D, 1W, 1M, 3M, 6M, 1Y buttons

5. **Real-time Updates**:
   - WebSocket connection for live data
   - LIVE/Delayed status indicator
   - Auto-refresh every 30 seconds as fallback

6. **Rich OHLCV Tooltip**:
   - Date with weekday
   - Open, High (green), Low (red), Close (bold), Volume (blue)
   - Smooth animations

7. **Trend Indicators**:
   - Green gradient for bullish
   - Red gradient for bearish
   - Trend label in footer

### Files Changed
- `/app/frontend/components/AdvancedStockChart.tsx` - NEW component
- `/app/frontend/app/stock/[symbol]/page.tsx` - Uses AdvancedStockChart
- `/app/frontend/app/screener/page.tsx` - Modal uses AdvancedStockChart
- `/app/frontend/app/watchlist/page.tsx` - Modal uses AdvancedStockChart

## Tech Stack
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS, recharts
- **Backend**: FastAPI (Python), WebSocket
- **Database**: Supabase (PostgreSQL)
- **Data**: yfinance, nsepython, pkscreener
- **Charts**: recharts (AreaChart, ComposedChart, Bar, Line)

## What's Working
- 61 PKScreener scanners
- Paper trading with ₹10 Lakh virtual capital
- Watchlist with per-user storage
- Advanced stock charts with multiple types
- OHLCV tooltips and volume bars
- Moving averages (MA20, MA50)
- Timeframe selection (1D-1Y)

## Known Issues
- WebSocket connections failing (graceful fallback to Delayed)
- Screener API returns 0 results sometimes

## Backlog
### P1 - High Priority
- [ ] Fix WebSocket connections for real-time data
- [ ] Complete Google Auth flow
- [ ] Price alerts for watchlist

### P2 - Medium Priority
- [ ] PDF report generation
- [ ] Add RSI/MACD chart overlays
- [ ] Candlestick patterns recognition

### P3 - Future
- [ ] Broker API integration (Zerodha, Upstox)
- [ ] Backtesting module
- [ ] SwingAI Bot

## Last Updated
January 26, 2025
