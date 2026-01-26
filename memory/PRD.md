# SwingAI - Product Requirements Document

## Original Problem Statement
Build a cutting-edge, institutional-grade AI swing trading platform for the Indian stock market (NSE).

## Latest Update - January 26, 2025
**Fixed Real-time Updates + Added RSI/MACD Indicators + Pattern Recognition**

### Real-time Updates Fix
- Implemented reliable 5-second polling mechanism (more stable than WebSocket in k8s)
- Added LIVE status indicator with green animated dot
- Auto-updates prices without manual refresh
- Graceful fallback from WebSocket to polling

### RSI Indicator (NEW)
- RSI(14) calculation with proper gain/loss averaging
- Cyan line in dedicated RSI panel
- Reference lines at 30 (oversold) and 70 (overbought)
- RSI badge showing "RSI: X.X • Oversold/Overbought/Neutral"
- Toggle button (gauge icon) - cyan when active
- RSI values in tooltip

### MACD Indicator (NEW)
- MACD(12, 26, 9) calculation
- Green MACD line
- Red Signal line
- Purple histogram bars
- Toggle button (activity icon) - cyan when active
- MACD values in tooltip

### Candlestick Pattern Recognition (NEW)
- Detects: Doji, Hammer, Hanging Man, Shooting Star
- Bullish Engulfing, Bearish Engulfing
- Strong Bullish/Bearish momentum candles
- Pattern badge in header when detected
- Pattern info in tooltip
- Toggle button (lightning icon) - amber when active

### Chart Controls Summary
- **Chart Types**: Area, Line, Candlestick
- **Volume**: Toggle blue volume bars
- **MA**: Toggle MA20 (orange) and MA50 (purple)
- **RSI**: Toggle RSI(14) panel
- **MACD**: Toggle MACD panel
- **Patterns**: Toggle pattern recognition
- **Timeframes**: 1D, 1W, 1M, 3M, 6M, 1Y

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
- Real-time price updates (5s polling)
- RSI(14) indicator with zones
- MACD(12,26,9) with histogram
- Candlestick pattern detection
- Moving averages (MA20, MA50)
- Volume bars
- OHLCV tooltips

## Testing Results
- Frontend: 100% (20/20 features passed)
- Backend: 100% (real-time polling working)

## Backlog
### P1 - High Priority
- [ ] Complete Google Auth flow
- [ ] Price alerts for watchlist
- [ ] Add Bollinger Bands indicator

### P2 - Medium Priority
- [ ] PDF report generation
- [ ] Stochastic oscillator
- [ ] Support/Resistance levels

### P3 - Future
- [ ] Broker API integration (Zerodha, Upstox)
- [ ] Backtesting module
- [ ] SwingAI Bot

## Last Updated
January 26, 2025
