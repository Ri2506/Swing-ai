// ============================================================================
// STOCK DETAIL PAGE - Full Analysis with TradingView
// ============================================================================

'use client'

import { useState, useEffect, useRef } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import {
  ArrowLeft, TrendingUp, TrendingDown, Activity, BarChart3,
  Bookmark, BookmarkCheck, RefreshCw, Clock, ExternalLink,
  ArrowUpRight, ArrowDownRight, Target,
  LineChart, Layers, Zap
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer
} from 'recharts'

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.REACT_APP_BACKEND_URL || ''

interface StockData {
  symbol: string
  name: string
  price: number
  change: number
  change_percent: number
  open: number
  high: number
  low: number
  volume: number
  prev_close?: number
  day_high?: number
  day_low?: number
  week_52_high?: number
  week_52_low?: number
  market_cap?: number
  pe_ratio?: number
  sector?: string
  industry?: string
}

interface TechnicalData {
  rsi: number
  macd: number
  macd_signal: number
  sma_20: number
  sma_50: number
  sma_200?: number
  trend: string
  volume_ratio: number
}

// Real-time Candlestick Chart using Lightweight Charts (by TradingView)
function TradingViewAdvancedChart({ symbol }: { symbol: string }) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)
  const candleSeriesRef = useRef<any>(null)
  const volumeSeriesRef = useRef<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [chartError, setChartError] = useState<string | null>(null)
  const [timeframe, setTimeframe] = useState('D')
  
  // Timeframe options
  const timeframes = [
    { label: '1H', value: '1h', period: '5d' },
    { label: '4H', value: '4h', period: '1mo' },
    { label: 'D', value: 'D', period: '1y' },
    { label: 'W', value: 'W', period: '2y' },
  ]
  
  useEffect(() => {
    const initChart = async () => {
      if (!chartContainerRef.current) return
      
      try {
        // Dynamically import lightweight-charts
        const LWC = await import('lightweight-charts')
        
        // Clear previous chart
        if (chartRef.current) {
          chartRef.current.remove()
        }
        chartContainerRef.current.innerHTML = ''
        
        // Create chart
        const chart = LWC.createChart(chartContainerRef.current, {
          layout: {
            background: { type: LWC.ColorType.Solid, color: '#0f0f23' },
            textColor: '#9ca3af',
          },
          grid: {
            vertLines: { color: '#1f2937' },
            horzLines: { color: '#1f2937' },
          },
          rightPriceScale: {
            borderColor: '#374151',
          },
          timeScale: {
            borderColor: '#374151',
            timeVisible: true,
          },
          width: chartContainerRef.current.clientWidth,
          height: 450,
        })
        
        chartRef.current = chart
        
        // Add candlestick series - v5 uses addSeries with type
        const candleSeries = chart.addSeries(LWC.CandlestickSeries, {
          upColor: '#22c55e',
          downColor: '#ef4444',
          borderUpColor: '#22c55e',
          borderDownColor: '#ef4444',
          wickUpColor: '#22c55e',
          wickDownColor: '#ef4444',
        })
        candleSeriesRef.current = candleSeries
        
        // Add volume histogram
        const volumeSeries = chart.addSeries(LWC.HistogramSeries, {
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
        })
        chart.priceScale('volume').applyOptions({
          scaleMargins: { top: 0.85, bottom: 0 },
        })
        volumeSeriesRef.current = volumeSeries
        
        // Fetch data
        await fetchData()
        
        // Handle resize
        const handleResize = () => {
          if (chartContainerRef.current && chartRef.current) {
            chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth })
          }
        }
        window.addEventListener('resize', handleResize)
        
        return () => {
          window.removeEventListener('resize', handleResize)
        }
      } catch (err: any) {
        console.error('Chart init error:', err)
        setChartError(err.message || 'Failed to load chart')
        setIsLoading(false)
      }
    }
    
    initChart()
    
    return () => {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [symbol])
  
  const fetchData = async () => {
    setIsLoading(true)
    try {
      const tf = timeframes.find(t => t.value === timeframe)
      const period = tf?.period || '1y'
      
      const res = await fetch(`${API_BASE}/api/screener/prices/${symbol}/history?period=${period}`)
      const data = await res.json()
      
      if (data.success && data.history?.length > 0) {
        // Format for lightweight-charts - YYYY-MM-DD string
        const candles = data.history.map((item: any) => ({
          time: item.date.split('T')[0],
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
        }))
        
        const volumes = data.history.map((item: any) => ({
          time: item.date.split('T')[0],
          value: item.volume,
          color: item.close >= item.open ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)',
        }))
        
        if (candleSeriesRef.current) {
          candleSeriesRef.current.setData(candles)
        }
        if (volumeSeriesRef.current) {
          volumeSeriesRef.current.setData(volumes)
        }
        if (chartRef.current) {
          chartRef.current.timeScale().fitContent()
        }
      }
    } catch (err) {
      console.error('Fetch error:', err)
    }
    setIsLoading(false)
  }
  
  // Re-fetch when timeframe changes
  useEffect(() => {
    if (candleSeriesRef.current) {
      fetchData()
    }
  }, [timeframe])
  
  return (
    <div className="w-full" data-testid="tradingview-advanced-chart">
      {/* Timeframe selector */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {timeframes.map((tf) => (
            <button
              key={tf.value}
              onClick={() => setTimeframe(tf.value)}
              className={`px-3 py-1.5 text-sm rounded-lg transition ${
                timeframe === tf.value
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-green-500 rounded-full"></span> Bullish
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-red-500 rounded-full"></span> Bearish
          </span>
        </div>
      </div>
      
      {/* Chart container */}
      <div className="relative bg-[#0f0f23] rounded-lg overflow-hidden border border-gray-800">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0f0f23]/80 z-10">
            <div className="text-center">
              <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin mx-auto mb-2" />
              <p className="text-gray-400 text-sm">Loading chart...</p>
            </div>
          </div>
        )}
        <div ref={chartContainerRef} className="w-full" style={{ height: '450px' }} />
      </div>
      
      {/* Chart info */}
      <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
        <span>Real-time NSE data • Powered by yfinance</span>
        <a
          href={`https://www.tradingview.com/chart/?symbol=NSE:${symbol}`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-indigo-400 hover:text-indigo-300"
        >
          Open in TradingView <ExternalLink className="w-3 h-3" />
        </a>
      </div>
    </div>
  )
}

// Custom Price Chart using Recharts with yfinance data
function PriceChart({ symbol, priceData }: { symbol: string; priceData: StockData | null }) {
  const [chartData, setChartData] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [timeframe, setTimeframe] = useState('1M')
  
  const tvSymbol = `NSE:${symbol}`
  
  useEffect(() => {
    fetchChartData()
  }, [symbol, timeframe])
  
  const fetchChartData = async () => {
    setIsLoading(true)
    try {
      // Fetch historical data from our backend
      const res = await fetch(`${API_BASE}/api/screener/prices/${symbol}/history?period=${timeframe.toLowerCase()}`)
      const data = await res.json()
      
      if (data.success && data.history) {
        const formattedData = data.history.map((item: any) => ({
          date: new Date(item.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }),
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          volume: item.volume,
          // For area chart
          price: item.close
        }))
        setChartData(formattedData)
      }
    } catch (error) {
      console.error('Error fetching chart data:', error)
      // Generate placeholder data if API fails
      const days = timeframe === '1W' ? 7 : timeframe === '1M' ? 30 : timeframe === '3M' ? 90 : 365
      const basePrice = priceData?.price || 1000
      const placeholderData = Array.from({ length: days }, (_, i) => {
        const variance = (Math.random() - 0.5) * 0.02 * basePrice
        const price = basePrice + variance
        return {
          date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }),
          price: price,
          open: price - Math.random() * 10,
          high: price + Math.random() * 20,
          low: price - Math.random() * 20,
          close: price,
          volume: Math.floor(Math.random() * 1000000)
        }
      })
      setChartData(placeholderData)
    }
    setIsLoading(false)
  }
  
  const timeframes = [
    { label: '1W', value: '1W' },
    { label: '1M', value: '1M' },
    { label: '3M', value: '3M' },
    { label: '1Y', value: '1Y' },
  ]
  
  const isPositive = chartData.length > 1 ? chartData[chartData.length - 1]?.close >= chartData[0]?.close : true
  
  return (
    <div className="w-full" data-testid="tradingview-chart">
      {/* Chart Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {timeframes.map((tf) => (
            <button
              key={tf.value}
              onClick={() => setTimeframe(tf.value)}
              className={`px-3 py-1.5 text-sm rounded-lg transition ${
                timeframe === tf.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>
        <a
          href={`https://www.tradingview.com/chart/?symbol=${tvSymbol}`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white rounded-lg text-sm font-medium transition shadow-lg shadow-blue-500/20"
        >
          <LineChart className="w-4 h-4" />
          Advanced Chart
          <ExternalLink className="w-3 h-3" />
        </a>
      </div>
      
      {/* Main Chart */}
      <div className="relative h-[400px] bg-gray-900/50 rounded-lg overflow-hidden border border-gray-800 p-4">
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10">
            <div className="text-center">
              <RefreshCw className="w-8 h-8 text-blue-400 animate-spin mx-auto mb-2" />
              <p className="text-gray-400 text-sm">Loading chart data...</p>
            </div>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis 
                dataKey="date" 
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#6b7280', fontSize: 11 }}
                interval="preserveStartEnd"
              />
              <YAxis 
                domain={['auto', 'auto']}
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#6b7280', fontSize: 11 }}
                tickFormatter={(value) => `₹${value.toLocaleString('en-IN')}`}
                width={80}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#fff'
                }}
                formatter={(value: any) => [`₹${Number(value).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`, 'Price']}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Area 
                type="monotone" 
                dataKey="price" 
                stroke={isPositive ? "#22c55e" : "#ef4444"}
                strokeWidth={2}
                fillOpacity={1} 
                fill="url(#colorPrice)" 
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
      
      {/* Info Banner */}
      <div className="mt-4 p-3 bg-blue-900/20 border border-blue-800/50 rounded-lg">
        <div className="flex items-center gap-2 text-sm text-blue-300">
          <Activity className="w-4 h-4" />
          <span>For advanced technical analysis, indicators, and drawing tools, open the full </span>
          <a 
            href={`https://www.tradingview.com/chart/?symbol=${tvSymbol}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:text-blue-300 font-medium underline"
          >
            TradingView Chart
          </a>
        </div>
      </div>
      
      {/* Additional chart links */}
      <div className="mt-4 flex flex-wrap gap-3">
        <a 
          href={`https://www.tradingview.com/symbols/${tvSymbol}/technicals/`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg text-sm transition"
        >
          <Activity className="w-4 h-4 text-purple-400" />
          Technical Analysis
        </a>
        <a 
          href={`https://www.tradingview.com/symbols/${tvSymbol}/financials/`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg text-sm transition"
        >
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          Financials
        </a>
        <a 
          href={`https://www.screener.in/company/${symbol}/`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg text-sm transition"
        >
          <Target className="w-4 h-4 text-yellow-400" />
          Screener.in
        </a>
      </div>
    </div>
  )
}

// Technical Indicator Card
function IndicatorCard({ 
  label, 
  value, 
  subValue,
  trend,
  icon: Icon 
}: { 
  label: string
  value: string | number
  subValue?: string
  trend?: 'up' | 'down' | 'neutral'
  icon?: any
}) {
  const trendColors = {
    up: 'text-green-400',
    down: 'text-red-400',
    neutral: 'text-gray-400'
  }
  
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        {Icon && <Icon className="w-4 h-4 text-gray-500" />}
        <span className="text-sm text-gray-500">{label}</span>
      </div>
      <div className={`text-xl font-bold ${trend ? trendColors[trend] : 'text-white'}`}>
        {value}
      </div>
      {subValue && (
        <div className="text-xs text-gray-500 mt-1">{subValue}</div>
      )}
    </div>
  )
}

export default function StockDetailPage() {
  const params = useParams()
  const router = useRouter()
  const symbol = (params.symbol as string)?.toUpperCase()
  
  const [stockData, setStockData] = useState<StockData | null>(null)
  const [technicals, setTechnicals] = useState<TechnicalData | null>(null)
  const [loading, setLoading] = useState(true)
  const [isInWatchlist, setIsInWatchlist] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [userId] = useState('ffb9e2ca-6733-4e84-9286-0aa134e6f57e')
  
  // WebSocket for real-time updates
  const wsRef = useRef<WebSocket | null>(null)
  const [wsConnected, setWsConnected] = useState(false)

  useEffect(() => {
    if (!symbol) return
    
    fetchStockData()
    checkWatchlist()
    connectWebSocket()
    
    // Polling fallback every 10 seconds
    const interval = setInterval(fetchStockData, 10000)
    
    return () => {
      clearInterval(interval)
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [symbol])

  const connectWebSocket = () => {
    try {
      const wsUrl = API_BASE.replace('https://', 'wss://').replace('http://', 'ws://')
      const ws = new WebSocket(`${wsUrl}/ws/prices/${Date.now()}`)
      
      ws.onopen = () => {
        setWsConnected(true)
        // Subscribe to this symbol
        ws.send(JSON.stringify({
          action: 'subscribe',
          symbols: [symbol]
        }))
      }
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.type === 'price_update' && data.prices?.[symbol]) {
          const price = data.prices[symbol]
          setStockData(prev => prev ? {
            ...prev,
            price: price.price,
            change: price.change,
            change_percent: price.change_percent,
            high: price.high,
            low: price.low,
            volume: price.volume
          } : null)
          setLastUpdate(new Date())
        }
      }
      
      ws.onerror = () => setWsConnected(false)
      ws.onclose = () => setWsConnected(false)
      
      wsRef.current = ws
    } catch (error) {
      console.error('WebSocket error:', error)
    }
  }

  const fetchStockData = async () => {
    try {
      // Fetch price data
      const priceRes = await fetch(`${API_BASE}/api/screener/prices/${symbol}`)
      const priceData = await priceRes.json()
      
      if (priceData.success) {
        setStockData(prev => ({
          ...prev,
          symbol: symbol,
          name: symbol, // Will be updated
          price: priceData.price,
          change: priceData.change,
          change_percent: priceData.change_percent,
          open: priceData.open,
          high: priceData.high,
          low: priceData.low,
          volume: priceData.volume
        } as StockData))
        setLastUpdate(new Date())
      }
      
      // Fetch technical analysis
      const techRes = await fetch(`${API_BASE}/api/screener/pk/scan/single?symbol=${symbol}&scanner_id=trend`, {
        method: 'POST'
      })
      const techData = await techRes.json()
      
      if (techData.success && techData.result) {
        const r = techData.result
        setTechnicals({
          rsi: r.rsi || 50,
          macd: r.macd || 0,
          macd_signal: r.macd_signal || 0,
          sma_20: r.sma_20 || 0,
          sma_50: r.sma_50 || 0,
          trend: r.trend || 'Neutral',
          volume_ratio: r.volume_ratio || 1
        })
        
        // Update stock data with more info
        setStockData(prev => prev ? {
          ...prev,
          name: r.name || symbol,
          sector: r.sector,
          week_52_high: r.high_52w,
          week_52_low: r.low_52w,
          market_cap: r.market_cap
        } : null)
      }
      
      setLoading(false)
    } catch (error) {
      console.error('Error fetching stock data:', error)
      setLoading(false)
    }
  }

  const checkWatchlist = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/watchlist/${userId}/check/${symbol}`)
      const data = await res.json()
      setIsInWatchlist(data.in_watchlist || false)
    } catch (error) {
      console.error('Error checking watchlist:', error)
    }
  }

  const toggleWatchlist = async () => {
    try {
      if (isInWatchlist) {
        await fetch(`${API_BASE}/api/watchlist/${userId}/${symbol}`, { method: 'DELETE' })
        setIsInWatchlist(false)
      } else {
        await fetch(`${API_BASE}/api/watchlist/add`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, symbol })
        })
        setIsInWatchlist(true)
      }
    } catch (error) {
      console.error('Error toggling watchlist:', error)
    }
  }

  const isPositive = (stockData?.change_percent || 0) >= 0

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <RefreshCw className="w-8 h-8 text-green-500 animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white" data-testid="stock-detail-page">
      {/* Background */}
      <div className="fixed inset-0 bg-gradient-to-b from-gray-900/50 via-black to-black pointer-events-none" />
      
      {/* Header */}
      <header className="sticky top-0 z-40 bg-black/80 backdrop-blur-xl border-b border-gray-800">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button onClick={() => router.back()} className="p-2 hover:bg-gray-800 rounded-lg">
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-2xl font-bold">{symbol}</h1>
                  <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded-full">NSE</span>
                  {wsConnected && (
                    <span className="flex items-center gap-1 px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full">
                      <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                      Live
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-500">{stockData?.name || symbol}</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={toggleWatchlist}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition ${
                  isInWatchlist 
                    ? 'bg-yellow-500/20 text-yellow-400' 
                    : 'bg-gray-800 hover:bg-gray-700'
                }`}
              >
                {isInWatchlist ? <BookmarkCheck className="w-4 h-4" /> : <Bookmark className="w-4 h-4" />}
                {isInWatchlist ? 'Watching' : 'Watch'}
              </button>
              <button
                onClick={fetchStockData}
                className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </header>
      
      <div className="container mx-auto px-4 py-6">
        {/* Price Section */}
        <div className="grid lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-2 bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="text-4xl font-bold text-white mb-2">
                  ₹{stockData?.price?.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                </div>
                <div className={`flex items-center gap-2 text-lg ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                  {isPositive ? <ArrowUpRight className="w-5 h-5" /> : <ArrowDownRight className="w-5 h-5" />}
                  <span>{isPositive ? '+' : ''}{stockData?.change?.toFixed(2)}</span>
                  <span>({isPositive ? '+' : ''}{stockData?.change_percent?.toFixed(2)}%)</span>
                </div>
              </div>
              {lastUpdate && (
                <div className="text-xs text-gray-500 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {lastUpdate.toLocaleTimeString()}
                </div>
              )}
            </div>
            
            <div className="grid grid-cols-4 gap-4">
              <div>
                <div className="text-gray-500 text-sm">Open</div>
                <div className="font-medium">₹{stockData?.open?.toLocaleString('en-IN')}</div>
              </div>
              <div>
                <div className="text-gray-500 text-sm">High</div>
                <div className="font-medium text-green-400">₹{stockData?.high?.toLocaleString('en-IN')}</div>
              </div>
              <div>
                <div className="text-gray-500 text-sm">Low</div>
                <div className="font-medium text-red-400">₹{stockData?.low?.toLocaleString('en-IN')}</div>
              </div>
              <div>
                <div className="text-gray-500 text-sm">Volume</div>
                <div className="font-medium">{(stockData?.volume || 0).toLocaleString()}</div>
              </div>
            </div>
          </div>
          
          {/* Quick Stats */}
          <div className="space-y-4">
            <IndicatorCard 
              label="Trend" 
              value={technicals?.trend || 'N/A'} 
              trend={technicals?.trend?.includes('Up') ? 'up' : technicals?.trend?.includes('Down') ? 'down' : 'neutral'}
              icon={TrendingUp}
            />
            <IndicatorCard 
              label="RSI (14)" 
              value={technicals?.rsi?.toFixed(1) || 'N/A'} 
              subValue={technicals?.rsi && technicals.rsi < 30 ? 'Oversold' : technicals?.rsi && technicals.rsi > 70 ? 'Overbought' : 'Neutral'}
              trend={technicals?.rsi && technicals.rsi < 30 ? 'up' : technicals?.rsi && technicals.rsi > 70 ? 'down' : 'neutral'}
              icon={Activity}
            />
            <IndicatorCard 
              label="Volume" 
              value={`${technicals?.volume_ratio?.toFixed(1) || 1}x`} 
              subValue="vs 20-day avg"
              trend={technicals?.volume_ratio && technicals.volume_ratio > 1.5 ? 'up' : 'neutral'}
              icon={BarChart3}
            />
          </div>
        </div>
        
        {/* TradingView Advanced Real-time Chart */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl overflow-hidden mb-6">
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <LineChart className="w-5 h-5 text-blue-400" />
              <h2 className="font-semibold">TradingView Real-time Chart</h2>
              <span className="text-xs px-2 py-0.5 bg-green-500/20 text-green-400 rounded-full">Live</span>
            </div>
            <a 
              href={`https://www.tradingview.com/chart/?symbol=NSE:${symbol}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-sm text-blue-400 hover:text-blue-300"
            >
              Open Full Chart <ExternalLink className="w-3 h-3" />
            </a>
          </div>
          <div className="p-4">
            <TradingViewAdvancedChart symbol={symbol} />
          </div>
        </div>

        {/* Price Overview Chart */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl overflow-hidden mb-6">
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              <h2 className="font-semibold">Price Overview</h2>
              <span className="text-xs text-gray-500">yfinance data</span>
            </div>
          </div>
          <div className="p-4">
            <PriceChart symbol={symbol} priceData={stockData} />
          </div>
        </div>
        
        {/* Technical Indicators */}
        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          {/* Key Levels */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Target className="w-4 h-4 text-purple-400" />
              Key Levels
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">52 Week High</span>
                <span className="font-medium text-green-400">
                  ₹{stockData?.week_52_high?.toLocaleString('en-IN') || 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">52 Week Low</span>
                <span className="font-medium text-red-400">
                  ₹{stockData?.week_52_low?.toLocaleString('en-IN') || 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">SMA 20</span>
                <span className="font-medium">₹{technicals?.sma_20?.toLocaleString('en-IN') || 'N/A'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">SMA 50</span>
                <span className="font-medium">₹{technicals?.sma_50?.toLocaleString('en-IN') || 'N/A'}</span>
              </div>
            </div>
          </div>
          
          {/* MACD */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              MACD Analysis
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">MACD Line</span>
                <span className={`font-medium ${(technicals?.macd || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {technicals?.macd?.toFixed(2) || 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Signal Line</span>
                <span className="font-medium">{technicals?.macd_signal?.toFixed(2) || 'N/A'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Histogram</span>
                <span className={`font-medium ${((technicals?.macd || 0) - (technicals?.macd_signal || 0)) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {((technicals?.macd || 0) - (technicals?.macd_signal || 0)).toFixed(2)}
                </span>
              </div>
              <div className="pt-2 border-t border-gray-800">
                <span className={`text-sm ${(technicals?.macd || 0) > (technicals?.macd_signal || 0) ? 'text-green-400' : 'text-red-400'}`}>
                  {(technicals?.macd || 0) > (technicals?.macd_signal || 0) ? '↑ Bullish Crossover' : '↓ Bearish Crossover'}
                </span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Quick Actions */}
        <div className="flex items-center justify-center gap-4">
          <Link 
            href={`/paper-trading?symbol=${symbol}`}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl font-medium hover:opacity-90"
          >
            <Zap className="w-4 h-4" />
            Paper Trade
          </Link>
          <Link 
            href="/screener"
            className="flex items-center gap-2 px-6 py-3 bg-gray-800 hover:bg-gray-700 rounded-xl"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Screener
          </Link>
        </div>
      </div>
    </div>
  )
}
