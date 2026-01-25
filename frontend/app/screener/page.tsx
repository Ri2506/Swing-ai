// ============================================================================
// AI MARKET SCREENER V2 - ALL 61 SCANNERS + WATCHLIST + CUSTOM CHARTS
// Full PKScreener integration with dynamic scanner fetching
// ============================================================================

'use client'

import { useState, useEffect, useCallback } from 'react'
import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search, TrendingUp, TrendingDown, ArrowUpRight, ArrowDownRight,
  RefreshCw, Zap, Target, BarChart3, Activity, Clock, Star,
  ChevronRight, ChevronDown, X, Play, Bookmark, BookmarkCheck,
  ArrowLeft, Sparkles, Database, Globe2, Shield, LineChart,
  Layers, Triangle, Cpu, ArrowRightLeft, Eye,
  Plus, Check, AlertCircle
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer
} from 'recharts'

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.REACT_APP_BACKEND_URL || ''

// Icon mapping for categories
const CATEGORY_ICONS: { [key: string]: any } = {
  breakout: TrendingUp,
  momentum: Zap,
  reversal: ArrowRightLeft,
  patterns: Triangle,
  ma_signals: LineChart,
  technical: BarChart3,
  signals: Activity,
  consolidation: Layers,
  trend: TrendingUp,
  ml: Cpu,
  short_sell: TrendingDown,
}

const CATEGORY_COLORS: { [key: string]: { bg: string; border: string; text: string; gradient: string } } = {
  breakout: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', gradient: 'from-emerald-500 to-green-600' },
  momentum: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', gradient: 'from-blue-500 to-indigo-600' },
  reversal: { bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400', gradient: 'from-orange-500 to-amber-600' },
  patterns: { bg: 'bg-pink-500/10', border: 'border-pink-500/30', text: 'text-pink-400', gradient: 'from-pink-500 to-rose-600' },
  ma_signals: { bg: 'bg-cyan-500/10', border: 'border-cyan-500/30', text: 'text-cyan-400', gradient: 'from-cyan-500 to-teal-600' },
  technical: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-400', gradient: 'from-purple-500 to-violet-600' },
  signals: { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', gradient: 'from-yellow-500 to-orange-500' },
  consolidation: { bg: 'bg-gray-500/10', border: 'border-gray-500/30', text: 'text-gray-400', gradient: 'from-gray-500 to-slate-600' },
  trend: { bg: 'bg-green-500/10', border: 'border-green-500/30', text: 'text-green-400', gradient: 'from-green-500 to-emerald-600' },
  ml: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', gradient: 'from-red-500 to-rose-600' },
  short_sell: { bg: 'bg-rose-500/10', border: 'border-rose-500/30', text: 'text-rose-400', gradient: 'from-rose-500 to-red-600' },
}

// AI Intelligence Features
const AI_FEATURES = [
  { id: 'nifty_prediction', name: 'AI Nifty Outlook', icon: LineChart, description: 'Directional bias with confidence', endpoint: '/api/screener/ai/nifty-prediction' },
  { id: 'market_regime', name: 'Market Regime', icon: Layers, description: 'Bull/Bear regime analysis', endpoint: '/api/screener/ai/market-regime' },
  { id: 'trend_analysis', name: 'Trend Analysis', icon: TrendingUp, description: 'Uptrend/Downtrend breakdown', endpoint: '/api/screener/ai/trend-analysis' },
]

// ============================================================================
// COMPONENTS
// ============================================================================

interface StockCardProps {
  stock: any
  index: number
  onAddToWatchlist: (symbol: string) => void
  onViewChart: (symbol: string) => void
  isInWatchlist: boolean
}

function StockCard({ stock, index, onAddToWatchlist, onViewChart, isInWatchlist }: StockCardProps) {
  const isPositive = (stock.change_percent || stock.change_pct || 0) >= 0
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.02 }}
      className="group relative bg-gray-900/50 backdrop-blur-sm border border-gray-800 rounded-xl p-4 hover:border-gray-600 transition-all hover:bg-gray-800/50 cursor-pointer"
      data-testid={`stock-card-${stock.symbol}`}
      onClick={() => window.location.href = `/stock/${stock.symbol}`}
    >
      <div className={`absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity ${
        isPositive ? 'bg-green-500/5' : 'bg-red-500/5'
      }`} />
      
      <div className="relative z-10">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="text-lg font-bold text-white">{stock.symbol}</h3>
            <p className="text-xs text-gray-500 truncate max-w-[120px]">{stock.name || stock.sector || '-'}</p>
          </div>
          <div className={`flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium ${
            isPositive ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
          }`}>
            {isPositive ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
            {Math.abs(stock.change_percent || stock.change_pct || 0).toFixed(2)}%
          </div>
        </div>
        
        {/* Price */}
        <div className="text-2xl font-bold text-white mb-3">
          ₹{(stock.current_price || stock.ltp || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
        </div>
        
        {/* Metrics */}
        <div className="grid grid-cols-2 gap-2 text-sm mb-3">
          <div className="bg-gray-800/50 rounded-lg p-2">
            <div className="text-gray-500 text-xs">RSI</div>
            <div className={`font-medium ${
              (stock.rsi || 50) > 70 ? 'text-red-400' : (stock.rsi || 50) < 30 ? 'text-green-400' : 'text-white'
            }`}>{stock.rsi?.toFixed(1) || '-'}</div>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-2">
            <div className="text-gray-500 text-xs">Volume</div>
            <div className="text-white font-medium">{stock.volume_ratio ? `${stock.volume_ratio}x` : '-'}</div>
          </div>
        </div>
        
        {/* Signal/Reason */}
        {(stock.reason || stock.trend) && (
          <div className="text-xs text-gray-400 mb-3 truncate">
            {stock.reason || stock.trend}
          </div>
        )}
        
        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => { e.stopPropagation(); onViewChart(stock.symbol) }}
            className="flex-1 flex items-center justify-center gap-1 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-white transition"
          >
            <Eye className="w-4 h-4" />
            Chart
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onAddToWatchlist(stock.symbol) }}
            className={`flex items-center justify-center gap-1 py-2 px-3 rounded-lg text-sm transition ${
              isInWatchlist 
                ? 'bg-yellow-500/20 text-yellow-400' 
                : 'bg-gray-800 hover:bg-gray-700 text-white'
            }`}
            data-testid={`watchlist-btn-${stock.symbol}`}
          >
            {isInWatchlist ? <BookmarkCheck className="w-4 h-4" /> : <Bookmark className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </motion.div>
  )
}

function StockChartModal({ symbol, onClose }: { symbol: string; onClose: () => void }) {
  const [chartData, setChartData] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [timeframe, setTimeframe] = useState('1M')
  const [stockInfo, setStockInfo] = useState<any>(null)
  
  useEffect(() => {
    fetchChartData()
  }, [symbol, timeframe])
  
  const fetchChartData = async () => {
    setIsLoading(true)
    try {
      // Fetch historical data
      const res = await fetch(`${API_BASE}/api/screener/prices/${symbol}/history?period=${timeframe.toLowerCase()}`)
      const data = await res.json()
      
      if (data.success && data.history) {
        const formattedData = data.history.map((item: any) => ({
          date: new Date(item.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }),
          fullDate: new Date(item.date).toLocaleDateString('en-IN', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' }),
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          volume: item.volume,
          price: item.close
        }))
        setChartData(formattedData)
      }
      
      // Fetch current price
      const priceRes = await fetch(`${API_BASE}/api/screener/prices/${symbol}`)
      const priceData = await priceRes.json()
      if (priceData.success) {
        setStockInfo(priceData)
      }
    } catch (error) {
      console.error('Error fetching chart data:', error)
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
  const minPrice = chartData.length > 0 ? Math.min(...chartData.map(d => d.low || d.price)) * 0.995 : 0
  const maxPrice = chartData.length > 0 ? Math.max(...chartData.map(d => d.high || d.price)) * 1.005 : 0
  
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-gray-900/95 backdrop-blur-sm border border-gray-700 rounded-xl p-4 shadow-2xl">
          <p className="text-gray-400 text-xs mb-2">{data.fullDate}</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <span className="text-gray-500">Open</span>
            <span className="text-white font-medium">₹{data.open?.toLocaleString('en-IN')}</span>
            <span className="text-gray-500">High</span>
            <span className="text-green-400 font-medium">₹{data.high?.toLocaleString('en-IN')}</span>
            <span className="text-gray-500">Low</span>
            <span className="text-red-400 font-medium">₹{data.low?.toLocaleString('en-IN')}</span>
            <span className="text-gray-500">Close</span>
            <span className="text-white font-bold">₹{data.close?.toLocaleString('en-IN')}</span>
            <span className="text-gray-500">Volume</span>
            <span className="text-blue-400">{(data.volume / 1000000).toFixed(2)}M</span>
          </div>
        </div>
      )
    }
    return null
  }
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95 }}
        animate={{ scale: 1 }}
        exit={{ scale: 0.95 }}
        className="w-full max-w-6xl h-[80vh] bg-gray-900 rounded-2xl overflow-hidden border border-gray-800"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <LineChart className="w-5 h-5 text-blue-400" />
              <h2 className="text-xl font-bold text-white">{symbol}</h2>
              <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs rounded-full">NSE</span>
            </div>
            {stockInfo && (
              <div className="flex items-center gap-3">
                <span className="text-xl font-bold text-white">₹{stockInfo.price?.toLocaleString('en-IN')}</span>
                <span className={`text-sm font-medium ${stockInfo.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {stockInfo.change >= 0 ? '+' : ''}{stockInfo.change?.toFixed(2)} ({stockInfo.change_percent?.toFixed(2)}%)
                </span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-3">
            {/* Timeframe Buttons */}
            <div className="flex items-center bg-gray-800/50 rounded-lg p-1">
              {timeframes.map((tf) => (
                <button
                  key={tf.value}
                  onClick={() => setTimeframe(tf.value)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
                    timeframe === tf.value
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                  }`}
                >
                  {tf.label}
                </button>
              ))}
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-800 rounded-lg transition"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>
        </div>
        
        {/* Chart Area */}
        <div className="h-[calc(100%-80px)] p-4">
          {isLoading ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <RefreshCw className="w-10 h-10 text-blue-400 animate-spin mx-auto mb-3" />
                <p className="text-gray-400">Loading chart data...</p>
              </div>
            </div>
          ) : chartData.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <BarChart3 className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-400">No chart data available</p>
                <button
                  onClick={() => fetchChartData()}
                  className="inline-flex items-center gap-2 mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition"
                >
                  <RefreshCw className="w-4 h-4" /> Retry
                </button>
              </div>
            </div>
          ) : (
            <div className="h-full bg-gradient-to-b from-gray-900 to-gray-950 rounded-xl overflow-hidden border border-gray-800">
              <div className={`absolute inset-0 opacity-10 ${isPositive ? 'bg-gradient-to-t from-green-500/20' : 'bg-gradient-to-t from-red-500/20'} to-transparent pointer-events-none`} />
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 20, right: 20, left: 10, bottom: 20 }}>
                  <defs>
                    <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.4}/>
                      <stop offset="50%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.1}/>
                      <stop offset="100%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis 
                    dataKey="date" 
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    interval="preserveStartEnd"
                    dy={10}
                  />
                  <YAxis 
                    domain={[minPrice, maxPrice]}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    tickFormatter={(value) => `₹${value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`}
                    width={70}
                    dx={-5}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area 
                    type="monotone" 
                    dataKey="price" 
                    stroke={isPositive ? "#22c55e" : "#ef4444"}
                    strokeWidth={2.5}
                    fill="url(#chartGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="px-4 pb-4 flex items-center justify-between text-xs text-gray-500 border-t border-gray-800/50 pt-3">
          <span>Real-time NSE data • Powered by yfinance</span>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full ${isPositive ? 'bg-green-500' : 'bg-red-500'}`}></span>
              {isPositive ? 'Bullish' : 'Bearish'} trend
            </span>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}

function NiftyPredictionPanel({ data }: { data: any }) {
  if (!data) return null
  
  const prediction = data.prediction || {}
  
  return (
    <div className="bg-gradient-to-br from-purple-900/30 to-blue-900/30 border border-purple-500/30 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <LineChart className="w-6 h-6 text-purple-400" />
        <h3 className="text-lg font-bold text-white">AI Nifty Outlook</h3>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Current Level</div>
          <div className="text-2xl font-bold text-white">{data.current_level?.toLocaleString()}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Direction</div>
          <div className={`text-2xl font-bold ${prediction.direction === 'BULLISH' ? 'text-green-400' : prediction.direction === 'BEARISH' ? 'text-red-400' : 'text-yellow-400'}`}>
            {prediction.direction} {prediction.direction === 'BULLISH' ? '↑' : prediction.direction === 'BEARISH' ? '↓' : '→'}
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Confidence</div>
          <div className="text-xl font-bold text-white">{prediction.confidence?.toFixed(1) || 0}%</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-1">Change</div>
          <div className={`text-xl font-bold ${(data.change_percent || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {data.change_percent?.toFixed(2)}%
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// MAIN PAGE
// ============================================================================

export default function ScreenerPage() {
  // Scanner state
  const [categories, setCategories] = useState<{ [key: string]: any }>({})
  const [totalScanners, setTotalScanners] = useState(0)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedScanner, setSelectedScanner] = useState<any | null>(null)
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<string | null>(null)
  
  // Watchlist state
  const [watchlist, setWatchlist] = useState<string[]>([])
  const [userId] = useState('ffb9e2ca-6733-4e84-9286-0aa134e6f57e') // Test user - replace with real auth
  
  // Chart state
  const [chartSymbol, setChartSymbol] = useState<string | null>(null)
  
  // AI state
  const [activeTab, setActiveTab] = useState<'scanners' | 'ai'>('scanners')
  const [aiData, setAiData] = useState<any>(null)
  const [aiLoading, setAiLoading] = useState(false)
  
  // Real-time price updates
  const [priceUpdateTime, setPriceUpdateTime] = useState<Date | null>(null)

  // Fetch categories on mount
  useEffect(() => {
    fetchCategories()
    fetchWatchlist()
    
    // Auto-refresh AI Swing on load
    runSwingCandidates()
  }, [])
  
  // Real-time price polling every 30 seconds
  useEffect(() => {
    if (results.length === 0) return
    
    const interval = setInterval(() => {
      refreshPrices()
    }, 30000) // Update every 30 seconds
    
    return () => clearInterval(interval)
  }, [results])
  
  const refreshPrices = async () => {
    if (results.length === 0) return
    
    try {
      const symbols = results.map(r => r.symbol).join(',')
      const res = await fetch(`${API_BASE}/api/screener/prices/live?symbols=${symbols}`)
      const data = await res.json()
      
      if (data.success && data.prices) {
        // Update results with new prices
        setResults(prev => prev.map(stock => {
          const newPrice = data.prices.find((p: any) => p.symbol === stock.symbol)
          if (newPrice) {
            return {
              ...stock,
              current_price: newPrice.price,
              ltp: newPrice.price,
              change_percent: newPrice.change_percent,
              change: newPrice.change,
            }
          }
          return stock
        }))
        setPriceUpdateTime(new Date())
      }
    } catch (error) {
      console.error('Error refreshing prices:', error)
    }
  }

  const fetchCategories = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/screener/pk/categories`)
      const data = await res.json()
      
      if (data.success) {
        setCategories(data.categories || {})
        setTotalScanners(data.total_scanners || 0)
      }
    } catch (error) {
      console.error('Error fetching categories:', error)
    }
  }

  const fetchWatchlist = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/watchlist/${userId}`)
      const data = await res.json()
      
      if (data.success) {
        setWatchlist(data.watchlist?.map((item: any) => item.symbol) || [])
      }
    } catch (error) {
      console.error('Error fetching watchlist:', error)
    }
  }

  const runScan = async (scanner: any) => {
    setSelectedScanner(scanner)
    setLoading(true)
    setResults([])
    
    try {
      const res = await fetch(
        `${API_BASE}/api/screener/pk/scan/batch?scanner_id=${scanner.id}&universe=nifty50&limit=50`,
        { method: 'POST' }
      )
      const data = await res.json()
      
      if (data.success) {
        setResults(data.results || [])
        setLastUpdated(data.timestamp || new Date().toISOString())
      }
    } catch (error) {
      console.error('Scan failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const runSwingCandidates = async () => {
    setSelectedScanner({ id: 'ai', name: 'AI Swing Candidates' })
    setLoading(true)
    setResults([])
    
    try {
      const res = await fetch(`${API_BASE}/api/screener/swing-candidates?limit=30`)
      const data = await res.json()
      
      if (data.success) {
        setResults(data.results || [])
        setLastUpdated(data.timestamp || new Date().toISOString())
      }
    } catch (error) {
      console.error('Swing candidates failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const runAIFeature = async (feature: any) => {
    setAiLoading(true)
    setAiData(null)
    
    try {
      const res = await fetch(`${API_BASE}${feature.endpoint}`)
      const data = await res.json()
      
      if (data.success) {
        setAiData({ type: feature.id, ...data })
      }
    } catch (error) {
      console.error('AI feature failed:', error)
    } finally {
      setAiLoading(false)
    }
  }

  const addToWatchlist = async (symbol: string) => {
    if (watchlist.includes(symbol)) {
      // Remove from watchlist
      try {
        await fetch(`${API_BASE}/api/watchlist/${userId}/${symbol}`, { method: 'DELETE' })
        setWatchlist(prev => prev.filter(s => s !== symbol))
      } catch (error) {
        console.error('Error removing from watchlist:', error)
      }
    } else {
      // Add to watchlist
      try {
        await fetch(`${API_BASE}/api/watchlist/add`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, symbol })
        })
        setWatchlist(prev => [...prev, symbol])
      } catch (error) {
        console.error('Error adding to watchlist:', error)
      }
    }
  }

  const currentCategory = selectedCategory ? categories[selectedCategory] : null

  return (
    <div className="min-h-screen bg-black text-white" data-testid="screener-page">
      {/* Background */}
      <div className="fixed inset-0 bg-gradient-to-b from-gray-900/50 via-black to-black pointer-events-none" />
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-green-900/10 via-transparent to-transparent pointer-events-none" />
      
      {/* Header */}
      <header className="sticky top-0 z-40 bg-black/80 backdrop-blur-xl border-b border-gray-800">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <Link href="/" className="flex items-center gap-2 text-gray-400 hover:text-white">
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600">
                  <Search className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold">AI Market Screener</h1>
                  <p className="text-xs text-gray-500">{totalScanners} Scanners • PKScreener Powered</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Link href="/watchlist" className="flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm">
                <Bookmark className="w-4 h-4 text-yellow-400" />
                <span className="hidden sm:inline">Watchlist</span>
                {watchlist.length > 0 && (
                  <span className="px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 text-xs rounded-full">
                    {watchlist.length}
                  </span>
                )}
              </Link>
              <Link href="/dashboard" className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 rounded-lg text-sm font-medium">
                Dashboard
              </Link>
            </div>
          </div>
        </div>
      </header>
      
      {/* Stats Bar */}
      <div className="border-b border-gray-800 bg-gray-900/30">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-green-500" />
                <span className="text-gray-400">Scanners:</span>
                <span className="font-semibold text-white">{totalScanners}</span>
              </div>
              <div className="flex items-center gap-2">
                <Globe2 className="w-4 h-4 text-blue-500" />
                <span className="text-gray-400">Categories:</span>
                <span className="font-semibold text-white">{Object.keys(categories).length}</span>
              </div>
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-purple-500" />
                <span className="text-gray-400">Universe:</span>
                <span className="font-semibold text-white">NSE 2200+</span>
              </div>
            </div>
            
            <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-1">
              <button
                onClick={() => setActiveTab('scanners')}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'scanners' ? 'bg-green-500 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                Scanners
              </button>
              <button
                onClick={() => setActiveTab('ai')}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all flex items-center gap-1 ${
                  activeTab === 'ai' ? 'bg-purple-500 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                <Sparkles className="w-3 h-3" />
                AI Intelligence
              </button>
            </div>
          </div>
        </div>
      </div>
      
      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        <div className="grid lg:grid-cols-12 gap-6">
          
          {/* Sidebar */}
          <div className="lg:col-span-3">
            <div className="sticky top-24 space-y-4">
              {activeTab === 'scanners' ? (
                <>
                  {/* AI Quick Scan */}
                  <motion.button
                    onClick={runSwingCandidates}
                    whileHover={{ scale: 1.02 }}
                    className="w-full p-4 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 text-white text-left"
                    data-testid="ai-swing-btn"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-white/20">
                        <Sparkles className="w-5 h-5" />
                      </div>
                      <div>
                        <div className="font-bold">AI Swing Candidates</div>
                        <div className="text-xs text-white/70">Best setups ranked by AI</div>
                      </div>
                    </div>
                  </motion.button>
                  
                  {/* Dynamic Categories from API */}
                  <div className="space-y-2">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase px-1">
                      Scanner Categories ({totalScanners} scanners)
                    </h3>
                    {Object.entries(categories).map(([catKey, category]: [string, any]) => {
                      const Icon = CATEGORY_ICONS[catKey] || Target
                      const colors = CATEGORY_COLORS[catKey] || CATEGORY_COLORS.technical
                      
                      return (
                        <button
                          key={catKey}
                          onClick={() => setSelectedCategory(selectedCategory === catKey ? null : catKey)}
                          className={`w-full p-3 rounded-xl text-left transition-all ${
                            selectedCategory === catKey 
                              ? `bg-gradient-to-br ${colors.gradient} border-2 border-white/20`
                              : `${colors.bg} border ${colors.border} hover:border-white/20`
                          }`}
                          data-testid={`category-${catKey}`}
                        >
                          <div className="flex items-center gap-3">
                            <Icon className={`w-5 h-5 ${selectedCategory === catKey ? 'text-white' : colors.text}`} />
                            <div className="flex-1">
                              <div className={`font-semibold ${selectedCategory === catKey ? 'text-white' : 'text-white'}`}>
                                {category.name}
                              </div>
                              <div className={`text-xs ${selectedCategory === catKey ? 'text-white/70' : 'text-gray-500'}`}>
                                {category.scanners?.length || 0} scanners
                              </div>
                            </div>
                            <ChevronRight className={`w-4 h-4 transition-transform ${
                              selectedCategory === catKey ? 'text-white rotate-90' : 'text-gray-500'
                            }`} />
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </>
              ) : (
                <>
                  <h3 className="text-xs font-semibold text-gray-500 uppercase px-1">AI Intelligence Tools</h3>
                  {AI_FEATURES.map(feature => {
                    const Icon = feature.icon
                    return (
                      <motion.button
                        key={feature.id}
                        onClick={() => runAIFeature(feature)}
                        disabled={aiLoading}
                        whileHover={{ scale: 1.02 }}
                        className="w-full p-4 rounded-xl bg-gradient-to-br from-purple-600/20 to-blue-600/20 border border-purple-500/30 text-left hover:border-purple-400/50 transition-all disabled:opacity-50"
                      >
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg bg-purple-500/20">
                            <Icon className="w-5 h-5 text-purple-400" />
                          </div>
                          <div>
                            <div className="font-semibold text-white">{feature.name}</div>
                            <div className="text-xs text-gray-400">{feature.description}</div>
                          </div>
                          {aiLoading && <RefreshCw className="w-4 h-4 text-purple-400 animate-spin ml-auto" />}
                        </div>
                      </motion.button>
                    )
                  })}
                </>
              )}
            </div>
          </div>
          
          {/* Scanner List */}
          <AnimatePresence mode="wait">
            {selectedCategory && currentCategory && (
              <motion.div
                key="scanner-list"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="lg:col-span-3"
              >
                <div className="sticky top-24 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-gray-400">{currentCategory.name}</h3>
                    <button onClick={() => setSelectedCategory(null)} className="p-1 hover:bg-gray-800 rounded">
                      <X className="w-4 h-4 text-gray-500" />
                    </button>
                  </div>
                  
                  <div className="space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto pr-2">
                    {currentCategory.scanners?.map((scanner: any) => (
                      <button
                        key={scanner.id}
                        onClick={() => runScan(scanner)}
                        className={`w-full p-3 rounded-lg text-left transition-all ${
                          selectedScanner?.id === scanner.id
                            ? 'bg-blue-500/20 border border-blue-500/50'
                            : 'bg-gray-800/50 hover:bg-gray-700/50'
                        }`}
                        data-testid={`scanner-${scanner.id}`}
                      >
                        <div className="flex items-center gap-3">
                          <div className="flex-1">
                            <span className={`font-medium ${selectedScanner?.id === scanner.id ? 'text-blue-400' : 'text-white'}`}>
                              {scanner.name}
                            </span>
                            {scanner.menu_code && (
                              <p className="text-xs text-gray-500 mt-0.5">{scanner.menu_code}</p>
                            )}
                          </div>
                          <Play className="w-4 h-4 text-gray-400" />
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Results */}
          <div className={`${selectedCategory ? 'lg:col-span-6' : 'lg:col-span-9'}`}>
            {activeTab === 'ai' && aiData ? (
              <NiftyPredictionPanel data={aiData} />
            ) : (
              <>
                {/* Results Header */}
                <div className="flex items-center justify-between mb-4">
                  <div>
                    {selectedScanner ? (
                      <div>
                        <h2 className="text-xl font-bold text-white">{selectedScanner.name}</h2>
                        <div className="flex items-center gap-3 text-sm text-gray-500 mt-1">
                          <span>{results.length} stocks found</span>
                          {lastUpdated && (
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {new Date(lastUpdated).toLocaleTimeString()}
                            </span>
                          )}
                          {priceUpdateTime && (
                            <span className="flex items-center gap-1 text-green-400">
                              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                              Live • {priceUpdateTime.toLocaleTimeString()}
                            </span>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div>
                        <h2 className="text-xl font-bold text-white">AI Swing Candidates</h2>
                        <p className="text-sm text-gray-500 mt-1">{results.length} stocks • Real-time prices</p>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <button
                      onClick={refreshPrices}
                      disabled={loading || results.length === 0}
                      className="flex items-center gap-2 px-3 py-2 bg-green-500/20 hover:bg-green-500/30 rounded-lg text-sm text-green-400 disabled:opacity-50"
                      title="Refresh prices"
                    >
                      <RefreshCw className="w-4 h-4" />
                      Live
                    </button>
                    {selectedScanner && (
                      <button
                        onClick={() => runScan(selectedScanner)}
                        disabled={loading}
                        className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm disabled:opacity-50"
                      >
                        <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                        Rescan
                      </button>
                    )}
                  </div>
                </div>
                
                {/* Loading */}
                {loading && (
                  <div className="flex flex-col items-center justify-center py-20">
                    <div className="relative">
                      <div className="w-16 h-16 border-4 border-gray-800 border-t-green-500 rounded-full animate-spin" />
                      <Search className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 text-green-500" />
                    </div>
                    <p className="mt-4 text-gray-400">Scanning stocks...</p>
                  </div>
                )}
                
                {/* Empty State */}
                {!loading && !selectedScanner && (
                  <div className="flex flex-col items-center justify-center py-20 text-center">
                    <div className="p-4 rounded-2xl bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/20 mb-4">
                      <Search className="w-12 h-12 text-green-500" />
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">Ready to Scan</h3>
                    <p className="text-gray-400 max-w-md mb-4">
                      Access {totalScanners} professional scanners including breakouts, momentum, ML signals, and more.
                    </p>
                  </div>
                )}
                
                {/* Results Grid */}
                {!loading && results.length > 0 && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                    {results.map((stock, index) => (
                      <StockCard
                        key={stock.symbol}
                        stock={stock}
                        index={index}
                        onAddToWatchlist={addToWatchlist}
                        onViewChart={setChartSymbol}
                        isInWatchlist={watchlist.includes(stock.symbol)}
                      />
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
      
      {/* Stock Chart Modal */}
      <AnimatePresence>
        {chartSymbol && (
          <StockChartModal symbol={chartSymbol} onClose={() => setChartSymbol(null)} />
        )}
      </AnimatePresence>
      
      {/* Footer */}
      <footer className="border-t border-gray-800 mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Search className="w-4 h-4 text-green-500" />
              <span className="text-sm text-gray-400">
                AI Market Screener • PKScreener Powered • {totalScanners} Scanners
              </span>
            </div>
            <div className="flex items-center gap-6 text-sm text-gray-500">
              <Link href="/watchlist" className="hover:text-white flex items-center gap-1">
                <Bookmark className="w-3 h-3" />
                Watchlist
              </Link>
              <Link href="/dashboard" className="hover:text-white flex items-center gap-1">
                Dashboard <ChevronRight className="w-3 h-3" />
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
