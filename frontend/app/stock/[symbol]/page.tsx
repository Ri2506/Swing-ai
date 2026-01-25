// ============================================================================
// STOCK DETAIL PAGE - Full Analysis with Advanced Real-time Charts
// ============================================================================

'use client'

import { useState, useEffect, useRef } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import {
  ArrowLeft, TrendingUp, TrendingDown, Activity, BarChart3,
  Bookmark, BookmarkCheck, RefreshCw, Clock,
  ArrowUpRight, ArrowDownRight, Target,
  LineChart, Layers, Zap
} from 'lucide-react'
import AdvancedStockChart from '@/components/AdvancedStockChart'

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
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'price_update' && data.prices?.[symbol]) {
            const priceUpdate = data.prices[symbol]
            setStockData(prev => prev ? {
              ...prev,
              price: priceUpdate.price,
              change: priceUpdate.change,
              change_percent: priceUpdate.change_percent
            } : null)
            setLastUpdate(new Date())
          }
        } catch (e) {
          // Ignore parse errors
        }
      }
      
      ws.onclose = () => {
        setWsConnected(false)
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000)
      }
      
      ws.onerror = () => {
        setWsConnected(false)
      }
      
      wsRef.current = ws
    } catch (error) {
      console.error('WebSocket connection error:', error)
    }
  }

  const fetchStockData = async () => {
    try {
      // Fetch price data
      const priceRes = await fetch(`${API_BASE}/api/screener/prices/${symbol}`)
      const priceData = await priceRes.json()
      
      if (priceData.success) {
        setStockData({
          symbol,
          name: priceData.name || symbol,
          price: priceData.price,
          change: priceData.change,
          change_percent: priceData.change_percent,
          open: priceData.open,
          high: priceData.high,
          low: priceData.low,
          volume: priceData.volume,
          prev_close: priceData.prev_close,
          week_52_high: priceData.week_52_high,
          week_52_low: priceData.week_52_low,
          market_cap: priceData.market_cap,
          pe_ratio: priceData.pe_ratio,
          sector: priceData.sector,
          industry: priceData.industry
        })
        setLastUpdate(new Date())
      }
      
      // Fetch technicals
      const techRes = await fetch(`${API_BASE}/api/screener/technicals/${symbol}`)
      const techData = await techRes.json()
      
      if (techData.success) {
        setTechnicals({
          rsi: techData.rsi,
          macd: techData.macd,
          macd_signal: techData.macd_signal,
          sma_20: techData.sma_20,
          sma_50: techData.sma_50,
          sma_200: techData.sma_200,
          trend: techData.trend,
          volume_ratio: techData.volume_ratio
        })
      }
    } catch (error) {
      console.error('Error fetching stock data:', error)
    }
    setLoading(false)
  }

  const checkWatchlist = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/watchlist/${userId}`)
      const data = await res.json()
      if (data.success && data.watchlist) {
        setIsInWatchlist(data.watchlist.some((item: any) => item.symbol === symbol))
      }
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
        await fetch(`${API_BASE}/api/watchlist/${userId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol, notes: '' })
        })
        setIsInWatchlist(true)
      }
    } catch (error) {
      console.error('Error toggling watchlist:', error)
    }
  }

  const isPositive = (stockData?.change || 0) >= 0

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-20 h-20 mx-auto">
            <div className="absolute inset-0 border-4 border-blue-500/20 rounded-full"></div>
            <div className="absolute inset-0 border-4 border-transparent border-t-blue-500 rounded-full animate-spin"></div>
          </div>
          <p className="text-gray-400 mt-4">Loading {symbol}...</p>
        </div>
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
        {/* Quick Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
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
          <IndicatorCard 
            label="Day Range" 
            value={`₹${stockData?.low?.toFixed(0)} - ₹${stockData?.high?.toFixed(0)}`} 
            subValue={`Open: ₹${stockData?.open?.toFixed(2)}`}
            icon={Target}
          />
        </div>
        
        {/* Advanced Stock Chart */}
        <div className="mb-6">
          <AdvancedStockChart 
            symbol={symbol} 
            showHeader={true}
            height="450px"
          />
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
              <div className="flex items-center justify-between">
                <span className="text-gray-400">SMA 200</span>
                <span className="font-medium">₹{technicals?.sma_200?.toLocaleString('en-IN') || 'N/A'}</span>
              </div>
            </div>
          </div>
          
          {/* MACD & Momentum */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-400" />
              Momentum Indicators
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">MACD</span>
                <span className={`font-medium ${(technicals?.macd || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {technicals?.macd?.toFixed(2) || 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Signal Line</span>
                <span className="font-medium">{technicals?.macd_signal?.toFixed(2) || 'N/A'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">RSI (14)</span>
                <span className={`font-medium ${
                  technicals?.rsi && technicals.rsi < 30 ? 'text-green-400' : 
                  technicals?.rsi && technicals.rsi > 70 ? 'text-red-400' : 'text-white'
                }`}>
                  {technicals?.rsi?.toFixed(2) || 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Trend</span>
                <span className={`font-medium ${
                  technicals?.trend?.includes('Up') ? 'text-green-400' : 
                  technicals?.trend?.includes('Down') ? 'text-red-400' : 'text-white'
                }`}>
                  {technicals?.trend || 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Volume Ratio</span>
                <span className={`font-medium ${(technicals?.volume_ratio || 1) > 1.5 ? 'text-yellow-400' : 'text-white'}`}>
                  {technicals?.volume_ratio?.toFixed(2) || '1.00'}x
                </span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Additional Info */}
        {(stockData?.sector || stockData?.market_cap) && (
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              Company Info
            </h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              {stockData?.sector && (
                <div>
                  <div className="text-gray-500 text-sm">Sector</div>
                  <div className="font-medium">{stockData.sector}</div>
                </div>
              )}
              {stockData?.industry && (
                <div>
                  <div className="text-gray-500 text-sm">Industry</div>
                  <div className="font-medium">{stockData.industry}</div>
                </div>
              )}
              {stockData?.market_cap && (
                <div>
                  <div className="text-gray-500 text-sm">Market Cap</div>
                  <div className="font-medium">
                    ₹{(stockData.market_cap / 10000000).toFixed(2)} Cr
                  </div>
                </div>
              )}
              {stockData?.pe_ratio && (
                <div>
                  <div className="text-gray-500 text-sm">P/E Ratio</div>
                  <div className="font-medium">{stockData.pe_ratio.toFixed(2)}</div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
