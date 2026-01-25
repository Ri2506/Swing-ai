'use client'

// ============================================================================
// ADVANCED STOCK CHART - Cutting Edge Real-time Trading Chart
// Features: WebSocket real-time updates, candlestick, area, volume bars
// ============================================================================

import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ComposedChart, Bar, Line, ReferenceLine, CartesianGrid
} from 'recharts'
import {
  TrendingUp, TrendingDown, RefreshCw, Clock, Zap,
  Activity, BarChart3, LineChart, CandlestickChart,
  Maximize2, Volume2, Target, AlertCircle
} from 'lucide-react'

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.REACT_APP_BACKEND_URL || ''
const WS_BASE = API_BASE.replace('https://', 'wss://').replace('http://', 'ws://')

interface ChartDataPoint {
  date: string
  fullDate: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  price: number
  ma20?: number
  ma50?: number
}

interface StockInfo {
  symbol: string
  price: number
  change: number
  change_percent: number
  volume?: number
  high?: number
  low?: number
  open?: number
}

interface AdvancedStockChartProps {
  symbol: string
  showHeader?: boolean
  height?: string
  onClose?: () => void
  isModal?: boolean
}

export default function AdvancedStockChart({ 
  symbol, 
  showHeader = true, 
  height = "500px",
  onClose,
  isModal = false
}: AdvancedStockChartProps) {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [timeframe, setTimeframe] = useState('1M')
  const [chartType, setChartType] = useState<'area' | 'candle' | 'line'>('area')
  const [showVolume, setShowVolume] = useState(true)
  const [showMA, setShowMA] = useState(false)
  const [stockInfo, setStockInfo] = useState<StockInfo | null>(null)
  const [isLive, setIsLive] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Calculate moving averages
  const calculateMA = (data: ChartDataPoint[], period: number) => {
    return data.map((item, index) => {
      if (index < period - 1) return null
      const slice = data.slice(index - period + 1, index + 1)
      const sum = slice.reduce((acc, curr) => acc + curr.close, 0)
      return sum / period
    })
  }

  // Fetch chart data
  const fetchChartData = useCallback(async () => {
    setIsLoading(true)
    try {
      const periodMap: Record<string, string> = {
        '1D': '1d',
        '1W': '1w', 
        '1M': '1mo',
        '3M': '3mo',
        '6M': '6mo',
        '1Y': '1y',
        'ALL': 'max'
      }
      
      const res = await fetch(`${API_BASE}/api/screener/prices/${symbol}/history?period=${periodMap[timeframe] || '1mo'}`)
      const data = await res.json()
      
      if (data.success && data.history) {
        const formattedData: ChartDataPoint[] = data.history.map((item: any) => ({
          date: new Date(item.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }),
          fullDate: new Date(item.date).toLocaleDateString('en-IN', { 
            weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' 
          }),
          open: parseFloat(item.open?.toFixed(2)) || 0,
          high: parseFloat(item.high?.toFixed(2)) || 0,
          low: parseFloat(item.low?.toFixed(2)) || 0,
          close: parseFloat(item.close?.toFixed(2)) || 0,
          volume: item.volume || 0,
          price: parseFloat(item.close?.toFixed(2)) || 0
        }))
        
        // Add moving averages
        const ma20 = calculateMA(formattedData, 20)
        const ma50 = calculateMA(formattedData, 50)
        
        formattedData.forEach((item, idx) => {
          item.ma20 = ma20[idx] ? parseFloat(ma20[idx]!.toFixed(2)) : undefined
          item.ma50 = ma50[idx] ? parseFloat(ma50[idx]!.toFixed(2)) : undefined
        })
        
        setChartData(formattedData)
      }
      
      // Fetch current price
      const priceRes = await fetch(`${API_BASE}/api/screener/prices/${symbol}`)
      const priceData = await priceRes.json()
      if (priceData.success) {
        setStockInfo({
          symbol,
          price: priceData.price,
          change: priceData.change,
          change_percent: priceData.change_percent,
          volume: priceData.volume,
          high: priceData.high,
          low: priceData.low,
          open: priceData.open
        })
      }
      
      setLastUpdate(new Date())
    } catch (error) {
      console.error('Error fetching chart data:', error)
    }
    setIsLoading(false)
  }, [symbol, timeframe])

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    
    try {
      const clientId = `chart_${symbol}_${Date.now()}`
      wsRef.current = new WebSocket(`${WS_BASE}/ws/prices/${clientId}`)
      
      wsRef.current.onopen = () => {
        setIsLive(true)
        // Subscribe to symbol
        wsRef.current?.send(JSON.stringify({
          action: 'subscribe',
          symbols: [symbol]
        }))
      }
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'price_update' && data.prices?.[symbol]) {
            const priceUpdate = data.prices[symbol]
            setStockInfo(prev => ({
              ...prev!,
              price: priceUpdate.price,
              change: priceUpdate.change,
              change_percent: priceUpdate.change_percent
            }))
            setLastUpdate(new Date())
            
            // Update last candle in chart
            setChartData(prev => {
              if (prev.length === 0) return prev
              const updated = [...prev]
              const lastCandle = { ...updated[updated.length - 1] }
              lastCandle.close = priceUpdate.price
              lastCandle.price = priceUpdate.price
              lastCandle.high = Math.max(lastCandle.high, priceUpdate.price)
              lastCandle.low = Math.min(lastCandle.low, priceUpdate.price)
              updated[updated.length - 1] = lastCandle
              return updated
            })
          }
        } catch (e) {
          console.error('WebSocket message error:', e)
        }
      }
      
      wsRef.current.onclose = () => {
        setIsLive(false)
        // Reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 5000)
      }
      
      wsRef.current.onerror = () => {
        setIsLive(false)
      }
    } catch (error) {
      console.error('WebSocket connection error:', error)
      setIsLive(false)
    }
  }, [symbol])

  useEffect(() => {
    fetchChartData()
    connectWebSocket()
    
    // Refresh data every 30 seconds as backup
    const interval = setInterval(fetchChartData, 30000)
    
    return () => {
      clearInterval(interval)
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [fetchChartData, connectWebSocket])

  const timeframes = [
    { label: '1D', value: '1D' },
    { label: '1W', value: '1W' },
    { label: '1M', value: '1M' },
    { label: '3M', value: '3M' },
    { label: '6M', value: '6M' },
    { label: '1Y', value: '1Y' },
  ]

  const isPositive = stockInfo ? stockInfo.change >= 0 : (chartData.length > 1 ? chartData[chartData.length - 1]?.close >= chartData[0]?.close : true)
  const minPrice = chartData.length > 0 ? Math.min(...chartData.map(d => d.low)) * 0.995 : 0
  const maxPrice = chartData.length > 0 ? Math.max(...chartData.map(d => d.high)) * 1.005 : 0
  const maxVolume = chartData.length > 0 ? Math.max(...chartData.map(d => d.volume)) : 0

  // Calculate period change
  const periodChange = chartData.length > 1 
    ? ((chartData[chartData.length - 1].close - chartData[0].close) / chartData[0].close * 100)
    : 0

  // Custom Tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-gray-900/98 backdrop-blur-xl border border-gray-700/50 rounded-2xl p-4 shadow-2xl min-w-[200px]"
        >
          <p className="text-gray-400 text-xs mb-3 font-medium">{data.fullDate}</p>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-gray-500 text-sm">Open</span>
              <span className="text-white font-semibold">₹{data.open?.toLocaleString('en-IN')}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500 text-sm">High</span>
              <span className="text-green-400 font-semibold">₹{data.high?.toLocaleString('en-IN')}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500 text-sm">Low</span>
              <span className="text-red-400 font-semibold">₹{data.low?.toLocaleString('en-IN')}</span>
            </div>
            <div className="flex justify-between items-center border-t border-gray-700/50 pt-2 mt-2">
              <span className="text-gray-500 text-sm">Close</span>
              <span className="text-white font-bold text-lg">₹{data.close?.toLocaleString('en-IN')}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500 text-sm">Volume</span>
              <span className="text-blue-400 font-medium">{(data.volume / 1000000).toFixed(2)}M</span>
            </div>
          </div>
        </motion.div>
      )
    }
    return null
  }

  // Candlestick shape for recharts
  const CandlestickShape = (props: any) => {
    const { x, y, width, payload, yAxis } = props
    if (!payload || !yAxis) return null
    
    const { open, close, high, low } = payload
    const scale = yAxis.scale
    if (!scale) return null
    
    const isUp = close >= open
    const color = isUp ? '#22c55e' : '#ef4444'
    const candleWidth = Math.max(width * 0.6, 2)
    const candleX = x + (width - candleWidth) / 2
    
    const yOpen = scale(open)
    const yClose = scale(close)
    const yHigh = scale(high)
    const yLow = scale(low)
    
    const bodyTop = Math.min(yOpen, yClose)
    const bodyHeight = Math.abs(yClose - yOpen) || 1
    
    return (
      <g>
        {/* Wick */}
        <line
          x1={x + width / 2}
          y1={yHigh}
          x2={x + width / 2}
          y2={yLow}
          stroke={color}
          strokeWidth={1}
        />
        {/* Body */}
        <rect
          x={candleX}
          y={bodyTop}
          width={candleWidth}
          height={bodyHeight}
          fill={isUp ? color : color}
          stroke={color}
          strokeWidth={1}
          rx={1}
        />
      </g>
    )
  }

  const chartContent = (
    <div className={`relative ${isModal ? '' : 'rounded-2xl border border-gray-800/50'} bg-gradient-to-b from-gray-900/95 to-gray-950/95 backdrop-blur-xl overflow-hidden`}>
      {/* Ambient glow effect */}
      <div className={`absolute inset-0 opacity-20 pointer-events-none ${isPositive ? 'bg-gradient-to-t from-green-500/10 via-transparent' : 'bg-gradient-to-t from-red-500/10 via-transparent'}`} />
      
      {/* Header */}
      {showHeader && (
        <div className="relative z-10 px-6 py-4 border-b border-gray-800/50">
          <div className="flex items-center justify-between">
            {/* Stock Info */}
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${isPositive ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
                  <Activity className={`w-5 h-5 ${isPositive ? 'text-green-400' : 'text-red-400'}`} />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-xl font-bold text-white">{symbol}</h2>
                    <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs rounded-full font-medium">NSE</span>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    {isLive ? (
                      <span className="flex items-center gap-1 text-xs text-green-400">
                        <span className="relative flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                        </span>
                        LIVE
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-xs text-gray-500">
                        <span className="w-2 h-2 rounded-full bg-gray-600"></span>
                        Delayed
                      </span>
                    )}
                    {lastUpdate && (
                      <span className="text-xs text-gray-500">
                        {lastUpdate.toLocaleTimeString('en-IN')}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              
              {stockInfo && (
                <div className="flex items-center gap-4 pl-6 border-l border-gray-800">
                  <div>
                    <span className="text-2xl font-bold text-white">
                      ₹{stockInfo.price?.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </span>
                    <div className={`flex items-center gap-1 text-sm font-semibold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                      {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                      {isPositive ? '+' : ''}{stockInfo.change?.toFixed(2)} ({stockInfo.change_percent?.toFixed(2)}%)
                    </div>
                  </div>
                  <div className={`px-3 py-1.5 rounded-lg text-sm font-medium ${periodChange >= 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                    {periodChange >= 0 ? '+' : ''}{periodChange.toFixed(2)}% ({timeframe})
                  </div>
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="flex items-center gap-3">
              {/* Chart Type */}
              <div className="flex items-center bg-gray-800/50 rounded-xl p-1">
                <button
                  onClick={() => setChartType('area')}
                  className={`p-2 rounded-lg transition-all ${chartType === 'area' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                  title="Area Chart"
                >
                  <BarChart3 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setChartType('line')}
                  className={`p-2 rounded-lg transition-all ${chartType === 'line' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                  title="Line Chart"
                >
                  <LineChart className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setChartType('candle')}
                  className={`p-2 rounded-lg transition-all ${chartType === 'candle' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                  title="Candlestick"
                >
                  <CandlestickChart className="w-4 h-4" />
                </button>
              </div>

              {/* Indicators */}
              <div className="flex items-center gap-1 bg-gray-800/50 rounded-xl p-1">
                <button
                  onClick={() => setShowVolume(!showVolume)}
                  className={`p-2 rounded-lg transition-all ${showVolume ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'}`}
                  title="Volume"
                >
                  <Volume2 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setShowMA(!showMA)}
                  className={`p-2 rounded-lg transition-all ${showMA ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'}`}
                  title="Moving Averages"
                >
                  <Target className="w-4 h-4" />
                </button>
              </div>

              {/* Timeframes */}
              <div className="flex items-center bg-gray-800/50 rounded-xl p-1">
                {timeframes.map((tf) => (
                  <button
                    key={tf.value}
                    onClick={() => setTimeframe(tf.value)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                      timeframe === tf.value
                        ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25'
                        : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                    }`}
                  >
                    {tf.label}
                  </button>
                ))}
              </div>

              {/* Refresh */}
              <button
                onClick={fetchChartData}
                className="p-2 bg-gray-800/50 hover:bg-gray-700 rounded-xl transition-all"
                title="Refresh"
              >
                <RefreshCw className={`w-4 h-4 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
              </button>

              {onClose && (
                <button
                  onClick={onClose}
                  className="p-2 bg-gray-800/50 hover:bg-red-500/20 hover:text-red-400 rounded-xl transition-all text-gray-400"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Chart Area */}
      <div className="relative" style={{ height }}>
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="relative">
                <div className="w-16 h-16 border-4 border-blue-500/20 rounded-full animate-pulse"></div>
                <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-t-blue-500 rounded-full animate-spin"></div>
              </div>
              <p className="text-gray-400 mt-4 text-sm">Loading chart data...</p>
            </div>
          </div>
        ) : chartData.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <AlertCircle className="w-12 h-12 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-400">No chart data available</p>
              <button
                onClick={fetchChartData}
                className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition flex items-center gap-2 mx-auto"
              >
                <RefreshCw className="w-4 h-4" /> Retry
              </button>
            </div>
          </div>
        ) : (
          <div className="h-full p-4">
            <ResponsiveContainer width="100%" height={showVolume ? "75%" : "100%"}>
              {chartType === 'candle' ? (
                <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.3}/>
                      <stop offset="100%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" opacity={0.5} />
                  <XAxis 
                    dataKey="date" 
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis 
                    domain={[minPrice, maxPrice]}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    tickFormatter={(value) => `₹${value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`}
                    width={70}
                    orientation="right"
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar 
                    dataKey="close" 
                    shape={<CandlestickShape />}
                    isAnimationActive={false}
                  />
                  {showMA && (
                    <>
                      <Line type="monotone" dataKey="ma20" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="MA20" />
                      <Line type="monotone" dataKey="ma50" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="MA50" />
                    </>
                  )}
                </ComposedChart>
              ) : (
                <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.4}/>
                      <stop offset="50%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.1}/>
                      <stop offset="100%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" opacity={0.3} />
                  <XAxis 
                    dataKey="date" 
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis 
                    domain={[minPrice, maxPrice]}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#6b7280', fontSize: 11 }}
                    tickFormatter={(value) => `₹${value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`}
                    width={70}
                    orientation="right"
                  />
                  <Tooltip content={<CustomTooltip />} />
                  {chartType === 'area' ? (
                    <Area 
                      type="monotone" 
                      dataKey="price" 
                      stroke={isPositive ? "#22c55e" : "#ef4444"}
                      strokeWidth={2.5}
                      fill="url(#colorPrice)"
                    />
                  ) : (
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke={isPositive ? "#22c55e" : "#ef4444"}
                      strokeWidth={2.5}
                      dot={false}
                    />
                  )}
                  {showMA && (
                    <>
                      <Line type="monotone" dataKey="ma20" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="MA20" />
                      <Line type="monotone" dataKey="ma50" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="MA50" />
                    </>
                  )}
                </AreaChart>
              )}
            </ResponsiveContainer>
            
            {/* Volume Chart */}
            {showVolume && (
              <ResponsiveContainer width="100%" height="22%">
                <ComposedChart data={chartData} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                  <XAxis dataKey="date" hide />
                  <YAxis 
                    domain={[0, maxVolume * 1.1]}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#6b7280', fontSize: 10 }}
                    tickFormatter={(value) => `${(value / 1000000).toFixed(0)}M`}
                    width={70}
                    orientation="right"
                  />
                  <Tooltip 
                    formatter={(value: number) => [`${(value / 1000000).toFixed(2)}M`, 'Volume']}
                    contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                    labelStyle={{ color: '#9ca3af' }}
                  />
                  <Bar 
                    dataKey="volume" 
                    fill="#3b82f6"
                    opacity={0.6}
                    radius={[2, 2, 0, 0]}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="relative z-10 px-6 py-3 border-t border-gray-800/50 flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <Zap className="w-3 h-3 text-yellow-500" />
            Real-time NSE data
          </span>
          {showMA && (
            <span className="flex items-center gap-2">
              <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-amber-500 rounded"></span> MA20</span>
              <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-purple-500 rounded"></span> MA50</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <span className={`flex items-center gap-1.5 ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
            {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {isPositive ? 'Bullish' : 'Bearish'} Trend
          </span>
        </div>
      </div>
    </div>
  )

  if (isModal) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="w-full max-w-7xl"
          onClick={(e) => e.stopPropagation()}
        >
          {chartContent}
        </motion.div>
      </motion.div>
    )
  }

  return chartContent
}
