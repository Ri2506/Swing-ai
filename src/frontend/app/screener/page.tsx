// ============================================================================
// SWINGAI - SCREENER PAGE
// PKScreener powered stock screener with 40+ scanners
// Beautiful UI that makes the app look cutting-edge
// ============================================================================

'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuth } from '../../contexts/AuthContext'
import {
  ArrowLeft,
  Search,
  Rocket,
  TrendingUp,
  Activity,
  RefreshCw,
  Building,
  Clock,
  DollarSign,
  BarChart3,
  Play,
  Star,
  Lock,
  ChevronRight,
  Filter,
  Download,
  Plus,
  Bell,
  ExternalLink,
  Zap,
  Target,
  Eye,
  Crown,
} from 'lucide-react'

// ============================================================================
// SCANNER DATA
// ============================================================================

const SCANNER_CATEGORIES = [
  {
    id: 'breakouts',
    name: 'Breakout Scanners',
    icon: Rocket,
    color: 'from-emerald-500 to-green-600',
    bgColor: 'bg-emerald-500/10',
    borderColor: 'border-emerald-500/30',
    textColor: 'text-emerald-400',
    description: 'Stocks breaking out of consolidation',
    scanners: [
      { id: 1, name: 'Probable Breakouts', description: 'Stocks near breakout levels', premium: false },
      { id: 2, name: "Today's Breakouts", description: 'Confirmed breakouts today', premium: false },
      { id: 17, name: '52-Week High Breakout', description: 'Breaking 52-week highs', premium: true },
      { id: 23, name: 'Breaking Out Now', description: 'Real-time breakouts', premium: true },
    ]
  },
  {
    id: 'momentum',
    name: 'Momentum Scanners',
    icon: TrendingUp,
    color: 'from-blue-500 to-cyan-600',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30',
    textColor: 'text-blue-400',
    description: 'High momentum stocks with volume',
    scanners: [
      { id: 5, name: 'RSI Screening', description: 'Overbought/Oversold stocks', premium: false },
      { id: 9, name: 'Volume Gainers', description: 'Unusual volume spikes', premium: false },
      { id: 13, name: 'Bullish RSI & MACD', description: 'Double momentum confirmation', premium: true },
      { id: 31, name: 'High Momentum', description: 'RSI + MFI + CCI combined', premium: true },
    ]
  },
  {
    id: 'patterns',
    name: 'Chart Patterns',
    icon: Activity,
    color: 'from-purple-500 to-violet-600',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30',
    textColor: 'text-purple-400',
    description: 'Classic chart patterns & setups',
    scanners: [
      { id: 14, name: 'NR4 Daily', description: 'Narrow Range compression', premium: false },
      { id: 3, name: 'VCP Patterns', description: 'Volatility Contraction (Minervini)', premium: true },
      { id: 7, name: 'Chart Patterns', description: 'H&S, Cup & Handle, Triangles', premium: true },
      { id: 24, name: 'SuperTrend Bullish', description: 'Higher Highs pattern', premium: true },
    ]
  },
  {
    id: 'reversals',
    name: 'Reversal Scanners',
    icon: RefreshCw,
    color: 'from-amber-500 to-orange-600',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/30',
    textColor: 'text-amber-400',
    description: 'Potential trend reversal candidates',
    scanners: [
      { id: 6, name: 'Reversal Signals', description: 'Potential trend reversals', premium: false },
      { id: 25, name: 'Watch for Reversal', description: 'Lower Highs pattern', premium: false },
      { id: 18, name: 'Aroon Crossover', description: 'Bullish Aroon(14) cross', premium: true },
      { id: 20, name: 'Bullish Tomorrow', description: 'AI prediction for next day', premium: true },
    ]
  },
  {
    id: 'institutional',
    name: 'Smart Money',
    icon: Building,
    color: 'from-cyan-500 to-teal-600',
    bgColor: 'bg-cyan-500/10',
    borderColor: 'border-cyan-500/30',
    textColor: 'text-cyan-400',
    description: 'Track institutional activity',
    scanners: [
      { id: 22, name: 'Stock Performance', description: 'Multi-timeframe analysis', premium: false },
      { id: 26, name: 'Corporate Actions', description: 'Splits, bonus, dividends', premium: false },
      { id: 21, name: 'MF/FII Popular', description: 'Institutional favorites', premium: true },
    ]
  },
  {
    id: 'intraday',
    name: 'Intraday Scanners',
    icon: Clock,
    color: 'from-red-500 to-rose-600',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/30',
    textColor: 'text-red-400',
    description: 'Real-time intraday setups',
    scanners: [
      { id: 12, name: 'Price & Volume Breakout', description: 'N-minute breakouts', premium: true },
      { id: 29, name: 'Bid/Ask Buildup', description: 'Order flow analysis', premium: true },
      { id: 32, name: 'Intraday Setup', description: 'Breakout/Breakdown setups', premium: true },
    ]
  },
  {
    id: 'value',
    name: 'Value Hunting',
    icon: DollarSign,
    color: 'from-green-500 to-emerald-600',
    bgColor: 'bg-green-500/10',
    borderColor: 'border-green-500/30',
    textColor: 'text-green-400',
    description: 'Undervalued & oversold stocks',
    scanners: [
      { id: 15, name: '52-Week Low', description: 'Near 52-week lows', premium: false },
      { id: 16, name: '10-Day Low Breakout', description: 'Short-term oversold', premium: false },
      { id: 33, name: 'Profitable Setups', description: 'High probability trades', premium: true },
    ]
  },
  {
    id: 'technical',
    name: 'Technical Indicators',
    icon: BarChart3,
    color: 'from-violet-500 to-purple-600',
    bgColor: 'bg-violet-500/10',
    borderColor: 'border-violet-500/30',
    textColor: 'text-violet-400',
    description: 'Advanced indicator scans',
    scanners: [
      { id: 8, name: 'CCI Scanner', description: 'CCI outside range', premium: false },
      { id: 27, name: 'ATR Cross', description: 'Volatility expansion', premium: false },
      { id: 11, name: 'Ichimoku Bullish', description: 'Cloud breakout', premium: true },
      { id: 30, name: 'ATR Trailing Stops', description: 'Swing trade levels', premium: true },
    ]
  },
]

// Mock scan results (replace with API call)
const MOCK_RESULTS = [
  { symbol: 'TRENT', ltp: 4567.89, change: 3.2, volume: 1234567, signal: 'Strong Buy', rsi: 62, pattern: 'Breakout' },
  { symbol: 'POLYCAB', ltp: 5678.90, change: 2.8, volume: 987654, signal: 'Buy', rsi: 58, pattern: 'VCP' },
  { symbol: 'PERSISTENT', ltp: 4321.00, change: 1.9, volume: 876543, signal: 'Buy', rsi: 55, pattern: 'Momentum' },
  { symbol: 'DIXON', ltp: 5432.10, change: 2.5, volume: 654321, signal: 'Buy', rsi: 61, pattern: 'Breakout' },
  { symbol: 'TATAELXSI', ltp: 6789.00, change: 1.5, volume: 543210, signal: 'Hold', rsi: 52, pattern: 'Consolidating' },
  { symbol: 'COFORGE', ltp: 5234.50, change: 2.1, volume: 432109, signal: 'Buy', rsi: 57, pattern: 'Momentum' },
  { symbol: 'ASTRAL', ltp: 1876.30, change: 1.8, volume: 765432, signal: 'Buy', rsi: 54, pattern: 'Support' },
  { symbol: 'LALPATHLAB', ltp: 2345.60, change: 1.2, volume: 321098, signal: 'Hold', rsi: 48, pattern: 'Range' },
]

// ============================================================================
// SCREENER PAGE
// ============================================================================

export default function ScreenerPage() {
  const router = useRouter()
  const { user, profile, loading: authLoading } = useAuth()
  
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedScanner, setSelectedScanner] = useState<any | null>(null)
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  // Check if user has premium access
  const isPremium = profile?.subscription_status === 'active' || profile?.subscription_status === 'trial'

  // Run a scan
  const runScan = async (scanner: any) => {
    if (scanner.premium && !isPremium) {
      router.push('/pricing')
      return
    }

    setSelectedScanner(scanner)
    setLoading(true)

    // Simulate API call (replace with real API)
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // In production, call: const result = await api.screener.runScan(scanner.id)
    setResults(MOCK_RESULTS)
    setLoading(false)
  }

  // Filter categories by search
  const filteredCategories = SCANNER_CATEGORIES.filter(cat =>
    cat.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    cat.scanners.some(s => s.name.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  // Redirect if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login')
    }
  }, [user, authLoading, router])

  if (authLoading) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <RefreshCw className="w-8 h-8 text-emerald-500 animate-spin" />
      </div>
    )
  }

  if (!user) return null

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-[#0a0a0f]/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/dashboard" className="p-2 hover:bg-white/5 rounded-xl transition-colors">
                <ArrowLeft className="w-5 h-5 text-gray-400" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                  Stock Screener
                  <span className="px-2 py-0.5 text-xs font-medium bg-emerald-500/20 text-emerald-400 rounded-full">
                    40+ Scanners
                  </span>
                </h1>
                <p className="text-sm text-gray-500">Powered by PKScreener • Find winning stocks</p>
              </div>
            </div>

            {/* Search */}
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search scanners..."
                  className="w-64 pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-xl text-white placeholder:text-gray-500 focus:outline-none focus:border-emerald-500/50"
                />
              </div>
              {!isPremium && (
                <Link
                  href="/pricing"
                  className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-amber-500/25 transition-all"
                >
                  <Crown className="w-4 h-4" />
                  Upgrade
                </Link>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Bar */}
        <div className="grid grid-cols-4 gap-4 mb-8">
          {[
            { label: 'Total Scanners', value: '40+', icon: BarChart3, color: 'text-emerald-400' },
            { label: 'Stocks Covered', value: '1,800+', icon: Target, color: 'text-blue-400' },
            { label: 'Updated', value: 'Real-time', icon: Zap, color: 'text-amber-400' },
            { label: 'Your Access', value: isPremium ? 'Full' : 'Limited', icon: isPremium ? Star : Lock, color: isPremium ? 'text-emerald-400' : 'text-gray-400' },
          ].map((stat, i) => (
            <div key={i} className="bg-white/[0.02] border border-white/5 rounded-2xl p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">{stat.label}</p>
                  <p className={`text-xl font-bold ${stat.color}`}>{stat.value}</p>
                </div>
                <stat.icon className={`w-8 h-8 ${stat.color} opacity-50`} />
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Scanner Categories */}
          <div className="lg:col-span-2 space-y-6">
            <h2 className="text-lg font-semibold text-white">Scanner Categories</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredCategories.map((category) => (
                <motion.div
                  key={category.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`group relative overflow-hidden rounded-2xl border ${
                    selectedCategory === category.id 
                      ? `${category.borderColor} ${category.bgColor}` 
                      : 'border-white/5 bg-white/[0.02] hover:border-white/10'
                  } transition-all cursor-pointer`}
                  onClick={() => setSelectedCategory(selectedCategory === category.id ? null : category.id)}
                >
                  {/* Gradient overlay */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${category.color} opacity-0 group-hover:opacity-5 transition-opacity`} />
                  
                  <div className="relative p-5">
                    <div className="flex items-start justify-between mb-3">
                      <div className={`p-2.5 rounded-xl ${category.bgColor}`}>
                        <category.icon className={`w-5 h-5 ${category.textColor}`} />
                      </div>
                      <span className="text-xs text-gray-500">{category.scanners.length} scanners</span>
                    </div>
                    
                    <h3 className="text-white font-semibold mb-1">{category.name}</h3>
                    <p className="text-sm text-gray-500 mb-4">{category.description}</p>
                    
                    {/* Scanners list */}
                    <AnimatePresence>
                      {selectedCategory === category.id && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className="space-y-2 pt-3 border-t border-white/5"
                        >
                          {category.scanners.map((scanner) => (
                            <button
                              key={scanner.id}
                              onClick={(e) => {
                                e.stopPropagation()
                                runScan(scanner)
                              }}
                              disabled={loading}
                              className={`w-full flex items-center justify-between p-3 rounded-xl transition-all ${
                                selectedScanner?.id === scanner.id
                                  ? `${category.bgColor} ${category.borderColor} border`
                                  : 'bg-white/[0.03] hover:bg-white/[0.05]'
                              }`}
                            >
                              <div className="flex items-center gap-3">
                                {scanner.premium && !isPremium ? (
                                  <Lock className="w-4 h-4 text-gray-500" />
                                ) : (
                                  <Play className={`w-4 h-4 ${category.textColor}`} />
                                )}
                                <div className="text-left">
                                  <p className="text-sm font-medium text-white">{scanner.name}</p>
                                  <p className="text-xs text-gray-500">{scanner.description}</p>
                                </div>
                              </div>
                              {scanner.premium && (
                                <span className={`px-2 py-0.5 text-xs rounded-full ${
                                  isPremium 
                                    ? 'bg-emerald-500/20 text-emerald-400' 
                                    : 'bg-amber-500/20 text-amber-400'
                                }`}>
                                  {isPremium ? 'PRO' : 'Upgrade'}
                                </span>
                              )}
                            </button>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                    
                    <div className="flex items-center justify-between mt-3">
                      <div className="flex -space-x-1">
                        {category.scanners.slice(0, 3).map((_, i) => (
                          <div key={i} className={`w-6 h-6 rounded-full ${category.bgColor} border-2 border-[#0a0a0f] flex items-center justify-center`}>
                            <span className={`text-xs ${category.textColor}`}>{i + 1}</span>
                          </div>
                        ))}
                        {category.scanners.length > 3 && (
                          <div className="w-6 h-6 rounded-full bg-white/10 border-2 border-[#0a0a0f] flex items-center justify-center">
                            <span className="text-xs text-gray-400">+{category.scanners.length - 3}</span>
                          </div>
                        )}
                      </div>
                      <ChevronRight className={`w-4 h-4 text-gray-500 transition-transform ${
                        selectedCategory === category.id ? 'rotate-90' : ''
                      }`} />
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Scan Results</h2>
              {results.length > 0 && (
                <button className="flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors">
                  <Download className="w-4 h-4" />
                  Export
                </button>
              )}
            </div>

            <div className="bg-white/[0.02] border border-white/5 rounded-2xl overflow-hidden">
              {loading ? (
                <div className="p-12 text-center">
                  <RefreshCw className="w-8 h-8 text-emerald-500 animate-spin mx-auto mb-4" />
                  <p className="text-gray-400">Scanning 1,800+ stocks...</p>
                  <p className="text-sm text-gray-500 mt-1">This may take a few seconds</p>
                </div>
              ) : results.length > 0 ? (
                <div>
                  {/* Scanner info */}
                  {selectedScanner && (
                    <div className="p-4 border-b border-white/5 bg-white/[0.02]">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-white">{selectedScanner.name}</p>
                          <p className="text-sm text-gray-500">{results.length} stocks found</p>
                        </div>
                        <button
                          onClick={() => runScan(selectedScanner)}
                          className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                        >
                          <RefreshCw className="w-4 h-4 text-gray-400" />
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Results list */}
                  <div className="divide-y divide-white/5">
                    {results.map((stock, i) => (
                      <motion.div
                        key={stock.symbol}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.05 }}
                        className="p-4 hover:bg-white/[0.02] transition-colors"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-blue-500/20 flex items-center justify-center">
                              <span className="text-sm font-bold text-white">{stock.symbol.slice(0, 2)}</span>
                            </div>
                            <div>
                              <p className="font-medium text-white">{stock.symbol}</p>
                              <p className="text-xs text-gray-500">{stock.pattern}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="font-medium text-white">₹{stock.ltp.toLocaleString()}</p>
                            <p className={`text-sm ${stock.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {stock.change >= 0 ? '+' : ''}{stock.change}%
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4 text-xs text-gray-500">
                            <span>Vol: {(stock.volume / 1000000).toFixed(1)}M</span>
                            <span>RSI: {stock.rsi}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <button className="p-1.5 hover:bg-white/5 rounded-lg transition-colors" title="Add to Watchlist">
                              <Plus className="w-4 h-4 text-gray-400" />
                            </button>
                            <button className="p-1.5 hover:bg-white/5 rounded-lg transition-colors" title="Set Alert">
                              <Bell className="w-4 h-4 text-gray-400" />
                            </button>
                            <button className="p-1.5 hover:bg-white/5 rounded-lg transition-colors" title="View Chart">
                              <ExternalLink className="w-4 h-4 text-gray-400" />
                            </button>
                            <button className="px-3 py-1.5 bg-emerald-500/20 text-emerald-400 text-xs font-medium rounded-lg hover:bg-emerald-500/30 transition-colors">
                              Trade
                            </button>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="p-12 text-center">
                  <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mx-auto mb-4">
                    <Search className="w-8 h-8 text-gray-600" />
                  </div>
                  <p className="text-gray-400 mb-2">No scan results yet</p>
                  <p className="text-sm text-gray-500">Select a scanner from the left to find stocks</p>
                </div>
              )}
            </div>

            {/* Quick Actions */}
            <div className="bg-gradient-to-br from-emerald-500/10 to-blue-500/10 border border-emerald-500/20 rounded-2xl p-5">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-xl bg-emerald-500/20">
                  <Zap className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <p className="font-medium text-white">AI Swing Candidates</p>
                  <p className="text-sm text-gray-400">Best stocks from all scanners</p>
                </div>
              </div>
              <button
                onClick={() => {
                  setSelectedScanner({ id: 'ai', name: 'AI Swing Candidates' })
                  setLoading(true)
                  setTimeout(() => {
                    setResults(MOCK_RESULTS.slice(0, 5))
                    setLoading(false)
                  }, 1500)
                }}
                className="w-full py-2.5 bg-gradient-to-r from-emerald-500 to-blue-500 text-white font-medium rounded-xl hover:shadow-lg hover:shadow-emerald-500/25 transition-all"
              >
                Find Best Swing Stocks
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
