'use client'

import { useState, useEffect } from 'react'
import {
  Search,
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Target,
  BarChart3,
  RefreshCw,
  ArrowUpRight,
  ArrowDownRight,
  Layers,
  Filter,
  ChevronRight,
  ChevronDown,
  Play,
  Cpu,
  LineChart,
  BarChart2,
  PieChart,
  Crosshair,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Info,
} from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.REACT_APP_BACKEND_URL || ''

interface ScannerCategory {
  name: string
  description: string
  scanners: Scanner[]
}

interface Scanner {
  id: string
  name: string
  menu_code: string
  category?: string
  category_name?: string
}

interface ScanResult {
  symbol: string
  name?: string
  current_price?: number
  ltp?: number
  change_percent?: number
  rsi?: number
  volume_ratio?: number
  reason?: string
  trend?: string
  passed?: boolean
  sector?: string
  high_52w?: number
  low_52w?: number
  macd?: number
}

const categoryIcons: { [key: string]: any } = {
  breakout: TrendingUp,
  momentum: Zap,
  reversal: Target,
  patterns: Layers,
  ma_signals: LineChart,
  technical: BarChart3,
  signals: Activity,
  consolidation: BarChart2,
  trend: TrendingUp,
  ml: Cpu,
  short_sell: TrendingDown,
}

const categoryColors: { [key: string]: string } = {
  breakout: 'text-green-400',
  momentum: 'text-yellow-400',
  reversal: 'text-purple-400',
  patterns: 'text-blue-400',
  ma_signals: 'text-cyan-400',
  technical: 'text-orange-400',
  signals: 'text-pink-400',
  consolidation: 'text-gray-400',
  trend: 'text-emerald-400',
  ml: 'text-red-400',
  short_sell: 'text-red-500',
}

export default function PKScreenerPage() {
  const [categories, setCategories] = useState<{ [key: string]: ScannerCategory }>({})
  const [allScanners, setAllScanners] = useState<{ [key: string]: Scanner }>({})
  const [expandedCategory, setExpandedCategory] = useState<string | null>('breakout')
  const [selectedScanner, setSelectedScanner] = useState<string>('probable_breakout')
  const [selectedScannerInfo, setSelectedScannerInfo] = useState<Scanner | null>(null)
  const [universe, setUniverse] = useState('nifty50')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<ScanResult[]>([])
  const [scanStats, setScanStats] = useState({ total: 0, passed: 0 })
  const [lastScanTime, setLastScanTime] = useState<string>('')
  const [serviceAvailable, setServiceAvailable] = useState(true)

  // Fetch scanner categories on mount
  useEffect(() => {
    fetchCategories()
  }, [])

  // Update scanner info when selection changes
  useEffect(() => {
    if (allScanners[selectedScanner]) {
      setSelectedScannerInfo(allScanners[selectedScanner])
    }
  }, [selectedScanner, allScanners])

  const fetchCategories = async () => {
    try {
      const res = await fetch(`${API_URL}/api/screener/pk/categories`)
      const data = await res.json()
      
      if (data.success) {
        setCategories(data.categories || {})
        setServiceAvailable(data.service_available)
        
        // Flatten scanners for easy lookup
        const scanners: { [key: string]: Scanner } = {}
        Object.entries(data.categories || {}).forEach(([catKey, catValue]: [string, any]) => {
          catValue.scanners?.forEach((scanner: Scanner) => {
            scanners[scanner.id] = {
              ...scanner,
              category: catKey,
              category_name: catValue.name
            }
          })
        })
        setAllScanners(scanners)
      }
    } catch (error) {
      console.error('Error fetching categories:', error)
    }
  }

  const runScan = async () => {
    setLoading(true)
    setResults([])
    
    try {
      const res = await fetch(
        `${API_URL}/api/screener/pk/scan/batch?scanner_id=${selectedScanner}&universe=${universe}&limit=100`,
        { method: 'POST' }
      )
      const data = await res.json()
      
      if (data.success) {
        setResults(data.results || [])
        setScanStats({
          total: data.total_scanned || 0,
          passed: data.results_count || 0
        })
        setLastScanTime(new Date().toLocaleTimeString())
      }
    } catch (error) {
      console.error('Error running scan:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleCategory = (category: string) => {
    setExpandedCategory(expandedCategory === category ? null : category)
  }

  const selectScanner = (scannerId: string) => {
    setSelectedScanner(scannerId)
    setResults([])
  }

  const getUniverseLabel = () => {
    switch (universe) {
      case 'nifty50': return 'Nifty 50'
      case 'nifty500': return 'Top 200'
      case 'all': return 'All NSE (2200+)'
      default: return universe
    }
  }

  return (
    <div className="min-h-screen bg-background pb-8" data-testid="pk-screener-page">
      {/* Header */}
      <div className="border-b border-border bg-background-card">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl">
                <Search className="w-8 h-8 text-blue-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-text-primary">PKScreener Pro</h1>
                <p className="text-text-secondary">40+ AI-powered stock scanners</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Universe Selector */}
              <select
                value={universe}
                onChange={(e) => setUniverse(e.target.value)}
                className="px-4 py-2 bg-background border border-border rounded-lg text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent"
                data-testid="universe-selector"
              >
                <option value="nifty50">Nifty 50</option>
                <option value="nifty500">Top 200 Stocks</option>
                <option value="all">All NSE (2200+)</option>
              </select>

              {/* Service Status */}
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs ${
                serviceAvailable 
                  ? 'bg-green-500/10 text-green-400' 
                  : 'bg-red-500/10 text-red-400'
              }`}>
                {serviceAvailable ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
                PKScreener {serviceAvailable ? 'Active' : 'Unavailable'}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          
          {/* Sidebar - Scanner Categories */}
          <div className="col-span-3">
            <div className="bg-background-card border border-border rounded-xl overflow-hidden">
              <div className="p-4 border-b border-border">
                <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider">
                  Scanner Categories
                </h3>
                <p className="text-xs text-text-secondary mt-1">
                  {Object.keys(allScanners).length} scanners available
                </p>
              </div>
              
              <div className="divide-y divide-border">
                {Object.entries(categories).map(([catKey, category]) => {
                  const Icon = categoryIcons[catKey] || Filter
                  const colorClass = categoryColors[catKey] || 'text-text-primary'
                  const isExpanded = expandedCategory === catKey
                  
                  return (
                    <div key={catKey}>
                      {/* Category Header */}
                      <button
                        onClick={() => toggleCategory(catKey)}
                        className="w-full flex items-center justify-between p-3 hover:bg-background transition"
                        data-testid={`category-${catKey}`}
                      >
                        <div className="flex items-center gap-3">
                          <Icon className={`w-4 h-4 ${colorClass}`} />
                          <div className="text-left">
                            <div className="text-sm font-medium text-text-primary">{category.name}</div>
                            <div className="text-xs text-text-secondary">
                              {category.scanners?.length || 0} scanners
                            </div>
                          </div>
                        </div>
                        <ChevronDown className={`w-4 h-4 text-text-secondary transition ${isExpanded ? 'rotate-180' : ''}`} />
                      </button>
                      
                      {/* Scanner List */}
                      {isExpanded && (
                        <div className="bg-background/50 py-1">
                          {category.scanners?.map((scanner) => (
                            <button
                              key={scanner.id}
                              onClick={() => selectScanner(scanner.id)}
                              className={`w-full text-left px-4 py-2 text-sm transition ${
                                selectedScanner === scanner.id
                                  ? 'bg-accent/15 text-accent border-l-2 border-accent'
                                  : 'text-text-secondary hover:bg-background hover:text-text-primary'
                              }`}
                              data-testid={`scanner-${scanner.id}`}
                            >
                              {scanner.name}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="col-span-9 space-y-6">
            
            {/* Scanner Info & Run Button */}
            <div className="bg-background-card border border-border rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-3">
                    {selectedScannerInfo && (
                      <>
                        {(() => {
                          const Icon = categoryIcons[selectedScannerInfo.category || ''] || Filter
                          return <Icon className={`w-6 h-6 ${categoryColors[selectedScannerInfo.category || '']}`} />
                        })()}
                      </>
                    )}
                    <div>
                      <h2 className="text-xl font-bold text-text-primary">
                        {selectedScannerInfo?.name || 'Select a Scanner'}
                      </h2>
                      <p className="text-sm text-text-secondary">
                        {selectedScannerInfo?.category_name} • {getUniverseLabel()}
                      </p>
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={runScan}
                  disabled={loading || !selectedScanner}
                  className="flex items-center gap-2 px-6 py-3 bg-accent text-background rounded-xl font-semibold hover:bg-accent/90 disabled:opacity-50 disabled:cursor-not-allowed transition"
                  data-testid="run-scan-button"
                >
                  {loading ? (
                    <RefreshCw className="w-5 h-5 animate-spin" />
                  ) : (
                    <Play className="w-5 h-5" />
                  )}
                  {loading ? 'Scanning...' : 'Run Scan'}
                </button>
              </div>

              {/* Scan Stats */}
              {lastScanTime && (
                <div className="mt-4 pt-4 border-t border-border flex items-center gap-6 text-sm">
                  <div className="flex items-center gap-2">
                    <BarChart2 className="w-4 h-4 text-text-secondary" />
                    <span className="text-text-secondary">Scanned:</span>
                    <span className="font-medium text-text-primary">{scanStats.total}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                    <span className="text-text-secondary">Passed:</span>
                    <span className="font-medium text-green-400">{scanStats.passed}</span>
                  </div>
                  <div className="flex items-center gap-2 text-text-secondary">
                    <Activity className="w-4 h-4" />
                    Last scan: {lastScanTime}
                  </div>
                </div>
              )}
            </div>

            {/* Results Table */}
            <div className="bg-background-card border border-border rounded-xl overflow-hidden">
              <div className="p-4 border-b border-border flex items-center justify-between">
                <h3 className="font-semibold text-text-primary">
                  Scan Results {results.length > 0 && `(${results.length})`}
                </h3>
                {results.length > 0 && (
                  <span className="text-xs text-text-secondary">
                    Sorted by relevance
                  </span>
                )}
              </div>

              {loading ? (
                <div className="flex items-center justify-center py-16">
                  <RefreshCw className="w-8 h-8 text-accent animate-spin" />
                </div>
              ) : results.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-background/50">
                      <tr className="text-left text-xs text-text-secondary uppercase tracking-wider">
                        <th className="px-4 py-3">Symbol</th>
                        <th className="px-4 py-3">Price</th>
                        <th className="px-4 py-3">Change</th>
                        <th className="px-4 py-3">RSI</th>
                        <th className="px-4 py-3">Volume</th>
                        <th className="px-4 py-3">Signal</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {results.map((stock, idx) => (
                        <tr 
                          key={stock.symbol} 
                          className="hover:bg-background/50 transition"
                          data-testid={`result-row-${idx}`}
                        >
                          <td className="px-4 py-3">
                            <div>
                              <div className="font-semibold text-text-primary">{stock.symbol}</div>
                              <div className="text-xs text-text-secondary truncate max-w-[150px]">
                                {stock.name || stock.sector || '-'}
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-3 font-medium text-text-primary">
                            ₹{(stock.current_price || stock.ltp)?.toLocaleString('en-IN', { maximumFractionDigits: 2 }) || '-'}
                          </td>
                          <td className="px-4 py-3">
                            <span className={`inline-flex items-center gap-1 ${
                              (stock.change_percent || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                            }`}>
                              {(stock.change_percent || 0) >= 0 ? (
                                <ArrowUpRight className="w-3 h-3" />
                              ) : (
                                <ArrowDownRight className="w-3 h-3" />
                              )}
                              {Math.abs(stock.change_percent || 0).toFixed(2)}%
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                              (stock.rsi || 50) < 30 ? 'bg-green-500/15 text-green-400' :
                              (stock.rsi || 50) > 70 ? 'bg-red-500/15 text-red-400' :
                              'bg-gray-500/15 text-gray-400'
                            }`}>
                              {stock.rsi?.toFixed(1) || '-'}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-text-secondary">
                            {stock.volume_ratio ? `${stock.volume_ratio}x` : '-'}
                          </td>
                          <td className="px-4 py-3">
                            <div className="text-sm text-text-secondary max-w-[200px] truncate">
                              {stock.reason || stock.trend || '-'}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                  <Search className="w-12 h-12 text-text-secondary/50 mb-4" />
                  <h4 className="text-lg font-medium text-text-primary mb-2">
                    No Results Yet
                  </h4>
                  <p className="text-text-secondary max-w-md">
                    Select a scanner from the sidebar and click "Run Scan" to find stocks matching your criteria
                  </p>
                </div>
              )}
            </div>

            {/* Quick Stats Cards */}
            {results.length > 0 && (
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-background-card border border-border rounded-xl p-4">
                  <div className="flex items-center gap-2 text-green-400 mb-2">
                    <TrendingUp className="w-4 h-4" />
                    <span className="text-xs font-medium">Top Gainer</span>
                  </div>
                  <div className="font-bold text-text-primary">
                    {results.sort((a, b) => (b.change_percent || 0) - (a.change_percent || 0))[0]?.symbol || '-'}
                  </div>
                  <div className="text-sm text-green-400">
                    +{results.sort((a, b) => (b.change_percent || 0) - (a.change_percent || 0))[0]?.change_percent?.toFixed(2) || 0}%
                  </div>
                </div>
                
                <div className="bg-background-card border border-border rounded-xl p-4">
                  <div className="flex items-center gap-2 text-red-400 mb-2">
                    <TrendingDown className="w-4 h-4" />
                    <span className="text-xs font-medium">Top Loser</span>
                  </div>
                  <div className="font-bold text-text-primary">
                    {results.sort((a, b) => (a.change_percent || 0) - (b.change_percent || 0))[0]?.symbol || '-'}
                  </div>
                  <div className="text-sm text-red-400">
                    {results.sort((a, b) => (a.change_percent || 0) - (b.change_percent || 0))[0]?.change_percent?.toFixed(2) || 0}%
                  </div>
                </div>
                
                <div className="bg-background-card border border-border rounded-xl p-4">
                  <div className="flex items-center gap-2 text-blue-400 mb-2">
                    <BarChart3 className="w-4 h-4" />
                    <span className="text-xs font-medium">Highest Volume</span>
                  </div>
                  <div className="font-bold text-text-primary">
                    {results.sort((a, b) => (b.volume_ratio || 0) - (a.volume_ratio || 0))[0]?.symbol || '-'}
                  </div>
                  <div className="text-sm text-blue-400">
                    {results.sort((a, b) => (b.volume_ratio || 0) - (a.volume_ratio || 0))[0]?.volume_ratio?.toFixed(1) || 0}x avg
                  </div>
                </div>
                
                <div className="bg-background-card border border-border rounded-xl p-4">
                  <div className="flex items-center gap-2 text-yellow-400 mb-2">
                    <Activity className="w-4 h-4" />
                    <span className="text-xs font-medium">Most Oversold</span>
                  </div>
                  <div className="font-bold text-text-primary">
                    {results.filter(r => r.rsi).sort((a, b) => (a.rsi || 50) - (b.rsi || 50))[0]?.symbol || '-'}
                  </div>
                  <div className="text-sm text-yellow-400">
                    RSI {results.filter(r => r.rsi).sort((a, b) => (a.rsi || 50) - (b.rsi || 50))[0]?.rsi?.toFixed(1) || '-'}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
