'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import {
  TrendingUp,
  TrendingDown,
  ArrowUp,
  ArrowDown,
  BarChart3,
  Sparkles,
  Target,
  Activity,
  PieChart,
  Zap,
  Eye,
  Clock,
} from 'lucide-react'

interface MarketData {
  indices: {
    nifty50: { value: number; change: number; change_percent: number }
    sensex: { value: number; change: number; change_percent: number }
    banknifty: { value: number; change: number; change_percent: number }
  }
  market_status: string
  market_sentiment: string
}

interface Stock {
  symbol: string
  name: string
  current_price: number
  day_change_percent: number
  volume: string
  ai_score: number
}

export default function DashboardPage() {
  const [marketData, setMarketData] = useState<MarketData | null>(null)
  const [trendingStocks, setTrendingStocks] = useState<Stock[]>([])
  const [topGainers, setTopGainers] = useState<Stock[]>([])
  const [topLosers, setTopLosers] = useState<Stock[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchMarketData()
  }, [])

  const fetchMarketData = async () => {
    try {
      const [overviewRes, trendingRes, moversRes] = await Promise.all([
        fetch('http://localhost:8001/api/market/overview'),
        fetch('http://localhost:8001/api/market/trending?limit=6'),
        fetch('http://localhost:8001/api/market/top-movers'),
      ])

      const overview = await overviewRes.json()
      const trending = await trendingRes.json()
      const movers = await moversRes.json()

      setMarketData(overview)
      setTrendingStocks(trending.trending_stocks || [])
      setTopGainers(movers.gainers || [])
      setTopLosers(movers.losers || [])
    } catch (error) {
      console.error('Error fetching market data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <Activity className="mx-auto h-12 w-12 animate-pulse text-accent" />
          <p className="mt-4 text-lg text-text-secondary">Loading market data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background-primary px-6 py-8">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="mb-2 text-4xl font-bold text-text-primary">
            <span className="gradient-text-professional">Market Dashboard</span>
          </h1>
          <p className="text-lg text-text-secondary">
            Real-time NSE/BSE market data powered by AI intelligence
          </p>
        </div>

        {/* Market Indices */}
        <div className="mb-8 grid gap-4 md:grid-cols-3">
          {marketData && (
            <>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6"
              >
                <div className="mb-2 text-sm font-medium text-text-secondary">NIFTY 50</div>
                <div className="mb-1 text-3xl font-bold text-text-primary">
                  {marketData.indices.nifty50.value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                </div>
                <div className={`flex items-center gap-1 text-sm font-semibold ${
                  marketData.indices.nifty50.change >= 0 ? 'text-success' : 'text-danger'
                }`}>
                  {marketData.indices.nifty50.change >= 0 ? <ArrowUp className="h-4 w-4" /> : <ArrowDown className="h-4 w-4" />}
                  {Math.abs(marketData.indices.nifty50.change).toFixed(2)} ({marketData.indices.nifty50.change_percent.toFixed(2)}%)
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6"
              >
                <div className="mb-2 text-sm font-medium text-text-secondary">SENSEX</div>
                <div className="mb-1 text-3xl font-bold text-text-primary">
                  {marketData.indices.sensex.value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                </div>
                <div className={`flex items-center gap-1 text-sm font-semibold ${
                  marketData.indices.sensex.change >= 0 ? 'text-success' : 'text-danger'
                }`}>
                  {marketData.indices.sensex.change >= 0 ? <ArrowUp className="h-4 w-4" /> : <ArrowDown className="h-4 w-4" />}
                  {Math.abs(marketData.indices.sensex.change).toFixed(2)} ({marketData.indices.sensex.change_percent.toFixed(2)}%)
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6"
              >
                <div className="mb-2 text-sm font-medium text-text-secondary">BANK NIFTY</div>
                <div className="mb-1 text-3xl font-bold text-text-primary">
                  {marketData.indices.banknifty.value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                </div>
                <div className={`flex items-center gap-1 text-sm font-semibold ${
                  marketData.indices.banknifty.change >= 0 ? 'text-success' : 'text-danger'
                }`}>
                  {marketData.indices.banknifty.change >= 0 ? <ArrowUp className="h-4 w-4" /> : <ArrowDown className="h-4 w-4" />}
                  {Math.abs(marketData.indices.banknifty.change).toFixed(2)} ({marketData.indices.banknifty.change_percent.toFixed(2)}%)
                </div>
              </motion.div>
            </>
          )}
        </div>

        {/* AI Top Picks - Trending Stocks */}
        <section className="mb-8">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-2xl font-bold text-text-primary">
              <span className="gradient-text-accent">AI Top Picks</span>
            </h2>
            <Link
              href="/stocks?filter=trending"
              className="text-sm font-medium text-accent transition hover:text-accent/80"
            >
              View All →
            </Link>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {trendingStocks.map((stock, index) => (
              <motion.div
                key={stock.symbol}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className="group cursor-pointer rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-5 transition hover:border-accent/40 hover:shadow-lg"
              >
                <div className="mb-3 flex items-start justify-between">
                  <div>
                    <div className="font-bold text-text-primary">{stock.symbol}</div>
                    <div className="text-xs text-text-secondary">{stock.name}</div>
                  </div>
                  <div className="flex items-center gap-1 rounded-full bg-accent/15 px-2 py-1">
                    <Sparkles className="h-3 w-3 text-accent" />
                    <span className="text-xs font-semibold text-accent">{stock.ai_score}</span>
                  </div>
                </div>
                <div className="mb-2 text-2xl font-bold text-text-primary">₹{stock.current_price.toFixed(2)}</div>
                <div className="flex items-center justify-between text-sm">
                  <div className={`flex items-center gap-1 font-semibold ${
                    stock.day_change_percent >= 0 ? 'text-success' : 'text-danger'
                  }`}>
                    {stock.day_change_percent >= 0 ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
                    {Math.abs(stock.day_change_percent).toFixed(2)}%
                  </div>
                  <div className="text-text-secondary">Vol: {stock.volume}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Top Gainers & Losers */}
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Top Gainers */}
          <section>
            <div className="mb-4 flex items-center gap-2">
              <TrendingUp className="h-6 w-6 text-success" />
              <h2 className="text-2xl font-bold text-text-primary">Top Gainers</h2>
            </div>
            <div className="space-y-3">
              {topGainers.map((stock, index) => (
                <motion.div
                  key={stock.symbol}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between rounded-lg border border-border/60 bg-background-surface p-4 transition hover:border-success/40"
                >
                  <div>
                    <div className="font-semibold text-text-primary">{stock.symbol}</div>
                    <div className="text-sm text-text-secondary">₹{stock.current_price.toFixed(2)}</div>
                  </div>
                  <div className="flex items-center gap-2 rounded-full bg-success/15 px-3 py-1">
                    <ArrowUp className="h-4 w-4 text-success" />
                    <span className="font-bold text-success">{stock.day_change_percent.toFixed(2)}%</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </section>

          {/* Top Losers */}
          <section>
            <div className="mb-4 flex items-center gap-2">
              <TrendingDown className="h-6 w-6 text-danger" />
              <h2 className="text-2xl font-bold text-text-primary">Top Losers</h2>
            </div>
            <div className="space-y-3">
              {topLosers.map((stock, index) => (
                <motion.div
                  key={stock.symbol}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between rounded-lg border border-border/60 bg-background-surface p-4 transition hover:border-danger/40"
                >
                  <div>
                    <div className="font-semibold text-text-primary">{stock.symbol}</div>
                    <div className="text-sm text-text-secondary">₹{stock.current_price.toFixed(2)}</div>
                  </div>
                  <div className="flex items-center gap-2 rounded-full bg-danger/15 px-3 py-1">
                    <ArrowDown className="h-4 w-4 text-danger" />
                    <span className="font-bold text-danger">{Math.abs(stock.day_change_percent).toFixed(2)}%</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </section>
        </div>

        {/* Quick Actions */}
        <section className="mt-8">
          <h2 className="mb-4 text-2xl font-bold text-text-primary">Quick Actions</h2>
          <div className="grid gap-4 md:grid-cols-4">
            <Link
              href="/screener"
              className="group flex items-center gap-3 rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-5 transition hover:border-accent/40 hover:shadow-lg"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-accent/15">
                <Sparkles className="h-6 w-6 text-accent" />
              </div>
              <div>
                <div className="font-semibold text-text-primary">AI Screener</div>
                <div className="text-xs text-text-secondary">43+ Scanners</div>
              </div>
            </Link>

            <Link
              href="/signals"
              className="group flex items-center gap-3 rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-5 transition hover:border-primary/40 hover:shadow-lg"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/15">
                <Target className="h-6 w-6 text-primary" />
              </div>
              <div>
                <div className="font-semibold text-text-primary">AI Signals</div>
                <div className="text-xs text-text-secondary">Live Trades</div>
              </div>
            </Link>

            <Link
              href="/stocks"
              className="group flex items-center gap-3 rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-5 transition hover:border-accent/40 hover:shadow-lg"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-accent/15">
                <BarChart3 className="h-6 w-6 text-accent" />
              </div>
              <div>
                <div className="font-semibold text-text-primary">All Stocks</div>
                <div className="text-xs text-text-secondary">NSE/BSE</div>
              </div>
            </Link>

            <Link
              href="/portfolio"
              className="group flex items-center gap-3 rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-5 transition hover:border-primary/40 hover:shadow-lg"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/15">
                <PieChart className="h-6 w-6 text-primary" />
              </div>
              <div>
                <div className="font-semibold text-text-primary">Portfolio</div>
                <div className="text-xs text-text-secondary">Track P&L</div>
              </div>
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}
