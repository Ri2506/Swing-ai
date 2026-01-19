'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import {
  History,
  TrendingUp,
  TrendingDown,
  ArrowUp,
  ArrowDown,
  Activity,
  Calendar,
  Filter,
} from 'lucide-react'

interface Trade {
  id: string
  symbol: string
  direction: 'LONG' | 'SHORT'
  entry_price: number
  exit_price: number
  quantity: number
  pnl: number
  pnl_percent: number
  entry_date: string
  exit_date: string
  status: 'win' | 'loss'
}

export default function TradesPage() {
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'win' | 'loss'>('all')

  useEffect(() => {
    // Mock trades data
    const mockTrades: Trade[] = [
      {
        id: '1',
        symbol: 'RELIANCE',
        direction: 'LONG',
        entry_price: 2720.00,
        exit_price: 2847.50,
        quantity: 50,
        pnl: 6375.00,
        pnl_percent: 4.69,
        entry_date: '2024-12-10',
        exit_date: '2024-12-15',
        status: 'win',
      },
      {
        id: '2',
        symbol: 'TCS',
        direction: 'SHORT',
        entry_price: 3720.00,
        exit_price: 3678.90,
        quantity: 30,
        pnl: 1233.00,
        pnl_percent: 1.10,
        entry_date: '2024-12-08',
        exit_date: '2024-12-12',
        status: 'win',
      },
      {
        id: '3',
        symbol: 'INFY',
        direction: 'LONG',
        entry_price: 1580.00,
        exit_price: 1523.45,
        quantity: 100,
        pnl: -5655.00,
        pnl_percent: -3.58,
        entry_date: '2024-12-05',
        exit_date: '2024-12-11',
        status: 'loss',
      },
      {
        id: '4',
        symbol: 'HDFCBANK',
        direction: 'LONG',
        entry_price: 1620.00,
        exit_price: 1678.00,
        quantity: 75,
        pnl: 4350.00,
        pnl_percent: 3.58,
        entry_date: '2024-12-01',
        exit_date: '2024-12-09',
        status: 'win',
      },
      {
        id: '5',
        symbol: 'ICICIBANK',
        direction: 'SHORT',
        entry_price: 1050.00,
        exit_price: 1089.75,
        quantity: 100,
        pnl: -3975.00,
        pnl_percent: -3.79,
        entry_date: '2024-11-28',
        exit_date: '2024-12-05',
        status: 'loss',
      },
    ]
    setTrades(mockTrades)
    setLoading(false)
  }, [])

  const filteredTrades = trades.filter(trade => {
    if (filter === 'all') return true
    return trade.status === filter
  })

  const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0)
  const winningTrades = trades.filter(t => t.status === 'win').length
  const winRate = (winningTrades / trades.length) * 100

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background-primary">
        <Activity className="h-12 w-12 animate-pulse text-accent" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background-primary px-6 py-8">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="mb-2 text-4xl font-bold text-text-primary">
                <span className="gradient-text-professional">Trade History</span>
              </h1>
              <p className="text-lg text-text-secondary">
                Complete record of your executed trades
              </p>
            </div>
            <Link
              href="/dashboard"
              className="rounded-lg border border-border/60 bg-background-surface px-4 py-2 text-sm font-medium text-text-primary transition hover:border-accent/60"
            >
              ← Back to Dashboard
            </Link>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="mb-8 grid gap-4 md:grid-cols-4">
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium text-text-secondary">
              <History className="h-4 w-4" />
              Total Trades
            </div>
            <div className="text-3xl font-bold text-text-primary">{trades.length}</div>
          </div>
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium text-text-secondary">
              <TrendingUp className="h-4 w-4" />
              Total P&L
            </div>
            <div className={`text-3xl font-bold ${totalPnL >= 0 ? 'text-success' : 'text-danger'}`}>
              {totalPnL >= 0 ? '+' : ''}₹{totalPnL.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </div>
          </div>
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6">
            <div className="mb-2 text-sm font-medium text-text-secondary">Win Rate</div>
            <div className="text-3xl font-bold text-accent">{winRate.toFixed(1)}%</div>
          </div>
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6">
            <div className="mb-2 text-sm font-medium text-text-secondary">Wins/Losses</div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-success">{winningTrades}</span>
              <span className="text-text-secondary">/</span>
              <span className="text-3xl font-bold text-danger">{trades.length - winningTrades}</span>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="mb-6 flex items-center gap-3">
          <Filter className="h-5 w-5 text-text-secondary" />
          <div className="flex gap-2">
            {(['all', 'win', 'loss'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                  filter === f
                    ? 'bg-accent/15 text-accent'
                    : 'bg-background-surface text-text-secondary hover:bg-background-elevated'
                }`}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Trades Table */}
        <div className="overflow-hidden rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-border/40 bg-background-elevated/50">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-text-secondary">Symbol</th>
                  <th className="px-6 py-4 text-center text-sm font-semibold text-text-secondary">Direction</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-text-secondary">Entry</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-text-secondary">Exit</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-text-secondary">Qty</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-text-secondary">P&L</th>
                  <th className="px-6 py-4 text-center text-sm font-semibold text-text-secondary">Dates</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {filteredTrades.map((trade, index) => (
                  <motion.tr
                    key={trade.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="transition-colors hover:bg-background-elevated/30"
                  >
                    <td className="px-6 py-4 font-semibold text-text-primary">{trade.symbol}</td>
                    <td className="px-6 py-4 text-center">
                      <span className={`inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-bold ${
                        trade.direction === 'LONG' ? 'bg-success/15 text-success' : 'bg-danger/15 text-danger'
                      }`}>
                        {trade.direction === 'LONG' ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                        {trade.direction}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right text-text-secondary">₹{trade.entry_price.toFixed(2)}</td>
                    <td className="px-6 py-4 text-right text-text-primary">₹{trade.exit_price.toFixed(2)}</td>
                    <td className="px-6 py-4 text-right text-text-secondary">{trade.quantity}</td>
                    <td className="px-6 py-4 text-right">
                      <div className={`font-bold ${trade.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                        {trade.pnl >= 0 ? '+' : ''}₹{trade.pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                      </div>
                      <div className={`flex items-center justify-end gap-1 text-xs ${trade.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                        {trade.pnl >= 0 ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
                        {Math.abs(trade.pnl_percent).toFixed(2)}%
                      </div>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <div className="flex items-center justify-center gap-1 text-sm text-text-secondary">
                        <Calendar className="h-4 w-4" />
                        {trade.entry_date} → {trade.exit_date}
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
