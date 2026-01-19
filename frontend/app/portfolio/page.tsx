'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  ArrowUp,
  ArrowDown,
  PieChart,
  BarChart3,
  Activity,
} from 'lucide-react'

interface Position {
  id: string
  symbol: string
  name: string
  quantity: number
  avg_price: number
  current_price: number
  pnl: number
  pnl_percent: number
  value: number
}

export default function PortfolioPage() {
  const [positions, setPositions] = useState<Position[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Mock portfolio data
    const mockPositions: Position[] = [
      {
        id: '1',
        symbol: 'RELIANCE',
        name: 'Reliance Industries Ltd',
        quantity: 50,
        avg_price: 2780.00,
        current_price: 2847.50,
        pnl: 3375.00,
        pnl_percent: 2.43,
        value: 142375.00,
      },
      {
        id: '2',
        symbol: 'TCS',
        name: 'Tata Consultancy Services',
        quantity: 30,
        avg_price: 3650.00,
        current_price: 3678.90,
        pnl: 867.00,
        pnl_percent: 0.79,
        value: 110367.00,
      },
      {
        id: '3',
        symbol: 'INFY',
        name: 'Infosys Ltd',
        quantity: 100,
        avg_price: 1550.00,
        current_price: 1523.45,
        pnl: -2655.00,
        pnl_percent: -1.71,
        value: 152345.00,
      },
      {
        id: '4',
        symbol: 'HDFCBANK',
        name: 'HDFC Bank Ltd',
        quantity: 75,
        avg_price: 1650.00,
        current_price: 1678.00,
        pnl: 2100.00,
        pnl_percent: 1.70,
        value: 125850.00,
      },
    ]
    setPositions(mockPositions)
    setLoading(false)
  }, [])

  const totalValue = positions.reduce((sum, p) => sum + p.value, 0)
  const totalPnL = positions.reduce((sum, p) => sum + p.pnl, 0)
  const totalInvested = positions.reduce((sum, p) => sum + (p.avg_price * p.quantity), 0)
  const overallPnLPercent = (totalPnL / totalInvested) * 100

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
                <span className="gradient-text-professional">Portfolio</span>
              </h1>
              <p className="text-lg text-text-secondary">
                Track your holdings and performance
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

        {/* Portfolio Summary */}
        <div className="mb-8 grid gap-4 md:grid-cols-4">
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium text-text-secondary">
              <Wallet className="h-4 w-4" />
              Portfolio Value
            </div>
            <div className="text-3xl font-bold text-text-primary">₹{totalValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
          </div>
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium text-text-secondary">
              <TrendingUp className="h-4 w-4" />
              Total P&L
            </div>
            <div className={`text-3xl font-bold ${totalPnL >= 0 ? 'text-success' : 'text-danger'}`}>
              {totalPnL >= 0 ? '+' : ''}₹{totalPnL.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </div>
            <div className={`mt-1 flex items-center gap-1 text-sm ${totalPnL >= 0 ? 'text-success' : 'text-danger'}`}>
              {totalPnL >= 0 ? <ArrowUp className="h-4 w-4" /> : <ArrowDown className="h-4 w-4" />}
              {Math.abs(overallPnLPercent).toFixed(2)}%
            </div>
          </div>
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium text-text-secondary">
              <PieChart className="h-4 w-4" />
              Positions
            </div>
            <div className="text-3xl font-bold text-accent">{positions.length}</div>
          </div>
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium text-text-secondary">
              <BarChart3 className="h-4 w-4" />
              Total Invested
            </div>
            <div className="text-3xl font-bold text-text-primary">₹{totalInvested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
          </div>
        </div>

        {/* Holdings Table */}
        <div className="overflow-hidden rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated">
          <div className="border-b border-border/40 bg-background-surface/50 px-6 py-4">
            <h2 className="text-xl font-bold text-text-primary">Holdings</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-border/40 bg-background-elevated/50">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-text-secondary">Stock</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-text-secondary">Qty</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-text-secondary">Avg Price</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-text-secondary">LTP</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-text-secondary">Current Value</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-text-secondary">P&L</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {positions.map((position, index) => (
                  <motion.tr
                    key={position.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="transition-colors hover:bg-background-elevated/30"
                  >
                    <td className="px-6 py-4">
                      <div className="font-semibold text-text-primary">{position.symbol}</div>
                      <div className="text-sm text-text-secondary">{position.name}</div>
                    </td>
                    <td className="px-6 py-4 text-right font-medium text-text-primary">{position.quantity}</td>
                    <td className="px-6 py-4 text-right text-text-secondary">₹{position.avg_price.toFixed(2)}</td>
                    <td className="px-6 py-4 text-right font-medium text-text-primary">₹{position.current_price.toFixed(2)}</td>
                    <td className="px-6 py-4 text-right font-medium text-text-primary">
                      ₹{position.value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className={`font-bold ${position.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                        {position.pnl >= 0 ? '+' : ''}₹{position.pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                      </div>
                      <div className={`flex items-center justify-end gap-1 text-sm ${position.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                        {position.pnl >= 0 ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
                        {Math.abs(position.pnl_percent).toFixed(2)}%
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
