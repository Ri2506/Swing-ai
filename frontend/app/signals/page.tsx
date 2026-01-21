'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import {
  Target,
  TrendingUp,
  ArrowUp,
  ArrowDown,
  Sparkles,
  Activity,
  Filter,
  Clock,
  AlertCircle,
} from 'lucide-react'

interface Signal {
  id: string
  symbol: string
  name: string
  direction: 'LONG'
  entry_price: number
  target_price: number
  stop_loss: number
  confidence: number
  risk_reward: number
  generated_at: string
  status: 'active' | 'triggered' | 'expired'
}

export default function SignalsPage() {
  const [signals, setSignals] = useState<Signal[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'active' | 'triggered'>('all')

  useEffect(() => {
    fetchSignals()
  }, [])

  const fetchSignals = async () => {
    try {
      // Mock signals data - LONG positions only
      const mockSignals: Signal[] = [
        {
          id: '1',
          symbol: 'RELIANCE',
          name: 'Reliance Industries Ltd',
          direction: 'LONG',
          entry_price: 2847.50,
          target_price: 3020.00,
          stop_loss: 2780.00,
          confidence: 89,
          risk_reward: 2.57,
          generated_at: new Date(Date.now() - 2 * 60000).toISOString(),
          status: 'active',
        },
        {
          id: '2',
          symbol: 'TCS',
          name: 'Tata Consultancy Services',
          direction: 'LONG',
          entry_price: 3678.90,
          target_price: 3850.00,
          stop_loss: 3580.00,
          confidence: 82,
          risk_reward: 1.73,
          generated_at: new Date(Date.now() - 45 * 60000).toISOString(),
          status: 'active',
        },
        {
          id: '3',
          symbol: 'HDFCBANK',
          name: 'HDFC Bank Ltd',
          direction: 'LONG',
          entry_price: 1650.00,
          target_price: 1780.00,
          stop_loss: 1600.00,
          confidence: 75,
          risk_reward: 2.60,
          generated_at: new Date(Date.now() - 2 * 3600000).toISOString(),
          status: 'triggered',
        },
        {
          id: '4',
          symbol: 'INFY',
          name: 'Infosys Ltd',
          direction: 'LONG',
          entry_price: 1523.45,
          target_price: 1620.00,
          stop_loss: 1480.00,
          confidence: 78,
          risk_reward: 2.22,
          generated_at: new Date(Date.now() - 4 * 3600000).toISOString(),
          status: 'active',
        },
        {
          id: '5',
          symbol: 'BHARTIARTL',
          name: 'Bharti Airtel Ltd',
          direction: 'LONG',
          entry_price: 1547.00,
          target_price: 1680.00,
          stop_loss: 1490.00,
          confidence: 84,
          risk_reward: 2.33,
          generated_at: new Date(Date.now() - 6 * 3600000).toISOString(),
          status: 'expired',
        },
      ]
      setSignals(mockSignals)
    } catch (error) {
      console.error('Error fetching signals:', error)
    } finally {
      setLoading(false)
    }
  }

  const filteredSignals = signals.filter(signal => {
    if (filter === 'all') return true
    return signal.status === filter
  })

  const getTimeAgo = (dateString: string) => {
    const diff = Date.now() - new Date(dateString).getTime()
    const minutes = Math.floor(diff / 60000)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    return `${Math.floor(hours / 24)}d ago`
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background-primary">
        <div className="text-center">
          <Activity className="mx-auto h-12 w-12 animate-pulse text-accent" />
          <p className="mt-4 text-lg text-text-secondary">Loading AI signals...</p>
        </div>
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
                <span className="gradient-text-professional">AI Trading Signals</span>
              </h1>
              <p className="text-lg text-text-secondary">
                Real-time swing trade signals powered by artificial intelligence
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

        {/* Stats Overview */}
        <div className="mb-8 grid gap-4 md:grid-cols-4">
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-5">
            <div className="mb-2 text-sm font-medium text-text-secondary">Active Signals</div>
            <div className="text-3xl font-bold text-primary">{signals.filter(s => s.status === 'active').length}</div>
          </div>
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-5">
            <div className="mb-2 text-sm font-medium text-text-secondary">Avg Confidence</div>
            <div className="text-3xl font-bold text-accent">{Math.round(signals.reduce((a, b) => a + b.confidence, 0) / signals.length)}%</div>
          </div>
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-5">
            <div className="mb-2 text-sm font-medium text-text-secondary">Triggered Today</div>
            <div className="text-3xl font-bold text-success">{signals.filter(s => s.status === 'triggered').length}</div>
          </div>
          <div className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-5">
            <div className="mb-2 text-sm font-medium text-text-secondary">Avg Risk:Reward</div>
            <div className="text-3xl font-bold text-text-primary">{(signals.reduce((a, b) => a + b.risk_reward, 0) / signals.length).toFixed(2)}:1</div>
          </div>
        </div>

        {/* Filters */}
        <div className="mb-6 flex items-center gap-3">
          <Filter className="h-5 w-5 text-text-secondary" />
          <div className="flex gap-2">
            {(['all', 'active', 'triggered'] as const).map((f) => (
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

        {/* Signals List */}
        <div className="space-y-4">
          {filteredSignals.map((signal, index) => (
            <motion.div
              key={signal.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="overflow-hidden rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated transition hover:border-accent/40"
            >
              <div className="border-b border-border/30 bg-background-surface/50 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-success/15">
                      <TrendingUp className="h-5 w-5 text-success" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-lg font-bold text-text-primary">{signal.symbol}</span>
                        <span className="rounded px-2 py-0.5 text-xs font-bold bg-success/15 text-success">
                          BUY
                        </span>
                        <span className={`rounded px-2 py-0.5 text-xs font-medium ${
                          signal.status === 'active' ? 'bg-accent/15 text-accent' :
                          signal.status === 'triggered' ? 'bg-success/15 text-success' :
                          'bg-text-secondary/15 text-text-secondary'
                        }`}>
                          {signal.status.toUpperCase()}
                        </span>
                      </div>
                      <span className="text-sm text-text-secondary">{signal.name}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1 text-sm text-text-secondary">
                      <Clock className="h-4 w-4" />
                      {getTimeAgo(signal.generated_at)}
                    </div>
                    <div className="flex items-center gap-1 rounded-full bg-accent/15 px-3 py-1">
                      <Sparkles className="h-4 w-4 text-accent" />
                      <span className="text-sm font-bold text-accent">{signal.confidence}%</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid gap-4 p-6 md:grid-cols-4">
                <div>
                  <div className="mb-1 text-xs font-medium text-text-secondary">Entry Price</div>
                  <div className="rounded-lg bg-accent/10 px-4 py-3 text-center">
                    <div className="text-lg font-bold text-accent">₹{signal.entry_price.toFixed(2)}</div>
                  </div>
                </div>
                <div>
                  <div className="mb-1 text-xs font-medium text-text-secondary">Target</div>
                  <div className="rounded-lg bg-success/10 px-4 py-3 text-center">
                    <div className="text-lg font-bold text-success">₹{signal.target_price.toFixed(2)}</div>
                  </div>
                </div>
                <div>
                  <div className="mb-1 text-xs font-medium text-text-secondary">Stop Loss</div>
                  <div className="rounded-lg bg-danger/10 px-4 py-3 text-center">
                    <div className="text-lg font-bold text-danger">₹{signal.stop_loss.toFixed(2)}</div>
                  </div>
                </div>
                <div>
                  <div className="mb-1 text-xs font-medium text-text-secondary">Risk:Reward</div>
                  <div className="rounded-lg bg-primary/10 px-4 py-3 text-center">
                    <div className="text-lg font-bold text-primary">{signal.risk_reward.toFixed(2)}:1</div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {filteredSignals.length === 0 && (
          <div className="rounded-xl border border-border/60 bg-background-surface p-12 text-center">
            <AlertCircle className="mx-auto h-12 w-12 text-text-secondary" />
            <p className="mt-4 text-lg text-text-secondary">No signals found for the selected filter</p>
          </div>
        )}
      </div>
    </div>
  )
}
