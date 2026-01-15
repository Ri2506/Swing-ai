// ============================================================================
// SWINGAI - ADMIN SIGNALS PAGE
// Signal analytics and performance tracking
// ============================================================================

'use client'

import { useEffect, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  Target,
  TrendingUp,
  TrendingDown,
  Activity,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  XCircle,
  BarChart3,
} from 'lucide-react'
import { SignalStats } from '@/types/admin'

export default function AdminSignalsPage() {
  const [stats, setStats] = useState<SignalStats | null>(null)
  const [recentSignals, setRecentSignals] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [period, setPeriod] = useState(30)

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || ''

      // Fetch stats
      const statsRes = await fetch(`${apiUrl}/api/admin/signals/stats?days=${period}`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      })

      if (statsRes.ok) {
        setStats(await statsRes.json())
      } else {
        setStats(getMockStats())
      }

      // Fetch recent signals
      const signalsRes = await fetch(`${apiUrl}/api/signals/history?limit=20`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      })

      if (signalsRes.ok) {
        const data = await signalsRes.json()
        setRecentSignals(data.signals || [])
      } else {
        setRecentSignals(getMockSignals())
      }
    } catch (err) {
      console.error('Failed to fetch signals data:', err)
      setStats(getMockStats())
      setRecentSignals(getMockSignals())
    } finally {
      setLoading(false)
    }
  }, [period])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const getToken = () => {
    if (typeof window === 'undefined') return ''
    return localStorage.getItem('sb-access-token') || ''
  }

  const getMockStats = (): SignalStats => ({
    period_days: period,
    total_signals: 342,
    target_hit: 198,
    sl_hit: 122,
    accuracy: 61.88,
    avg_per_day: 11.4,
  })

  const getMockSignals = () => [
    {
      id: '1',
      symbol: 'RELIANCE',
      direction: 'LONG',
      entry_price: 2450,
      stop_loss: 2400,
      target_1: 2550,
      confidence: 78,
      status: 'target_hit',
      date: '2025-08-15',
    },
    {
      id: '2',
      symbol: 'TCS',
      direction: 'SHORT',
      entry_price: 3680,
      stop_loss: 3750,
      target_1: 3550,
      confidence: 72,
      status: 'sl_hit',
      date: '2025-08-15',
    },
    {
      id: '3',
      symbol: 'INFY',
      direction: 'LONG',
      entry_price: 1480,
      stop_loss: 1450,
      target_1: 1550,
      confidence: 85,
      status: 'active',
      date: '2025-08-15',
    },
    {
      id: '4',
      symbol: 'HDFC',
      direction: 'LONG',
      entry_price: 1650,
      stop_loss: 1600,
      target_1: 1750,
      confidence: 68,
      status: 'target_hit',
      date: '2025-08-14',
    },
  ]

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'target_hit':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-500/10 text-green-500 rounded-full text-xs font-medium">
            <CheckCircle className="w-3 h-3" /> Target Hit
          </span>
        )
      case 'sl_hit':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-red-500/10 text-red-500 rounded-full text-xs font-medium">
            <XCircle className="w-3 h-3" /> SL Hit
          </span>
        )
      case 'active':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-500/10 text-blue-500 rounded-full text-xs font-medium">
            <Activity className="w-3 h-3" /> Active
          </span>
        )
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 bg-gray-500/10 text-gray-500 rounded-full text-xs font-medium">
            {status}
          </span>
        )
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-500"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Signal Analytics</h1>
          <p className="text-gray-400 mt-1">AI signal performance and accuracy tracking</p>
        </div>
        <button
          onClick={fetchData}
          className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RefreshCw className="w-5 h-5 text-gray-400" />
        </button>
      </div>

      {/* Period Selector */}
      <div className="flex gap-2">
        {[7, 30, 90, 365].map((days) => (
          <button
            key={days}
            onClick={() => setPeriod(days)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              period === days
                ? 'bg-red-500 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {days === 365 ? '1 Year' : `${days} Days`}
          </button>
        ))}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-900 rounded-2xl border border-gray-800 p-6"
        >
          <div className="flex items-center gap-3 mb-2">
            <Target className="w-5 h-5 text-blue-500" />
            <span className="text-sm text-gray-400">Total Signals</span>
          </div>
          <p className="text-3xl font-bold text-white">{stats?.total_signals || 0}</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gray-900 rounded-2xl border border-gray-800 p-6"
        >
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-5 h-5 text-green-500" />
            <span className="text-sm text-gray-400">Target Hit</span>
          </div>
          <p className="text-3xl font-bold text-green-500">{stats?.target_hit || 0}</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gray-900 rounded-2xl border border-gray-800 p-6"
        >
          <div className="flex items-center gap-3 mb-2">
            <TrendingDown className="w-5 h-5 text-red-500" />
            <span className="text-sm text-gray-400">SL Hit</span>
          </div>
          <p className="text-3xl font-bold text-red-500">{stats?.sl_hit || 0}</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-900 rounded-2xl border border-gray-800 p-6"
        >
          <div className="flex items-center gap-3 mb-2">
            <BarChart3 className="w-5 h-5 text-purple-500" />
            <span className="text-sm text-gray-400">Accuracy</span>
          </div>
          <p
            className={`text-3xl font-bold ${
              (stats?.accuracy || 0) >= 55 ? 'text-green-500' : 'text-yellow-500'
            }`}
          >
            {stats?.accuracy.toFixed(1) || 0}%
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-gray-900 rounded-2xl border border-gray-800 p-6"
        >
          <div className="flex items-center gap-3 mb-2">
            <Activity className="w-5 h-5 text-orange-500" />
            <span className="text-sm text-gray-400">Avg/Day</span>
          </div>
          <p className="text-3xl font-bold text-white">{stats?.avg_per_day.toFixed(1) || 0}</p>
        </motion.div>
      </div>

      {/* Accuracy Gauge */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Accuracy Overview</h2>
        <div className="flex items-center gap-6">
          <div className="relative w-32 h-32">
            <svg className="w-32 h-32 transform -rotate-90">
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="currentColor"
                strokeWidth="8"
                fill="none"
                className="text-gray-800"
              />
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="currentColor"
                strokeWidth="8"
                fill="none"
                strokeDasharray={`${(stats?.accuracy || 0) * 3.52} 352`}
                className={(stats?.accuracy || 0) >= 55 ? 'text-green-500' : 'text-yellow-500'}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold text-white">{stats?.accuracy.toFixed(0) || 0}%</span>
            </div>
          </div>
          <div className="flex-1 space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Winners</span>
                <span className="text-green-500">{stats?.target_hit || 0}</span>
              </div>
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 rounded-full"
                  style={{
                    width: `${((stats?.target_hit || 0) / (stats?.total_signals || 1)) * 100}%`,
                  }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Losers</span>
                <span className="text-red-500">{stats?.sl_hit || 0}</span>
              </div>
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-red-500 rounded-full"
                  style={{
                    width: `${((stats?.sl_hit || 0) / (stats?.total_signals || 1)) * 100}%`,
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Signals */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">
        <div className="p-6 border-b border-gray-800">
          <h2 className="text-lg font-semibold text-white">Recent Signals</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Symbol</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Direction</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Entry</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">SL</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Target</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Confidence</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {recentSignals.map((signal) => (
                <tr key={signal.id} className="hover:bg-gray-800/30 transition-colors">
                  <td className="px-4 py-4">
                    <span className="text-white font-medium">{signal.symbol}</span>
                  </td>
                  <td className="px-4 py-4">
                    <span
                      className={`inline-flex items-center gap-1 ${
                        signal.direction === 'LONG' ? 'text-green-500' : 'text-red-500'
                      }`}
                    >
                      {signal.direction === 'LONG' ? (
                        <TrendingUp className="w-4 h-4" />
                      ) : (
                        <TrendingDown className="w-4 h-4" />
                      )}
                      {signal.direction}
                    </span>
                  </td>
                  <td className="px-4 py-4 text-gray-300">₹{signal.entry_price}</td>
                  <td className="px-4 py-4 text-red-400">₹{signal.stop_loss}</td>
                  <td className="px-4 py-4 text-green-400">₹{signal.target_1}</td>
                  <td className="px-4 py-4">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            signal.confidence >= 75
                              ? 'bg-green-500'
                              : signal.confidence >= 60
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                          }`}
                          style={{ width: `${signal.confidence}%` }}
                        />
                      </div>
                      <span className="text-gray-400 text-sm">{signal.confidence}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-4">{getStatusBadge(signal.status)}</td>
                  <td className="px-4 py-4 text-gray-500 text-sm">{signal.date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
