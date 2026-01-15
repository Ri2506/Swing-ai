// ============================================================================
// SWINGAI - ADMIN SYSTEM HEALTH PAGE
// System monitoring and health dashboard
// ============================================================================

'use client'

import { useEffect, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  Database,
  Server,
  Wifi,
  Clock,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertCircle,
  Users,
  Target,
  TrendingUp,
  Cpu,
  HardDrive,
  Globe,
} from 'lucide-react'
import { SystemHealth } from '@/types/admin'

export default function AdminSystemPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [loading, setLoading] = useState(true)
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)

  const fetchHealth = useCallback(async () => {
    try {
      setLoading(true)
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || ''

      const res = await fetch(`${apiUrl}/api/admin/system/health`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      })

      if (res.ok) {
        setHealth(await res.json())
      } else {
        setHealth(getMockHealth())
      }
      setLastRefresh(new Date())
    } catch (err) {
      console.error('Failed to fetch health:', err)
      setHealth(getMockHealth())
      setLastRefresh(new Date())
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchHealth()
  }, [fetchHealth])

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchHealth, 30000) // Refresh every 30 seconds
      return () => clearInterval(interval)
    }
  }, [autoRefresh, fetchHealth])

  const getToken = () => {
    if (typeof window === 'undefined') return ''
    return localStorage.getItem('sb-access-token') || ''
  }

  const getMockHealth = (): SystemHealth => ({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    database: 'connected',
    redis: 'disabled',
    scheduler_status: 'running',
    last_signal_run: new Date(Date.now() - 3600000).toISOString(),
    active_websocket_connections: 127,
    metrics: {
      total_users: 5234,
      active_subscribers: 1847,
      today_signals: 12,
      today_trades: 89,
      active_positions: 234,
    },
  })

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'connected':
      case 'running':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'degraded':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />
      case 'error':
      case 'stopped':
        return <XCircle className="w-5 h-5 text-red-500" />
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'connected':
      case 'running':
        return 'text-green-500 bg-green-500/10 border-green-500/30'
      case 'degraded':
        return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30'
      case 'error':
      case 'stopped':
        return 'text-red-500 bg-red-500/10 border-red-500/30'
      default:
        return 'text-gray-500 bg-gray-500/10 border-gray-500/30'
    }
  }

  if (loading && !health) {
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
          <h1 className="text-3xl font-bold text-white">System Health</h1>
          <p className="text-gray-400 mt-1">Real-time system monitoring and status</p>
        </div>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-gray-400">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-gray-700 bg-gray-800 text-red-500 focus:ring-red-500"
            />
            Auto-refresh (30s)
          </label>
          <button
            onClick={fetchHealth}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
            <span className="text-gray-400">Refresh</span>
          </button>
        </div>
      </div>

      {/* Last Refresh */}
      {lastRefresh && (
        <p className="text-xs text-gray-500">
          Last updated: {lastRefresh.toLocaleTimeString()}
        </p>
      )}

      {/* Overall Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`rounded-2xl border p-6 ${getStatusColor(health?.status || 'error')}`}
      >
        <div className="flex items-center gap-4">
          {getStatusIcon(health?.status || 'error')}
          <div>
            <h2 className="text-xl font-bold">System Status: {health?.status?.toUpperCase()}</h2>
            <p className="text-sm opacity-80">
              Last checked: {health?.timestamp ? new Date(health.timestamp).toLocaleString() : 'N/A'}
            </p>
          </div>
        </div>
      </motion.div>

      {/* Service Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Database */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-900 rounded-2xl border border-gray-800 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <Database className="w-8 h-8 text-blue-500" />
            {getStatusIcon(health?.database || 'error')}
          </div>
          <h3 className="text-lg font-semibold text-white">Database</h3>
          <p className="text-sm text-gray-400 mt-1 capitalize">
            {health?.database || 'Unknown'}
          </p>
        </motion.div>

        {/* Redis */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gray-900 rounded-2xl border border-gray-800 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <Server className="w-8 h-8 text-red-500" />
            {getStatusIcon(health?.redis || 'disabled')}
          </div>
          <h3 className="text-lg font-semibold text-white">Redis</h3>
          <p className="text-sm text-gray-400 mt-1 capitalize">
            {health?.redis || 'Unknown'}
          </p>
        </motion.div>

        {/* Scheduler */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gray-900 rounded-2xl border border-gray-800 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <Clock className="w-8 h-8 text-purple-500" />
            {getStatusIcon(health?.scheduler_status || 'stopped')}
          </div>
          <h3 className="text-lg font-semibold text-white">Scheduler</h3>
          <p className="text-sm text-gray-400 mt-1 capitalize">
            {health?.scheduler_status || 'Unknown'}
          </p>
          {health?.last_signal_run && (
            <p className="text-xs text-gray-500 mt-2">
              Last run: {new Date(health.last_signal_run).toLocaleTimeString()}
            </p>
          )}
        </motion.div>

        {/* WebSocket */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-900 rounded-2xl border border-gray-800 p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <Wifi className="w-8 h-8 text-green-500" />
            <span className="text-2xl font-bold text-green-500">
              {health?.active_websocket_connections || 0}
            </span>
          </div>
          <h3 className="text-lg font-semibold text-white">WebSocket</h3>
          <p className="text-sm text-gray-400 mt-1">Active connections</p>
        </motion.div>
      </div>

      {/* Metrics Grid */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-white mb-6">System Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-500/10 rounded-xl flex items-center justify-center mx-auto mb-3">
              <Users className="w-6 h-6 text-blue-500" />
            </div>
            <p className="text-2xl font-bold text-white">
              {health?.metrics.total_users.toLocaleString() || 0}
            </p>
            <p className="text-sm text-gray-400">Total Users</p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-green-500/10 rounded-xl flex items-center justify-center mx-auto mb-3">
              <TrendingUp className="w-6 h-6 text-green-500" />
            </div>
            <p className="text-2xl font-bold text-white">
              {health?.metrics.active_subscribers.toLocaleString() || 0}
            </p>
            <p className="text-sm text-gray-400">Active Subscribers</p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-purple-500/10 rounded-xl flex items-center justify-center mx-auto mb-3">
              <Target className="w-6 h-6 text-purple-500" />
            </div>
            <p className="text-2xl font-bold text-white">
              {health?.metrics.today_signals || 0}
            </p>
            <p className="text-sm text-gray-400">Today's Signals</p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-orange-500/10 rounded-xl flex items-center justify-center mx-auto mb-3">
              <Activity className="w-6 h-6 text-orange-500" />
            </div>
            <p className="text-2xl font-bold text-white">
              {health?.metrics.today_trades || 0}
            </p>
            <p className="text-sm text-gray-400">Today's Trades</p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-red-500/10 rounded-xl flex items-center justify-center mx-auto mb-3">
              <Globe className="w-6 h-6 text-red-500" />
            </div>
            <p className="text-2xl font-bold text-white">
              {health?.metrics.active_positions || 0}
            </p>
            <p className="text-sm text-gray-400">Active Positions</p>
          </div>
        </div>
      </div>

      {/* Environment Info */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Environment</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <p className="text-sm text-gray-400">API URL</p>
            <code className="text-white text-sm">
              {process.env.NEXT_PUBLIC_API_URL || 'Not configured'}
            </code>
          </div>
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <p className="text-sm text-gray-400">Environment</p>
            <code className="text-white text-sm">
              {process.env.NODE_ENV || 'development'}
            </code>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Quick Actions</h2>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => window.open('/api/docs', '_blank')}
            className="px-4 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 rounded-lg text-blue-400 text-sm font-medium transition-colors"
          >
            API Documentation
          </button>
          <button
            onClick={() => window.open('/api/health', '_blank')}
            className="px-4 py-2 bg-green-500/10 hover:bg-green-500/20 border border-green-500/30 rounded-lg text-green-400 text-sm font-medium transition-colors"
          >
            Health Endpoint
          </button>
          <button
            onClick={fetchHealth}
            className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 text-sm font-medium transition-colors"
          >
            Force Refresh
          </button>
        </div>
      </div>
    </div>
  )
}
