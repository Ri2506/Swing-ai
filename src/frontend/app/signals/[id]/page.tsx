// ============================================================================
// SWINGAI - SIGNAL DETAIL PAGE
// Detailed view of individual signal with chart and analysis
// ============================================================================

'use client'

import { useState, useEffect } from 'react'
import { useRouter, useParams } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { useAuth } from '../../../contexts/AuthContext'
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Target,
  Shield,
  Clock,
  BarChart3,
  CheckCircle,
  XCircle,
  AlertCircle,
  Zap,
  Brain,
  Activity,
} from 'lucide-react'

export default function SignalDetailPage() {
  const router = useRouter()
  const params = useParams()
  const { user } = useAuth()
  const [signal, setSignal] = useState<any>(null)

  useEffect(() => {
    if (!user) {
      router.push('/login')
      return
    }

    // Mock signal data (in real app, fetch from API)
    setSignal({
      id: params.id,
      symbol: 'RELIANCE',
      exchange: 'NSE',
      segment: 'EQUITY',
      direction: 'LONG',
      entry_price: 2456.75,
      current_price: 2468.30,
      stop_loss: 2400.00,
      target: 2550.00,
      confidence: 85,
      risk_reward_ratio: 2.5,
      position_size: 100,
      status: 'active',
      created_at: new Date(Date.now() - 3600000).toISOString(),
      valid_until: new Date(Date.now() + 82800000).toISOString(),
      model_predictions: {
        catboost: { prediction: 'LONG', confidence: 82 },
        tft: { prediction: 'LONG', confidence: 88 },
        stockformer: { prediction: 'LONG', confidence: 85 },
        ensemble_confidence: 85,
        model_agreement: 1.0,
      },
      technical_analysis: {
        rsi: 58.3,
        macd: { value: 12.5, signal: 10.2, histogram: 2.3 },
        volume_ratio: 1.8,
        support_levels: [2400, 2420, 2440],
        resistance_levels: [2490, 2520, 2550],
      },
    })
  }, [user, router, params.id])

  if (!user || !signal) return null

  const isProfitable = signal.current_price > signal.entry_price
  const pnl = (signal.current_price - signal.entry_price) * signal.position_size
  const pnlPercent = ((signal.current_price - signal.entry_price) / signal.entry_price) * 100

  return (
    <div className="min-h-screen bg-background-primary">
      {/* Header */}
      <div className="border-b border-gray-800 bg-background-surface/50 backdrop-blur-xl sticky top-0 z-10">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/signals"
                className="p-2 rounded-lg hover:bg-background-elevated transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-text-secondary" />
              </Link>
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-3xl font-bold text-text-primary">{signal.symbol}</h1>
                  <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                    signal.direction === 'LONG'
                      ? 'bg-success/20 text-success'
                      : 'bg-danger/20 text-danger'
                  }`}>
                    {signal.direction}
                  </span>
                  <span className="px-3 py-1 rounded-full text-sm bg-gray-800 text-text-secondary">
                    {signal.segment}
                  </span>
                </div>
                <p className="text-text-secondary mt-1">{signal.exchange} • Signal #{signal.id}</p>
              </div>
            </div>

            <button className="px-6 py-3 bg-gradient-primary text-white rounded-xl font-medium hover:shadow-glow-md transition-all">
              Execute Trade
            </button>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Price Info */}
            <div className="bg-background-surface/50 backdrop-blur-xl rounded-2xl border border-gray-800 p-6">
              <h2 className="text-xl font-bold text-text-primary mb-6">Price Information</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-text-muted text-sm mb-1">Entry Price</div>
                  <div className="text-xl font-bold text-text-primary font-mono">
                    ₹{signal.entry_price.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-text-muted text-sm mb-1">Current Price</div>
                  <div className={`text-xl font-bold font-mono ${isProfitable ? 'text-success' : 'text-danger'}`}>
                    ₹{signal.current_price.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-text-muted text-sm mb-1">Stop Loss</div>
                  <div className="text-xl font-bold text-danger font-mono">
                    ₹{signal.stop_loss.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-text-muted text-sm mb-1">Target</div>
                  <div className="text-xl font-bold text-success font-mono">
                    ₹{signal.target.toFixed(2)}
                  </div>
                </div>
              </div>

              {/* P&L */}
              <div className="mt-6 p-4 rounded-xl bg-background-elevated border border-gray-800">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-text-muted text-sm mb-1">Unrealized P&L</div>
                    <div className={`text-2xl font-bold font-mono ${isProfitable ? 'text-success' : 'text-danger'}`}>
                      {isProfitable ? '+' : ''}₹{pnl.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-text-muted text-sm mb-1">Return</div>
                    <div className={`text-2xl font-bold ${isProfitable ? 'text-success' : 'text-danger'}`}>
                      {pnlPercent > 0 ? '+' : ''}{pnlPercent.toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* AI Model Predictions */}
            <div className="bg-background-surface/50 backdrop-blur-xl rounded-2xl border border-gray-800 p-6">
              <h2 className="text-xl font-bold text-text-primary mb-6">AI Model Predictions</h2>

              <div className="space-y-4">
                {/* CatBoost */}
                <div className="p-4 rounded-xl bg-background-elevated border border-gray-800">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-blue-500/10">
                        <BarChart3 className="w-5 h-5 text-blue-500" />
                      </div>
                      <div>
                        <div className="font-bold text-text-primary">CatBoost</div>
                        <div className="text-sm text-text-muted">Gradient Boosting</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${
                        signal.model_predictions.catboost.prediction === 'LONG'
                          ? 'text-success'
                          : 'text-danger'
                      }`}>
                        {signal.model_predictions.catboost.prediction}
                      </div>
                      <div className="text-sm text-text-secondary">
                        {signal.model_predictions.catboost.confidence}% confidence
                      </div>
                    </div>
                  </div>
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-blue-400"
                      style={{ width: `${signal.model_predictions.catboost.confidence}%` }}
                    />
                  </div>
                </div>

                {/* TFT */}
                <div className="p-4 rounded-xl bg-background-elevated border border-gray-800">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-purple-500/10">
                        <Activity className="w-5 h-5 text-purple-500" />
                      </div>
                      <div>
                        <div className="font-bold text-text-primary">Temporal Fusion Transformer</div>
                        <div className="text-sm text-text-muted">Time Series Forecasting</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${
                        signal.model_predictions.tft.prediction === 'LONG'
                          ? 'text-success'
                          : 'text-danger'
                      }`}>
                        {signal.model_predictions.tft.prediction}
                      </div>
                      <div className="text-sm text-text-secondary">
                        {signal.model_predictions.tft.confidence}% confidence
                      </div>
                    </div>
                  </div>
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-purple-400"
                      style={{ width: `${signal.model_predictions.tft.confidence}%` }}
                    />
                  </div>
                </div>

                {/* Stockformer */}
                <div className="p-4 rounded-xl bg-background-elevated border border-gray-800">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-green-500/10">
                        <TrendingUp className="w-5 h-5 text-green-500" />
                      </div>
                      <div>
                        <div className="font-bold text-text-primary">Stockformer</div>
                        <div className="text-sm text-text-muted">Stock-Specific Transformer</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${
                        signal.model_predictions.stockformer.prediction === 'LONG'
                          ? 'text-success'
                          : 'text-danger'
                      }`}>
                        {signal.model_predictions.stockformer.prediction}
                      </div>
                      <div className="text-sm text-text-secondary">
                        {signal.model_predictions.stockformer.confidence}% confidence
                      </div>
                    </div>
                  </div>
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-500 to-green-400"
                      style={{ width: `${signal.model_predictions.stockformer.confidence}%` }}
                    />
                  </div>
                </div>

                {/* Ensemble */}
                <div className="p-4 rounded-xl bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-500/30">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-blue-500/20">
                        <Brain className="w-5 h-5 text-blue-400" />
                      </div>
                      <div>
                        <div className="font-bold text-white">Ensemble Prediction</div>
                        <div className="text-sm text-blue-200">
                          {(signal.model_predictions.model_agreement * 100).toFixed(0)}% Model Agreement
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-white">
                        {signal.model_predictions.ensemble_confidence}%
                      </div>
                      <div className="text-sm text-blue-200">Confidence</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Technical Analysis */}
            <div className="bg-background-surface/50 backdrop-blur-xl rounded-2xl border border-gray-800 p-6">
              <h2 className="text-xl font-bold text-text-primary mb-6">Technical Analysis</h2>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-background-elevated border border-gray-800">
                  <div className="text-text-muted text-sm mb-1">RSI (14)</div>
                  <div className="text-2xl font-bold text-text-primary">{signal.technical_analysis.rsi}</div>
                  <div className="text-xs text-text-secondary mt-1">
                    {signal.technical_analysis.rsi > 70 ? 'Overbought' : signal.technical_analysis.rsi < 30 ? 'Oversold' : 'Neutral'}
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-background-elevated border border-gray-800">
                  <div className="text-text-muted text-sm mb-1">MACD</div>
                  <div className="text-2xl font-bold text-text-primary">{signal.technical_analysis.macd.value}</div>
                  <div className="text-xs text-success mt-1">Bullish Crossover</div>
                </div>

                <div className="p-4 rounded-xl bg-background-elevated border border-gray-800">
                  <div className="text-text-muted text-sm mb-1">Volume Ratio</div>
                  <div className="text-2xl font-bold text-text-primary">{signal.technical_analysis.volume_ratio}x</div>
                  <div className="text-xs text-text-secondary mt-1">Above Average</div>
                </div>

                <div className="p-4 rounded-xl bg-background-elevated border border-gray-800">
                  <div className="text-text-muted text-sm mb-1">Risk:Reward</div>
                  <div className="text-2xl font-bold text-success">1:{signal.risk_reward_ratio}</div>
                  <div className="text-xs text-text-secondary mt-1">Favorable</div>
                </div>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Signal Status */}
            <div className="bg-background-surface/50 backdrop-blur-xl rounded-2xl border border-gray-800 p-6">
              <h3 className="text-lg font-bold text-text-primary mb-4">Signal Status</h3>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Status</span>
                  <span className="px-3 py-1 rounded-full text-sm font-bold bg-success/20 text-success">
                    Active
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Created</span>
                  <span className="text-text-primary font-mono text-sm">
                    {new Date(signal.created_at).toLocaleString()}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Valid Until</span>
                  <span className="text-text-primary font-mono text-sm">
                    {new Date(signal.valid_until).toLocaleString()}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Position Size</span>
                  <span className="text-text-primary font-bold">{signal.position_size} shares</span>
                </div>
              </div>
            </div>

            {/* Risk Management */}
            <div className="bg-background-surface/50 backdrop-blur-xl rounded-2xl border border-gray-800 p-6">
              <h3 className="text-lg font-bold text-text-primary mb-4">Risk Management</h3>

              <div className="space-y-3">
                <div className="flex items-center gap-2 text-danger">
                  <Shield className="w-4 h-4" />
                  <span className="text-sm">Stop Loss at ₹{signal.stop_loss.toFixed(2)}</span>
                </div>

                <div className="flex items-center gap-2 text-success">
                  <Target className="w-4 h-4" />
                  <span className="text-sm">Target at ₹{signal.target.toFixed(2)}</span>
                </div>

                <div className="flex items-center gap-2 text-text-secondary">
                  <BarChart3 className="w-4 h-4" />
                  <span className="text-sm">Max Risk: ₹{((signal.entry_price - signal.stop_loss) * signal.position_size).toFixed(2)}</span>
                </div>

                <div className="flex items-center gap-2 text-text-secondary">
                  <TrendingUp className="w-4 h-4" />
                  <span className="text-sm">Potential Profit: ₹{((signal.target - signal.entry_price) * signal.position_size).toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="bg-background-surface/50 backdrop-blur-xl rounded-2xl border border-gray-800 p-6">
              <h3 className="text-lg font-bold text-text-primary mb-4">Quick Actions</h3>

              <div className="space-y-3">
                <button className="w-full px-4 py-3 bg-gradient-primary text-white rounded-xl font-medium hover:shadow-glow-md transition-all">
                  Execute Trade
                </button>

                <button className="w-full px-4 py-3 bg-background-elevated border border-gray-800 text-text-primary rounded-xl font-medium hover:border-gray-700 transition-all">
                  Add to Watchlist
                </button>

                <button className="w-full px-4 py-3 bg-background-elevated border border-gray-800 text-text-primary rounded-xl font-medium hover:border-gray-700 transition-all">
                  Set Price Alert
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
