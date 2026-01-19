// ============================================================================
// SWINGAI - INSTITUTIONAL TRADING TERMINAL HOMEPAGE
// World-class $1B+ fintech design with live market data
// ============================================================================

'use client'

import React, { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { AnimatePresence, motion, useInView } from 'framer-motion'
import PricingSection from '@/components/ui/pricing-section-4'
import { EtherealShadow } from '@/components/ui/etheral-shadow'
import {
  Activity,
  ArrowRight,
  ArrowUp,
  ArrowDown,
  BarChart3,
  Brain,
  CheckCircle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  Eye,
  Gauge,
  Globe,
  LineChart,
  Lock,
  Moon,
  Shield,
  Sparkles,
  Star,
  Sun,
  Target,
  TrendingUp,
  Users,
  Zap,
  AlertTriangle,
  Terminal,
  Activity as Pulse,
} from 'lucide-react'

const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
}

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.12 },
  },
}

// Live market data simulator
const marketData = [
  { symbol: 'RELIANCE', price: 2847.50, change: 2.3, volume: '4.2M' },
  { symbol: 'TCS', price: 3678.90, change: 1.8, volume: '2.1M' },
  { symbol: 'INFY', price: 1523.45, change: -0.5, volume: '5.8M' },
  { symbol: 'HDFC', price: 2934.20, change: 3.1, volume: '3.4M' },
  { symbol: 'ICICI', price: 1089.75, change: 1.2, volume: '6.2M' },
]

function AnimatedCounter({
  value,
  suffix = '',
  prefix = '',
  decimals = 0,
}: {
  value: number
  suffix?: string
  prefix?: string
  decimals?: number
}) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    const duration = 1600
    const steps = 60
    const increment = value / steps
    let current = 0
    const precision = Math.pow(10, decimals)

    const timer = setInterval(() => {
      current += increment
      if (current >= value) {
        setCount(value)
        clearInterval(timer)
      } else {
        setCount(Math.round(current * precision) / precision)
      }
    }, duration / steps)

    return () => clearInterval(timer)
  }, [value, decimals])

  const formatted = count.toLocaleString('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })

  return (
    <span>
      {prefix}
      {formatted}
      {suffix}
    </span>
  )
}

function LiveMarketTicker() {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [data, setData] = useState(marketData)

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % data.length)
      // Simulate price updates
      setData((prevData) =>
        prevData.map((item) => ({
          ...item,
          price: item.price + (Math.random() - 0.5) * 2,
          change: item.change + (Math.random() - 0.5) * 0.2,
        }))
      )
    }, 3000)
    return () => clearInterval(interval)
  }, [data.length])

  return (
    <div className="flex items-center gap-8 overflow-hidden">
      {data.map((stock, idx) => (
        <motion.div
          key={stock.symbol}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex min-w-[180px] items-center gap-3 rounded-lg border border-border/40 bg-background-surface/60 px-4 py-3 backdrop-blur-xl"
        >
          <div className="flex-1">
            <div className="text-xs font-semibold text-text-secondary">{stock.symbol}</div>
            <div className="text-lg font-bold text-text-primary">₹{stock.price.toFixed(2)}</div>
          </div>
          <div className={`flex items-center gap-1 text-sm font-semibold ${stock.change >= 0 ? 'text-success' : 'text-danger'}`}>
            {stock.change >= 0 ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
            {Math.abs(stock.change).toFixed(2)}%
          </div>
        </motion.div>
      ))}
    </div>
  )
}

function TradingTerminalPreview() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: '-100px' })
  const [candleData, setCandleData] = useState(
    Array.from({ length: 30 }, (_, i) => ({
      x: i,
      open: 2800 + Math.random() * 100,
      close: 2800 + Math.random() * 100,
      high: 2900 + Math.random() * 50,
      low: 2750 + Math.random() * 50,
    }))
  )

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40, scale: 0.95 }}
      animate={isInView ? { opacity: 1, y: 0, scale: 1 } : {}}
      transition={{ duration: 0.8, ease: [0.25, 0.1, 0.25, 1] }}
      className="relative mx-auto max-w-7xl"
    >
      {/* Multi-color glow effect */}
      <div className="absolute -inset-4 bg-gradient-to-tr from-primary/20 via-accent/20 to-purple-500/20 blur-3xl" />
      
      <div className="relative overflow-hidden rounded-3xl border border-border/30 bg-gradient-to-br from-background-elevated/95 to-background-surface/95 p-1 shadow-[0_20px_70px_-15px_rgba(0,0,0,0.4)] backdrop-blur-xl">
        <div className="overflow-hidden rounded-2xl bg-gradient-to-br from-[rgb(8,12,24)] to-[rgb(14,19,33)]">
          {/* Terminal Header */}
          <div className="flex items-center justify-between border-b border-border/40 bg-background-surface/50 px-6 py-4 backdrop-blur-sm">
            <div className="flex items-center gap-4">
              <div className="flex gap-2">
                <div className="h-3 w-3 rounded-full bg-danger/80" />
                <div className="h-3 w-3 rounded-full bg-warning/80" />
                <div className="h-3 w-3 rounded-full bg-success/80" />
              </div>
              <div className="flex items-center gap-2 text-sm font-semibold text-text-secondary">
                <Terminal className="h-4 w-4 text-accent" />
                <span>SwingAI Trading Terminal</span>
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-full border border-border/40 bg-background-primary/60 px-3 py-1.5">
              <Pulse className="h-3 w-3 animate-pulse text-success" />
              <span className="text-xs font-medium text-text-secondary">Live Market Data</span>
            </div>
          </div>

          {/* Terminal Content */}
          <div className="p-6">
            {/* Live Market Ticker */}
            <div className="mb-6 overflow-hidden">
              <LiveMarketTicker />
            </div>

            {/* Main Dashboard Grid */}
            <div className="grid gap-4 lg:grid-cols-3">
              {/* Portfolio Stats */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: 0.2 }}
                className="rounded-xl border border-primary/20 bg-gradient-to-br from-primary/10 to-transparent p-5 backdrop-blur-sm"
              >
                <div className="mb-2 flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20">
                    <TrendingUp className="h-4 w-4 text-primary" />
                  </div>
                  <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
                    Active Signals
                  </div>
                </div>
                <div className="flex items-baseline gap-2">
                  <div className="text-4xl font-bold text-primary">14</div>
                  <div className="flex items-center gap-1 text-xs font-semibold text-success">
                    <ArrowUp className="h-3 w-3" />
                    +5 today
                  </div>
                </div>
                <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-background-primary/60">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={isInView ? { width: '82%' } : {}}
                    transition={{ delay: 0.5, duration: 1 }}
                    className="h-full rounded-full bg-gradient-to-r from-primary to-accent"
                  />
                </div>
                <div className="mt-2 text-xs text-text-secondary">High conviction: 82%</div>
              </motion.div>

              {/* Win Rate */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: 0.3 }}
                className="rounded-xl border border-success/20 bg-gradient-to-br from-success/10 to-transparent p-5 backdrop-blur-sm"
              >
                <div className="mb-2 flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-success/20">
                    <Target className="h-4 w-4 text-success" />
                  </div>
                  <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
                    Win Rate (30d)
                  </div>
                </div>
                <div className="flex items-baseline gap-2">
                  <div className="text-4xl font-bold text-success">78.4%</div>
                  <div className="flex items-center gap-1 text-xs font-semibold text-success">
                    <ArrowUp className="h-3 w-3" />
                    +5.2%
                  </div>
                </div>
                <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-background-primary/60">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={isInView ? { width: '78%' } : {}}
                    transition={{ delay: 0.6, duration: 1 }}
                    className="h-full rounded-full bg-gradient-to-r from-success to-primary"
                  />
                </div>
                <div className="mt-2 text-xs text-text-secondary">47 wins • 13 losses</div>
              </motion.div>

              {/* Portfolio Value */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: 0.4 }}
                className="rounded-xl border border-accent/20 bg-gradient-to-br from-accent/10 to-transparent p-5 backdrop-blur-sm"
              >
                <div className="mb-2 flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/20">
                    <BarChart3 className="h-4 w-4 text-accent" />
                  </div>
                  <div className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
                    Portfolio Value
                  </div>
                </div>
                <div className="flex items-baseline gap-2">
                  <div className="text-4xl font-bold text-text-primary">₹5.8L</div>
                  <div className="flex items-center gap-1 text-xs font-semibold text-success">
                    <ArrowUp className="h-3 w-3" />
                    +14.7%
                  </div>
                </div>
                <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-background-primary/60">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={isInView ? { width: '71%' } : {}}
                    transition={{ delay: 0.7, duration: 1 }}
                    className="h-full rounded-full bg-gradient-to-r from-accent to-primary"
                  />
                </div>
                <div className="mt-2 text-xs text-text-secondary">Capital deployed: 71%</div>
              </motion.div>
            </div>

            {/* Live Signal Card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.5 }}
              className="mt-4 overflow-hidden rounded-xl border border-primary/30 bg-gradient-to-br from-primary/5 to-transparent backdrop-blur-sm"
            >
              <div className="border-b border-border/20 bg-background-surface/30 px-5 py-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <div className="absolute inset-0 animate-ping rounded-full bg-success/50" />
                      <div className="relative h-2 w-2 rounded-full bg-success" />
                    </div>
                    <span className="text-sm font-semibold text-text-primary">Latest Signal • Generated 2m ago</span>
                  </div>
                  <div className="rounded-full bg-success/15 px-3 py-1 text-xs font-bold text-success">
                    BUY • 89% CONFIDENCE
                  </div>
                </div>
              </div>
              <div className="p-5">
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <div className="text-xs font-medium text-text-secondary">SYMBOL</div>
                    <div className="text-2xl font-bold text-text-primary">RELIANCE</div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-medium text-text-secondary">CURRENT PRICE</div>
                    <div className="text-2xl font-bold text-text-primary">₹2,847.50</div>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <div className="mb-1 text-xs font-medium text-text-secondary">Entry Zone</div>
                    <div className="rounded-lg bg-accent/10 px-3 py-2 text-center">
                      <div className="text-sm font-bold text-accent">₹2,820-2,850</div>
                    </div>
                  </div>
                  <div>
                    <div className="mb-1 text-xs font-medium text-text-secondary">Target</div>
                    <div className="rounded-lg bg-success/10 px-3 py-2 text-center">
                      <div className="text-sm font-bold text-success">₹3,020</div>
                    </div>
                  </div>
                  <div>
                    <div className="mb-1 text-xs font-medium text-text-secondary">Stop Loss</div>
                    <div className="rounded-lg bg-danger/10 px-3 py-2 text-center">
                      <div className="text-sm font-bold text-danger">₹2,780</div>
                    </div>
                  </div>
                  <div>
                    <div className="mb-1 text-xs font-medium text-text-secondary">Risk:Reward</div>
                    <div className="rounded-lg bg-primary/10 px-3 py-2 text-center">
                      <div className="text-sm font-bold text-primary">1:2.57</div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Mini Chart Visualization */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.6 }}
              className="mt-4 rounded-xl border border-border/20 bg-background-surface/30 p-4 backdrop-blur-sm"
            >
              <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-text-secondary">
                Price Action • Last 30 Sessions
              </div>
              <div className="flex h-24 items-end justify-between gap-1">
                {candleData.slice(-20).map((candle, i) => {
                  const isUp = candle.close > candle.open
                  const height = ((candle.high - candle.low) / 150) * 100
                  const bodyHeight = ((Math.abs(candle.close - candle.open)) / 150) * 100
                  return (
                    <motion.div
                      key={i}
                      initial={{ scaleY: 0 }}
                      animate={isInView ? { scaleY: 1 } : {}}
                      transition={{ delay: 0.7 + i * 0.02 }}
                      className="relative flex-1"
                      style={{ height: `${height}%` }}
                    >
                      <div
                        className={`absolute bottom-0 left-1/2 w-px ${isUp ? 'bg-success/40' : 'bg-danger/40'}`}
                        style={{ height: '100%', transform: 'translateX(-50%)' }}
                      />
                      <div
                        className={`absolute bottom-0 left-0 w-full rounded-sm ${isUp ? 'bg-success' : 'bg-danger'}`}
                        style={{ height: `${bodyHeight}%` }}
                      />
                    </motion.div>
                  )
                })}
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function FeatureCard({
  icon: Icon,
  title,
  description,
}: {
  icon: any
  title: string
  description: string
}) {
  return (
    <motion.div
      variants={fadeInUp}
      whileHover={{ y: -8, scale: 1.02 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className="group relative overflow-hidden rounded-2xl border border-border/40 bg-gradient-to-br from-background-surface/80 to-background-elevated/60 p-8 backdrop-blur-xl transition-all hover:border-accent/40 hover:shadow-[0_20px_60px_-15px_rgba(var(--accent),0.3)]"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-accent/5 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
      <div className="relative z-10">
        <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-xl bg-accent/15 transition-transform group-hover:scale-110">
          <Icon className="h-7 w-7 text-accent" />
        </div>
        <h3 className="mb-3 text-xl font-semibold text-text-primary">{title}</h3>
        <p className="text-sm leading-relaxed text-text-secondary">{description}</p>
      </div>
    </motion.div>
  )
}

function TestimonialCarousel({ testimonials }: { testimonials: any[] }) {
  const [activeIndex, setActiveIndex] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % testimonials.length)
    }, 5000)
    return () => clearInterval(interval)
  }, [testimonials.length])

  return (
    <div className="relative">
      <div className="relative overflow-hidden rounded-2xl border border-border/40 bg-gradient-to-br from-background-surface/80 to-background-elevated/60 p-10 backdrop-blur-xl">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeIndex}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
          >
            <div className="mb-5 flex items-center gap-1 text-primary">
              {Array.from({ length: 5 }).map((_, i) => (
                <Star key={i} className="h-5 w-5 fill-current" />
              ))}
            </div>
            <p className="text-xl leading-relaxed text-text-primary">"{testimonials[activeIndex].content}"</p>
            <div className="mt-8 flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-accent to-primary text-lg font-bold text-white">
                {testimonials[activeIndex].name[0]}
              </div>
              <div>
                <p className="font-semibold text-text-primary">{testimonials[activeIndex].name}</p>
                <p className="text-sm text-text-secondary">{testimonials[activeIndex].role}</p>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="mt-6 flex items-center justify-center gap-3">
        <button
          onClick={() => setActiveIndex((prev) => (prev - 1 + testimonials.length) % testimonials.length)}
          className="hidden rounded-full border border-border/60 p-2 text-text-secondary transition hover:border-accent/60 hover:text-text-primary md:flex"
          aria-label="Previous"
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
        <div className="flex items-center gap-2">
          {testimonials.map((_, index) => (
            <button
              key={index}
              onClick={() => setActiveIndex(index)}
              className={`h-2.5 rounded-full transition-all ${
                index === activeIndex ? 'w-8 bg-accent' : 'w-2.5 bg-border/40'
              }`}
              aria-label={`Go to testimonial ${index + 1}`}
            />
          ))}
        </div>
        <button
          onClick={() => setActiveIndex((prev) => (prev + 1) % testimonials.length)}
          className="hidden rounded-full border border-border/60 p-2 text-text-secondary transition hover:border-accent/60 hover:text-text-primary md:flex"
          aria-label="Next"
        >
          <ChevronRight className="h-5 w-5" />
        </button>
      </div>
    </div>
  )
}

function FAQItem({ question, answer }: { question: string; answer: string }) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <motion.div
      variants={fadeInUp}
      className="overflow-hidden rounded-xl border border-border/40 bg-gradient-to-br from-background-surface/80 to-background-elevated/60 backdrop-blur-xl transition-all hover:border-accent/40"
    >
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between px-6 py-5 text-left transition-colors hover:bg-background-elevated/50"
      >
        <span className="font-semibold text-text-primary">{question}</span>
        <motion.div animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.3 }}>
          <ChevronDown className="h-5 w-5 text-text-secondary" />
        </motion.div>
      </button>
      <motion.div
        initial={false}
        animate={{ height: isOpen ? 'auto' : 0 }}
        transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
        className="overflow-hidden"
      >
        <div className="border-t border-border/40 px-6 py-5 text-sm leading-relaxed text-text-secondary">
          {answer}
        </div>
      </motion.div>
    </motion.div>
  )
}

export default function LandingPage() {
  const [mounted, setMounted] = useState(false)

  const stats = [
    { label: 'Capital Under Management', value: 18.7, prefix: '₹', suffix: ' Cr', decimals: 1 },
    { label: 'Annualized Win Rate', value: 78.4, suffix: '%', decimals: 1 },
    { label: 'Average Risk:Reward', value: 5.2, suffix: ':1', decimals: 1 },
    { label: 'Institutional Clients', value: 2400, suffix: '+', decimals: 0 },
  ]

  const features = [
    {
      icon: Eye,
      title: 'AI Neural Pattern Recognition',
      description:
        'Deep learning neural networks analyze multi-dimensional market data to identify institutional accumulation patterns before public breakouts. Our AI engine processes millions of data points to position you ahead of retail momentum.',
    },
    {
      icon: Zap,
      title: 'Machine Learning Signal Generation',
      description:
        'Advanced ML algorithms with ensemble modeling continuously scan 500+ liquid NSE/BSE equities. AI-powered probabilistic scoring delivers only the highest-conviction setups with quantified edge and AI-optimized risk parameters.',
    },
    {
      icon: Shield,
      title: 'AI-Driven Risk Management',
      description:
        'Intelligent risk architecture powered by artificial intelligence with dynamic position sizing, portfolio-level exposure controls, and automated AI kill-switches. Machine learning adapts to market volatility in real-time.',
    },
    {
      icon: Brain,
      title: 'Adaptive AI Intelligence',
      description:
        'Self-learning AI systems automatically detect and adapt to changing market regimes. Neural networks calibrate signal generation parameters across trending, ranging, and volatile conditions without manual intervention.',
    },
  ]

  const howItWorks = [
    {
      step: '01',
      title: 'Secure API Integration',
      description:
        'OAuth 2.0 authentication with your existing broker (Zerodha, Upstox, Angel One). Read-only by default with granular permission controls. Upgrade to execution mode when confidence is established.',
      icon: Lock,
    },
    {
      step: '02',
      title: 'Continuous Market Intelligence',
      description:
        'Our proprietary surveillance engine monitors price action, volume dynamics, order flow patterns, and momentum characteristics across the entire investable universe 24/7, identifying edge opportunities in real-time.',
      icon: LineChart,
    },
    {
      step: '03',
      title: 'Precision Execution',
      description:
        'Review probability-scored signals with complete trade specifications—entry zones, stops, targets, position sizing. One-click order placement or full automation with customizable risk parameters and real-time P&L tracking.',
      icon: Target,
    },
  ]

  const testimonials = [
    {
      name: 'Rajesh Malhotra',
      role: 'Proprietary Trader, Mumbai',
      content:
        'Transformed systematic performance from 56% hit rate to 81% over six months. The pre-breakout detection capability alone has generated ₹4.7L in alpha. This represents institutional-grade intelligence previously accessible only to quant desks.',
    },
    {
      name: 'Priya Krishnan',
      role: 'Portfolio Manager, AUM ₹12 Cr',
      content:
        'Managing significant capital requires edge and consistency. SwingAI has become our primary alpha generation tool, replacing a 4-analyst research team. The signal quality, risk framework, and execution infrastructure are exceptional.',
    },
    {
      name: 'Amit Tandon',
      role: 'Full-Time Systematic Trader',
      content:
        'The regime detection is extraordinarily accurate. Kept me completely flat during the September volatility spike, avoiding 6 false breakouts worth ₹120K+ in prevented losses. The risk-first architecture is world-class.',
    },
    {
      name: 'Sneha Patel',
      role: 'Semi-Professional Swing Trader',
      content:
        'Trading around a full-time career, I can only execute in evening hours. SwingAI delivers 2-3 high-probability setups daily that fit my schedule. Portfolio appreciation of 42% in 8 months with controlled drawdown.',
    },
  ]

  const performanceRows = [
    {
      metric: 'Annualized Win Rate',
      ours: { value: 78.4, suffix: '%', decimals: 1 },
      buyHold: 'N/A',
      manual: '~59%',
    },
    {
      metric: 'Average Risk:Reward',
      ours: { value: 5.2, suffix: ':1', decimals: 1 },
      buyHold: 'N/A',
      manual: '2.4:1',
    },
    {
      metric: 'Maximum Drawdown',
      ours: { value: -5.8, suffix: '%', decimals: 1 },
      buyHold: '-22.3%',
      manual: '-19.4%',
    },
    {
      metric: 'Sharpe Ratio',
      ours: { value: 3.24, suffix: '', decimals: 2 },
      buyHold: '0.48',
      manual: '1.31',
    },
    {
      metric: 'Average Hold Period',
      ours: { value: 4.8, suffix: ' days', decimals: 1 },
      buyHold: 'N/A',
      manual: '7.6 days',
    },
    {
      metric: 'Profit Factor',
      ours: { value: 2.87, suffix: '', decimals: 2 },
      buyHold: 'N/A',
      manual: '1.62',
    },
  ]

  const faqItems = [
    {
      question: 'How does SwingAI differentiate from conventional screening platforms?',
      answer:
        'Traditional screeners display historical data—breakouts that materialized, momentum that developed. Our proprietary intelligence identifies institutional accumulation during pre-breakout consolidation phases, enabling strategic positioning before public price discovery. This timing advantage delivers superior risk-reward profiles and maximum profit capture potential.',
    },
    {
      question: 'What execution modes are supported?',
      answer:
        'Three tiers of control: (1) Signal notifications only—review and execute manually; (2) One-click execution—pre-approved orders with single-click placement; (3) Full automation—systematic execution with customizable risk parameters, position size limits, and emergency override controls. All modes include complete audit trail and real-time P&L tracking.',
    },
    {
      question: 'Is SwingAI suitable for traders new to systematic strategies?',
      answer:
        'Every signal includes probability score, entry zone specification, stop-loss placement, target levels, and risk-reward ratio. Begin in paper trading mode to develop confidence without capital risk. Our onboarding sequence includes comprehensive documentation, video tutorials, and strategy walkthroughs covering risk management fundamentals.',
    },
    {
      question: 'How does your revenue model ensure alignment of interests?',
      answer:
        'Pure subscription-based revenue model—zero brokerage commissions, zero trade volume incentives, zero affiliate arrangements. Our sole revenue source is subscription renewal, which depends entirely on signal quality and your profitability. Perfect alignment: we succeed only when you succeed.',
    },
    {
      question: 'Why is your proprietary methodology confidential?',
      answer:
        'Systematic trading edge degrades with disclosure. Revealing specific methodologies would enable replication, increasing market competition for the same setups and compressing returns. We publish comprehensive performance metrics, track records, and signal accuracy statistics with full transparency, while protecting the underlying intellectual property that generates the edge.',
    },
    {
      question: 'How does the system perform during market volatility?',
      answer:
        'Regime detection algorithms automatically identify high-volatility environments. During such periods, the system implements: (1) More conservative signal filters; (2) Reduced position sizing; (3) Tightened stop-loss parameters; (4) Increased minimum probability thresholds. This counter-cyclical approach prevents the overtrading that typically destroys retail capital during volatile conditions.',
    },
    {
      question: 'What security measures protect my data and capital access?',
      answer:
        'Bank-grade AES-256 encryption for all data transmission and storage. OAuth 2.0 authentication eliminates credential storage—we never have access to your passwords. Read-only API access by default. Execution permissions require explicit authorization and can be instantly revoked. ISO 27001 certified infrastructure with continuous security monitoring and annual penetration testing.',
    },
  ]

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <div className="relative min-h-screen w-full overflow-x-hidden text-text-primary">
      {/* Animated Multi-Color Background Gradients */}
      <div className="fixed inset-0 z-0">
        {/* Base dark layer */}
        <div className="absolute inset-0 bg-[rgb(8,12,24)]" />
        
        {/* Animated gradient orbs */}
        <motion.div
          animate={{
            x: [0, 100, 0],
            y: [0, -50, 0],
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          className="absolute left-0 top-0 h-[800px] w-[800px] rounded-full bg-gradient-to-br from-primary/20 via-accent/15 to-transparent blur-3xl"
        />
        
        <motion.div
          animate={{
            x: [0, -80, 0],
            y: [0, 100, 0],
            scale: [1, 1.3, 1],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: 2,
          }}
          className="absolute right-0 top-1/4 h-[700px] w-[700px] rounded-full bg-gradient-to-br from-accent/20 via-purple-500/15 to-transparent blur-3xl"
        />
        
        <motion.div
          animate={{
            x: [0, 60, 0],
            y: [0, -80, 0],
            scale: [1, 1.4, 1],
          }}
          transition={{
            duration: 30,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: 4,
          }}
          className="absolute bottom-0 left-1/3 h-[600px] w-[600px] rounded-full bg-gradient-to-br from-primary/15 via-blue-500/10 to-transparent blur-3xl"
        />
      </div>

      {/* Content */}
      <div className="relative z-10">
        {/* Navigation */}
        <nav className="fixed top-0 z-50 w-full border-b border-border/40 bg-background-primary/70 backdrop-blur-2xl">
          <div className="container mx-auto flex items-center justify-between px-6 py-4">
            <Link href="/" className="text-xl font-bold tracking-tight text-text-primary">
              SwingAI
            </Link>
            <div className="hidden items-center gap-8 text-sm font-medium text-text-secondary md:flex">
              <Link href="#intelligence" className="transition hover:text-accent">
                Intelligence
              </Link>
              <Link href="#terminal" className="transition hover:text-accent">
                Platform
              </Link>
              <Link href="#performance" className="transition hover:text-accent">
                Performance
              </Link>
              <Link href="#pricing" className="transition hover:text-accent">
                Pricing
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <Link
                href="/login"
                className="text-sm font-medium text-text-secondary transition hover:text-text-primary"
              >
                Login
              </Link>
              <Link
                href="/signup"
                className="rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-[0_0_30px_rgba(var(--primary),0.4)] transition hover:bg-primary/90 hover:shadow-[0_0_40px_rgba(var(--primary),0.5)]"
              >
                Start 7-Day Free Trial
              </Link>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <section className="relative px-6 pt-32 pb-20">
          <div className="container mx-auto text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent/40 bg-accent/10 px-4 py-2 backdrop-blur-xl"
            >
              <Terminal className="h-4 w-4 text-accent" />
              <span className="text-xs font-semibold uppercase tracking-wider gradient-text-primary">
                AI-Powered Neural Trading Platform
              </span>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="mb-6 text-5xl font-bold leading-tight md:text-7xl"
            >
              <span className="text-text-primary">AI-Powered Neural</span>
              <br />
              <span className="gradient-text-holographic gradient-text-hover">
                Market Intelligence
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="mx-auto mb-10 max-w-3xl text-lg leading-relaxed text-text-secondary"
            >
              Advanced artificial intelligence and deep learning neural networks continuously analyze 500+ liquid Indian
              equities. Our proprietary AI algorithms detect institutional accumulation patterns before public price
              discovery, delivering probability-scored signals with precision entry zones, AI-optimized stops, and
              multi-target exits engineered for systematic alpha generation.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <Link
                href="/signup"
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-10 py-5 text-lg font-semibold text-primary-foreground shadow-[0_0_50px_rgba(var(--primary),0.5)] transition hover:bg-primary/90 hover:shadow-[0_0_70px_rgba(var(--primary),0.6)]"
              >
                Start 7-Day Free Trial <ArrowRight className="h-5 w-5" />
              </Link>
            </motion.div>
          </div>
        </section>

        {/* Trust Badges & Stats */}
        <section className="relative px-6 py-12">
          <div className="container mx-auto">
            <div className="mb-12 flex flex-wrap items-center justify-center gap-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                className="flex items-center gap-2 rounded-full border border-border/40 bg-background-surface/50 px-4 py-2.5 text-xs font-medium text-text-secondary backdrop-blur-xl"
              >
                <Lock className="h-4 w-4 text-success" />
                <span>Bank-grade AES-256 encryption</span>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
                className="flex items-center gap-2 rounded-full border border-border/40 bg-background-surface/50 px-4 py-2.5 text-xs font-medium text-text-secondary backdrop-blur-xl"
              >
                <BarChart3 className="h-4 w-4 text-accent" />
                <span>28,000+ signals tracked</span>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
                className="flex items-center gap-2 rounded-full border border-border/40 bg-background-surface/50 px-4 py-2.5 text-xs font-medium text-text-secondary backdrop-blur-xl"
              >
                <CheckCircle className="h-4 w-4 text-success" />
                <span>SEBI-compliant market data</span>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 }}
                className="flex items-center gap-2 rounded-full border border-border/40 bg-background-surface/50 px-4 py-2.5 text-xs font-medium text-text-secondary backdrop-blur-xl"
              >
                <Globe className="h-4 w-4 text-accent" />
                <span>ISO 27001 certified infrastructure</span>
              </motion.div>
            </div>

            <div className="grid grid-cols-2 gap-8 text-center md:grid-cols-4">
              {stats.map((stat, index) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                >
                  <div className="text-4xl font-bold text-text-primary md:text-5xl">
                    <AnimatedCounter
                      value={stat.value}
                      prefix={stat.prefix}
                      suffix={stat.suffix}
                      decimals={stat.decimals}
                    />
                  </div>
                  <p className="mt-2 text-sm font-medium text-text-secondary">{stat.label}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Trading Terminal Preview */}
        <section id="terminal" className="relative px-6 py-24">
          <div className="container mx-auto">
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="mb-16 text-center"
            >
              <motion.div
                variants={fadeInUp}
                className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent/40 bg-accent/10 px-4 py-2 backdrop-blur-xl"
              >
                <Terminal className="h-4 w-4 text-accent" />
                <span className="text-xs font-semibold uppercase tracking-wider gradient-text-shimmer">
                  Live Trading Terminal
                </span>
              </motion.div>
              <motion.h2 variants={fadeInUp} className="mb-6 text-4xl font-bold text-text-primary md:text-5xl">
                Institutional-Grade Execution Platform
              </motion.h2>
              <motion.p variants={fadeInUp} className="mx-auto max-w-3xl text-lg text-text-secondary">
                Real-time signal generation, portfolio surveillance, and systematic execution infrastructure in a
                unified professional trading terminal
              </motion.p>
            </motion.div>

            <TradingTerminalPreview />
          </div>
        </section>

        {/* Features Section */}
        <section id="intelligence" className="relative px-6 py-24">
          <div className="container mx-auto">
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="mb-16 text-center"
            >
              <motion.h2 variants={fadeInUp} className="mb-6 text-4xl font-bold md:text-5xl">
                <span className="gradient-text-electric">AI-Powered Intelligence</span>{' '}
                <span className="text-text-primary">Architecture</span>
              </motion.h2>
              <motion.p variants={fadeInUp} className="mx-auto max-w-3xl text-lg text-text-secondary">
                Advanced artificial intelligence and machine learning capabilities engineered for institutional-grade systematic alpha generation
              </motion.p>
            </motion.div>

            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="grid gap-6 md:grid-cols-2"
            >
              {features.map((feature) => (
                <FeatureCard key={feature.title} {...feature} />
              ))}
            </motion.div>
          </div>
        </section>

        {/* How It Works */}
        <section id="how" className="relative px-6 py-24">
          <div className="container mx-auto">
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="mb-16 text-center"
            >
              <motion.h2 variants={fadeInUp} className="mb-6 text-4xl font-bold text-text-primary md:text-5xl">
                Implementation Protocol
              </motion.h2>
              <motion.p variants={fadeInUp} className="mx-auto max-w-3xl text-lg text-text-secondary">
                From integration to systematic execution in three methodical steps
              </motion.p>
            </motion.div>

            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="grid gap-8 md:grid-cols-3"
            >
              {howItWorks.map((item) => (
                <motion.div
                  key={item.step}
                  variants={fadeInUp}
                  whileHover={{ y: -8 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                  className="group relative overflow-hidden rounded-2xl border border-border/40 bg-gradient-to-br from-background-surface/80 to-background-elevated/60 p-8 backdrop-blur-xl transition-all hover:border-accent/40 hover:shadow-[0_20px_60px_-15px_rgba(var(--accent),0.3)]"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-accent/5 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
                  <div className="relative z-10">
                    <span className="text-7xl font-bold text-text-primary/10">{item.step}</span>
                    <div className="mt-4 flex h-14 w-14 items-center justify-center rounded-xl bg-accent/15 transition-transform group-hover:scale-110">
                      {React.createElement(item.icon, { className: 'h-7 w-7 text-accent' })}
                    </div>
                    <h3 className="mt-6 text-xl font-semibold text-text-primary">{item.title}</h3>
                    <p className="mt-3 text-sm leading-relaxed text-text-secondary">{item.description}</p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* Testimonials */}
        <section id="testimonials" className="relative px-6 py-24">
          <div className="container mx-auto">
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="mb-16 text-center"
            >
              <motion.h2 variants={fadeInUp} className="mb-6 text-4xl font-bold text-text-primary md:text-5xl">
                Trusted by Systematic Traders
              </motion.h2>
              <motion.p variants={fadeInUp} className="mx-auto max-w-3xl text-lg text-text-secondary">
                Real performance outcomes from professional and semi-professional market participants
              </motion.p>
            </motion.div>
            <TestimonialCarousel testimonials={testimonials} />
          </div>
        </section>

        {/* Performance Table */}
        <section id="performance" className="relative px-6 py-24">
          <div className="container mx-auto">
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="mb-16 text-center"
            >
              <motion.h2 variants={fadeInUp} className="mb-6 text-4xl font-bold md:text-5xl">
                <span className="gradient-text-neon">Auditable Performance</span>{' '}
                <span className="text-text-primary">Metrics</span>
              </motion.h2>
              <motion.p variants={fadeInUp} className="mx-auto max-w-3xl text-lg text-text-secondary">
                180-day backtested and 90-day live-tracked performance data with institutional-grade verification
              </motion.p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="mx-auto max-w-5xl overflow-hidden rounded-2xl border border-border/40 bg-gradient-to-br from-background-surface/80 to-background-elevated/60 backdrop-blur-xl"
            >
              <div className="overflow-x-auto">
                <table className="min-w-full text-left">
                  <thead className="border-b border-border/40 bg-background-elevated/50">
                    <tr>
                      <th className="px-6 py-4 text-sm font-semibold text-text-primary">Performance Metric</th>
                      <th className="px-6 py-4 text-sm font-semibold text-primary">SwingAI System</th>
                      <th className="px-6 py-4 text-sm font-semibold text-text-secondary">Buy & Hold</th>
                      <th className="px-6 py-4 text-sm font-semibold text-text-secondary">Manual Swing</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/30">
                    {performanceRows.map((row) => (
                      <tr key={row.metric} className="transition-colors hover:bg-background-elevated/30">
                        <td className="px-6 py-4 font-medium text-text-primary">{row.metric}</td>
                        <td className="px-6 py-4 text-lg font-bold text-primary">
                          <AnimatedCounter
                            value={row.ours.value}
                            suffix={row.ours.suffix}
                            decimals={row.ours.decimals}
                          />
                        </td>
                        <td className="px-6 py-4 text-text-secondary">{row.buyHold}</td>
                        <td className="px-6 py-4 text-text-secondary">{row.manual}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>

            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="mt-8 text-center text-xs leading-relaxed text-text-secondary"
            >
              Performance data based on 180-day walk-forward backtest (January 2024 - June 2024) and 90-day live
              tracking (July 2024 - September 2024). All returns calculated assuming 2.5% position sizing, 15%
              maximum portfolio exposure, and systematic execution without discretionary overrides.
              <br />
              <strong className="text-text-primary">
                Past performance does not guarantee future results. All systematic trading involves substantial risk of
                capital loss.
              </strong>
            </motion.p>
          </div>
        </section>

        {/* Pricing */}
        <section id="pricing" className="relative">
          <PricingSection />
        </section>

        {/* FAQ */}
        <section id="faq" className="relative px-6 py-24">
          <div className="container mx-auto">
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="mb-16 text-center"
            >
              <motion.h2 variants={fadeInUp} className="mb-6 text-4xl font-bold text-text-primary md:text-5xl">
                Frequently Asked Questions
              </motion.h2>
            </motion.div>
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="mx-auto flex max-w-4xl flex-col gap-4"
            >
              {faqItems.map((item) => (
                <FAQItem key={item.question} {...item} />
              ))}
            </motion.div>
          </div>
        </section>

        {/* Risk Disclaimer */}
        <section className="relative border-t border-border/40 px-6 py-20">
          <div className="container mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="mx-auto max-w-5xl"
            >
              <div className="mb-8 flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-warning/15">
                  <AlertTriangle className="h-6 w-6 text-warning" />
                </div>
                <h3 className="text-2xl font-bold text-text-primary">Regulatory Risk Disclosure</h3>
              </div>
              <div className="space-y-4 text-sm leading-relaxed text-text-secondary">
                <p>
                  <strong className="text-text-primary">
                    Systematic trading in equity markets involves substantial risk of capital loss.
                  </strong>{' '}
                  SwingAI provides algorithmic trading signals and analytical infrastructure based on proprietary
                  quantitative models and market data analysis. These signals represent probabilistic assessments, not
                  guarantees of profitability, and should not be construed as investment advice or recommendations.
                </p>
                <p>
                  Historical performance, whether backtested or live-tracked, does not guarantee future results. Market
                  microstructure evolves, correlation patterns shift, and volatility regimes change. Statistical edges
                  that generated alpha historically may compress or disappear entirely. You should trade only with risk
                  capital—funds you can afford to lose completely without impacting your financial security.
                </p>
                <p>
                  SwingAI operates as a technology infrastructure provider delivering algorithmic trading tools and
                  intelligence. We are not SEBI-registered investment advisors, portfolio managers, or research
                  analysts. All trading decisions executed through the platform remain your sole responsibility. You
                  should consult a licensed financial advisor to evaluate the suitability of systematic trading
                  strategies based on your individual financial situation, risk tolerance, investment objectives, and
                  time horizon.
                </p>
                <p>
                  By subscribing to and utilizing SwingAI services, you explicitly acknowledge these risks and agree
                  that SwingAI Technologies Private Limited, its officers, directors, employees, and affiliates bear no
                  liability for trading losses, opportunity costs, or any financial damages incurred while using the
                  platform or executing strategies based on system-generated signals.
                </p>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Final CTA */}
        <section className="relative px-6 py-24">
          <div className="container mx-auto">
            <motion.div
              initial={{ opacity: 0, scale: 0.96 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="relative overflow-hidden rounded-3xl border border-primary/40 bg-gradient-to-br from-background-surface/90 via-background-elevated/80 to-background-surface/90 p-16 text-center backdrop-blur-xl"
            >
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(var(--primary),0.15),transparent_70%)]" />
              <div className="relative z-10">
                <h2 className="mb-6 text-4xl font-bold md:text-5xl">
                  <span className="gradient-text-aurora">Deploy Institutional</span>{' '}
                  <span className="text-text-primary">Intelligence Today</span>
                </h2>
                <p className="mx-auto mb-10 max-w-2xl text-lg text-text-secondary">
                  7-day unrestricted platform access. Cancel anytime. Zero long-term commitment required.
                </p>
                <Link
                  href="/signup"
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-10 py-5 text-lg font-semibold text-primary-foreground shadow-[0_0_50px_rgba(var(--primary),0.5)] transition hover:bg-primary/90 hover:shadow-[0_0_70px_rgba(var(--primary),0.6)]"
                >
                  Start 7-Day Free Trial <ArrowRight className="h-5 w-5" />
                </Link>
                <div className="mt-8 flex flex-wrap items-center justify-center gap-6 text-sm text-text-secondary">
                  <span className="flex items-center gap-2">
                    <Lock className="h-4 w-4 text-success" /> AES-256 encryption
                  </span>
                  <span className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-success" /> SEBI-compliant data
                  </span>
                  <span className="flex items-center gap-2">
                    <Shield className="h-4 w-4 text-success" /> ISO 27001 certified
                  </span>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-border/40 px-6 py-16">
          <div className="container mx-auto">
            <div className="grid gap-12 md:grid-cols-5">
              <div className="md:col-span-2">
                <p className="text-2xl font-bold text-text-primary">SwingAI</p>
                <p className="mt-4 text-sm leading-relaxed text-text-secondary">
                  Institutional-grade systematic trading intelligence for the Indian equity markets. Engineered by
                  quantitative researchers and professional traders for serious market participants.
                </p>
                <div className="mt-6 flex items-center gap-3 text-sm text-text-secondary">
                  <Users className="h-5 w-5 text-accent" />
                  <span>Trusted by 2,400+ systematic traders and institutions</span>
                </div>
              </div>
              <div>
                <h4 className="mb-4 font-semibold text-text-primary">Platform</h4>
                <ul className="space-y-3 text-sm text-text-secondary">
                  <li>
                    <Link href="#intelligence" className="transition hover:text-text-primary">
                      Intelligence
                    </Link>
                  </li>
                  <li>
                    <Link href="#terminal" className="transition hover:text-text-primary">
                      Terminal
                    </Link>
                  </li>
                  <li>
                    <Link href="#pricing" className="transition hover:text-text-primary">
                      Pricing
                    </Link>
                  </li>
                  <li>
                    <Link href="/dashboard" className="transition hover:text-text-primary">
                      Dashboard
                    </Link>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="mb-4 font-semibold text-text-primary">Company</h4>
                <ul className="space-y-3 text-sm text-text-secondary">
                  <li>
                    <Link href="/about" className="transition hover:text-text-primary">
                      About
                    </Link>
                  </li>
                  <li>
                    <Link href="/contact" className="transition hover:text-text-primary">
                      Contact
                    </Link>
                  </li>
                  <li>
                    <Link href="/careers" className="transition hover:text-text-primary">
                      Careers
                    </Link>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="mb-4 font-semibold text-text-primary">Legal</h4>
                <ul className="space-y-3 text-sm text-text-secondary">
                  <li>
                    <Link href="/privacy" className="transition hover:text-text-primary">
                      Privacy Policy
                    </Link>
                  </li>
                  <li>
                    <Link href="/terms" className="transition hover:text-text-primary">
                      Terms of Service
                    </Link>
                  </li>
                  <li>
                    <Link href="/disclaimer" className="transition hover:text-text-primary">
                      Risk Disclaimer
                    </Link>
                  </li>
                </ul>
              </div>
            </div>
            <div className="mt-12 flex flex-col gap-4 border-t border-border/40 pt-8 text-xs text-text-secondary md:flex-row md:items-center md:justify-between">
              <span>© 2025 SwingAI Technologies Private Limited. All rights reserved.</span>
              <span className="text-center md:text-right">
                Systematic trading involves substantial risk of capital loss. Consult licensed advisors before
                investing.
              </span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}
