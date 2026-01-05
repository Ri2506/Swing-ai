// ============================================================================
// SWINGAI - CUTTING-EDGE LANDING PAGE
// Next.js 14 + shadcn/ui + Framer Motion
// ============================================================================

'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion, useScroll, useTransform } from 'framer-motion'
import {
  ArrowRight, TrendingUp, Shield, Zap, BarChart3,
  ChevronRight, Play, Check, Star, Users, Target,
  Brain, LineChart, Bell, Smartphone, Globe, Lock,
  Activity, Sparkles, Search, Filter, ChevronDown,
  ChevronUp, Code, Database, Cpu
} from 'lucide-react'

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
}

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
}

// Stats counter component
function AnimatedCounter({ value, suffix = '' }: { value: number; suffix?: string }) {
  const [count, setCount] = useState(0)
  
  useEffect(() => {
    const duration = 2000
    const steps = 60
    const increment = value / steps
    let current = 0
    
    const timer = setInterval(() => {
      current += increment
      if (current >= value) {
        setCount(value)
        clearInterval(timer)
      } else {
        setCount(Math.floor(current))
      }
    }, duration / steps)
    
    return () => clearInterval(timer)
  }, [value])
  
  return <span>{count.toLocaleString()}{suffix}</span>
}

// Pricing card
function PricingCard({ plan, popular = false }: { plan: any; popular?: boolean }) {
  return (
    <motion.div
      variants={fadeInUp}
      className={`relative rounded-2xl p-8 ${
        popular 
          ? 'bg-gradient-to-b from-blue-600 to-blue-700 text-white ring-4 ring-blue-500/50' 
          : 'bg-gray-900 text-white border border-gray-800'
      }`}
    >
      {popular && (
        <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-yellow-500 text-black text-sm font-bold rounded-full">
          MOST POPULAR
        </div>
      )}
      
      <h3 className="text-xl font-bold mb-2">{plan.name}</h3>
      <p className={`text-sm mb-4 ${popular ? 'text-blue-100' : 'text-gray-400'}`}>{plan.description}</p>
      
      <div className="mb-6">
        <span className="text-4xl font-bold">₹{plan.price}</span>
        <span className={popular ? 'text-blue-100' : 'text-gray-400'}>/month</span>
      </div>
      
      <ul className="space-y-3 mb-8">
        {plan.features.map((feature: string, i: number) => (
          <li key={i} className="flex items-center gap-2">
            <Check className={`w-5 h-5 ${popular ? 'text-blue-200' : 'text-green-500'}`} />
            <span className={popular ? 'text-blue-50' : 'text-gray-300'}>{feature}</span>
          </li>
        ))}
      </ul>
      
      <Link
        href="/signup"
        className={`block w-full py-3 text-center rounded-lg font-semibold transition-all ${
          popular
            ? 'bg-white text-blue-600 hover:bg-blue-50'
            : 'bg-blue-600 text-white hover:bg-blue-700'
        }`}
      >
        Get Started
      </Link>
    </motion.div>
  )
}

// Feature card
function FeatureCard({ icon: Icon, title, description }: { icon: any; title: string; description: string }) {
  return (
    <motion.div
      variants={fadeInUp}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      className="p-6 rounded-2xl bg-gray-900/50 border border-gray-800 hover:border-blue-500/50 transition-colors"
    >
      <div className="w-12 h-12 rounded-xl bg-blue-600/20 flex items-center justify-center mb-4">
        <Icon className="w-6 h-6 text-blue-500" />
      </div>
      <h3 className="text-lg font-bold text-white mb-2">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </motion.div>
  )
}

// Testimonial card
function TestimonialCard({ name, role, content, avatar }: any) {
  return (
    <motion.div
      variants={fadeInUp}
      className="p-6 rounded-2xl bg-gray-900 border border-gray-800"
    >
      <div className="flex items-center gap-1 mb-4">
        {[...Array(5)].map((_, i) => (
          <Star key={i} className="w-4 h-4 fill-yellow-500 text-yellow-500" />
        ))}
      </div>
      <p className="text-gray-300 mb-4">"{content}"</p>
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold">
          {name[0]}
        </div>
        <div>
          <p className="font-semibold text-white">{name}</p>
          <p className="text-sm text-gray-400">{role}</p>
        </div>
      </div>
    </motion.div>
  )
}

// Live Stats Ticker
function LiveStatsTicker() {
  const [currentIndex, setCurrentIndex] = useState(0)

  const signals = [
    { symbol: 'TRENT', direction: 'BUY', confidence: 78, change: '+2.3%', color: 'text-green-500' },
    { symbol: 'POLYCAB', direction: 'BUY', confidence: 82, change: '+1.8%', color: 'text-green-500' },
    { symbol: 'VEDL', direction: 'SELL', confidence: 75, change: '-1.2%', color: 'text-red-500' },
    { symbol: 'INFY', direction: 'BUY', confidence: 71, change: '+0.9%', color: 'text-green-500' },
    { symbol: 'RELIANCE', direction: 'BUY', confidence: 85, change: '+1.5%', color: 'text-green-500' },
    { symbol: 'TCS', direction: 'SELL', confidence: 68, change: '-0.7%', color: 'text-red-500' },
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % signals.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="bg-background-surface/50 backdrop-blur-xl border border-gray-800 rounded-xl px-6 py-4 overflow-hidden">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2 text-sm font-medium text-text-muted">
          <Activity className="w-4 h-4 text-primary animate-pulse" />
          <span>Live Signals</span>
        </div>

        <motion.div
          key={currentIndex}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          className="flex items-center gap-4 font-mono"
        >
          <span className="font-bold text-text-primary">{signals[currentIndex].symbol}</span>
          <div className={`px-2 py-1 rounded ${
            signals[currentIndex].direction === 'BUY' ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'
          }`}>
            {signals[currentIndex].direction}
          </div>
          <span className="text-text-secondary">{signals[currentIndex].confidence}% Confidence</span>
          <span className={signals[currentIndex].color}>{signals[currentIndex].change}</span>
        </motion.div>
      </div>
    </div>
  )
}

// AI Model Card
function AIModelCard({ name, description, icon: Icon, accuracy, features }: any) {
  return (
    <motion.div
      variants={fadeInUp}
      whileHover={{ y: -8, transition: { duration: 0.3 } }}
      className="relative group p-8 rounded-2xl bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 hover:border-blue-500/50 transition-all"
    >
      {/* Glow effect on hover */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-blue-500/0 to-purple-500/0 group-hover:from-blue-500/10 group-hover:to-purple-500/10 transition-all" />

      <div className="relative z-10">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center mb-4">
          <Icon className="w-8 h-8 text-white" />
        </div>

        <h3 className="text-2xl font-bold text-white mb-2">{name}</h3>
        <p className="text-gray-400 mb-4">{description}</p>

        <div className="flex items-center gap-2 mb-4">
          <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              whileInView={{ width: `${accuracy}%` }}
              transition={{ duration: 1, delay: 0.2 }}
              className="h-full bg-gradient-to-r from-green-500 to-green-400"
            />
          </div>
          <span className="text-sm font-bold text-green-500">{accuracy}%</span>
        </div>

        <ul className="space-y-2">
          {features.map((feature: string, i: number) => (
            <li key={i} className="flex items-center gap-2 text-sm text-gray-300">
              <Check className="w-4 h-4 text-green-500" />
              {feature}
            </li>
          ))}
        </ul>
      </div>
    </motion.div>
  )
}

// Scanner Category Card
function ScannerCategoryCard({ name, count, icon: Icon, color }: any) {
  return (
    <motion.div
      variants={fadeInUp}
      whileHover={{ scale: 1.05 }}
      className={`p-6 rounded-xl bg-gradient-to-br ${color} border border-white/10 cursor-pointer`}
    >
      <Icon className="w-8 h-8 text-white mb-3" />
      <h3 className="text-lg font-bold text-white mb-1">{name}</h3>
      <p className="text-white/80 text-sm">{count} scanners</p>
    </motion.div>
  )
}

// FAQ Item
function FAQItem({ question, answer }: any) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <motion.div
      variants={fadeInUp}
      className="border border-gray-800 rounded-xl overflow-hidden"
    >
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-gray-900/50 transition-colors"
      >
        <span className="font-semibold text-white pr-4">{question}</span>
        {isOpen ? (
          <ChevronUp className="w-5 h-5 text-gray-400 flex-shrink-0" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-400 flex-shrink-0" />
        )}
      </button>
      <motion.div
        initial={false}
        animate={{ height: isOpen ? 'auto' : 0 }}
        className="overflow-hidden"
      >
        <div className="px-6 pb-4 text-gray-400">
          {answer}
        </div>
      </motion.div>
    </motion.div>
  )
}

// Main Landing Page
export default function LandingPage() {
  const { scrollY } = useScroll()
  const opacity = useTransform(scrollY, [0, 200], [1, 0])
  const scale = useTransform(scrollY, [0, 200], [1, 0.95])
  
  const plans = [
    {
      name: 'Free',
      description: 'Get started with basic signals',
      price: 0,
      features: ['3 signals/day', '2 max positions', 'Email alerts', 'Basic dashboard']
    },
    {
      name: 'Starter',
      description: 'Perfect for beginners',
      price: 999,
      features: ['10 signals/day', '5 max positions', 'Semi-auto trading', 'Telegram alerts', '₹5L max capital']
    },
    {
      name: 'Pro',
      description: 'For serious traders',
      price: 1999,
      features: ['25 signals/day', '10 max positions', 'Full-auto trading', 'F&O Futures', 'Priority support', '₹25L max capital']
    },
    {
      name: 'Elite',
      description: 'Maximum power',
      price: 4999,
      features: ['50 signals/day', '20 max positions', 'Options trading', 'API access', 'Dedicated support', 'Unlimited capital']
    }
  ]
  
  const features = [
    { icon: Brain, title: '3-Model AI Ensemble', description: 'CatBoost + TFT + Stockformer working together for maximum accuracy' },
    { icon: Target, title: '60+ Features', description: 'SMC/ICT patterns, price action, volume, FII/DII flows analyzed in real-time' },
    { icon: Shield, title: '5-Layer Risk Management', description: 'Automatic position sizing, stop losses, and circuit breakers' },
    { icon: Zap, title: 'Auto-Execution', description: 'Connect Zerodha, Angel One, or Upstox for automatic trading' },
    { icon: LineChart, title: 'F&O Trading', description: 'Trade Futures and Options with our advanced signals' },
    { icon: Bell, title: 'Real-time Alerts', description: 'Instant notifications via Telegram, Email, and Push' }
  ]
  
  const testimonials = [
    { name: 'Rajesh Kumar', role: 'Day Trader, Mumbai', content: 'SwingAI has completely transformed my trading. The AI signals are incredibly accurate and the auto-execution saves me hours every day.' },
    { name: 'Priya Sharma', role: 'Part-time Trader', content: 'As someone with a full-time job, SwingAI lets me trade professionally without staring at charts all day. Amazing product!' },
    { name: 'Amit Patel', role: 'F&O Trader, Ahmedabad', content: 'The F&O signals are game-changing. Finally, a platform that understands options trading properly.' }
  ]

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      {/* Gradient Background */}
      <div className="fixed inset-0 bg-gradient-to-b from-blue-950/20 via-black to-black pointer-events-none" />
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-transparent to-transparent pointer-events-none" />
      
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-lg border-b border-gray-800">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
            SwingAI
          </Link>
          <div className="hidden md:flex items-center gap-8">
            <Link href="#features" className="text-gray-400 hover:text-white transition">Features</Link>
            <Link href="#pricing" className="text-gray-400 hover:text-white transition">Pricing</Link>
            <Link href="#testimonials" className="text-gray-400 hover:text-white transition">Testimonials</Link>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/login" className="text-gray-300 hover:text-white transition">Login</Link>
            <Link href="/signup" className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition">
              Start Free Trial
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center pt-20">
        <div className="container mx-auto px-6 py-20">
          {/* Floating 3D Chart Elements */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            <motion.div
              animate={{ y: [0, -20, 0], rotate: [0, 5, 0] }}
              transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
              className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl backdrop-blur-sm border border-blue-500/30"
              style={{ transform: 'perspective(1000px) rotateY(20deg)' }}
            />
            <motion.div
              animate={{ y: [0, 20, 0], rotate: [0, -5, 0] }}
              transition={{ duration: 8, repeat: Infinity, ease: "easeInOut", delay: 1 }}
              className="absolute top-40 right-20 w-24 h-24 bg-gradient-to-br from-green-500/20 to-blue-500/20 rounded-2xl backdrop-blur-sm border border-green-500/30"
              style={{ transform: 'perspective(1000px) rotateY(-20deg)' }}
            />
            <motion.div
              animate={{ y: [0, -15, 0], rotate: [0, 3, 0] }}
              transition={{ duration: 7, repeat: Infinity, ease: "easeInOut", delay: 2 }}
              className="absolute bottom-40 left-1/4 w-28 h-28 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl backdrop-blur-sm border border-purple-500/30"
              style={{ transform: 'perspective(1000px) rotateY(15deg)' }}
            />
          </div>

          <motion.div
            style={{ opacity, scale }}
            className="text-center max-w-5xl mx-auto relative z-10"
          >
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600/20 border border-blue-500/30 rounded-full text-blue-400 text-sm mb-8"
            >
              <Zap className="w-4 h-4" />
              Powered by 3 AI Models • 58-62% Accuracy
            </motion.div>
            
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="text-5xl md:text-7xl font-bold mb-6 leading-tight"
            >
              AI-Powered{' '}
              <span className="bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
                Swing Trading
              </span>
              <br />for Indian Markets
            </motion.h1>
            
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto"
            >
              Automated trading signals and execution for NSE/BSE. 
              Connect your broker and let AI handle the rest.
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="flex flex-col sm:flex-row justify-center gap-4 mb-16"
            >
              <Link
                href="/signup"
                className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 rounded-xl font-semibold text-lg flex items-center justify-center gap-2 transition-all hover:scale-105"
              >
                Start 7-Day Free Trial <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                href="#demo"
                className="px-8 py-4 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-xl font-semibold text-lg flex items-center justify-center gap-2 transition-all"
              >
                <Play className="w-5 h-5" /> Watch Demo
              </Link>
            </motion.div>
            
            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-8"
            >
              <div className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-blue-500">
                  <AnimatedCounter value={58} suffix="%" />
                </div>
                <div className="text-gray-400">Avg Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-green-500">
                  <AnimatedCounter value={5000} suffix="+" />
                </div>
                <div className="text-gray-400">Active Traders</div>
              </div>
              <div className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-purple-500">
                  <AnimatedCounter value={150000} />
                </div>
                <div className="text-gray-400">Signals Generated</div>
              </div>
              <div className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-orange-500">
                  1:2
                </div>
                <div className="text-gray-400">Risk:Reward</div>
              </div>
            </motion.div>
          </motion.div>
        </div>
        
        {/* Scroll indicator */}
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <ChevronRight className="w-6 h-6 text-gray-500 rotate-90" />
        </motion.div>
      </section>

      {/* Live Stats Ticker Section */}
      <section className="relative py-8">
        <div className="container mx-auto px-6">
          <LiveStatsTicker />
        </div>
      </section>

      {/* AI Models Section */}
      <section className="relative py-20">
        <div className="container mx-auto px-6">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.div
              variants={fadeInUp}
              className="inline-flex items-center gap-2 px-4 py-2 bg-purple-600/20 border border-purple-500/30 rounded-full text-purple-400 text-sm mb-4"
            >
              <Sparkles className="w-4 h-4" />
              3-Model Ensemble AI
            </motion.div>
            <motion.h2 variants={fadeInUp} className="text-4xl md:text-5xl font-bold mb-4">
              Powered by <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-purple-500">Advanced AI Models</span>
            </motion.h2>
            <motion.p variants={fadeInUp} className="text-gray-400 text-lg max-w-2xl mx-auto">
              Three cutting-edge deep learning models working together to deliver the most accurate trading signals
            </motion.p>
          </motion.div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid md:grid-cols-3 gap-8"
          >
            <AIModelCard
              name="CatBoost"
              description="Gradient boosting model optimized for tabular data with categorical features"
              icon={BarChart3}
              accuracy={62}
              features={[
                'Fast inference speed',
                'Handles categorical data natively',
                'Robust to overfitting',
                'Best for feature-rich data'
              ]}
            />
            <AIModelCard
              name="Temporal Fusion Transformer (TFT)"
              description="State-of-the-art attention mechanism for multi-horizon time series forecasting"
              icon={Activity}
              accuracy={59}
              features={[
                'Multi-horizon forecasting',
                'Temporal attention layers',
                'Variable selection',
                'Interprets feature importance'
              ]}
            />
            <AIModelCard
              name="Stockformer"
              description="Specialized transformer architecture designed specifically for stock price prediction"
              icon={TrendingUp}
              accuracy={58}
              features={[
                'Stock-specific architecture',
                'Price pattern recognition',
                'Market regime detection',
                'Long-range dependencies'
              ]}
            />
          </motion.div>

          {/* Ensemble Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-12 p-8 rounded-2xl bg-gradient-to-r from-blue-900/20 to-purple-900/20 border border-blue-500/30"
          >
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-blue-500/20">
                <Brain className="w-8 h-8 text-blue-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white mb-2">Ensemble Voting System</h3>
                <p className="text-gray-400 mb-4">
                  All three models independently analyze each stock. A signal is generated only when at least 2 out of 3 models agree,
                  ensuring higher accuracy and reducing false signals. The ensemble confidence score reflects model agreement strength.
                </p>
                <div className="flex items-center gap-6 text-sm">
                  <div className="flex items-center gap-2">
                    <Check className="w-5 h-5 text-green-500" />
                    <span className="text-gray-300">3/3 Agreement = 85%+ Confidence</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Check className="w-5 h-5 text-yellow-500" />
                    <span className="text-gray-300">2/3 Agreement = 70-85% Confidence</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* PKScreener Section */}
      <section className="relative py-20 bg-gradient-to-b from-gray-900/50 to-black">
        <div className="container mx-auto px-6">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.div
              variants={fadeInUp}
              className="inline-flex items-center gap-2 px-4 py-2 bg-green-600/20 border border-green-500/30 rounded-full text-green-400 text-sm mb-4"
            >
              <Search className="w-4 h-4" />
              40+ Professional Scanners
            </motion.div>
            <motion.h2 variants={fadeInUp} className="text-4xl md:text-5xl font-bold mb-4">
              Powered by <span className="text-green-400">PKScreener</span>
            </motion.h2>
            <motion.p variants={fadeInUp} className="text-gray-400 text-lg max-w-2xl mx-auto">
              Industry-leading stock screening with 40+ pre-built scanners for every trading strategy
            </motion.p>
          </motion.div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-12"
          >
            <ScannerCategoryCard
              name="Breakouts"
              count={7}
              icon={TrendingUp}
              color="from-green-600 to-green-700"
            />
            <ScannerCategoryCard
              name="Reversals"
              count={6}
              icon={Activity}
              color="from-orange-600 to-orange-700"
            />
            <ScannerCategoryCard
              name="Momentum"
              count={8}
              icon={Zap}
              color="from-blue-600 to-blue-700"
            />
            <ScannerCategoryCard
              name="Volume"
              count={5}
              icon={BarChart3}
              color="from-purple-600 to-purple-700"
            />
            <ScannerCategoryCard
              name="Smart Money"
              count={4}
              icon={Target}
              color="from-pink-600 to-pink-700"
            />
            <ScannerCategoryCard
              name="Patterns"
              count={6}
              icon={Filter}
              color="from-indigo-600 to-indigo-700"
            />
          </motion.div>

          {/* Example Scanners */}
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="bg-gray-900/50 backdrop-blur-sm border border-gray-800 rounded-2xl p-8"
          >
            <h3 className="text-2xl font-bold text-white mb-6">Popular Scanners Include:</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[
                '52-Week High Breakout',
                'Unusual Volume Spike',
                'Bullish Reversal Patterns',
                'RSI Oversold (<30)',
                'MACD Bullish Crossover',
                'FII/DII Net Buying',
                'Opening Range Breakout',
                'Cup & Handle Pattern',
                'Volume Breakout with Price'
              ].map((scanner, i) => (
                <motion.div
                  key={i}
                  variants={fadeInUp}
                  className="flex items-center gap-2 p-3 rounded-lg bg-gray-800/50 border border-gray-700"
                >
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                  <span className="text-gray-300 text-sm">{scanner}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative py-20">
        <div className="container mx-auto px-6">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.h2 variants={fadeInUp} className="text-4xl md:text-5xl font-bold mb-4">
              Why Choose <span className="text-blue-500">SwingAI</span>?
            </motion.h2>
            <motion.p variants={fadeInUp} className="text-gray-400 text-lg max-w-2xl mx-auto">
              Built by traders, for traders. Our AI analyzes 60+ features to deliver high-probability swing trading signals.
            </motion.p>
          </motion.div>
          
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {features.map((feature, i) => (
              <FeatureCard key={i} {...feature} />
            ))}
          </motion.div>
        </div>
      </section>

      {/* How It Works */}
      <section className="relative py-20 bg-gradient-to-b from-gray-900/50 to-black">
        <div className="container mx-auto px-6">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.h2 variants={fadeInUp} className="text-4xl md:text-5xl font-bold mb-4">
              How It Works
            </motion.h2>
            <motion.p variants={fadeInUp} className="text-gray-400 text-lg">
              Three simple steps to automated trading
            </motion.p>
          </motion.div>
          
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid md:grid-cols-3 gap-8"
          >
            {[
              { step: '01', title: 'Sign Up & Connect', desc: 'Create account and connect your broker (Zerodha, Angel One, Upstox)' },
              { step: '02', title: 'Configure Settings', desc: 'Set your capital, risk profile, and preferred trading mode' },
              { step: '03', title: 'Start Trading', desc: 'Receive AI signals and let the system execute trades automatically' }
            ].map((item, i) => (
              <motion.div key={i} variants={fadeInUp} className="relative">
                <div className="text-8xl font-bold text-gray-800 absolute -top-4 left-0">{item.step}</div>
                <div className="relative z-10 pt-16">
                  <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                  <p className="text-gray-400">{item.desc}</p>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="relative py-20">
        <div className="container mx-auto px-6">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.h2 variants={fadeInUp} className="text-4xl md:text-5xl font-bold mb-4">
              Simple, Transparent Pricing
            </motion.h2>
            <motion.p variants={fadeInUp} className="text-gray-400 text-lg">
              Start free, upgrade when you're ready
            </motion.p>
          </motion.div>
          
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto"
          >
            {plans.map((plan, i) => (
              <PricingCard key={i} plan={plan} popular={i === 2} />
            ))}
          </motion.div>
        </div>
      </section>

      {/* Testimonials */}
      <section id="testimonials" className="relative py-20 bg-gradient-to-b from-gray-900/50 to-black">
        <div className="container mx-auto px-6">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.h2 variants={fadeInUp} className="text-4xl md:text-5xl font-bold mb-4">
              Loved by <span className="text-blue-500">Traders</span>
            </motion.h2>
          </motion.div>
          
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto"
          >
            {testimonials.map((t, i) => (
              <TestimonialCard key={i} {...t} />
            ))}
          </motion.div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="relative py-20">
        <div className="container mx-auto px-6">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.h2 variants={fadeInUp} className="text-4xl md:text-5xl font-bold mb-4">
              Frequently Asked Questions
            </motion.h2>
            <motion.p variants={fadeInUp} className="text-gray-400 text-lg">
              Everything you need to know about SwingAI
            </motion.p>
          </motion.div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="max-w-3xl mx-auto space-y-4"
          >
            <FAQItem
              question="How accurate are the AI trading signals?"
              answer="Our 3-model ensemble system achieves 58-62% accuracy across different market conditions. We prioritize quality over quantity - only generating signals when at least 2 out of 3 models agree. This conservative approach results in fewer but more reliable signals with proper risk:reward ratios (minimum 1:2)."
            />
            <FAQItem
              question="Which brokers are supported for auto-trading?"
              answer="We currently support Zerodha, Angel One, and Upstox for automatic trade execution. You'll need an active trading account with API access enabled. Our system connects securely using your broker's official APIs and never stores your trading credentials."
            />
            <FAQItem
              question="Can I trade F&O (Futures & Options) with SwingAI?"
              answer="Yes! Pro and Elite plans include F&O trading signals for both Futures and Options. Our AI models are specifically trained on F&O data including Open Interest analysis, option Greeks, and derivatives market structure. F&O signals come with enhanced risk management given the leverage involved."
            />
            <FAQItem
              question="How does the risk management system work?"
              answer="SwingAI implements 5-layer risk management: (1) Position sizing based on your capital and risk per trade, (2) Automatic stop-loss placement with every signal, (3) Daily loss limits to prevent catastrophic drawdowns, (4) Maximum concurrent positions based on your plan, (5) Circuit breakers that pause trading if daily loss threshold is exceeded."
            />
            <FAQItem
              question="What's the difference between signal-only and auto-trading modes?"
              answer="Signal-only mode sends you notifications when new trading opportunities are detected - you manually place trades through your broker. Semi-auto mode shows signals with a 'one-click execute' button requiring approval. Full-auto mode automatically executes trades without manual intervention (requires Pro/Elite plans and broker connection)."
            />
            <FAQItem
              question="Do I need prior trading experience?"
              answer="While prior market knowledge helps, SwingAI is designed for traders of all levels. Our signals include complete entry/exit prices and risk management. We recommend starting with the Free or Starter plan in signal-only mode to understand the system before enabling auto-trading."
            />
            <FAQItem
              question="What's included in the 7-day free trial?"
              answer="The free trial gives you full access to the Starter plan features: 10 signals/day, semi-auto trading, Telegram alerts, and complete dashboard access. No credit card required. After 7 days, you can continue with the Free plan or upgrade to any paid tier."
            />
            <FAQItem
              question="How are the 40+ PKScreener scans different from AI signals?"
              answer="PKScreener scans help you discover stocks matching specific technical criteria (breakouts, reversals, volume spikes, etc.) in real-time. AI signals are actionable trade recommendations with entry/exit/stop-loss levels. Use scanners to build watchlists and discover opportunities, then wait for AI signals for optimal entry timing."
            />
            <FAQItem
              question="Can I backtest the AI models' performance?"
              answer="Yes! Your dashboard includes detailed performance analytics showing historical signal accuracy, win rate, profit factor, and equity curve. Elite plan users get access to our backtesting API to test strategies on historical data before deploying them live."
            />
            <FAQItem
              question="What happens if my broker connection fails?"
              answer="If your broker API connection drops, SwingAI automatically switches to signal-only mode and sends you alerts via Telegram/Email. You can manually execute trades through your broker platform. The system continuously attempts to reconnect and will resume auto-trading once the connection is restored."
            />
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-20">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="relative rounded-3xl bg-gradient-to-r from-blue-600 to-purple-600 p-12 text-center overflow-hidden"
          >
            <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />
            <div className="relative z-10">
              <h2 className="text-4xl md:text-5xl font-bold mb-4">
                Ready to Transform Your Trading?
              </h2>
              <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
                Join 5,000+ traders already using SwingAI. Start your 7-day free trial today.
              </p>
              <Link
                href="/signup"
                className="inline-flex items-center gap-2 px-8 py-4 bg-white text-blue-600 rounded-xl font-semibold text-lg hover:bg-blue-50 transition"
              >
                Get Started Free <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-12">
        <div className="container mx-auto px-6">
          {/* Newsletter Section */}
          <div className="mb-12 p-8 rounded-2xl bg-gradient-to-r from-gray-900 to-gray-800 border border-gray-700">
            <div className="max-w-2xl mx-auto text-center">
              <Bell className="w-12 h-12 text-blue-500 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-white mb-2">Stay Updated</h3>
              <p className="text-gray-400 mb-6">
                Get weekly trading insights, AI signal updates, and exclusive tips delivered to your inbox
              </p>
              <form className="flex flex-col sm:flex-row gap-3">
                <input
                  type="email"
                  placeholder="Enter your email"
                  className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                />
                <button
                  type="submit"
                  className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all"
                >
                  Subscribe
                </button>
              </form>
              <p className="text-xs text-gray-500 mt-3">
                No spam. Unsubscribe anytime. Your email is safe with us.
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-5 gap-8 mb-8">
            <div className="md:col-span-2">
              <div className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent mb-4">
                SwingAI
              </div>
              <p className="text-gray-400 mb-4">
                AI-powered swing trading for Indian markets. Built by traders, for traders.
              </p>
              <div className="flex items-center gap-4">
                <Link href="https://twitter.com" target="_blank" className="text-gray-400 hover:text-blue-400 transition">
                  <Globe className="w-5 h-5" />
                </Link>
                <Link href="https://telegram.org" target="_blank" className="text-gray-400 hover:text-blue-400 transition">
                  <Bell className="w-5 h-5" />
                </Link>
                <Link href="https://github.com" target="_blank" className="text-gray-400 hover:text-white transition">
                  <Code className="w-5 h-5" />
                </Link>
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-4 text-white">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="#features" className="hover:text-white transition">Features</Link></li>
                <li><Link href="#pricing" className="hover:text-white transition">Pricing</Link></li>
                <li><Link href="/dashboard" className="hover:text-white transition">Dashboard</Link></li>
                <li><Link href="/docs" className="hover:text-white transition">Documentation</Link></li>
                <li><Link href="/api" className="hover:text-white transition">API</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4 text-white">Company</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/about" className="hover:text-white transition">About</Link></li>
                <li><Link href="/blog" className="hover:text-white transition">Blog</Link></li>
                <li><Link href="/contact" className="hover:text-white transition">Contact</Link></li>
                <li><Link href="/careers" className="hover:text-white transition">Careers</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4 text-white">Legal</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/privacy" className="hover:text-white transition">Privacy Policy</Link></li>
                <li><Link href="/terms" className="hover:text-white transition">Terms of Service</Link></li>
                <li><Link href="/disclaimer" className="hover:text-white transition">Risk Disclaimer</Link></li>
                <li><Link href="/refund" className="hover:text-white transition">Refund Policy</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-gray-500 text-sm text-center md:text-left">
              © 2025 SwingAI. All rights reserved. Trading involves risk. Past performance doesn't guarantee future results.
            </p>
            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span>Made with ❤️ in India</span>
              <span>•</span>
              <span>SEBI Registered</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
