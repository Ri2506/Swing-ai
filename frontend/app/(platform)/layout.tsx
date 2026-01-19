'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  BarChart3,
  TrendingUp,
  Search,
  Bell,
  User,
  ChevronDown,
  Sparkles,
  LineChart,
  Target,
  Settings,
  LogOut,
  Menu,
  X,
  Calculator,
  Shield,
} from 'lucide-react'
import CalculatorModal from '@/components/CalculatorModal'

export default function PlatformLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [calculatorType, setCalculatorType] = useState<'position' | 'risk' | null>(null)

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { name: 'AI Screener', href: '/screener', icon: Sparkles },
    { name: 'Signals', href: '/signals', icon: Target },
    { name: 'Stocks', href: '/stocks', icon: TrendingUp },
    { name: 'Portfolio', href: '/portfolio', icon: LineChart },
  ]

  return (
    <div className="min-h-screen bg-background-primary">
      {/* Calculator Modals */}
      {calculatorType && (
        <CalculatorModal
          isOpen={!!calculatorType}
          onClose={() => setCalculatorType(null)}
          type={calculatorType}
        />
      )}

      {/* Top Navigation */}
      <nav className="fixed top-0 z-50 w-full border-b border-border/60 bg-background-primary/80 backdrop-blur-xl">
        <div className="mx-auto px-6">
          <div className="flex h-16 items-center justify-between">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-2">
              <span className="text-xl font-bold gradient-text-professional">SwingAI</span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden items-center gap-6 md:flex">
              {navigation.map((item) => {
                const Icon = item.icon
                const isActive = pathname === item.href
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`flex items-center gap-2 text-sm font-medium transition ${
                      isActive
                        ? 'text-accent'
                        : 'text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {item.name}
                  </Link>
                )
              })}
            </div>

            {/* Right Actions */}
            <div className="flex items-center gap-3">
              {/* Calculator Buttons */}
              <button
                onClick={() => setCalculatorType('position')}
                className="hidden items-center gap-2 rounded-lg border border-primary/60 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition hover:bg-primary/20 md:flex"
              >
                <Calculator className="h-4 w-4" />
                Position Size
              </button>
              <button
                onClick={() => setCalculatorType('risk')}
                className="hidden items-center gap-2 rounded-lg border border-accent/60 bg-accent/10 px-4 py-2 text-sm font-medium text-accent transition hover:bg-accent/20 md:flex"
              >
                <Shield className="h-4 w-4" />
                Risk Manager
              </button>

              <button className="flex h-9 w-9 items-center justify-center rounded-lg border border-border/60 text-text-secondary transition hover:text-accent">
                <Bell className="h-4 w-4" />
              </button>
              <Link
                href="/settings"
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-border/60 text-text-secondary transition hover:text-accent"
              >
                <User className="h-4 w-4" />
              </Link>

              {/* Mobile Menu Button */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-border/60 text-text-secondary transition hover:text-accent md:hidden"
              >
                {mobileMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="border-t border-border/60 py-4 md:hidden">
              {navigation.map((item) => {
                const Icon = item.icon
                const isActive = pathname === item.href
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`flex items-center gap-3 px-4 py-3 text-sm font-medium transition ${
                      isActive
                        ? 'text-accent bg-accent/10'
                        : 'text-text-secondary hover:bg-background-surface'
                    }`}
                  >
                    <Icon className="h-5 w-5" />
                    {item.name}
                  </Link>
                )
              })}
              
              {/* Mobile Calculator Buttons */}
              <div className="mt-4 space-y-2 px-4">
                <button
                  onClick={() => {
                    setCalculatorType('position')
                    setMobileMenuOpen(false)
                  }}
                  className="flex w-full items-center gap-2 rounded-lg border border-primary/60 bg-primary/10 px-4 py-3 text-sm font-medium text-primary"
                >
                  <Calculator className="h-5 w-5" />
                  Position Size Calculator
                </button>
                <button
                  onClick={() => {
                    setCalculatorType('risk')
                    setMobileMenuOpen(false)
                  }}
                  className="flex w-full items-center gap-2 rounded-lg border border-accent/60 bg-accent/10 px-4 py-3 text-sm font-medium text-accent"
                >
                  <Shield className="h-5 w-5" />
                  Risk Management Calculator
                </button>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-16">{children}</main>
    </div>
  )
}

