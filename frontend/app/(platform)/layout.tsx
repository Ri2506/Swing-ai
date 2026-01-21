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
  Menu,
  X,
  Calculator,
  Shield,
  Zap,
  Brain,
  TrendingDown,
  Briefcase,
  PieChart,
  LayoutGrid,
} from 'lucide-react'
import CalculatorModal from '@/components/CalculatorModal'

export default function PlatformLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [calculatorType, setCalculatorType] = useState<'position' | 'risk' | null>(null)
  const [quickMenuOpen, setQuickMenuOpen] = useState(false)

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { name: 'AI Screener', href: '/screener', icon: Sparkles },
    { name: 'AI Intelligence', href: '/ai-intelligence', icon: Brain },
    { name: 'Signals', href: '/signals', icon: Target },
    { name: 'Stocks', href: '/stocks', icon: TrendingUp },
    { name: 'Paper Trading', href: '/paper-trading', icon: Briefcase },
    { name: 'Tools', href: '/tools', icon: Calculator },
  ]

  const quickAccessMenu = [
    {
      category: 'Trading Tools',
      items: [
        { name: 'Position Sizing', action: () => setCalculatorType('position'), icon: Calculator },
        { name: 'Risk Manager', action: () => setCalculatorType('risk'), icon: Shield },
        { name: 'AI Signals', href: '/signals', icon: Zap },
      ],
    },
    {
      category: 'Market Data',
      items: [
        { name: 'Market Overview', href: '/dashboard', icon: BarChart3 },
        { name: 'Top Gainers', href: '/stocks?filter=gainers', icon: TrendingUp },
        { name: 'Top Losers', href: '/stocks?filter=losers', icon: TrendingDown },
        { name: 'Trending Stocks', href: '/stocks?filter=trending', icon: Sparkles },
      ],
    },
    {
      category: 'Analysis',
      items: [
        { name: 'AI Screener', href: '/screener', icon: Brain },
        { name: 'Sector Performance', href: '/dashboard?view=sectors', icon: PieChart },
        { name: 'My Portfolio', href: '/portfolio', icon: Briefcase },
      ],
    },
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
              {/* Quick Access Dropdown */}
              <div className="relative">
                <button
                  onClick={() => setQuickMenuOpen(!quickMenuOpen)}
                  className="hidden items-center gap-2 rounded-lg border border-border/60 bg-background-surface/60 px-4 py-2 text-sm font-medium text-text-primary transition hover:border-accent/60 md:flex"
                >
                  <LayoutGrid className="h-4 w-4" />
                  Quick Access
                  <ChevronDown className={`h-4 w-4 transition-transform ${quickMenuOpen ? 'rotate-180' : ''}`} />
                </button>

                {/* Dropdown Menu */}
                {quickMenuOpen && (
                  <>
                    <div
                      className="fixed inset-0 z-10"
                      onClick={() => setQuickMenuOpen(false)}
                    />
                    <div className="absolute right-0 top-full z-20 mt-2 w-80 rounded-xl border border-border/60 bg-background-surface shadow-2xl">
                      <div className="p-4">
                        {quickAccessMenu.map((section) => (
                          <div key={section.category} className="mb-4 last:mb-0">
                            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-secondary">
                              {section.category}
                            </div>
                            <div className="space-y-1">
                              {section.items.map((item) => {
                                const Icon = item.icon
                                if (item.action) {
                                  return (
                                    <button
                                      key={item.name}
                                      onClick={() => {
                                        item.action()
                                        setQuickMenuOpen(false)
                                      }}
                                      className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-text-secondary transition hover:bg-background-elevated hover:text-text-primary"
                                    >
                                      <Icon className="h-4 w-4" />
                                      {item.name}
                                    </button>
                                  )
                                }
                                return (
                                  <Link
                                    key={item.name}
                                    href={item.href || '#'}
                                    onClick={() => setQuickMenuOpen(false)}
                                    className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-text-secondary transition hover:bg-background-elevated hover:text-text-primary"
                                  >
                                    <Icon className="h-4 w-4" />
                                    {item.name}
                                  </Link>
                                )
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Calculator Buttons */}
              <button
                onClick={() => setCalculatorType('position')}
                className="hidden items-center gap-2 rounded-lg border border-primary/60 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition hover:bg-primary/20 lg:flex"
              >
                <Calculator className="h-4 w-4" />
                Position
              </button>
              <button
                onClick={() => setCalculatorType('risk')}
                className="hidden items-center gap-2 rounded-lg border border-accent/60 bg-accent/10 px-4 py-2 text-sm font-medium text-accent transition hover:bg-accent/20 lg:flex"
              >
                <Shield className="h-4 w-4" />
                Risk
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
              {/* Main Navigation */}
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
              
              {/* Quick Access in Mobile */}
              <div className="mt-4 border-t border-border/60 pt-4">
                {quickAccessMenu.map((section) => (
                  <div key={section.category} className="mb-4">
                    <div className="mb-2 px-4 text-xs font-semibold uppercase tracking-wider text-text-secondary">
                      {section.category}
                    </div>
                    {section.items.map((item) => {
                      const Icon = item.icon
                      if (item.action) {
                        return (
                          <button
                            key={item.name}
                            onClick={() => {
                              item.action()
                              setMobileMenuOpen(false)
                            }}
                            className="flex w-full items-center gap-3 px-4 py-2 text-sm text-text-secondary transition hover:bg-background-surface"
                          >
                            <Icon className="h-4 w-4" />
                            {item.name}
                          </button>
                        )
                      }
                      return (
                        <Link
                          key={item.name}
                          href={item.href || '#'}
                          onClick={() => setMobileMenuOpen(false)}
                          className="flex items-center gap-3 px-4 py-2 text-sm text-text-secondary transition hover:bg-background-surface"
                        >
                          <Icon className="h-4 w-4" />
                          {item.name}
                        </Link>
                      )
                    })}
                  </div>
                ))}
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


