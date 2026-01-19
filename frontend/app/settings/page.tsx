'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import {
  Settings,
  User,
  Bell,
  Shield,
  Wallet,
  Moon,
  Sun,
  Globe,
  Mail,
  Smartphone,
  Save,
  Check,
} from 'lucide-react'

export default function SettingsPage() {
  const [notifications, setNotifications] = useState({
    email: true,
    push: true,
    signals: true,
    trades: true,
    news: false,
  })
  const [riskSettings, setRiskSettings] = useState({
    maxPositionSize: 10,
    maxDailyLoss: 2,
    defaultStopLoss: 3,
  })
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="min-h-screen bg-background-primary px-6 py-8">
      <div className="mx-auto max-w-4xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="mb-2 text-4xl font-bold text-text-primary">
                <span className="gradient-text-professional">Settings</span>
              </h1>
              <p className="text-lg text-text-secondary">
                Manage your account and preferences
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

        <div className="space-y-6">
          {/* Profile Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6"
          >
            <div className="mb-6 flex items-center gap-3">
              <User className="h-6 w-6 text-accent" />
              <h2 className="text-xl font-bold text-text-primary">Profile</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <label className="mb-2 block text-sm font-medium text-text-secondary">Full Name</label>
                <input
                  type="text"
                  defaultValue="Rahul Sharma"
                  className="w-full rounded-lg border border-border/60 bg-background-primary/60 px-4 py-3 text-text-primary focus:border-accent/60 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium text-text-secondary">Email</label>
                <input
                  type="email"
                  defaultValue="rahul@example.com"
                  className="w-full rounded-lg border border-border/60 bg-background-primary/60 px-4 py-3 text-text-primary focus:border-accent/60 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium text-text-secondary">Phone</label>
                <input
                  type="tel"
                  defaultValue="+91 98765 43210"
                  className="w-full rounded-lg border border-border/60 bg-background-primary/60 px-4 py-3 text-text-primary focus:border-accent/60 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium text-text-secondary">Trading Capital (₹)</label>
                <input
                  type="number"
                  defaultValue="500000"
                  className="w-full rounded-lg border border-border/60 bg-background-primary/60 px-4 py-3 text-text-primary focus:border-accent/60 focus:outline-none"
                />
              </div>
            </div>
          </motion.div>

          {/* Notification Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6"
          >
            <div className="mb-6 flex items-center gap-3">
              <Bell className="h-6 w-6 text-accent" />
              <h2 className="text-xl font-bold text-text-primary">Notifications</h2>
            </div>
            <div className="space-y-4">
              {[
                { key: 'email', icon: Mail, label: 'Email Notifications' },
                { key: 'push', icon: Smartphone, label: 'Push Notifications' },
                { key: 'signals', label: 'New Signal Alerts' },
                { key: 'trades', label: 'Trade Execution Updates' },
                { key: 'news', label: 'Market News & Updates' },
              ].map(({ key, icon: Icon, label }) => (
                <div key={key} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {Icon && <Icon className="h-5 w-5 text-text-secondary" />}
                    <span className="text-text-primary">{label}</span>
                  </div>
                  <button
                    onClick={() => setNotifications(prev => ({ ...prev, [key]: !prev[key as keyof typeof prev] }))}
                    className={`h-6 w-11 rounded-full transition ${
                      notifications[key as keyof typeof notifications] ? 'bg-accent' : 'bg-background-primary'
                    }`}
                  >
                    <div
                      className={`h-5 w-5 transform rounded-full bg-white shadow transition ${
                        notifications[key as keyof typeof notifications] ? 'translate-x-5' : 'translate-x-0.5'
                      }`}
                    />
                  </button>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Risk Management Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="rounded-xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-6"
          >
            <div className="mb-6 flex items-center gap-3">
              <Shield className="h-6 w-6 text-accent" />
              <h2 className="text-xl font-bold text-text-primary">Risk Management</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <label className="mb-2 block text-sm font-medium text-text-secondary">Max Position Size (%)</label>
                <input
                  type="number"
                  value={riskSettings.maxPositionSize}
                  onChange={(e) => setRiskSettings(prev => ({ ...prev, maxPositionSize: parseInt(e.target.value) }))}
                  className="w-full rounded-lg border border-border/60 bg-background-primary/60 px-4 py-3 text-text-primary focus:border-accent/60 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium text-text-secondary">Max Daily Loss (%)</label>
                <input
                  type="number"
                  value={riskSettings.maxDailyLoss}
                  onChange={(e) => setRiskSettings(prev => ({ ...prev, maxDailyLoss: parseInt(e.target.value) }))}
                  className="w-full rounded-lg border border-border/60 bg-background-primary/60 px-4 py-3 text-text-primary focus:border-accent/60 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium text-text-secondary">Default Stop Loss (%)</label>
                <input
                  type="number"
                  value={riskSettings.defaultStopLoss}
                  onChange={(e) => setRiskSettings(prev => ({ ...prev, defaultStopLoss: parseInt(e.target.value) }))}
                  className="w-full rounded-lg border border-border/60 bg-background-primary/60 px-4 py-3 text-text-primary focus:border-accent/60 focus:outline-none"
                />
              </div>
            </div>
          </motion.div>

          {/* Save Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex justify-end"
          >
            <button
              onClick={handleSave}
              className={`flex items-center gap-2 rounded-xl px-6 py-3 font-semibold transition ${
                saved
                  ? 'bg-success text-white'
                  : 'bg-accent text-accent-foreground hover:bg-accent/90'
              }`}
            >
              {saved ? (
                <>
                  <Check className="h-5 w-5" />
                  Saved!
                </>
              ) : (
                <>
                  <Save className="h-5 w-5" />
                  Save Changes
                </>
              )}
            </button>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
