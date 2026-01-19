'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Mail, ArrowLeft } from 'lucide-react'

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitted(true)
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background-primary px-4">
      <div className="w-full max-w-md">
        <Link
          href="/login"
          className="mb-8 inline-flex items-center gap-2 text-sm text-text-secondary transition hover:text-text-primary"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Login
        </Link>

        <div className="rounded-2xl border border-border/60 bg-gradient-to-br from-background-surface to-background-elevated p-8">
          <h1 className="mb-2 text-2xl font-bold text-text-primary">Reset Password</h1>
          <p className="mb-6 text-text-secondary">
            Enter your email address and we'll send you a link to reset your password.
          </p>

          {submitted ? (
            <div className="rounded-lg bg-success/15 p-4 text-center">
              <p className="text-success">
                If an account exists for {email}, you will receive a password reset link shortly.
              </p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium text-text-secondary">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-text-secondary" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    required
                    className="w-full rounded-lg border border-border/60 bg-background-primary/60 py-3 pl-10 pr-4 text-text-primary placeholder-text-secondary focus:border-accent/60 focus:outline-none"
                  />
                </div>
              </div>
              <button
                type="submit"
                className="w-full rounded-lg bg-accent py-3 font-semibold text-accent-foreground transition hover:bg-accent/90"
              >
                Send Reset Link
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  )
}
