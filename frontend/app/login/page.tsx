'use client'

import { useState, useEffect, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { TrendingUp, Shield, Zap, ArrowRight } from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.REACT_APP_BACKEND_URL || ''

/**
 * Login Content Component
 * 
 * REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
 */
function LoginContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Check if user is already logged in
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await fetch(`${API_URL}/api/auth/me`, {
          credentials: 'include',
        })
        
        if (response.ok) {
          // Already logged in, redirect to dashboard
          router.push('/dashboard')
        }
      } catch {
        // Not logged in, stay on login page
      }
    }

    checkAuth()

    // Check for error in URL
    const errorParam = searchParams.get('error')
    if (errorParam) {
      const errorMessages: Record<string, string> = {
        'no_session': 'Authentication session not found. Please try again.',
        'auth_failed': 'Authentication failed. Please try again.',
        'invalid_response': 'Invalid response from server. Please try again.',
        'network': 'Network error. Please check your connection.',
      }
      setError(errorMessages[errorParam] || 'An error occurred. Please try again.')
    }
  }, [router, searchParams])

  const handleGoogleLogin = () => {
    setIsLoading(true)
    setError(null)
    
    // REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
    const redirectUrl = window.location.origin + '/auth/callback'
    window.location.href = `https://auth.emergentagent.com/?redirect=${encodeURIComponent(redirectUrl)}`
  }

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-accent/20 via-primary/10 to-background p-12 flex-col justify-between">
        <div>
          <div className="flex items-center gap-3 mb-12">
            <div className="w-10 h-10 bg-accent rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-text-primary">SwingAI</span>
          </div>
          
          <h1 className="text-4xl font-bold text-text-primary mb-4">
            AI-Powered Trading Intelligence
          </h1>
          <p className="text-xl text-text-secondary mb-12">
            Advanced stock screening and swing trading signals for the Indian market.
          </p>

          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="p-2 bg-accent/15 rounded-lg">
                <Zap className="w-5 h-5 text-accent" />
              </div>
              <div>
                <h3 className="font-semibold text-text-primary">2200+ NSE Stocks</h3>
                <p className="text-text-secondary text-sm">Scan the entire NSE market in seconds</p>
              </div>
            </div>
            
            <div className="flex items-start gap-4">
              <div className="p-2 bg-success/15 rounded-lg">
                <TrendingUp className="w-5 h-5 text-success" />
              </div>
              <div>
                <h3 className="font-semibold text-text-primary">AI Screeners</h3>
                <p className="text-text-secondary text-sm">43+ intelligent stock scanners</p>
              </div>
            </div>
            
            <div className="flex items-start gap-4">
              <div className="p-2 bg-primary/15 rounded-lg">
                <Shield className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold text-text-primary">Paper Trading</h3>
                <p className="text-text-secondary text-sm">Practice with ₹10 Lakh virtual money</p>
              </div>
            </div>
          </div>
        </div>

        <p className="text-text-secondary text-sm">
          © 2025 SwingAI. All rights reserved.
        </p>
      </div>

      {/* Right side - Login */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          {/* Mobile logo */}
          <div className="lg:hidden flex items-center gap-3 mb-8 justify-center">
            <div className="w-10 h-10 bg-accent rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-text-primary">SwingAI</span>
          </div>

          <div className="bg-background-card border border-border rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-text-primary text-center mb-2">
              Welcome Back
            </h2>
            <p className="text-text-secondary text-center mb-8">
              Sign in to access your trading dashboard
            </p>

            {error && (
              <div className="mb-6 p-4 bg-danger/10 border border-danger/30 rounded-lg text-danger text-sm">
                {error}
              </div>
            )}

            <button
              onClick={handleGoogleLogin}
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-white hover:bg-gray-50 text-gray-800 font-medium rounded-xl border border-gray-200 transition disabled:opacity-50"
              data-testid="google-login-btn"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path
                      fill="#4285F4"
                      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    />
                    <path
                      fill="#34A853"
                      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    />
                    <path
                      fill="#FBBC05"
                      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                    />
                    <path
                      fill="#EA4335"
                      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    />
                  </svg>
                  Continue with Google
                </>
              )}
            </button>

            <div className="relative my-8">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-border"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-background-card text-text-secondary">or</span>
              </div>
            </div>

            <button
              onClick={() => router.push('/paper-trading')}
              className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-background hover:bg-background-surface text-text-primary font-medium rounded-xl border border-border transition"
            >
              Try Paper Trading
              <ArrowRight className="w-4 h-4" />
            </button>

            <p className="text-xs text-text-secondary text-center mt-6">
              By signing in, you agree to our Terms of Service and Privacy Policy.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
