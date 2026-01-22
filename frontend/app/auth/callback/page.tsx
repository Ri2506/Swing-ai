'use client'

import { useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.REACT_APP_BACKEND_URL || ''

/**
 * AuthCallback Component
 * 
 * Handles the OAuth callback after Google sign-in.
 * Extracts session_id from URL hash, exchanges it for user data,
 * and redirects to dashboard.
 * 
 * REMINDER: DO NOT HARDCODE THE URL, OR ADD ANY FALLBACKS OR REDIRECT URLS, THIS BREAKS THE AUTH
 */
export default function AuthCallback() {
  const router = useRouter()
  const hasProcessed = useRef(false)

  useEffect(() => {
    // Prevent double processing in StrictMode
    if (hasProcessed.current) return
    hasProcessed.current = true

    const processAuth = async () => {
      try {
        // Get session_id from URL hash
        const hash = window.location.hash
        const sessionIdMatch = hash.match(/session_id=([^&]+)/)
        
        if (!sessionIdMatch) {
          console.error('No session_id found in URL')
          router.push('/login?error=no_session')
          return
        }

        const sessionId = sessionIdMatch[1]

        // Exchange session_id for user data
        const response = await fetch(`${API_URL}/api/auth/session`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({ session_id: sessionId }),
        })

        if (!response.ok) {
          const error = await response.json()
          console.error('Auth failed:', error)
          router.push('/login?error=auth_failed')
          return
        }

        const data = await response.json()

        if (data.success && data.user) {
          // Store user in localStorage for quick access
          localStorage.setItem('swingai_user', JSON.stringify(data.user))
          localStorage.setItem('swingai_user_id', data.user.user_id)
          
          // Clear the hash and redirect to dashboard
          window.history.replaceState(null, '', window.location.pathname)
          router.push('/dashboard')
        } else {
          router.push('/login?error=invalid_response')
        }
      } catch (error) {
        console.error('Auth error:', error)
        router.push('/login?error=network')
      }
    }

    processAuth()
  }, [router])

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 border-4 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-text-secondary">Signing you in...</p>
      </div>
    </div>
  )
}
