// ============================================================================
// SWINGAI - AUTH CONTEXT
// Global authentication state management
// ============================================================================

'use client'

import { createContext, useContext, useEffect, useState } from 'react'
import { User } from '@supabase/supabase-js'
import { supabase, getUserProfile } from '../lib/supabase'
import { UserProfile } from '../types'
import { useRouter } from 'next/navigation'

// ============================================================================
// TYPES
// ============================================================================

interface AuthContextType {
  user: User | null
  profile: UserProfile | null
  loading: boolean
  signUp: (email: string, password: string, fullName: string) => Promise<void>
  signIn: (email: string, password: string) => Promise<void>
  signInWithGoogle: () => Promise<void>
  signOut: () => Promise<void>
  refreshProfile: () => Promise<void>
}

// ============================================================================
// CONTEXT
// ============================================================================

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// ============================================================================
// PROVIDER
// ============================================================================

// DEV MODE - Check if Supabase is configured
const isDevMode = !process.env.NEXT_PUBLIC_SUPABASE_URL

// Mock user for development
const MOCK_USER: User = {
  id: 'dev-user-123',
  email: 'demo@swingai.com',
  aud: 'authenticated',
  role: 'authenticated',
  app_metadata: {},
  user_metadata: { full_name: 'Demo Trader' },
  created_at: new Date().toISOString(),
} as User

const MOCK_PROFILE: UserProfile = {
  id: 'dev-user-123',
  email: 'demo@swingai.com',
  full_name: 'Demo Trader',
  phone: '+91 98765 43210',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  capital: 500000,
  risk_profile: 'moderate',
  trading_mode: 'semi_auto',
  max_positions: 10,
  risk_per_trade: 2,
  fo_enabled: true,
  preferred_option_type: 'both',
  daily_loss_limit: 5,
  weekly_loss_limit: 10,
  monthly_loss_limit: 20,
  trailing_sl_enabled: true,
  notifications_enabled: true,
  telegram_chat_id: '',
  subscription_status: 'active',
  subscription_plan_id: undefined,
  broker_connected: true,
  broker_name: 'zerodha',
  total_trades: 42,
  winning_trades: 26,
  total_pnl: 12500,
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  // ============================================================================
  // LOAD USER ON MOUNT
  // ============================================================================

  useEffect(() => {
    // Get initial session
    const loadUser = async () => {
      try {
        // DEV MODE: Use mock user
        if (isDevMode) {
          console.log('🚀 DEV MODE: Using mock user')
          setUser(MOCK_USER)
          setProfile(MOCK_PROFILE)
          setLoading(false)
          return
        }

        const { data: { session } } = await supabase.auth.getSession()

        if (session?.user) {
          setUser(session.user)
          await loadProfile(session.user.id)
        }
      } catch (error) {
        console.error('Error loading user:', error)
        // In case of error, use mock user in dev
        if (isDevMode) {
          setUser(MOCK_USER)
          setProfile(MOCK_PROFILE)
        }
      } finally {
        setLoading(false)
      }
    }

    loadUser()

    // Skip auth listener in dev mode
    if (isDevMode) return

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        console.log('Auth state changed:', event)

        if (session?.user) {
          setUser(session.user)
          await loadProfile(session.user.id)
        } else {
          setUser(null)
          setProfile(null)
        }

        setLoading(false)
      }
    )

    return () => {
      subscription.unsubscribe()
    }
  }, [])

  // ============================================================================
  // LOAD USER PROFILE
  // ============================================================================

  const loadProfile = async (userId: string) => {
    try {
      const profileData = await getUserProfile(userId)
      setProfile(profileData as UserProfile)
    } catch (error) {
      console.error('Error loading profile:', error)
      setProfile(null)
    }
  }

  // ============================================================================
  // REFRESH PROFILE
  // ============================================================================

  const refreshProfile = async () => {
    if (user) {
      await loadProfile(user.id)
    }
  }

  // ============================================================================
  // SIGN UP
  // ============================================================================

  const signUp = async (email: string, password: string, fullName: string) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            full_name: fullName,
          },
          emailRedirectTo: `${window.location.origin}/auth/callback`,
        },
      })

      if (error) throw error

      if (data.user) {
        // Create profile in database
        const { error: profileError } = await supabase
          .from('user_profiles')
          .insert({
            id: data.user.id,
            email: data.user.email,
            full_name: fullName,
          })

        if (profileError) {
          console.error('Error creating profile:', profileError)
        }
      }
    } catch (error: any) {
      throw new Error(error.message || 'Failed to sign up')
    }
  }

  // ============================================================================
  // SIGN IN
  // ============================================================================

  const signIn = async (email: string, password: string) => {
    try {
      // DEV MODE: Auto login with mock user
      if (isDevMode) {
        setUser(MOCK_USER)
        setProfile(MOCK_PROFILE)
        router.push('/dashboard')
        return
      }

      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })

      if (error) throw error

      // Redirect to dashboard after successful login
      router.push('/dashboard')
    } catch (error: any) {
      throw new Error(error.message || 'Failed to sign in')
    }
  }

  // ============================================================================
  // SIGN IN WITH GOOGLE
  // ============================================================================

  const signInWithGoogle = async () => {
    try {
      // DEV MODE: Auto login with mock user
      if (isDevMode) {
        setUser(MOCK_USER)
        setProfile(MOCK_PROFILE)
        router.push('/dashboard')
        return
      }

      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/auth/callback`,
        },
      })

      if (error) throw error
    } catch (error: any) {
      throw new Error(error.message || 'Failed to sign in with Google')
    }
  }

  // ============================================================================
  // SIGN OUT
  // ============================================================================

  const signOut = async () => {
    try {
      // DEV MODE: Just clear local state
      if (isDevMode) {
        setUser(null)
        setProfile(null)
        router.push('/')
        return
      }

      const { error } = await supabase.auth.signOut()
      if (error) throw error

      setUser(null)
      setProfile(null)

      // Redirect to landing page
      router.push('/')
    } catch (error: any) {
      throw new Error(error.message || 'Failed to sign out')
    }
  }

  // ============================================================================
  // CONTEXT VALUE
  // ============================================================================

  const value: AuthContextType = {
    user,
    profile,
    loading,
    signUp,
    signIn,
    signInWithGoogle,
    signOut,
    refreshProfile,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// ============================================================================
// HOOK
// ============================================================================

export function useAuth() {
  const context = useContext(AuthContext)

  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }

  return context
}

export default AuthContext
