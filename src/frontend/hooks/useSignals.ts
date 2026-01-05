// ============================================================================
// SWINGAI - SIGNALS HOOK
// Fetch and manage trading signals
// ============================================================================

'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, handleApiError } from '../lib/api'
import { Signal, SignalFilters } from '../types'
import { toast } from 'sonner'

// ============================================================================
// GET ALL SIGNALS
// ============================================================================

export function useSignals(filters?: SignalFilters) {
  return useQuery({
    queryKey: ['signals', filters],
    queryFn: () => api.signals.getAll(filters),
    staleTime: 30000, // 30 seconds
    refetchInterval: 60000, // Refetch every 60 seconds
  })
}

// ============================================================================
// GET SIGNAL BY ID
// ============================================================================

export function useSignal(id: string) {
  return useQuery({
    queryKey: ['signal', id],
    queryFn: () => api.signals.getById(id),
    enabled: !!id,
  })
}

// ============================================================================
// EXECUTE SIGNAL
// ============================================================================

export function useExecuteSignal() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ signalId, data }: { signalId: string; data: any }) =>
      api.signals.execute(signalId, data),
    onSuccess: () => {
      toast.success('Signal executed successfully!')
      queryClient.invalidateQueries({ queryKey: ['signals'] })
      queryClient.invalidateQueries({ queryKey: ['positions'] })
    },
    onError: (error) => {
      const message = handleApiError(error)
      toast.error(`Failed to execute signal: ${message}`)
    },
  })
}

// ============================================================================
// GET SIGNAL HISTORY
// ============================================================================

export function useSignalHistory(filters?: any) {
  return useQuery({
    queryKey: ['signal-history', filters],
    queryFn: () => api.signals.getHistory(filters),
    staleTime: 60000, // 1 minute
  })
}
