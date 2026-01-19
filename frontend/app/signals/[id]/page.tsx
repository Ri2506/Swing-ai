'use client'

import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

export default function SignalDetailPage() {
  return (
    <div className="min-h-screen bg-background-primary px-6 py-8">
      <div className="mx-auto max-w-4xl">
        <Link
          href="/signals"
          className="mb-8 inline-flex items-center gap-2 text-text-secondary transition hover:text-text-primary"
        >
          <ArrowLeft className="h-5 w-5" />
          Back to Signals
        </Link>
        <h1 className="text-3xl font-bold text-text-primary">Signal Details</h1>
        <p className="mt-4 text-text-secondary">
          This page will show detailed signal information with charts and analysis.
        </p>
      </div>
    </div>
  )
}
