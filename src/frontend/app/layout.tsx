import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Providers } from './providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'SwingAI - AI-Powered Swing Trading for Indian Markets',
  description: 'Automated trading signals and execution for NSE/BSE. Connect your broker and let AI handle the rest. 58-62% accuracy with 3-model ensemble AI.',
  keywords: ['swing trading', 'AI trading', 'NSE', 'BSE', 'stock market', 'automated trading', 'India'],
  authors: [{ name: 'SwingAI' }],
  openGraph: {
    title: 'SwingAI - AI-Powered Swing Trading',
    description: 'Automated trading signals and execution for Indian markets',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-black text-white antialiased`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
