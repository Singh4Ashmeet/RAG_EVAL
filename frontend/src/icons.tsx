import type { SVGProps } from 'react'

export function BrandLogo(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 32 32" width="28" height="28" fill="none" aria-hidden="true" {...props}>
      <rect width="32" height="32" rx="7" fill="#18271E" />
      <rect x="1" y="1" width="30" height="30" rx="6" stroke="#2F5D46" strokeWidth="1.5" />
      <path d="M 8 11 L 16 7 L 24 11 L 16 15 Z" fill="#2F5D46" stroke="#5CD692" strokeWidth="1.2" strokeLinejoin="round" />
      <path d="M 8 16 L 16 20 L 24 16" fill="none" stroke="#5CD692" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M 8 21 L 16 25 L 24 21" fill="none" stroke="#5CD692" strokeWidth="1.5" strokeLinejoin="round" />
      <circle cx="16" cy="11" r="1.5" fill="#E2B155" />
    </svg>
  )
}

export function GoogleIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" {...props}>
      <path
        fill="#EA4335"
        d="M12 5c1.6 0 3 .6 4.1 1.7l3.1-3.1C17.3 1.8 14.8 1 12 1 7.5 1 3.7 3.6 1.9 7.3l3.7 2.9C6.5 7.4 9 5 12 5z"
      />
      <path
        fill="#4285F4"
        d="M23.5 12.3c0-.8-.1-1.7-.2-2.3H12v4.6h6.5c-.3 1.5-1.1 2.8-2.4 3.7l3.7 2.9c2.2-2 3.7-5 3.7-8.9z"
      />
      <path
        fill="#FBBC05"
        d="M5.6 14.8c-.3-.8-.4-1.8-.4-2.8s.1-1.9.4-2.8L1.9 6.3C.7 8.7 0 10.3 0 12s.7 3.3 1.9 5.7l3.7-2.9z"
      />
      <path
        fill="#34A853"
        d="M12 23c3.2 0 6-1.1 8-3l-3.7-2.9c-1.1.7-2.5 1.2-4.3 1.2-3 0-5.5-2.4-6.4-5.2L1.9 16c1.8 3.7 5.6 7 10.1 7z"
      />
    </svg>
  )
}

export function GitHubIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true" {...props}>
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
      />
    </svg>
  )
}

export function DocumentIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true" {...props}>
      <path d="M4 4a2 2 0 012-2h5l5 5v9a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" />
      <path d="M11 2v5h5" />
      <path d="M7 11h6M7 14h4" />
    </svg>
  )
}

export function AskIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true" {...props}>
      <circle cx="9" cy="9" r="6" />
      <path d="M13.5 13.5L17.5 17.5" strokeLinecap="round" />
      <path d="M7 8.5a2 2 0 013.5-1.2c.5.6.5 1.2 0 1.7-.5.5-1 1-1 1.8M9.5 13h.01" strokeLinecap="round" />
    </svg>
  )
}

export function DatasetIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true" {...props}>
      <ellipse cx="10" cy="5" rx="7" ry="2.5" />
      <path d="M3 5v5c0 1.38 3.13 2.5 7 2.5s7-1.12 7-2.5V5" />
      <path d="M3 10v5c0 1.38 3.13 2.5 7 2.5s7-1.12 7-2.5v-5" />
    </svg>
  )
}

export function RunsIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true" {...props}>
      <path d="M3 10h3l2-5 3 10 2-5h4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function ExperimentIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true" {...props}>
      <path d="M7 3h6M9 3v4.5L4.5 15A2 2 0 006.3 18h7.4a2 2 0 001.8-3L11 7.5V3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M6 14h8" />
    </svg>
  )
}

export function SettingsIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true" {...props}>
      <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
      <path d="M16.2 12a1.2 1.2 0 00.2 1.3l.1.1a1.5 1.5 0 010 2.1 1.5 1.5 0 01-2.1 0l-.1-.1a1.2 1.2 0 00-1.3-.2 1.2 1.2 0 00-.8 1.1v.2a1.5 1.5 0 01-1.5 1.5 1.5 1.5 0 01-1.5-1.5v-.2a1.2 1.2 0 00-.8-1.1 1.2 1.2 0 00-1.3.2l-.1.1a1.5 1.5 0 01-2.1 0 1.5 1.5 0 010-2.1l.1-.1a1.2 1.2 0 00.2-1.3 1.2 1.2 0 00-1.1-.8h-.2a1.5 1.5 0 01-1.5-1.5 1.5 1.5 0 011.5-1.5h.2a1.2 1.2 0 001.1-.8 1.2 1.2 0 00-.2-1.3l-.1-.1a1.5 1.5 0 010-2.1 1.5 1.5 0 012.1 0l.1.1a1.2 1.2 0 001.3.2h.1a1.2 1.2 0 00.7-1.1v-.2a1.5 1.5 0 011.5-1.5 1.5 1.5 0 011.5 1.5v.2a1.2 1.2 0 00.8 1.1 1.2 1.2 0 001.3-.2l.1-.1a1.5 1.5 0 012.1 0 1.5 1.5 0 010 2.1l-.1.1a1.2 1.2 0 00-.2 1.3v.1a1.2 1.2 0 001.1.7h.2a1.5 1.5 0 011.5 1.5 1.5 1.5 0 01-1.5 1.5h-.2a1.2 1.2 0 00-1.1.8z" />
    </svg>
  )
}

export function SignOutIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true" {...props}>
      <path d="M7 17H4a1 1 0 01-1-1V4a1 1 0 011-1h3M13 14l4-4-4-4M17 10H7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function UploadCloudIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true" {...props}>
      <path d="M7 16a4 4 0 01-.88-7.9A5 5 0 0115.9 6 4 4 0 0119 12a3 3 0 01-.5 5.95" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 12v9m0-9l-3 3m3-3l3 3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function CheckCircleIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true" {...props}>
      <circle cx="10" cy="10" r="8" />
      <path d="M6.5 10l2.5 2.5 5-5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function ShieldCheckIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true" {...props}>
      <path d="M10 2L4 4.5v5.5c0 4 3 6.8 6 7.5 3-.7 6-3.5 6-7.5V4.5L10 2z" strokeLinejoin="round" />
      <path d="M7.5 10l2 2 3.5-3.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function SparklesIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true" {...props}>
      <path d="M10 2l1.8 4.2L16 8l-4.2 1.8L10 14l-1.8-4.2L4 8l4.2-1.8L10 2zM15 14l.9 2.1L18 17l-2.1.9L15 20l-.9-2.1L12 17l2.1-.9L15 14z" strokeLinejoin="round" />
    </svg>
  )
}

