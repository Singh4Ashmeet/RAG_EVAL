import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import App from './App'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('RAG_EVAL workbench', () => {
  it('shows invite-only sign in when no session is present', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Unauthenticated')))
    render(<App />)
    expect(await screen.findByRole('link', { name: 'Continue with GitHub' })).toHaveAttribute(
      'href',
      '/api/v1/auth/github',
    )
    expect(screen.getByRole('link', { name: 'Terms' })).toHaveAttribute('href', '/terms.html')
  })

  it('opens the corpus first and navigates to the empty datasets state', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/auth/me')) return Response.json({ id: 'workspace', login: 'ashmeet' })
        if (url.endsWith('/documents') || url.endsWith('/datasets')) return Response.json([])
        return Response.json({ detail: 'Not found' }, { status: 404 })
      }),
    )
    render(<App />)
    expect(await screen.findByRole('heading', { name: 'Source corpus' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Datasets' }))
    expect(await screen.findByRole('heading', { name: 'No evaluation datasets' })).toBeInTheDocument()
  })
})
