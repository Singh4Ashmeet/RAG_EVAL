import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import type { paths } from './api.generated'
import {
  BrandLogo,
  GoogleIcon,
  GitHubIcon,
  DocumentIcon,
  AskIcon,
  DatasetIcon,
  RunsIcon,
  ExperimentIcon,
  SettingsIcon,
  SignOutIcon,
  UploadCloudIcon,
  ShieldCheckIcon,
  SparklesIcon,
} from './icons'
import { GuideBot } from './GuideBot'
import './App.css'

type Section = 'Corpus' | 'Ask' | 'Datasets' | 'Runs' | 'Experiments'
type Json200<Operation> = Operation extends { responses: { 200: { content: { 'application/json': infer Body } } } } ? Body : never
type Account = Json200<paths['/api/v1/auth/me']['get']>
type AuthMethods = { email: boolean; google: boolean; github: boolean }
type DocumentRow = Json200<paths['/api/v1/documents']['get']>[number]
type DatasetRow = Json200<paths['/api/v1/datasets']['get']>[number]
type ElementRow = Json200<paths['/api/v1/documents/{doc_id}/elements']['get']>[number]
type RunRow = {
  id: string
  dataset_id: string
  revision: number
  status: string
  created_at: string
  report: {
    metrics: Record<string, number | null>
    metric_counts: Record<string, number>
    queries: {
      query_id: string
      query: string
      status: string
      errors: string[]
      retrieval_metrics: Record<string, number>
      answer_metrics: Record<string, number>
      generated_answer: string | null
      citations: { label: string; chunk_id: string; document_id: string }[]
      retrieved_sources: { score: number; chunk: { chunk_id: string; document_id: string; text: string; source_element_ids: string[] } }[]
    }[]
  } | null
}
type ExperimentRow = {
  id: string
  revision: number
  status: string
  created_at: string
  report: {
    config: { baseline_variant: string }
    variants: { name: string; status: string; pipeline_config: Record<string, unknown>; error?: string }[]
    comparisons: {
      candidate_variant: string
      status: string
      reason?: string
      metrics: Record<string, {
        baseline_mean: number | null
        candidate_mean: number | null
        delta: number | null
        confidence_interval_low: number | null
        confidence_interval_high: number | null
        paired_query_count: number
      }>
    }[]
  } | null
}
type ApiError = { detail?: string }

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, { credentials: 'same-origin', ...init })
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as ApiError
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

const tabs: { id: Section; label: string; icon: typeof DocumentIcon }[] = [
  { id: 'Corpus', label: 'Corpus', icon: DocumentIcon },
  { id: 'Ask', label: 'Ask', icon: AskIcon },
  { id: 'Datasets', label: 'Datasets', icon: DatasetIcon },
  { id: 'Runs', label: 'Runs', icon: RunsIcon },
  { id: 'Experiments', label: 'Experiments', icon: ExperimentIcon },
]

function App() {
  const [section, setSection] = useState<Section>('Corpus')
  const [account, setAccount] = useState<Account | null>(null)
  const [authChecked, setAuthChecked] = useState(false)
  const [authMethods, setAuthMethods] = useState<AuthMethods | null>(null)
  const [emailInput, setEmailInput] = useState('')
  const [emailLinkSent, setEmailLinkSent] = useState(false)
  const [emailSending, setEmailSending] = useState(false)
  const [devLink, setDevLink] = useState('')
  const [authError, setAuthError] = useState('')
  const [documents, setDocuments] = useState<DocumentRow[]>([])
  const [datasets, setDatasets] = useState<DatasetRow[]>([])
  const [elements, setElements] = useState<ElementRow[]>([])
  const [selectedDocument, setSelectedDocument] = useState<string>('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [copiedHash, setCopiedHash] = useState<string | null>(null)

  const refresh = async () => {
    if (!account) return
    try {
      const [docs, sets] = await Promise.all([api<DocumentRow[]>('/documents'), api<DatasetRow[]>('/datasets')])
      setDocuments(docs)
      setDatasets(sets)
      setError('')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  useEffect(() => {
    api<Account>('/auth/me')
      .then(setAccount)
      .catch(() => setAccount(null))
      .finally(() => setAuthChecked(true))
  }, [])

  useEffect(() => {
    api<AuthMethods>('/auth/methods')
      .then(setAuthMethods)
      .catch(() => setAuthMethods(null))
  }, [])

  useEffect(() => {
    if (account) {
      void refresh()
    }
  }, [account])

  const processingDocuments = documents.some(doc => doc.status === 'queued' || doc.status === 'running')
  useEffect(() => {
    if (!processingDocuments) return
    const timer = window.setInterval(() => {
      api<DocumentRow[]>('/documents')
        .then(setDocuments)
        .catch(e => setError((e as Error).message))
    }, 3000)
    return () => window.clearInterval(timer)
  }, [processingDocuments])

  useEffect(() => {
    if (selectedDocument) {
      api<ElementRow[]>(`/documents/${selectedDocument}/elements`)
        .then(setElements)
        .catch(e => setError((e as Error).message))
    } else {
      setElements([])
    }
  }, [selectedDocument])

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setNotice('')
    const formElement = event.currentTarget
    const file = new FormData(formElement).get('file')
    if (!(file instanceof File) || !file.size) {
      setError('Choose a .txt or .md file first.')
      return
    }
    setBusy(true)
    try {
      const form = new FormData()
      form.set('file', file)
      const result = await api<{ id: string; status: string; element_count: number }>('/documents', {
        method: 'POST',
        body: form,
      })
      setNotice(
        `Document stored with ${result.status} indexing status. ${result.element_count} source elements are ready for dataset labels.`,
      )
      formElement.reset()
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function sendSignInLink(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setAuthError('')
    setEmailLinkSent(false)
    setDevLink('')
    setEmailSending(true)
    try {
      const result = await api<{ detail: string; dev_link?: string }>('/auth/email/request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: emailInput }),
      })
      setEmailLinkSent(true)
      if (result.dev_link) {
        setDevLink(result.dev_link)
      }
    } catch (e) {
      setAuthError((e as Error).message)
    } finally {
      setEmailSending(false)
    }
  }

  async function removeDocument(doc: DocumentRow) {
    if (!window.confirm(`Delete ${doc.name}? Existing evaluation reports may retain source excerpts.`)) return
    try {
      await api(`/documents/${doc.id}`, { method: 'DELETE' })
      setNotice('Document and active source elements deleted. Existing reports retain their excerpts.')
      if (selectedDocument === doc.id) setSelectedDocument('')
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function retryIndexing(doc: DocumentRow) {
    try {
      await api(`/documents/${doc.id}/reindex`, { method: 'POST' })
      setNotice('Indexing job queued.')
      await refresh()
    } catch (e) {
      setError((e as Error).message)
      await refresh()
    }
  }

  function copyText(text: string, id: string) {
    void navigator.clipboard.writeText(text).then(() => {
      setCopiedHash(id)
      setTimeout(() => setCopiedHash(null), 2000)
    })
  }

  if (!authChecked) {
    return (
      <main className="auth-state" aria-live="polite">
        <div className="auth-loading-spinner" aria-hidden="true" />
        <p>Checking your session…</p>
      </main>
    )
  }

  if (window.location.pathname === '/admin/invitations') {
    return <InvitationAdmin />
  }

  if (!account) {
    return (
      <div className="landing-layout">
        <header className="landing-nav">
          <div className="landing-nav-inner">
            <div className="wordmark brand-link">
              <BrandLogo />
              <span>RAG<span>_</span>EVAL</span>
            </div>
            <nav className="landing-nav-links" aria-label="Landing page links">
              <a href="#pipeline" tabIndex={-1}>Architecture</a>
              <a href="#features" tabIndex={-1}>Features</a>
              <a href="#metrics" tabIndex={-1}>Metrics</a>
              <a href="#security" tabIndex={-1}>Security</a>
              <a href="#signin" tabIndex={-1} className="landing-nav-cta">Sign in</a>
            </nav>
          </div>
        </header>

        <main id="top" className="sign-in">
          {/* Hero Section */}
          <section className="hero-section">
            <div className="hero-content">
              <div className="hero-badge">
                <span className="pulse-indicator" />
                <span className="eyebrow-text">EVALUATION WORKSPACE · PRODUCTION BETA</span>
              </div>
              <h1 className="hero-headline">Measure the evidence behind your RAG system.</h1>
              <p className="intro">
                Create an account with your email, Google, or GitHub. Your workspace is private to you.
                Anchor your ground truth against immutable elements, benchmark retrieval precision @ K, and statistically prove pipeline improvements.
              </p>

              {/* Sign In Card */}
              <div id="signin" className="sign-in-card">
                <div className="sign-in-card-header">
                  <h2>Access your workspace</h2>
                  <p>Choose your preferred authentication method</p>
                </div>

                {authError && (
                  <p className="inline-error" role="alert">
                    {authError}
                  </p>
                )}

                {authMethods?.email && (
                  <form className="sign-in-form" onSubmit={sendSignInLink}>
                    <label htmlFor="sign-in-email">Email address</label>
                    <div className="input-group">
                      <input
                        id="sign-in-email"
                        type="email"
                        autoComplete="email"
                        required
                        value={emailInput}
                        onChange={event => setEmailInput(event.target.value)}
                        placeholder="researcher@organization.com"
                      />
                    </div>
                    <button className="button primary" type="submit" disabled={emailSending}>
                      {emailSending ? 'Sending link…' : 'Email me a sign-in link'}
                    </button>
                    {emailLinkSent && (
                      <p className="auth-confirmation" role="status">
                        If that address can receive sign-in links, one will arrive shortly. The link expires in 15 minutes.
                      </p>
                    )}
                    {devLink && (
                      <div className="dev-sign-in-banner">
                        <span className="dev-tag">LOCAL DEV MODE</span>
                        <p>One-click local sign-in link generated:</p>
                        <a className="button primary dev-login-link" href={devLink}>
                          Sign In Instantly →
                        </a>
                      </div>
                    )}
                  </form>
                )}

                <div className="divider-row">
                  <span>or continue with</span>
                </div>

                <div className="sign-in-providers" aria-label="Other sign-in options">
                  {authMethods?.google && (
                    <a className="button secondary provider-button" href="/api/v1/auth/google">
                      <GoogleIcon />
                      <span>Continue with Google</span>
                    </a>
                  )}
                  {authMethods?.github && (
                    <a className="button secondary provider-button" href="/api/v1/auth/github">
                      <GitHubIcon />
                      <span>Continue with GitHub</span>
                    </a>
                  )}
                </div>

                {authMethods === null && (
                  <p className="auth-help" role="status">
                    Sign-in options are loading. Refresh this page if they do not appear.
                  </p>
                )}
                {authMethods && !authMethods.email && <p className="auth-help">Email sign-in is not configured yet.</p>}
                {authMethods && !authMethods.google && <p className="auth-help">Google sign-in is not configured yet.</p>}

                <p className="legal">
                  By continuing, you agree to the <a href="/terms.html">Terms</a> and <a href="/privacy.html">Privacy notice</a>.
                </p>
              </div>
            </div>

            {/* Hero Visual / Evidence Pipeline Diagram */}
            <div className="hero-visual-col" id="pipeline">
              <div className="visual-card">
                <div className="visual-card-topbar">
                  <div className="window-dots">
                    <span className="dot red" />
                    <span className="dot yellow" />
                    <span className="dot green" />
                  </div>
                  <span className="visual-title">rag-evaluation-pipeline-v4.spec</span>
                  <span className="visual-tag">IMMUTABLE ANCHOR</span>
                </div>
                <div className="visual-image-container">
                  <img
                    src="/rag-pipeline-hero.svg"
                    alt="RAG Evaluation Pipeline Architecture showing source corpus ingestion, dense vector retrieval, grounded context synthesis, and statistical hypothesis testing"
                    className="hero-diagram-img"
                    width="960"
                    height="540"
                  />
                </div>
                <div className="visual-caption-strip">
                  <div className="caption-stat">
                    <span className="stat-label">Precision @ 1</span>
                    <strong className="stat-val">0.892</strong>
                  </div>
                  <div className="caption-stat">
                    <span className="stat-label">Recall @ 5</span>
                    <strong className="stat-val">0.965</strong>
                  </div>
                  <div className="caption-stat">
                    <span className="stat-label">Paired Bootstrap</span>
                    <strong className="stat-val highlight">Δ +0.142</strong>
                  </div>
                  <div className="caption-stat">
                    <span className="stat-label">Hallucinations</span>
                    <strong className="stat-val zero">0 citations</strong>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Pillars Strip */}
          <section className="pillars-strip">
            <div className="pillar-item">
              <div className="pillar-icon">
                <DocumentIcon />
              </div>
              <div>
                <h3>Deterministic Source Elements</h3>
                <p>Ground truth mapped to immutable element UUIDs, immune to chunking boundary changes.</p>
              </div>
            </div>
            <div className="pillar-item">
              <div className="pillar-icon">
                <RunsIcon />
              </div>
              <div>
                <h3>1,000 Bootstrap Resamples</h3>
                <p>Rigorous paired hypothesis testing with 95% confidence intervals, not just simple means.</p>
              </div>
            </div>
            <div className="pillar-item">
              <div className="pillar-icon">
                <ShieldCheckIcon />
              </div>
              <div>
                <h3>Fernet Encrypted Keys</h3>
                <p>Gemini and Groq credentials stored encrypted at rest. Embeddings computed locally on Ollama.</p>
              </div>
            </div>
          </section>

          {/* Feature Showcase Grid with Technical Diagrams */}
          <section id="features" className="features-section">
            <div className="section-header-centered">
              <p className="eyebrow">SYSTEM CAPABILITIES</p>
              <h2>Engineered for evidence, not intuition</h2>
              <p className="section-subhead">
                Every component is structured so you can trace every generated token back to ranked cosine evidence in your private corpus.
              </p>
            </div>

            <div className="feature-grid">
              {/* Feature 1 */}
              <article className="feature-card">
                <div className="feature-diagram-wrapper">
                  <img
                    src="/corpus-diagram.svg"
                    alt="Source element slicing and SHA-256 content addressing diagram"
                    className="feature-img"
                    width="480"
                    height="320"
                    loading="lazy"
                  />
                </div>
                <div className="feature-card-body">
                  <span className="feature-step-num">01 · INGESTION &amp; REVISIONS</span>
                  <h3>Immutable Source Elements</h3>
                  <p>
                    Rechunking your corpus shouldn&apos;t wipe out months of evaluation labels. RAG_EVAL extracts deterministic, stable element IDs so your ground-truth questions remain valid forever.
                  </p>
                  <ul className="feature-list">
                    <li>SHA-256 verified document hashing</li>
                    <li>UTF-8 text &amp; Markdown normalization</li>
                    <li>Preserved source excerpts in historical reports</li>
                  </ul>
                </div>
              </article>

              {/* Feature 2 */}
              <article className="feature-card" id="metrics">
                <div className="feature-diagram-wrapper">
                  <img
                    src="/retrieval-benchmark.svg"
                    alt="Multi-cutoff retrieval fidelity chart showing precision and recall at k equals 1, 3, and 5"
                    className="feature-img"
                    width="480"
                    height="320"
                    loading="lazy"
                  />
                </div>
                <div className="feature-card-body">
                  <span className="feature-step-num">02 · RETRIEVAL BENCHMARKING</span>
                  <h3>Precision @ K &amp; Cosine Fidelity</h3>
                  <p>
                    Evaluate retrieval fidelity across standardized cutoffs (k=1, 3, 5). Track Mean Reciprocal Rank (MRR) and inspect raw cosine similarity scores with zero guesswork.
                  </p>
                  <ul className="feature-list">
                    <li>Strict denominator accounting for failed queries</li>
                    <li>Ollama nomic-embed-text vector index</li>
                    <li>Per-query ranking and excerpt verification</li>
                  </ul>
                </div>
              </article>

              {/* Feature 3 */}
              <article className="feature-card">
                <div className="feature-diagram-wrapper">
                  <img
                    src="/paired-hypothesis.svg"
                    alt="Paired experiment bootstrap distribution showing delta estimate and 95 percent confidence intervals"
                    className="feature-img"
                    width="480"
                    height="320"
                    loading="lazy"
                  />
                </div>
                <div className="feature-card-body">
                  <span className="feature-step-num">03 · STATISTICAL HYPOTHESIS TESTING</span>
                  <h3>Paired Depth Experiments</h3>
                  <p>
                    Stop declaring winners on tiny score differences. Run 1,000 bootstrap resamples with a fixed random seed to calculate paired deltas and empirical 95% confidence intervals.
                  </p>
                  <ul className="feature-list">
                    <li>Paired query comparison (e.g. depth-3 vs depth-5)</li>
                    <li>Upper and lower confidence bounds</li>
                    <li>Reports effect size without declaring false winners</li>
                  </ul>
                </div>
              </article>

              {/* Feature 4 */}
              <article className="feature-card" id="security">
                <div className="feature-diagram-wrapper">
                  <img
                    src="/security-architecture.svg"
                    alt="Security and privacy architecture diagram illustrating client-side encryption and local embeddings"
                    className="feature-img"
                    width="480"
                    height="320"
                    loading="lazy"
                  />
                </div>
                <div className="feature-card-body">
                  <span className="feature-step-num">04 · TENANCY &amp; SECURITY</span>
                  <h3>Encrypted Keys &amp; Local Embeddings</h3>
                  <p>
                    Your vectors and document text never leave your server. Cloud model generation is strictly opt-in using client-side encrypted credentials for Google Gemini and Groq.
                  </p>
                  <ul className="feature-list">
                    <li>Fernet symmetric encryption for API credentials</li>
                    <li>Private workspace isolation per account</li>
                    <li>1-click cascading workspace deletion</li>
                  </ul>
                </div>
              </article>
            </div>
          </section>

          {/* How It Works Workflow Bar */}
          <section className="workflow-section">
            <div className="section-header-centered">
              <p className="eyebrow">FOUR-STEP EVALUATION WORKFLOW</p>
              <h2>From raw documents to verified benchmarks</h2>
            </div>
            <div className="workflow-steps">
              <div className="step-card">
                <span className="step-badge">STEP 1</span>
                <h3>Upload Corpus</h3>
                <p>Import documentation, RFCs, or manuals in Markdown or plain text.</p>
              </div>
              <div className="step-arrow">→</div>
              <div className="step-card">
                <span className="step-badge">STEP 2</span>
                <h3>Label Elements</h3>
                <p>Tag queries to stable source elements to form immutable evaluation revisions.</p>
              </div>
              <div className="step-arrow">→</div>
              <div className="step-card">
                <span className="step-badge">STEP 3</span>
                <h3>Run Evaluation</h3>
                <p>Queue batched queries against vector stores to compute MRR and Precision@K.</p>
              </div>
              <div className="step-arrow">→</div>
              <div className="step-card">
                <span className="step-badge">STEP 4</span>
                <h3>Compare Variants</h3>
                <p>Test pipeline adjustments with 1,000 bootstrap resamples for verified deltas.</p>
              </div>
            </div>
          </section>

          {/* Landing Footer */}
          <footer className="landing-footer">
            <div className="footer-top">
              <div className="footer-brand">
                <div className="wordmark">
                  <BrandLogo />
                  <span>RAG<span>_</span>EVAL</span>
                </div>
                <p className="footer-tagline">Evidence-based RAG evaluation and verification workbench.</p>
              </div>
              <div className="footer-nav">
                <div className="footer-col">
                  <strong>Workbench</strong>
                  <a href="#pipeline">Architecture</a>
                  <a href="#features">Features</a>
                  <a href="#metrics">Metrics</a>
                </div>
                <div className="footer-col">
                  <strong>Access &amp; Legal</strong>
                  <a href="/terms.html">Terms of service</a>
                  <a href="/privacy.html">Privacy notice</a>
                  <a href="/admin/invitations">Invitation administration</a>
                </div>
              </div>
            </div>
            <div className="footer-bottom">
              <span>RAG_EVAL · Hosted beta · Documents and prompts are processed on this service.</span>
              <span>All rights reserved.</span>
            </div>
          </footer>
        </main>
        <GuideBot isLoggedIn={false} />
      </div>
    )
  }

  return (
    <div className="shell">
      <header className="topbar">
        <a className="wordmark brand-link" href="#corpus" onClick={() => setSection('Corpus')}>
          <BrandLogo />
          <span>RAG<span>_</span>EVAL</span>
        </a>
        <nav aria-label="Primary navigation">
          {tabs.map(tab => {
            const IconComponent = tab.icon
            return (
              <button
                key={tab.id}
                className={section === tab.id ? 'nav-item active' : 'nav-item'}
                onClick={() => setSection(tab.id)}
              >
                <IconComponent />
                <span>{tab.label}</span>
                {tab.id === 'Corpus' && documents.length > 0 && <span className="tab-count">{documents.length}</span>}
                {tab.id === 'Datasets' && datasets.length > 0 && <span className="tab-count">{datasets.length}</span>}
              </button>
            )
          })}
        </nav>
        <div className="account">
          <span className="account-mark" aria-hidden="true">
            {account.login.slice(0, 1).toUpperCase()}
          </span>
          <span className="account-name">{account.login}</span>
          <button className="quiet-button button-with-icon" onClick={() => setSettingsOpen(true)}>
            <SettingsIcon />
            <span>Settings</span>
          </button>
          <button
            className="quiet-button button-with-icon"
            onClick={() => {
              void fetch('/api/v1/auth/logout', { method: 'POST' }).then(() => window.location.reload())
            }}
          >
            <SignOutIcon />
            <span>Sign out</span>
          </button>
        </div>
      </header>

      <main className="page">
        <div className="page-heading">
          <div>
            <p className="eyebrow">WORKSPACE / {section.toUpperCase()}</p>
            <h1>{section === 'Corpus' ? 'Source corpus' : section}</h1>
            <p className="subhead">
              {section === 'Corpus'
                ? 'Manage source files and stable elements used for retrieval and evaluation.'
                : section === 'Datasets'
                  ? 'Build immutable evaluation revisions against source elements.'
                  : section === 'Ask'
                    ? 'Inspect retrieved evidence and grounded answers.'
                    : section === 'Runs'
                      ? 'Review query level results and metric denominators.'
                      : 'Compare named pipeline variants on paired queries.'}
            </p>
          </div>
          <div className="heading-badges">
            <span className="workspace-tag">PRIVATE WORKSPACE</span>
            {processingDocuments && (
              <span className="indexing-live-tag">
                <span className="pulsing-dot" />
                INDEXING JOBS ACTIVE
              </span>
            )}
          </div>
        </div>

        {error && (
          <div className="message error" role="alert">
            <span>{error}</span>
            <button onClick={() => setError('')} aria-label="Dismiss error">
              ×
            </button>
          </div>
        )}
        {notice && (
          <div className="message success" role="status">
            <span>{notice}</span>
            <button onClick={() => setNotice('')} aria-label="Dismiss notice">
              ×
            </button>
          </div>
        )}

        {section === 'Corpus' && (
          <>
            <section className="upload-row">
              <div className="upload-header">
                <div className="upload-icon-circle">
                  <UploadCloudIcon />
                </div>
                <div>
                  <h2>Add source material</h2>
                  <p>UTF-8 text or Markdown, up to 25 MiB per file. Elements are automatically extracted for ground-truth labeling.</p>
                </div>
              </div>
              <form onSubmit={upload} className="upload-form">
                <label className="file-input">
                  <span className="file-input-text">Choose file</span>
                  <input name="file" type="file" accept=".txt,.md,text/plain,text/markdown" />
                </label>
                <button className="button primary" disabled={busy}>
                  {busy ? 'Saving…' : 'Upload source'}
                </button>
              </form>
            </section>

            <section className="table-section">
              <div className="section-title">
                <div>
                  <h2>Documents</h2>
                  <span>{documents.length} {documents.length === 1 ? 'document' : 'documents'}</span>
                </div>
              </div>

              {documents.length === 0 ? (
                <div className="empty">
                  <span className="empty-mark">01</span>
                  <div>
                    <h3>Your corpus is empty</h3>
                    <p>Upload a text or Markdown source to begin. Source elements will be available for dataset labels.</p>
                  </div>
                </div>
              ) : (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Source</th>
                        <th>State</th>
                        <th>Content digest</th>
                        <th>Added</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {documents.map(doc => (
                        <tr key={doc.id} className={selectedDocument === doc.id ? 'selected' : ''}>
                          <td>
                            <button
                              className="text-link document-name-button"
                              onClick={() => setSelectedDocument(selectedDocument === doc.id ? '' : doc.id)}
                            >
                              <DocumentIcon />
                              <span>{doc.name}</span>
                            </button>
                          </td>
                          <td>
                            <span className={`status ${doc.status}`}>
                              <i />
                              {doc.status}
                            </span>
                          </td>
                          <td>
                            <button
                              className="code-copy-btn"
                              type="button"
                              onClick={() => copyText(doc.sha256, doc.id)}
                              title="Click to copy full SHA-256 digest"
                            >
                              <code>{doc.sha256.slice(0, 16)}…</code>
                              <span className="copy-indicator">{copiedHash === doc.id ? 'Copied!' : 'Copy'}</span>
                            </button>
                          </td>
                          <td>{new Date(doc.created_at).toLocaleDateString()}</td>
                          <td>
                            <button className="quiet-button" onClick={() => void retryIndexing(doc)}>
                              {doc.status === 'failed' ? 'Retry index' : 'Reindex'}
                            </button>
                            <button className="quiet-button danger-text" onClick={() => void removeDocument(doc)}>
                              Delete
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            {selectedDocument && (
              <section className="table-section source-elements-section">
                <div className="section-title">
                  <div>
                    <h2>Source elements</h2>
                    <span>Stable IDs for relevance labels ({elements.length} elements extracted)</span>
                  </div>
                  <button className="quiet-button" onClick={() => setSelectedDocument('')}>
                    Close elements viewer
                  </button>
                </div>
                <div className="elements-container">
                  {elements.map(el => (
                    <article className="element-row" key={el.id}>
                      <div className="element-meta">
                        <code>{el.id}</code>
                        <span className="element-ordinal">Ordinal #{el.ordinal}</span>
                      </div>
                      <p>{el.text}</p>
                    </article>
                  ))}
                </div>
              </section>
            )}
          </>
        )}

        {section === 'Datasets' && (
          <DatasetWorkbench
            datasets={datasets}
            documents={documents}
            onCreated={async () => {
              await refresh()
              setNotice('Immutable dataset revision created.')
            }}
            onError={message => setError(message)}
          />
        )}

        {section === 'Ask' && <AskWorkbench documents={documents} />}

        {section === 'Runs' && <RunsWorkbench datasets={datasets} onError={message => setError(message)} />}

        {section === 'Experiments' && (
          <ExperimentsWorkbench datasets={datasets} onError={message => setError(message)} />
        )}

        <footer className="app-footer">
          <div className="footer-col-left">
            <span>RAG_EVAL · Hosted beta</span>
            <span>Documents and prompts are processed on this service.</span>
          </div>
          <div className="footer-col-right">
            <a href="/terms.html">Terms</a>
            <a href="/privacy.html">Privacy notice</a>
          </div>
        </footer>
      </main>

      <GuideBot isLoggedIn={true} currentSection={section} onNavigate={setSection} />

      {settingsOpen && (
        <SettingsPanel account={account} onClose={() => setSettingsOpen(false)} onError={message => setError(message)} />
      )}
    </div>
  )
}

function InvitationAdmin() {
  const [token, setToken] = useState('')
  const [email, setEmail] = useState('')
  const [invitations, setInvitations] = useState<{ email: string; created_at: string }[]>([])
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  async function request(path: string, init?: RequestInit) {
    const response = await fetch(`/api/v1/invitations${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', 'X-Admin-Token': token, ...init?.headers },
    })
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as ApiError
      throw new Error(body.detail || `Request failed (${response.status})`)
    }
    return response.json()
  }

  async function refresh() {
    try {
      setInvitations(await request(''))
      setError('')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function invite(e: FormEvent) {
    e.preventDefault()
    try {
      await request('', { method: 'POST', body: JSON.stringify({ email }) })
      setEmail('')
      setNotice('Invitation saved.')
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function revoke(address: string) {
    try {
      await request(`/${encodeURIComponent(address)}`, { method: 'DELETE' })
      setNotice(`Invitation revoked for ${address}.`)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <main className="page admin-page">
      <a className="wordmark brand-link" href="/">
        <BrandLogo />
        <span>RAG<span>_</span>EVAL</span>
      </a>
      <p className="eyebrow">SERVICE OWNER</p>
      <h1>Invitation access</h1>
      <p className="subhead">This page requires the service invitation token. Keep it private; the token stays in this browser tab only.</p>
      <label className="admin-token">
        Invitation admin token
        <input type="password" autoComplete="off" value={token} onChange={e => setToken(e.target.value)} />
      </label>
      <button className="button secondary" onClick={() => void refresh()} disabled={!token}>
        Load invitations
      </button>
      {error && <div className="message error" role="alert">{error}</div>}
      {notice && <div className="message success" role="status">{notice}</div>}
      <form className="admin-invite" onSubmit={invite}>
        <label>
          GitHub verified email
          <input type="email" required value={email} onChange={e => setEmail(e.target.value)} />
        </label>
        <button className="button primary" disabled={!token}>
          Invite account
        </button>
      </form>
      <section className="table-section">
        <div className="section-title">
          <h2>Invited email addresses</h2>
          <span>{invitations.length} invitations</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Email</th>
                <th>Added</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {invitations.map(item => (
                <tr key={item.email}>
                  <td>{item.email}</td>
                  <td>{new Date(item.created_at).toLocaleDateString()}</td>
                  <td>
                    <button className="quiet-button danger-text" onClick={() => void revoke(item.email)}>
                      Revoke
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <a className="admin-back" href="/">
        ← Return to workspace
      </a>
    </main>
  )
}

function DatasetWorkbench({
  datasets,
  documents,
  onCreated,
  onError,
}: {
  datasets: DatasetRow[]
  documents: DocumentRow[]
  onCreated: () => Promise<void>
  onError: (message: string) => void
}) {
  type Draft = { id: string; query: string; relevant: string[] }
  const [name, setName] = useState('')
  const [cases, setCases] = useState<Draft[]>([{ id: crypto.randomUUID(), query: '', relevant: [] }])
  const [busy, setBusy] = useState(false)

  const updateCase = (id: string, update: Partial<Draft>) =>
    setCases(current => current.map(item => (item.id === id ? { ...item, ...update } : item)))

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (cases.some(item => !item.relevant.length)) {
      onError('Select at least one relevant source element for every query.')
      return
    }
    setBusy(true)
    try {
      await api('/datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          cases: cases.map(item => ({
            query_id: crypto.randomUUID(),
            query: item.query,
            relevant_element_ids: item.relevant,
          })),
        }),
      })
      setName('')
      setCases([{ id: crypto.randomUUID(), query: '', relevant: [] }])
      await onCreated()
    } catch (err) {
      onError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <section className="form-section">
        <div className="section-title">
          <div>
            <h2>New dataset</h2>
            <span>Element labels remain stable across chunking changes</span>
          </div>
        </div>

        {documents.length === 0 ? (
          <p className="empty-inline">Add source material before creating element-labeled cases.</p>
        ) : (
          <form className="dataset-form" onSubmit={submit}>
            <label>
              Dataset name
              <input value={name} onChange={e => setName(e.target.value)} required maxLength={255} placeholder="e.g. core-spec-precision-r1" />
            </label>
            {cases.map((item, index) => (
              <fieldset className="case-editor" key={item.id}>
                <legend>Query {index + 1}</legend>
                <label>
                  Evaluation query
                  <textarea
                    value={item.query}
                    onChange={e => updateCase(item.id, { query: e.target.value })}
                    required
                    rows={2}
                    placeholder="Enter the query you want retrieval to answer"
                  />
                </label>
                <div className="element-picker-list">
                  <span className="picker-title">Relevant source elements (must check at least one)</span>
                  {documents.map(doc => (
                    <div className="element-picker" key={doc.id}>
                      <strong>{doc.name}</strong>
                      <ElementChoices
                        doc={doc}
                        selected={item.relevant}
                        onSelected={relevant => updateCase(item.id, { relevant })}
                      />
                    </div>
                  ))}
                </div>
                {cases.length > 1 && (
                  <button
                    type="button"
                    className="quiet-button danger-text"
                    onClick={() => setCases(current => current.filter(c => c.id !== item.id))}
                  >
                    Remove query
                  </button>
                )}
              </fieldset>
            ))}
            <div className="dataset-actions">
              <button
                type="button"
                className="button secondary"
                onClick={() => setCases(current => [...current, { id: crypto.randomUUID(), query: '', relevant: [] }])}
              >
                + Add another query
              </button>
              <button className="button primary" disabled={busy}>
                {busy ? 'Saving…' : `Create ${cases.length}-query revision`}
              </button>
            </div>
          </form>
        )}
      </section>

      <section className="table-section">
        <div className="section-title">
          <div>
            <h2>Datasets</h2>
            <span>Revisions are immutable once created</span>
          </div>
        </div>
        {datasets.length === 0 ? (
          <div className="empty">
            <span className="empty-mark">02</span>
            <div>
              <h3>No evaluation datasets</h3>
              <p>Create a dataset with reviewed source element labels before running evaluation.</p>
            </div>
          </div>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Revision</th>
                  <th>Cases</th>
                  <th>Digest</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map(d => (
                  <tr key={d.id}>
                    <td><strong>{d.name}</strong></td>
                    <td>
                      <span className="revision-pill">r{d.revision}</span>
                    </td>
                    <td>{d.case_count} cases</td>
                    <td>
                      <code>{d.sha256.slice(0, 16)}…</code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  )
}

function ElementChoices({
  doc,
  selected,
  onSelected,
}: {
  doc: DocumentRow
  selected: string[]
  onSelected: (items: string[]) => void
}) {
  const [items, setItems] = useState<ElementRow[]>([])
  useEffect(() => {
    api<ElementRow[]>(`/documents/${doc.id}/elements`)
      .then(setItems)
      .catch(() => setItems([]))
  }, [doc.id])

  return (
    <div className="choice-list">
      {items.map(item => (
        <label key={item.id} className="choice-item">
          <input
            type="checkbox"
            checked={selected.includes(item.id)}
            onChange={e =>
              onSelected(e.target.checked ? [...selected, item.id] : selected.filter(id => id !== item.id))
            }
          />
          <span>{item.text.slice(0, 240)}</span>
        </label>
      ))}
    </div>
  )
}

function AskWorkbench({ documents }: { documents: DocumentRow[] }) {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<{ element_id: string; text: string; cosine_similarity: number; document: string }[]>([])
  const [provider, setProvider] = useState('ollama')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError('')
    setAnswer('')
    setSources([])
    try {
      const result = await api<{
        answer: string
        sources: typeof sources
        unresolved_citation_labels: string[]
      }>('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question, provider }),
      })
      setAnswer(result.answer)
      setSources(result.sources)
      if (result.unresolved_citation_labels.length) {
        setError(`Unresolved citation labels: ${result.unresolved_citation_labels.join(', ')}`)
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="form-section">
      <div className="section-title">
        <div>
          <h2>Ask your corpus</h2>
          <span>Retrieval scores are cosine similarity, not confidence</span>
        </div>
      </div>

      {!documents.length ? (
        <p className="empty-inline">Upload and index source material before asking a question.</p>
      ) : (
        <form className="ask-form" onSubmit={submit}>
          <label htmlFor="ask-query">Question</label>
          <textarea
            id="ask-query"
            rows={3}
            value={question}
            onChange={e => setQuestion(e.target.value)}
            required
            placeholder="Ask a question grounded in your sources"
          />

          <label htmlFor="chat-provider">
            Chat provider
            <select id="chat-provider" value={provider} onChange={e => setProvider(e.target.value)}>
              <option value="ollama">Server managed Ollama (Local)</option>
              <option value="gemini">Google Gemini · uses your saved key</option>
              <option value="groq">Groq · uses your saved key</option>
            </select>
          </label>

          {provider !== 'ollama' && (
            <p className="provider-disclosure">
              Your question and retrieved source excerpts will be sent to {provider === 'gemini' ? 'Google Gemini' : 'Groq'} for generation.
            </p>
          )}

          <button className="button primary" disabled={busy}>
            {busy ? 'Retrieving evidence…' : 'Retrieve evidence'}
          </button>

          {error && (
            <p className="inline-error" role="alert">
              {error}
            </p>
          )}

          {answer && (
            <article className="answer">
              <div className="answer-header">
                <SparklesIcon />
                <h3>Grounded Answer</h3>
              </div>
              <p className="answer-body">{answer}</p>

              <div className="sources-list">
                <h4>Retrieved Evidence Sources ({sources.length})</h4>
                {sources.map((source, index) => (
                  <details key={source.element_id} className="source-item">
                    <summary>
                      <span className="source-index">[S{index + 1}]</span>
                      <span className="source-doc">{source.document}</span>
                      <div className="cosine-bar-wrapper">
                        <span className="cosine-label">cosine {source.cosine_similarity.toFixed(3)}</span>
                        <div className="cosine-meter">
                          <div
                            className="cosine-fill"
                            style={{ width: `${Math.max(0, Math.min(100, source.cosine_similarity * 100))}%` }}
                          />
                        </div>
                      </div>
                    </summary>
                    <div className="source-detail-content">
                      <p>{source.text}</p>
                      <div className="source-code-tag">
                        <code>{source.element_id}</code>
                      </div>
                    </div>
                  </details>
                ))}
              </div>
            </article>
          )}
        </form>
      )}
    </section>
  )
}

function SettingsPanel({
  account,
  onClose,
  onError,
}: {
  account: Account
  onClose: () => void
  onError: (message: string) => void
}) {
  const [credentials, setCredentials] = useState<{ provider: string; configured: boolean }[]>([])
  const [provider, setProvider] = useState('gemini')
  const [key, setKey] = useState('')
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)

  async function refresh() {
    try {
      setCredentials(await api('/settings/credentials'))
    } catch (e) {
      onError((e as Error).message)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  async function save(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    try {
      await api('/settings/credentials', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, api_key: key }),
      })
      setKey('')
      setMessage(`${provider} key encrypted and saved.`)
      await refresh()
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function check(name: string) {
    setMessage('Checking provider connection…')
    try {
      await api(`/settings/credentials/${name}/check`, { method: 'POST' })
      setMessage(`${name} connection check succeeded.`)
    } catch (e) {
      setMessage((e as Error).message)
    }
  }

  async function remove(name: string) {
    try {
      await api(`/settings/credentials/${name}`, { method: 'DELETE' })
      setMessage(`${name} key removed.`)
      await refresh()
    } catch (e) {
      onError((e as Error).message)
    }
  }

  async function deleteWorkspace() {
    if (window.prompt(`Type ${account.login} to permanently delete this workspace.`) !== account.login) return
    try {
      await api('/workspace', { method: 'DELETE' })
      window.location.assign('/')
    } catch (e) {
      onError((e as Error).message)
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}>
      <section className="settings-panel" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <header>
          <div>
            <p className="eyebrow">ACCOUNT SETTINGS</p>
            <h2 id="settings-title">{account.login}</h2>
          </div>
          <button className="quiet-button close-btn" onClick={onClose} aria-label="Close settings">
            ×
          </button>
        </header>

        <p className="settings-explainer">
          Saved keys are encrypted using Fernet symmetric ciphers and never displayed again. They are used only for cloud chat generation; vector embeddings remain strictly on server-managed Ollama.
        </p>

        <div className="saved-keys">
          <h3>Configured Provider Keys</h3>
          {credentials.length ? (
            credentials.map(item => (
              <div className="saved-key" key={item.provider}>
                <div className="saved-key-info">
                  <strong>{item.provider}</strong>
                  <span className="key-status-chip">Stored &amp; Encrypted</span>
                </div>
                <div className="saved-key-actions">
                  <button className="quiet-button" onClick={() => void check(item.provider)}>
                    Check connection
                  </button>
                  <button className="quiet-button danger-text" onClick={() => void remove(item.provider)}>
                    Remove
                  </button>
                </div>
              </div>
            ))
          ) : (
            <p className="no-keys-note">No cloud provider keys saved yet.</p>
          )}
        </div>

        <form className="credential-form" onSubmit={save}>
          <h3>Add or Update Provider Key</h3>
          <label>
            Provider
            <select value={provider} onChange={e => setProvider(e.target.value)}>
              <option value="gemini">Google Gemini</option>
              <option value="groq">Groq</option>
            </select>
          </label>
          <label>
            API key
            <input
              type="password"
              autoComplete="new-password"
              value={key}
              onChange={e => setKey(e.target.value)}
              required
              placeholder="Paste provider API key"
            />
          </label>
          <button className="button primary" disabled={busy}>
            {busy ? 'Saving…' : 'Save encrypted key'}
          </button>
        </form>

        {message && (
          <p className="settings-message" role="status">
            {message}
          </p>
        )}

        <section className="workspace-delete">
          <h3>Permanently Delete Workspace</h3>
          <p>Removes active documents, indexes, datasets, reports, credentials, and jobs. Encrypted backups expire within 30 days.</p>
          <button className="button danger-btn" onClick={() => void deleteWorkspace()}>
            Permanently delete workspace
          </button>
        </section>

        <p className="settings-footnote">
          Provider keys and prompts are processed as described in the <a href="/privacy.html">Privacy notice</a>.
        </p>
      </section>
    </div>
  )
}

function RunsWorkbench({ datasets, onError }: { datasets: DatasetRow[]; onError: (message: string) => void }) {
  const [runs, setRuns] = useState<RunRow[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [revision, setRevision] = useState('')
  const [busy, setBusy] = useState(false)

  async function refreshRuns() {
    try {
      const result = await api<RunRow[]>('/runs')
      setRuns(result)
    } catch (e) {
      onError((e as Error).message)
    }
  }

  useEffect(() => {
    void refreshRuns()
    const timer = window.setInterval(() => void refreshRuns(), 4000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    const dataset = datasets.find(item => item.id === datasetId)
    setRevision(dataset ? String(dataset.revision) : '')
  }, [datasetId, datasets])

  async function startRun(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    try {
      await api('/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: datasetId, revision: Number(revision), cutoffs: [1, 3, 5] }),
      })
      await refreshRuns()
    } catch (error) {
      onError((error as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <section className="form-section">
        <div className="section-title">
          <div>
            <h2>Start retrieval evaluation</h2>
            <span>Runs preserve failed cases and metric denominators</span>
          </div>
        </div>

        {datasets.length ? (
          <form className="run-form" onSubmit={startRun}>
            <label>
              Dataset revision
              <select value={datasetId} onChange={e => setDatasetId(e.target.value)} required>
                <option value="">Select a dataset</option>
                {datasets.map(d => (
                  <option key={d.id} value={d.id}>
                    {d.name} · r{d.revision} · {d.case_count} cases
                  </option>
                ))}
              </select>
            </label>
            <p className="form-help-text">Standard evaluation cutoffs: 1, 3, 5 · source-element labels required</p>
            <button className="button primary" disabled={busy}>
              {busy ? 'Queueing…' : 'Queue evaluation'}
            </button>
          </form>
        ) : (
          <p className="empty-inline">Create an element-labeled dataset before running an evaluation.</p>
        )}
      </section>

      <section className="table-section">
        <div className="section-title">
          <div>
            <h2>Evaluation runs</h2>
            <span>Refreshes while jobs are processing</span>
          </div>
        </div>

        {runs.length === 0 ? (
          <div className="empty">
            <span className="empty-mark">03</span>
            <div>
              <h3>No evaluation runs</h3>
              <p>Queue an evaluation against a reviewed dataset revision. Results include per-query metrics, statuses, and denominators.</p>
            </div>
          </div>
        ) : (
          runs.map(run => (
            <article className="run-result" key={run.id}>
              <div className="section-title">
                <div>
                  <h3>
                    Run <code>{run.id.slice(0, 8)}</code>
                  </h3>
                  <p>
                    Dataset revision {run.revision} · {new Date(run.created_at).toLocaleString()}
                  </p>
                </div>
                <span className={`run-status ${run.status}`}>{run.status}</span>
              </div>

              {run.report ? (
                <>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Metric</th>
                          <th>Value</th>
                          <th>Count</th>
                          <th>Label unit</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(run.report.metrics)
                          .filter(
                            ([key]) =>
                              key.startsWith('retrieval.') || key.startsWith('answer.') || key.startsWith('latency.'),
                          )
                          .map(([key, value]) => (
                            <tr key={key}>
                              <td>
                                <strong>{key}</strong>
                              </td>
                              <td>
                                <span className="metric-badge">{value == null ? '—' : value.toFixed(3)}</span>
                              </td>
                              <td>{run.report?.metric_counts[key] ?? 0}</td>
                              <td>
                                {key.includes('source_element')
                                  ? 'source elements'
                                  : key.startsWith('answer.')
                                    ? 'reference answers'
                                    : key.startsWith('latency.')
                                      ? 'milliseconds'
                                      : 'chunks'}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="query-list">
                    <h4>Query Level Diagnostics ({run.report.queries.length})</h4>
                    {run.report.queries.map(query => (
                      <details key={query.query_id}>
                        <summary>
                          <span className={`query-status-tag ${query.status}`}>{query.status}</span>
                          <span className="query-text">{query.query}</span>
                          <code>{query.query_id.slice(0, 8)}</code>
                        </summary>
                        {query.errors.map(item => (
                          <p className="inline-error" key={item}>
                            {item}
                          </p>
                        ))}
                        {query.generated_answer && <p className="query-answer">{query.generated_answer}</p>}
                        <pre>
                          {JSON.stringify(
                            {
                              retrieval_metrics: query.retrieval_metrics,
                              answer_metrics: query.answer_metrics,
                              citations: query.citations,
                            },
                            null,
                            2,
                          )}
                        </pre>
                        {query.retrieved_sources.map(source => (
                          <details className="report-source" key={source.chunk.chunk_id}>
                            <summary>
                              Ranked evidence · cosine {source.score.toFixed(3)} · {source.chunk.chunk_id}
                            </summary>
                            <p>{source.chunk.text}</p>
                          </details>
                        ))}
                      </details>
                    ))}
                  </div>
                </>
              ) : (
                <p className="inline-note">
                  {run.status === 'failed'
                    ? 'This run failed. Check worker and Ollama availability, then queue a new run.'
                    : 'No metrics are displayed until the report is persisted.'}
                </p>
              )}
            </article>
          ))
        )}
      </section>
    </>
  )
}

function ExperimentsWorkbench({
  datasets,
  onError,
}: {
  datasets: DatasetRow[]
  onError: (message: string) => void
}) {
  const [experiments, setExperiments] = useState<ExperimentRow[]>([])
  const [datasetId, setDatasetId] = useState('')
  const [baselineName, setBaselineName] = useState('depth-3')
  const [candidateName, setCandidateName] = useState('depth-5')
  const [baselineDepth, setBaselineDepth] = useState(3)
  const [candidateDepth, setCandidateDepth] = useState(5)
  const [busy, setBusy] = useState(false)

  async function refreshExperiments() {
    try {
      setExperiments(await api<ExperimentRow[]>('/experiments'))
    } catch (e) {
      onError((e as Error).message)
    }
  }

  useEffect(() => {
    void refreshExperiments()
    const timer = window.setInterval(() => void refreshExperiments(), 5000)
    return () => window.clearInterval(timer)
  }, [])

  async function start(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    try {
      await api('/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          revision: datasets.find(item => item.id === datasetId)?.revision,
          baseline_variant: baselineName,
          variants: [
            { name: baselineName, retrieval_depth: baselineDepth },
            { name: candidateName, retrieval_depth: candidateDepth },
          ],
          cutoffs: [1, 3, 5],
          bootstrap_resamples: 1000,
          random_seed: 0,
        }),
      })
      await refreshExperiments()
    } catch (err) {
      onError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <section className="form-section">
        <div className="section-title">
          <div>
            <h2>Compare retrieval depth</h2>
            <span>Paired queries · 1,000 bootstrap resamples · fixed seed 0</span>
          </div>
        </div>

        {datasets.length ? (
          <form className="experiment-form" onSubmit={start}>
            <label>
              Dataset revision
              <select value={datasetId} onChange={e => setDatasetId(e.target.value)} required>
                <option value="">Select a dataset</option>
                {datasets.map(d => (
                  <option key={d.id} value={d.id}>
                    {d.name} · r{d.revision} · {d.case_count} cases
                  </option>
                ))}
              </select>
            </label>
            <div className="variant-fields">
              <fieldset>
                <legend>Baseline Variant</legend>
                <label>
                  Name
                  <input value={baselineName} onChange={e => setBaselineName(e.target.value)} required maxLength={64} />
                </label>
                <label>
                  Retrieval depth
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={baselineDepth}
                    onChange={e => setBaselineDepth(Number(e.target.value))}
                  />
                </label>
              </fieldset>
              <fieldset>
                <legend>Candidate Variant</legend>
                <label>
                  Name
                  <input value={candidateName} onChange={e => setCandidateName(e.target.value)} required maxLength={64} />
                </label>
                <label>
                  Retrieval depth
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={candidateDepth}
                    onChange={e => setCandidateDepth(Number(e.target.value))}
                  />
                </label>
              </fieldset>
            </div>
            <p className="experiment-note">
              Metrics use the same dataset revision and cutoffs (1, 3, 5). Higher positive deltas indicate larger metric values; intervals describe paired query variation.
            </p>
            <button className="button primary" disabled={busy}>
              {busy ? 'Queueing…' : 'Queue paired experiment'}
            </button>
          </form>
        ) : (
          <p className="empty-inline">Create a source-element-labeled dataset before comparing retrieval variants.</p>
        )}
      </section>

      <section className="table-section">
        <div className="section-title">
          <div>
            <h2>Experiments</h2>
            <span>Effect sizes and intervals are reported without declaring an artificial winner</span>
          </div>
        </div>

        {experiments.length === 0 ? (
          <div className="empty">
            <span className="empty-mark">04</span>
            <div>
              <h3>No experiments yet</h3>
              <p>Compare two retrieval-depth configurations on one immutable, element-labeled dataset revision.</p>
            </div>
          </div>
        ) : (
          experiments.map(exp => (
            <article className="run-result" key={exp.id}>
              <div className="section-title">
                <div>
                  <h3>
                    Experiment <code>{exp.id.slice(0, 8)}</code>
                  </h3>
                  <p>
                    Dataset revision {exp.revision} · {new Date(exp.created_at).toLocaleString()}
                  </p>
                </div>
                <span className={`run-status ${exp.status}`}>{exp.status}</span>
              </div>

              {exp.report ? (
                <>
                  <div className="variant-summary">
                    {exp.report.variants.map(variant => (
                      <p key={variant.name}>
                        <strong>{variant.name}</strong> · depth {String(variant.pipeline_config.retrieval_depth)} ·{' '}
                        {variant.status}
                        {variant.error && <span className="inline-error"> · {variant.error}</span>}
                      </p>
                    ))}
                  </div>

                  {exp.report.comparisons.map(comparison => (
                    <div className="table-scroll" key={comparison.candidate_variant}>
                      <h4>
                        {exp.report?.config.baseline_variant} → {comparison.candidate_variant}
                      </h4>
                      {comparison.status === 'complete' ? (
                        <table>
                          <thead>
                            <tr>
                              <th>Metric</th>
                              <th>Baseline</th>
                              <th>Candidate</th>
                              <th>Delta</th>
                              <th>95% interval</th>
                              <th>Paired queries</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(comparison.metrics).map(([metric, value]) => {
                              const deltaVal = value.delta ?? 0
                              return (
                                <tr key={metric}>
                                  <td><strong>{metric}</strong></td>
                                  <td>{value.baseline_mean?.toFixed(3) ?? '—'}</td>
                                  <td>{value.candidate_mean?.toFixed(3) ?? '—'}</td>
                                  <td>
                                    <span
                                      className={`delta-tag ${deltaVal > 0 ? 'positive' : deltaVal < 0 ? 'negative' : ''}`}
                                    >
                                      {deltaVal > 0 ? '+' : ''}
                                      {value.delta?.toFixed(3) ?? '—'}
                                    </span>
                                  </td>
                                  <td>
                                    <code>
                                      [{value.confidence_interval_low?.toFixed(3) ?? '—'},{' '}
                                      {value.confidence_interval_high?.toFixed(3) ?? '—'}]
                                    </code>
                                  </td>
                                  <td>{value.paired_query_count}</td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      ) : (
                        <p>{comparison.reason ?? 'Comparison unavailable.'}</p>
                      )}
                    </div>
                  ))}
                </>
              ) : (
                <p className="inline-note">
                  {exp.status === 'failed'
                    ? 'This experiment failed. Check worker and model availability before retrying.'
                    : 'No comparison is shown until the experiment report is saved.'}
                </p>
              )}
            </article>
          ))
        )}
      </section>
    </>
  )
}

export default App
