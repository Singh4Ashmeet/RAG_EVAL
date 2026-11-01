import { useState, useEffect, useCallback } from 'react'
import type { CSSProperties } from 'react'

type Section = 'Corpus' | 'Ask' | 'Datasets' | 'Runs' | 'Experiments'

/* --------------------------------------------------------------------------
   Tour Step Definitions
   -------------------------------------------------------------------------- */

interface TourStep {
  id: string
  target: string | null
  title: string
  subtitle: string
  body: string
  section?: Section
  actionLabel?: string
  tip?: string
}

const WORKSPACE_STEPS: TourStep[] = [
  {
    id: 'welcome',
    target: null,
    title: 'Welcome to RAG_EVAL',
    subtitle: 'Evaluation Workflow Walkthrough',
    body: 'RAG_EVAL provides rigorous, reproducible evaluation for your RAG pipelines with deterministic ground truth and statistical significance testing. Let\'s walk through the workbench in 8 quick steps.',
    actionLabel: 'Begin Walkthrough →',
    tip: 'Press Escape anytime to pause or close the tour.',
  },
  {
    id: 'nav',
    target: '[data-tour="nav"]',
    title: '1. Pipeline Stages Navigation',
    subtitle: 'Corpus · Ask · Datasets · Runs · Experiments',
    body: 'Navigate between the five evaluation stages: manage source documents, test real-time retrieval in Ask, build immutable ground-truth datasets, queue quantitative runs, and measure statistical effect sizes in Experiments.',
    actionLabel: 'Next: Ingestion →',
  },
  {
    id: 'upload',
    target: '[data-tour="upload"]',
    title: '2. Upload Source Material',
    subtitle: 'Stable, Deterministic Element Extraction',
    body: 'Upload UTF-8 text (.txt) or Markdown (.md) documents up to 25 MiB. Each file is parsed into stable elements with immutable IDs (elem_...). When you adjust chunk sizes later, your ground-truth labels remain valid.',
    section: 'Corpus',
    actionLabel: 'Next: Document Registry →',
    tip: 'Select a document file above to begin.',
  },
  {
    id: 'documents',
    target: '[data-tour="documents"]',
    title: '3. Document Registry & Elements',
    subtitle: 'Content-Addressed Document Registry',
    body: 'Review processing state, SHA-256 content digests, and indexed status. Click any document name to inspect its individual extracted elements and verify the exact text spans available for evaluation queries.',
    section: 'Corpus',
    actionLabel: 'Next: Ask & Evidence →',
  },
  {
    id: 'ask',
    target: '[data-tour="ask-section"]',
    title: '4. Ask & Evidence Retrieval',
    subtitle: 'Real-Time Cosine Scores & Provenance',
    body: 'Type a question to test retrieval live. Select local Ollama (private) or configured cloud models (Gemini, Groq). Review retrieved passages ranked by exact cosine similarity with visual match gauges and grounded citations.',
    section: 'Ask',
    actionLabel: 'Next: Gold Datasets →',
  },
  {
    id: 'datasets',
    target: '[data-tour="datasets-section"]',
    title: '5. Build Ground-Truth Datasets',
    subtitle: 'Immutable, Content-Addressed Revisions',
    body: 'Map realistic evaluation questions directly to source element checkboxes. Revisions (r1, r2, ...) are content-addressed with SHA-256 digests so your benchmark metrics remain comparable over time.',
    section: 'Datasets',
    actionLabel: 'Next: Evaluation Runs →',
  },
  {
    id: 'runs',
    target: '[data-tour="runs-section"]',
    title: '6. Quantitative Benchmark Runs',
    subtitle: 'Precision @ 1/3/5 · Recall @ 5 · MRR',
    body: 'Queue batch evaluation jobs against dataset revisions. Every run produces standardized metrics (Precision@K, Mean Reciprocal Rank) and audits generated answers for hallucinated citation references [S1].',
    section: 'Runs',
    actionLabel: 'Next: Paired Experiments →',
  },
  {
    id: 'experiments',
    target: '[data-tour="experiments-section"]',
    title: '7. Paired A/B Experiments',
    subtitle: '1,000 Bootstrap Resamples · 95% Confidence Intervals',
    body: 'Compare Baseline vs. Candidate retrieval configurations on the identical dataset revision. The engine runs 1,000 bootstrap resamples to calculate empirical 95% confidence intervals, proving whether improvements are statistically genuine.',
    section: 'Experiments',
    actionLabel: 'Next: Configuration →',
  },
  {
    id: 'settings',
    target: '[data-tour="settings-btn"]',
    title: '8. Settings & Encryption Keys',
    subtitle: 'Fernet-Encrypted Key Storage',
    body: 'Add your Gemini or Groq API keys here. Secrets are encrypted at rest with your CREDENTIAL_ENCRYPTION_KEY. Private Ollama embeddings run locally through your configured OLLAMA_BASE_URL.',
    actionLabel: 'Finish Tour →',
  },
  {
    id: 'complete',
    target: null,
    title: 'You\'re All Set!',
    subtitle: 'Ready to Evaluate',
    body: 'You now know the full evaluation workflow. Start by uploading a source document in Corpus, then create a dataset revision and compare pipeline variants.',
    actionLabel: 'Enter Workspace',
  },
]

const LANDING_STEPS: TourStep[] = [
  {
    id: 'landing-welcome',
    target: null,
    title: 'Welcome to RAG_EVAL',
    subtitle: 'Local-First RAG Evaluation Workbench',
    body: 'RAG_EVAL evaluates Retrieval-Augmented Generation systems with mathematical rigor: element-level ground truth, deterministic metrics, and paired bootstrap significance tests.',
    actionLabel: 'See How to Start →',
  },
  {
    id: 'landing-signin',
    target: '#signin',
    title: 'Sign In to Your Workspace',
    subtitle: 'Email Link · Google · GitHub · Local Dev Mode',
    body: 'Access your private workspace here. For local testing, AUTH_DEV_MODE prints an instant one-click login link directly to your terminal without needing email credentials.',
    actionLabel: 'Explore Features →',
  },
  {
    id: 'landing-features',
    target: '#features',
    title: 'Rigorous Evaluation Features',
    subtitle: 'Beyond Synthetic LLM-as-a-Judge',
    body: 'Explore deterministic element extraction, content-addressed datasets, paired hypothesis testing, and encryption at rest. Scroll down to review the architecture and documentation.',
    actionLabel: 'Got It!',
  },
]

/* --------------------------------------------------------------------------
   Viewport & Geometry Helpers
   -------------------------------------------------------------------------- */

interface ViewportRect {
  top: number
  left: number
  width: number
  height: number
  bottom: number
  right: number
}

function getTargetViewportRect(selector: string): ViewportRect | null {
  const el = document.querySelector(selector)
  if (!el) return null
  const r = el.getBoundingClientRect()
  return {
    top: r.top,
    left: r.left,
    width: r.width,
    height: r.height,
    bottom: r.bottom,
    right: r.right,
  }
}

function computeTooltipPosition(rect: ViewportRect | null): CSSProperties {
  const viewportW = typeof window !== 'undefined' ? window.innerWidth : 1200
  const viewportH = typeof window !== 'undefined' ? window.innerHeight : 800
  const CARD_W = Math.min(420, viewportW - 32)
  const CARD_APPROX_H = 260
  const GAP = 12

  // Center modal if no target element
  if (!rect) {
    return {
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      width: CARD_W,
      maxHeight: `calc(100vh - 32px)`,
      zIndex: 10003,
    }
  }

  // Horizontal position: center relative to target element, clamped inside viewport
  let left = rect.left + rect.width / 2 - CARD_W / 2
  left = Math.max(16, Math.min(left, viewportW - CARD_W - 16))

  // Vertical position: prefer below target if space permits, else above
  const spaceBelow = viewportH - rect.bottom
  const spaceAbove = rect.top
  const placeAbove = spaceBelow < CARD_APPROX_H + GAP && spaceAbove > spaceBelow

  let top: number
  if (placeAbove) {
    top = Math.max(16, rect.top - CARD_APPROX_H - GAP)
  } else {
    top = Math.min(viewportH - CARD_APPROX_H - 16, rect.bottom + GAP)
  }

  return {
    position: 'fixed',
    top,
    left,
    width: CARD_W,
    maxHeight: `calc(100vh - 32px)`,
    zIndex: 10003,
  }
}

/* --------------------------------------------------------------------------
   GuideBot Component
   -------------------------------------------------------------------------- */

interface GuideBotProps {
  isLoggedIn: boolean
  currentSection?: Section
  onNavigate?: (section: Section) => void
}

const STORAGE_TOUR_DONE = 'rag_eval_tour_seen'
const STORAGE_COMPLETED_STEPS = 'rag_eval_guide_completed'
const STORAGE_BANNER_DISMISSED = 'rag_eval_welcome_dismissed'

export function GuideBot({ isLoggedIn, onNavigate }: GuideBotProps) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [tourActive, setTourActive] = useState(false)
  const [tourIndex, setTourIndex] = useState(0)
  const [targetRect, setTargetRect] = useState<ViewportRect | null>(null)
  const [completedSteps, setCompletedSteps] = useState<string[]>([])
  const [showWelcomeBanner, setShowWelcomeBanner] = useState(false)

  const steps = isLoggedIn ? WORKSPACE_STEPS : LANDING_STEPS
  const currentStep = steps[tourIndex] || steps[0]

  // Check stored states
  useEffect(() => {
    try {
      const savedSteps = localStorage.getItem(STORAGE_COMPLETED_STEPS)
      if (savedSteps) {
        setCompletedSteps(JSON.parse(savedSteps) as string[])
      }

      if (isLoggedIn) {
        const dismissed = localStorage.getItem(STORAGE_BANNER_DISMISSED)
        const tourDone = localStorage.getItem(STORAGE_TOUR_DONE)
        if (!dismissed && !tourDone) {
          setShowWelcomeBanner(true)
        }
      }
    } catch {
      // Ignore storage errors
    }
  }, [isLoggedIn])

  // Measure element in viewport coordinates
  const updateMeasurement = useCallback(() => {
    if (!tourActive) return
    const cur = steps[tourIndex]
    if (!cur?.target) {
      setTargetRect(null)
      return
    }
    const rect = getTargetViewportRect(cur.target)
    setTargetRect(rect)
  }, [tourActive, tourIndex, steps])

  useEffect(() => {
    updateMeasurement()
    window.addEventListener('resize', updateMeasurement)
    window.addEventListener('scroll', updateMeasurement, true)
    return () => {
      window.removeEventListener('resize', updateMeasurement)
      window.removeEventListener('scroll', updateMeasurement, true)
    }
  }, [updateMeasurement])

  // Scroll target element smoothly into view
  useEffect(() => {
    if (!tourActive) return
    const cur = steps[tourIndex]
    if (cur?.target) {
      const el = document.querySelector(cur.target)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        const timer = setTimeout(updateMeasurement, 250)
        return () => clearTimeout(timer)
      }
    }
  }, [tourActive, tourIndex, steps, updateMeasurement])

  // Keyboard navigation: Escape closes, Arrows navigate
  useEffect(() => {
    if (!tourActive) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        endTour()
      } else if (e.key === 'ArrowRight' && tourIndex < steps.length - 1) {
        stepTo(tourIndex + 1)
      } else if (e.key === 'ArrowLeft' && tourIndex > 0) {
        stepTo(tourIndex - 1)
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  })

  function stepTo(nextIdx: number) {
    if (nextIdx < 0 || nextIdx >= steps.length) return
    const targetStep = steps[nextIdx]

    // Record completed step
    if (currentStep && !completedSteps.includes(currentStep.id)) {
      const updated = [...completedSteps, currentStep.id]
      setCompletedSteps(updated)
      try {
        localStorage.setItem(STORAGE_COMPLETED_STEPS, JSON.stringify(updated))
      } catch {
        // Ignore
      }
    }

    // Switch workspace section if step requires it
    if (targetStep?.section && onNavigate && isLoggedIn) {
      onNavigate(targetStep.section)
      setTimeout(() => setTourIndex(nextIdx), 100)
    } else {
      setTourIndex(nextIdx)
    }
  }

  function startTourFrom(index: number = 0) {
    setDrawerOpen(false)
    setShowWelcomeBanner(false)
    setTourIndex(index)
    setTourActive(true)
    const stepObj = steps[index]
    if (stepObj?.section && onNavigate && isLoggedIn) {
      onNavigate(stepObj.section)
    }
  }

  function endTour() {
    setTourActive(false)
    try {
      localStorage.setItem(STORAGE_TOUR_DONE, 'true')
    } catch {
      // Ignore
    }
  }

  function dismissWelcomeBanner() {
    setShowWelcomeBanner(false)
    try {
      localStorage.setItem(STORAGE_BANNER_DISMISSED, 'true')
    } catch {
      // Ignore
    }
  }

  function toggleStepCheck(id: string) {
    const updated = completedSteps.includes(id)
      ? completedSteps.filter(s => s !== id)
      : [...completedSteps, id]
    setCompletedSteps(updated)
    try {
      localStorage.setItem(STORAGE_COMPLETED_STEPS, JSON.stringify(updated))
    } catch {
      // Ignore
    }
  }

  const isFirst = tourIndex === 0
  const isLast = tourIndex === steps.length - 1
  const tooltipStyle = computeTooltipPosition(targetRect)

  const actionSteps = steps.filter(s => s.target !== null)
  const completedCount = actionSteps.filter(s => completedSteps.includes(s.id)).length
  const progressPercent = actionSteps.length > 0 ? Math.round((completedCount / actionSteps.length) * 100) : 0

  return (
    <>
      {/* --------------------------------------------------------------------
          1. Optional First-Visit Welcome Banner (Polite & Non-intrusive)
          -------------------------------------------------------------------- */}
      {showWelcomeBanner && !tourActive && !drawerOpen && (
        <aside className="guide-welcome-toast" aria-label="Welcome guide prompt">
          <div className="guide-welcome-toast-icon">🧭</div>
          <div className="guide-welcome-toast-text">
            <strong>Welcome to RAG_EVAL!</strong>
            <span>Take the 1-minute walkthrough to learn how to evaluate your RAG pipeline.</span>
          </div>
          <div className="guide-welcome-toast-actions">
            <button
              type="button"
              className="button primary guide-welcome-btn-start"
              onClick={() => startTourFrom(0)}
            >
              Start Tour →
            </button>
            <button
              type="button"
              className="guide-welcome-btn-dismiss"
              onClick={dismissWelcomeBanner}
              aria-label="Dismiss banner"
            >
              Dismiss
            </button>
          </div>
        </aside>
      )}

      {/* --------------------------------------------------------------------
          2. Floating Guide Bot Trigger Pill (Bottom Right)
          -------------------------------------------------------------------- */}
      {!tourActive && (
        <aside className="guide-bot-anchor" aria-label="Evaluation Workflow Assistant">
          <button
            type="button"
            className="guide-bot-pill"
            onClick={() => setDrawerOpen(prev => !prev)}
            aria-expanded={drawerOpen}
          >
            <span className="guide-bot-avatar" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            </span>
            <span className="guide-bot-label">
              <strong>Guide Bot</strong>
              <span className="guide-bot-sub">{isLoggedIn ? `${progressPercent}% complete` : 'Step-by-step help'}</span>
            </span>
            <span className="guide-bot-badge">{isLoggedIn ? `${completedCount}/${actionSteps.length}` : 'Help'}</span>
          </button>
        </aside>
      )}

      {/* --------------------------------------------------------------------
          3. Step-by-Step Checklist Drawer
          -------------------------------------------------------------------- */}
      {drawerOpen && !tourActive && (
        <div
          className="guide-drawer"
          role="dialog"
          aria-modal="false"
          aria-labelledby="guide-drawer-heading"
        >
          <header className="guide-drawer-header">
            <div className="guide-drawer-title-group">
              <span className="guide-drawer-tag">EVALUATION ASSISTANT</span>
              <h3 id="guide-drawer-heading">Evaluation Workflow Guide</h3>
              <p className="guide-drawer-desc">
                {isLoggedIn
                  ? 'Follow these steps to establish rigorous, reproducible RAG quality benchmarks.'
                  : 'Follow these steps to access your workspace and evaluate your RAG pipeline.'}
              </p>
            </div>
            <button
              type="button"
              className="guide-drawer-close"
              onClick={() => setDrawerOpen(false)}
              aria-label="Close guide assistant"
            >
              ×
            </button>
          </header>

          {/* Quick Tour Launcher Card */}
          <div className="guide-tour-cta-card">
            <div className="guide-tour-cta-content">
              <h4>Interactive Spotlight Tour</h4>
              <p>Walk through every section with guided highlights and tips.</p>
            </div>
            <button
              type="button"
              className="button primary guide-launch-tour-btn"
              onClick={() => startTourFrom(0)}
            >
              Start Walkthrough →
            </button>
          </div>

          {/* Checklist Progress */}
          {isLoggedIn && (
            <div className="guide-drawer-progress">
              <div className="guide-drawer-progress-header">
                <span>Evaluation Checklist</span>
                <strong>{progressPercent}% Complete</strong>
              </div>
              <div className="guide-drawer-progress-track">
                <div
                  className="guide-drawer-progress-bar"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>
          )}

          {/* Step Cards */}
          <div className="guide-drawer-steps-list">
            {steps
              .filter(s => s.target !== null)
              .map((step, idx) => {
                const isDone = completedSteps.includes(step.id)
                return (
                  <article key={step.id} className={`guide-step-item ${isDone ? 'done' : ''}`}>
                    <label className="guide-step-check">
                      <input
                        type="checkbox"
                        checked={isDone}
                        onChange={() => toggleStepCheck(step.id)}
                      />
                      <span className="guide-step-num">{idx + 1}</span>
                    </label>

                    <div className="guide-step-details">
                      <div className="guide-step-header-line">
                        <h5>{step.title}</h5>
                        {step.section && (
                          <span className="guide-step-section-pill">{step.section}</span>
                        )}
                      </div>
                      <p className="guide-step-summary">{step.body}</p>
                      {step.tip && <div className="guide-step-tip">💡 {step.tip}</div>}

                      <div className="guide-step-action-row">
                        <button
                          type="button"
                          className="guide-step-spotlight-btn"
                          onClick={() => {
                            const foundIndex = steps.findIndex(s => s.id === step.id)
                            if (foundIndex >= 0) startTourFrom(foundIndex)
                          }}
                        >
                          Highlight on screen ↗
                        </button>
                        {step.section && onNavigate && isLoggedIn && (
                          <button
                            type="button"
                            className="guide-step-nav-btn"
                            onClick={() => onNavigate(step.section!)}
                          >
                            Open {step.section}
                          </button>
                        )}
                      </div>
                    </div>
                  </article>
                )
              })}
          </div>

          <footer className="guide-drawer-footer">
            <span className="guide-status-chip">
              <span className="status-dot-green" /> Local .env active
            </span>
            <button
              type="button"
              className="quiet-button"
              onClick={() => {
                setCompletedSteps([])
                try {
                  localStorage.removeItem(STORAGE_COMPLETED_STEPS)
                  localStorage.removeItem(STORAGE_TOUR_DONE)
                  localStorage.removeItem(STORAGE_BANNER_DISMISSED)
                } catch {
                  // Ignore
                }
              }}
            >
              Reset Guide Progress
            </button>
          </footer>
        </div>
      )}

      {/* --------------------------------------------------------------------
          4. Robust CSS Box-Shadow Spotlight Tour Mode
          -------------------------------------------------------------------- */}
      {tourActive && (
        <div className="tour-layer" role="dialog" aria-modal="true" aria-label="Guided Tour">
          {/* Clickable backdrop dismiss */}
          <div
            className="tour-backdrop"
            onClick={() => (isLast ? endTour() : stepTo(tourIndex + 1))}
            aria-hidden="true"
          />

          {/* Bulletproof CSS Box-Shadow Spotlight Cutout */}
          {targetRect && (
            <div
              className="tour-spotlight-cutout"
              style={{
                top: targetRect.top - 6,
                left: targetRect.left - 6,
                width: targetRect.width + 12,
                height: targetRect.height + 12,
              }}
              aria-hidden="true"
            />
          )}

          {/* Floating Tooltip Card */}
          <aside
            className="tour-card"
            style={tooltipStyle}
            onClick={e => e.stopPropagation()}
          >
            {/* Top progress strip */}
            <div className="tour-card-progress">
              <div
                className="tour-card-progress-fill"
                style={{ width: `${((tourIndex + 1) / steps.length) * 100}%` }}
              />
            </div>

            <div className="tour-card-header">
              <span className="tour-card-step-badge">
                STEP {tourIndex + 1} OF {steps.length}
              </span>
              <button
                type="button"
                className="tour-card-close"
                onClick={endTour}
                aria-label="Close walkthrough"
              >
                ×
              </button>
            </div>

            <div className="tour-card-content">
              <h4 className="tour-card-title">{currentStep.title}</h4>
              <p className="tour-card-subtitle">{currentStep.subtitle}</p>
              <p className="tour-card-body">{currentStep.body}</p>
              {currentStep.tip && (
                <div className="tour-card-tip">
                  <strong>Tip:</strong> {currentStep.tip}
                </div>
              )}
            </div>

            <footer className="tour-card-footer">
              <button
                type="button"
                className="tour-btn tour-btn-skip"
                onClick={endTour}
              >
                Skip tour
              </button>

              <div className="tour-btn-group">
                {!isFirst && (
                  <button
                    type="button"
                    className="tour-btn tour-btn-prev"
                    onClick={() => stepTo(tourIndex - 1)}
                  >
                    ← Back
                  </button>
                )}
                {!isLast && (
                  <button
                    type="button"
                    className="tour-btn tour-btn-next"
                    onClick={() => stepTo(tourIndex + 1)}
                  >
                    {currentStep.actionLabel || 'Next →'}
                  </button>
                )}
                {isLast && (
                  <button
                    type="button"
                    className="tour-btn tour-btn-next"
                    onClick={endTour}
                  >
                    Get Started →
                  </button>
                )}
              </div>
            </footer>

            {/* Stepper dots */}
            <nav className="tour-stepper-dots" aria-label="Tour step indicators">
              {steps.map((s, i) => (
                <button
                  key={s.id}
                  type="button"
                  className={`tour-dot ${i === tourIndex ? 'active' : ''} ${i < tourIndex ? 'done' : ''}`}
                  onClick={() => stepTo(i)}
                  aria-label={`Jump to step ${i + 1}: ${s.title}`}
                />
              ))}
            </nav>
          </aside>
        </div>
      )}
    </>
  )
}
