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
  placement: 'top' | 'bottom' | 'left' | 'right' | 'center'
  section?: Section
  actionLabel?: string
  tip?: string
}

const WORKSPACE_STEPS: TourStep[] = [
  {
    id: 'welcome',
    target: null,
    title: 'Welcome to RAG_EVAL',
    subtitle: 'Interactive Workbench Walkthrough',
    body: 'RAG_EVAL is a rigorous evaluation platform for retrieval-augmented generation. Follow this guided walkthrough to learn how to measure retrieval precision, verify grounded answers, and run paired bootstrap experiments.',
    placement: 'center',
    actionLabel: 'Start Walkthrough →',
    tip: 'Press Escape anytime to pause or close the guide.',
  },
  {
    id: 'nav',
    target: '[data-tour="nav"]',
    title: 'Pipeline Stages Navigation',
    subtitle: 'Corpus · Ask · Datasets · Runs · Experiments',
    body: 'Each tab represents a stage in your evaluation pipeline: manage source documents, test real-time retrieval in Ask, build immutable ground-truth datasets, queue quantitative runs, and measure statistical effect sizes in Experiments.',
    placement: 'bottom',
    actionLabel: 'Next: Ingestion →',
  },
  {
    id: 'upload',
    target: '[data-tour="upload"]',
    title: 'Step 1: Upload Source Corpus',
    subtitle: 'Stable, Deterministic Element Extraction',
    body: 'Upload UTF-8 text (.txt) or Markdown (.md) documents up to 25 MiB. Each document is deterministically parsed into stable elements with immutable UUIDs (elem_...). When you adjust chunk sizes later, your ground-truth labels remain valid.',
    placement: 'bottom',
    section: 'Corpus',
    actionLabel: 'Next: Document Library →',
    tip: 'Upload technical specs or documentation to start evaluating.',
  },
  {
    id: 'documents',
    target: '[data-tour="documents"]',
    title: 'Step 2: Inspect Extracted Elements',
    subtitle: 'Content-Addressed Document Registry',
    body: 'Review processing state, SHA-256 content digests, and indexed status. Click any document name to expand its extracted source elements and inspect the exact text spans available for evaluation queries.',
    placement: 'top',
    section: 'Corpus',
    actionLabel: 'Next: Ask & Evidence →',
  },
  {
    id: 'ask',
    target: '[data-tour="ask-section"]',
    title: 'Step 3: Interactive Query & Evidence',
    subtitle: 'Inspect Real-Time Cosine Scores & Provenance',
    body: 'Type a question to test retrieval live. Select local Ollama (private) or configured cloud models (Gemini, Groq). Review retrieved passages ranked by exact cosine similarity with visual match gauges and grounded citations.',
    placement: 'bottom',
    section: 'Ask',
    actionLabel: 'Next: Gold Datasets →',
  },
  {
    id: 'datasets',
    target: '[data-tour="datasets-section"]',
    title: 'Step 4: Build Ground-Truth Datasets',
    subtitle: 'Immutable, SHA-256 Revisions',
    body: 'Map realistic evaluation questions directly to source element checkboxes. Revisions (r1, r2, ...) are content-addressed and immutable so your benchmark metrics remain comparable over time.',
    placement: 'bottom',
    section: 'Datasets',
    actionLabel: 'Next: Evaluation Runs →',
  },
  {
    id: 'runs',
    target: '[data-tour="runs-section"]',
    title: 'Step 5: Quantitative Benchmark Runs',
    subtitle: 'Precision @ 1/3/5 · Recall @ 5 · MRR',
    body: 'Queue batch evaluation jobs against dataset revisions. Every run produces standardized metrics (Precision@K, Mean Reciprocal Rank) and audits generated answers for hallucinated citation references [S1].',
    placement: 'bottom',
    section: 'Runs',
    actionLabel: 'Next: Paired Experiments →',
  },
  {
    id: 'experiments',
    target: '[data-tour="experiments-section"]',
    title: 'Step 6: Paired A/B Experiments',
    subtitle: '1,000 Bootstrap Resamples · 95% Confidence Intervals',
    body: 'Compare Baseline vs. Candidate retrieval configurations on the identical dataset revision. The engine runs 1,000 bootstrap resamples to calculate empirical 95% confidence intervals, proving whether improvements are statistically genuine.',
    placement: 'bottom',
    section: 'Experiments',
    actionLabel: 'Next: Configuration →',
  },
  {
    id: 'settings',
    target: '[data-tour="settings-btn"]',
    title: 'Settings & Encryption Keys',
    subtitle: 'Fernet-Encrypted Key Storage',
    body: 'Add your Gemini or Groq API keys here. Secrets are encrypted at rest with your CREDENTIAL_ENCRYPTION_KEY. Private Ollama embeddings run locally through your configured OLLAMA_BASE_URL.',
    placement: 'bottom',
    actionLabel: 'Finish Tour →',
  },
  {
    id: 'complete',
    target: null,
    title: 'Ready to Evaluate',
    subtitle: 'You are all set!',
    body: 'You now know the full evaluation lifecycle. Start by uploading a document in Corpus, then build a dataset revision and compare pipeline variants.',
    placement: 'center',
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
    placement: 'center',
    actionLabel: 'See How to Start →',
  },
  {
    id: 'landing-signin',
    target: '#signin',
    title: 'Sign In to Your Workspace',
    subtitle: 'Email Link · Google · GitHub · Local Dev Mode',
    body: 'Access your private workspace here. For local testing, AUTH_DEV_MODE prints an instant one-click login link directly to your terminal without needing email credentials.',
    placement: 'right',
    actionLabel: 'Explore Features →',
  },
  {
    id: 'landing-features',
    target: '#features',
    title: 'Rigorous Evaluation Features',
    subtitle: 'Beyond Synthetic LLM-as-a-Judge',
    body: 'Explore deterministic element extraction, content-addressed datasets, paired hypothesis testing, and encryption at rest. Scroll down to review the architecture and documentation.',
    placement: 'top',
    actionLabel: 'Got It!',
  },
]

/* --------------------------------------------------------------------------
   Geometry & Positioning Helpers
   -------------------------------------------------------------------------- */

interface Rect {
  top: number
  left: number
  width: number
  height: number
  bottom: number
  right: number
}

function getElementRect(selector: string): Rect | null {
  const el = document.querySelector(selector)
  if (!el) return null
  const r = el.getBoundingClientRect()
  return {
    top: r.top + window.scrollY,
    left: r.left + window.scrollX,
    width: r.width,
    height: r.height,
    bottom: r.bottom + window.scrollY,
    right: r.right + window.scrollX,
  }
}

function computeTooltipPosition(
  rect: Rect | null,
  placement: TourStep['placement'],
): CSSProperties {
  const GAP = 16
  const TOOLTIP_W = 380
  const viewportW = typeof window !== 'undefined' ? window.innerWidth : 1200

  if (!rect || placement === 'center') {
    return {
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      width: Math.min(TOOLTIP_W, viewportW - 32),
      zIndex: 10002,
    }
  }

  // Clamped horizontal center relative to target
  let left = rect.left + rect.width / 2 - TOOLTIP_W / 2
  if (left < 16) left = 16
  if (left + TOOLTIP_W > viewportW - 16) {
    left = Math.max(16, viewportW - TOOLTIP_W - 16)
  }

  const base: CSSProperties = {
    position: 'absolute',
    width: Math.min(TOOLTIP_W, viewportW - 32),
    zIndex: 10002,
  }

  switch (placement) {
    case 'bottom':
      return {
        ...base,
        top: rect.bottom + GAP,
        left,
      }
    case 'top':
      return {
        ...base,
        top: rect.top - GAP,
        left,
        transform: 'translateY(-100%)',
      }
    case 'right':
      if (rect.right + GAP + TOOLTIP_W < viewportW) {
        return {
          ...base,
          top: Math.max(16, rect.top + rect.height / 2 - 120),
          left: rect.right + GAP,
        }
      }
      // Fallback to bottom if right edge overflows
      return {
        ...base,
        top: rect.bottom + GAP,
        left,
      }
    case 'left':
      if (rect.left - GAP - TOOLTIP_W > 16) {
        return {
          ...base,
          top: Math.max(16, rect.top + rect.height / 2 - 120),
          left: rect.left - GAP - TOOLTIP_W,
        }
      }
      return {
        ...base,
        top: rect.bottom + GAP,
        left,
      }
  }
}

/* --------------------------------------------------------------------------
   Spotlight SVG Mask Overlay
   -------------------------------------------------------------------------- */

function SpotlightOverlay({ rect }: { rect: Rect | null }) {
  if (!rect) {
    return <div className="tour-backdrop" />
  }

  const PAD = 10
  const RADIUS = 10
  const x = Math.max(0, rect.left - PAD)
  const y = Math.max(0, rect.top - PAD)
  const w = rect.width + PAD * 2
  const h = rect.height + PAD * 2

  const docW = typeof document !== 'undefined' ? Math.max(document.documentElement.scrollWidth, window.innerWidth) : 2000
  const docH = typeof document !== 'undefined' ? Math.max(document.documentElement.scrollHeight, window.innerHeight) : 2000

  return (
    <>
      <svg
        className="tour-svg-mask"
        viewBox={`0 0 ${docW} ${docH}`}
        aria-hidden="true"
      >
        <defs>
          <mask id="tour-spotlight-cutout">
            <rect x="0" y="0" width="100%" height="100%" fill="white" />
            <rect x={x} y={y} width={w} height={h} rx={RADIUS} ry={RADIUS} fill="black" />
          </mask>
        </defs>
        <rect
          x="0"
          y="0"
          width="100%"
          height="100%"
          fill="rgba(14, 23, 17, 0.65)"
          mask="url(#tour-spotlight-cutout)"
        />
      </svg>
      {/* Animated beacon ring around target element */}
      <div
        className="tour-beacon-ring"
        style={{
          top: y,
          left: x,
          width: w,
          height: h,
          borderRadius: RADIUS,
        }}
        aria-hidden="true"
      />
    </>
  )
}

/* --------------------------------------------------------------------------
   GuideBot Props
   -------------------------------------------------------------------------- */

interface GuideBotProps {
  isLoggedIn: boolean
  currentSection?: Section
  onNavigate?: (section: Section) => void
}

const STORAGE_TOUR_DONE = 'rag_eval_tour_seen'
const STORAGE_COMPLETED_STEPS = 'rag_eval_guide_completed'

export function GuideBot({ isLoggedIn, onNavigate }: GuideBotProps) {
  // Modal / Drawer state vs. Interactive Tour state
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [tourActive, setTourActive] = useState(false)
  const [tourIndex, setTourIndex] = useState(0)
  const [targetRect, setTargetRect] = useState<Rect | null>(null)
  const [completedSteps, setCompletedSteps] = useState<string[]>([])

  const steps = isLoggedIn ? WORKSPACE_STEPS : LANDING_STEPS
  const currentStep = steps[tourIndex] || steps[0]

  // Load completed step ids from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_COMPLETED_STEPS)
      if (saved) {
        setCompletedSteps(JSON.parse(saved) as string[])
      }
    } catch {
      // Ignore
    }
  }, [])

  // Auto-prompt tour on first workspace visit
  useEffect(() => {
    if (!isLoggedIn) return
    try {
      const seen = localStorage.getItem(STORAGE_TOUR_DONE)
      if (!seen) {
        const timer = setTimeout(() => {
          setTourActive(true)
          setTourIndex(0)
        }, 800)
        return () => clearTimeout(timer)
      }
    } catch {
      // Ignore
    }
  }, [isLoggedIn])

  // Recalculate spotlight measurement
  const updateMeasurement = useCallback(() => {
    if (!tourActive) return
    const cur = steps[tourIndex]
    if (!cur?.target) {
      setTargetRect(null)
      return
    }
    const rect = getElementRect(cur.target)
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
        const timer = setTimeout(updateMeasurement, 320)
        return () => clearTimeout(timer)
      }
    }
  }, [tourActive, tourIndex, steps, updateMeasurement])

  // Keyboard navigation (Escape = close, Arrows = next/prev)
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

    // Mark current step as completed
    if (currentStep && !completedSteps.includes(currentStep.id)) {
      const updated = [...completedSteps, currentStep.id]
      setCompletedSteps(updated)
      try {
        localStorage.setItem(STORAGE_COMPLETED_STEPS, JSON.stringify(updated))
      } catch {
        // Ignore
      }
    }

    // If step belongs to another section, navigate first
    if (targetStep?.section && onNavigate && isLoggedIn) {
      onNavigate(targetStep.section)
      setTimeout(() => setTourIndex(nextIdx), 120)
    } else {
      setTourIndex(nextIdx)
    }
  }

  function startTourFrom(index: number = 0) {
    setDrawerOpen(false)
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
  const tooltipStyle = computeTooltipPosition(targetRect, currentStep.placement)

  // Calculate checklist progress
  const actionSteps = steps.filter(s => s.target !== null)
  const completedCount = actionSteps.filter(s => completedSteps.includes(s.id)).length
  const progressPercent = actionSteps.length > 0 ? Math.round((completedCount / actionSteps.length) * 100) : 0

  return (
    <>
      {/* --------------------------------------------------------------------
          1. Floating Guide Bot Trigger Pill (Always available)
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
              <span className="guide-bot-sub">{isLoggedIn ? `${progressPercent}% completed` : 'Step-by-step help'}</span>
            </span>
            <span className="guide-bot-badge">{isLoggedIn ? `${completedCount}/${actionSteps.length}` : 'Start'}</span>
          </button>
        </aside>
      )}

      {/* --------------------------------------------------------------------
          2. Step-by-Step Interactive Guide Drawer / Panel
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
              <h3 id="guide-drawer-heading">Step-by-Step Workflow Guide</h3>
              <p className="guide-drawer-desc">
                {isLoggedIn
                  ? 'Follow these six steps to establish reproducible RAG quality benchmarks.'
                  : 'Welcome to RAG_EVAL. Follow these steps to sign in and begin evaluating.'}
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

          {/* Quick Launch Interactive Tour Banner */}
          <div className="guide-tour-cta-card">
            <div className="guide-tour-cta-content">
              <h4>Interactive Spotlight Tour</h4>
              <p>Let the Guide Bot walk you through every UI element on screen.</p>
            </div>
            <button
              type="button"
              className="button primary guide-launch-tour-btn"
              onClick={() => startTourFrom(0)}
            >
              Start Interactive Tour →
            </button>
          </div>

          {/* Progress Bar */}
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

          {/* Steps Checklist */}
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
                          Show on screen ↗
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
          3. Interactive Spotlight Tour Mode
          -------------------------------------------------------------------- */}
      {tourActive && (
        <div className="tour-layer" role="dialog" aria-modal="true" aria-label="Guided Tour">
          {/* Spotlight overlay with cut-out hole */}
          <SpotlightOverlay rect={targetRect} />

          {/* Clickable backdrop dismiss */}
          <div
            className="tour-shield"
            onClick={() => (isLast ? endTour() : stepTo(tourIndex + 1))}
            aria-hidden="true"
          />

          {/* Floating Tooltip Card */}
          <aside
            className={`tour-card tour-placement-${currentStep.placement}`}
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
