import { useState, useEffect } from 'react'
import type { ReactNode } from 'react'
import {
  DocumentIcon,
  AskIcon,
  DatasetIcon,
  RunsIcon,
  ExperimentIcon,
  ShieldCheckIcon,
  SparklesIcon,
} from './icons'

type Section = 'Corpus' | 'Ask' | 'Datasets' | 'Runs' | 'Experiments'

interface GuideStep {
  id: number
  title: string
  subtitle: string
  section?: Section
  icon: typeof DocumentIcon
  content: ReactNode
  actionLabel?: string
}

interface GuideBotProps {
  currentSection?: Section
  onNavigate?: (section: Section) => void
  isLoggedIn: boolean
}

export function GuideBot({ currentSection, onNavigate, isLoggedIn }: GuideBotProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [completedSteps, setCompletedSteps] = useState<number[]>([])

  // Load saved state from localStorage
  useEffect(() => {
    try {
      const savedCompleted = localStorage.getItem('rag_eval_guide_completed')
      if (savedCompleted) {
        setCompletedSteps(JSON.parse(savedCompleted) as number[])
      }
      const savedStep = localStorage.getItem('rag_eval_guide_step')
      if (savedStep) {
        const stepNum = parseInt(savedStep, 10)
        if (!isNaN(stepNum) && stepNum >= 0 && stepNum < 6) {
          setCurrentStepIndex(stepNum)
        }
      }
    } catch {
      // Ignore localStorage errors
    }
  }, [])

  // Sync step with current active section when navigating in app
  useEffect(() => {
    if (!currentSection) return
    const sectionToStep: Record<Section, number> = {
      Corpus: 1,
      Datasets: 2,
      Ask: 3,
      Runs: 4,
      Experiments: 5,
    }
    const targetStep = sectionToStep[currentSection]
    if (targetStep !== undefined && targetStep !== currentStepIndex) {
      // Optionally update step or keep user choice
    }
  }, [currentSection, currentStepIndex])

  function toggleComplete(stepId: number) {
    const updated = completedSteps.includes(stepId)
      ? completedSteps.filter(id => id !== stepId)
      : [...completedSteps, stepId]
    setCompletedSteps(updated)
    try {
      localStorage.setItem('rag_eval_guide_completed', JSON.stringify(updated))
    } catch {
      // Ignore
    }
  }

  function goToStep(index: number) {
    setCurrentStepIndex(index)
    try {
      localStorage.setItem('rag_eval_guide_step', index.toString())
    } catch {
      // Ignore
    }
  }

  const steps: GuideStep[] = [
    {
      id: 0,
      title: 'Step 1: Environment & .env Setup',
      subtitle: 'Configure local encryption, database, and model connections',
      icon: ShieldCheckIcon,
      content: (
        <div className="guide-content-block">
          <p className="guide-p">
            RAG_EVAL requires a few key environment variables in your <code>.env</code> file to run smoothly:
          </p>
          <ul className="guide-ul">
            <li>
              <strong>CREDENTIAL_ENCRYPTION_KEY:</strong> A 32-byte Fernet key used to encrypt provider keys (Gemini, Groq) in SQLite.
            </li>
            <li>
              <strong>DATABASE_URL:</strong> <code>sqlite:///./rag_eval_hosted.db</code> for simple local storage, or PostgreSQL for teams.
            </li>
            <li>
              <strong>COOKIE_SECURE:</strong> Set to <code>false</code> for local HTTP testing so authentication cookies persist.
            </li>
            <li>
              <strong>AUTH_DEV_MODE:</strong> Set to <code>true</code> to enable instant sign-in with any email via terminal log link.
            </li>
            <li>
              <strong>OLLAMA_BASE_URL:</strong> Default <code>http://localhost:11434</code> for private local embeddings.
            </li>
          </ul>
          <div className="guide-tip-box">
            <SparklesIcon />
            <div>
              <strong>Quick Start Tip:</strong> An active <code>.env</code> file has already been generated in your workspace with valid Fernet keys and dev authentication enabled!
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 1,
      title: 'Step 2: Source Corpus Ingestion',
      subtitle: 'Upload Markdown or plain text documents up to 25 MiB',
      section: 'Corpus',
      icon: DocumentIcon,
      actionLabel: 'Go to Corpus Upload',
      content: (
        <div className="guide-content-block">
          <p className="guide-p">
            Upload your technical documents, specifications, or knowledge base files:
          </p>
          <ul className="guide-ul">
            <li>Accepted formats: <code>.md</code> (Markdown) and <code>.txt</code> (UTF-8).</li>
            <li>
              <strong>Deterministic Element Extraction:</strong> Documents are automatically parsed into discrete, stable source elements (<code>elem_...</code>).
            </li>
            <li>
              <strong>Why This Matters:</strong> When you modify chunking sizes or overlap later, the source element UUIDs remain identical, preserving your ground-truth labels forever.
            </li>
          </ul>
          <div className="guide-tip-box">
            <strong>Action:</strong> Click on any document in the table to inspect its individual extracted elements.
          </div>
        </div>
      ),
    },
    {
      id: 2,
      title: 'Step 3: Ground-Truth Dataset Labeling',
      subtitle: 'Map evaluation questions directly to source elements',
      section: 'Datasets',
      icon: DatasetIcon,
      actionLabel: 'Go to Datasets Workbench',
      content: (
        <div className="guide-content-block">
          <p className="guide-p">
            Build immutable test datasets with verified ground truth:
          </p>
          <ol className="guide-ol">
            <li>Name your dataset (e.g., <code>core-architecture-r1</code>).</li>
            <li>Write a realistic query a user would ask your RAG pipeline.</li>
            <li>Check the checkboxes for the exact source elements that contain the factual answer.</li>
            <li>Click <strong>Create revision</strong> to persist an immutable revision (<code>r1</code>).</li>
          </ol>
          <div className="guide-tip-box">
            <strong>Key Benefit:</strong> Revisions are content-addressed with SHA-256 digests. You can never accidentally alter benchmark test sets retroactively.
          </div>
        </div>
      ),
    },
    {
      id: 3,
      title: 'Step 4: Interactive Querying & Evidence',
      subtitle: 'Inspect real-time cosine scores and citations',
      section: 'Ask',
      icon: AskIcon,
      actionLabel: 'Go to Ask Workbench',
      content: (
        <div className="guide-content-block">
          <p className="guide-p">
            Test how the vector store retrieves evidence for individual queries:
          </p>
          <ul className="guide-ul">
            <li>Type any question in the prompt textarea.</li>
            <li>Select your Chat Provider (Server-managed Ollama, Gemini, or Groq).</li>
            <li>Review retrieved passages ranked by <strong>Cosine Similarity</strong>.</li>
            <li>Inspect visual similarity gauges (e.g. <code>0.920</code> strong match).</li>
          </ul>
          <div className="guide-tip-box">
            <strong>Privacy Note:</strong> Embeddings are computed locally via Ollama. Only synthesis prompts are sent to Gemini/Groq if selected.
          </div>
        </div>
      ),
    },
    {
      id: 4,
      title: 'Step 5: Quantitative Benchmark Runs',
      subtitle: 'Evaluate retrieval precision @ 1, 3, 5 and Mean Reciprocal Rank',
      section: 'Runs',
      icon: RunsIcon,
      actionLabel: 'Go to Evaluation Runs',
      content: (
        <div className="guide-content-block">
          <p className="guide-p">
            Queue automated batch evaluations against your dataset revisions:
          </p>
          <ul className="guide-ul">
            <li>
              <strong>Precision @ 1:</strong> Percentage of top-1 results that are relevant ground-truth elements.
            </li>
            <li>
              <strong>Precision @ 5:</strong> Proportion of top-5 retrieved passages that are relevant.
            </li>
            <li>
              <strong>Recall @ 5:</strong> Proportion of all labeled ground-truth elements discovered in the top 5.
            </li>
            <li>
              <strong>Citation Audit:</strong> Flags any hallucinated citation references (<code>[S1]</code>) not grounded in retrieved excerpts.
            </li>
          </ul>
        </div>
      ),
    },
    {
      id: 5,
      title: 'Step 6: Paired A/B Hypothesis Testing',
      subtitle: 'Compare pipeline variants with 1,000 bootstrap resamples',
      section: 'Experiments',
      icon: ExperimentIcon,
      actionLabel: 'Go to Experiments Workbench',
      content: (
        <div className="guide-content-block">
          <p className="guide-p">
            Statistically prove whether a retrieval configuration actually improves performance:
          </p>
          <ul className="guide-ul">
            <li>Configure Baseline (e.g. <code>depth-3</code>) and Candidate (e.g. <code>depth-5</code>).</li>
            <li>RAG_EVAL runs paired queries on the identical dataset revision.</li>
            <li>Executes <strong>1,000 bootstrap resamples</strong> with fixed seed 0.</li>
            <li>
              Reports empirical <strong>95% Confidence Intervals</strong> (<code>[Δ_low, Δ_high]</code>) to verify if the improvement is statistically significant without random noise.
            </li>
          </ul>
        </div>
      ),
    },
  ]

  const currentStep = steps[currentStepIndex]
  const progressPercent = Math.round((completedSteps.length / steps.length) * 100)

  return (
    <aside className="guide-bot-wrapper" aria-label="Workflow Guide Assistant">
      {!isOpen ? (
        <button
          type="button"
          className="guide-bot-trigger"
          onClick={() => setIsOpen(true)}
          aria-expanded="false"
        >
          <span className="guide-trigger-icon" aria-hidden="true">⚡</span>
          <span className="guide-trigger-text">Evaluation Guide</span>
          <span className="guide-step-pill">
            {completedSteps.length}/{steps.length} done
          </span>
        </button>
      ) : (
        <div className="guide-bot-panel" role="dialog" aria-modal="false" aria-labelledby="guide-bot-title">
          {/* Header */}
          <header className="guide-bot-header">
            <div className="guide-header-title">
              <span className="guide-badge">WORKFLOW GUIDE</span>
              <h3 id="guide-bot-title">Evaluation Workflow Assistant</h3>
            </div>
            <button
              type="button"
              className="guide-close-btn"
              onClick={() => setIsOpen(false)}
              aria-label="Close guide"
            >
              ×
            </button>
          </header>

          {/* Progress Strip */}
          <div className="guide-progress-strip">
            <div className="guide-progress-label">
              <span>{progressPercent}% Completed</span>
              <span>Step {currentStepIndex + 1} of {steps.length}</span>
            </div>
            <div className="guide-progress-bar">
              <div className="guide-progress-fill" style={{ width: `${progressPercent}%` }} />
            </div>
          </div>

          {/* Step Selector Tabs */}
          <nav className="guide-steps-nav" aria-label="Guide steps">
            {steps.map((s, idx) => {
              const isCurrent = idx === currentStepIndex
              const isCompleted = completedSteps.includes(s.id)
              return (
                <button
                  key={s.id}
                  type="button"
                  className={`guide-step-tab ${isCurrent ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
                  onClick={() => goToStep(idx)}
                  title={s.title}
                >
                  <span className="tab-num">{isCompleted ? '✓' : idx + 1}</span>
                  <span className="tab-title-short">Step {idx + 1}</span>
                </button>
              )
            })}
          </nav>

          {/* Active Step Card */}
          <div className="guide-step-body">
            <div className="step-card-header">
              <h4>{currentStep.title}</h4>
              <p className="step-subtitle">{currentStep.subtitle}</p>
            </div>

            <div className="step-content-area">{currentStep.content}</div>

            {/* Action Bar */}
            <div className="guide-step-footer">
              <label className="guide-complete-checkbox">
                <input
                  type="checkbox"
                  checked={completedSteps.includes(currentStep.id)}
                  onChange={() => toggleComplete(currentStep.id)}
                />
                <span>Mark step as completed</span>
              </label>

              {currentStep.section && onNavigate && isLoggedIn && (
                <button
                  type="button"
                  className="button primary guide-action-btn"
                  onClick={() => {
                    if (currentStep.section) {
                      onNavigate(currentStep.section)
                    }
                  }}
                >
                  {currentStep.actionLabel || `Open ${currentStep.section}`}
                </button>
              )}
            </div>
          </div>

          {/* Pagination Controls */}
          <footer className="guide-nav-footer">
            <button
              type="button"
              className="quiet-button"
              disabled={currentStepIndex === 0}
              onClick={() => goToStep(currentStepIndex - 1)}
            >
              ← Previous
            </button>
            <div className="guide-quick-diag">
              <span className="diag-pill online">● .env active</span>
              <span className="diag-pill">Fernet key ready</span>
            </div>
            <button
              type="button"
              className="quiet-button"
              disabled={currentStepIndex === steps.length - 1}
              onClick={() => goToStep(currentStepIndex + 1)}
            >
              Next →
            </button>
          </footer>
        </div>
      )}
    </aside>
  )
}

