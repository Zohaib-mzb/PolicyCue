import { type ChangeEvent, type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { ApiError, askQuestion, deleteAnalysis, ingestPdf, ingestText, ingestWebsite } from './api/client'
import type { ActiveAnalysis, AnalysisType, AnswerResponse, CoverageStatus, SourceAttribution, WebsiteIngestion } from './api/types'
import './App.css'

const NO_ANSWER = 'I could not find that information in the provided document.'
const MAX_TEXT_LENGTH = 100_000
const MIN_TEXT_USEFUL_CHARACTERS = 50
const MAX_PDF_SIZE = 10 * 1024 * 1024
const tabs: { id: AnalysisType; label: string }[] = [
  { id: 'website', label: 'Website' }, { id: 'pdf', label: 'PDF' }, { id: 'text', label: 'Paste Text' },
]
const suggestions = ['What personal data is collected?', 'Can I delete my account?', 'What is the refund policy?', 'Can my content be removed?']
const features = [
  ['Website policy discovery', 'Analyze accessible policy pages from a website homepage or a direct policy URL.'],
  ['PDF and text analysis', 'Upload a policy PDF or paste policy text directly into the analyzer.'],
  ['Grounded AI answers', 'Ask natural-language questions answered only from the material PolicyCue analyzed.'],
  ['Source attribution', 'Review the policy page, document, or text source supporting each answer.'],
  ['Honest abstention', "When the answer isn't in the analyzed corpus, PolicyCue says so plainly."],
  ['Private session isolation', 'PDFs and pasted text are kept within your anonymous, server-controlled session.'],
]
const steps = [
  ['01', 'Add a source', 'Enter a website, upload a PDF, or paste policy text.'],
  ['02', 'PolicyCue analyzes it', 'Relevant content is discovered, processed, and prepared for retrieval.'],
  ['03', 'Ask questions', 'Ask natural-language questions about the analyzed policies.'],
  ['04', 'Verify the answer', 'Supported answers include source attribution so you can inspect the material.'],
]
const faqs = [
  ['What is PolicyCue?', 'PolicyCue is an AI policy analyzer that lets you ask questions about website policies, PDFs, and pasted policy text. It uses retrieval-augmented generation (RAG) to answer from the analyzed material.'],
  ['What can PolicyCue analyze?', 'It can analyze accessible public policy pages on a website, direct privacy policy or terms URLs, PDF files up to 10 MB, and pasted policy text up to 100,000 characters.'],
  ['How does PolicyCue answer questions?', 'PolicyCue retrieves relevant passages from the analyzed source and asks the AI model to answer only when those passages contain enough evidence.'],
  ['Does PolicyCue use information outside the analyzed policies?', 'No. Answers are instructed to use only retrieved content from the current analysis, without an outside-knowledge fallback.'],
  ['What happens if PolicyCue cannot find an answer?', `It returns: “${NO_ANSWER}” This is a normal grounded result, not an application error.`],
  ['Why might a website show partial coverage?', 'Some policy pages can be blocked, unavailable, too large, or outside the same website host. PolicyCue reports partial coverage rather than claiming it found everything.'],
  ['What can I do if a website cannot be fully analyzed?', 'Try a direct policy URL, paste the relevant policy text, or upload the policy as a PDF.'],
  ['Can I analyze a direct privacy policy or terms URL?', 'Yes. The website input accepts both homepages and direct policy URLs when the page is publicly accessible.'],
  ['Is PolicyCue legal advice?', 'No. PolicyCue helps explain analyzed policy content. It does not provide legal advice or guarantee that a website corpus is complete.'],
  ['How are PDFs and pasted text handled?', 'They are associated with your anonymous signed session and are not stored in browser history or browser storage by this interface. You can delete the active analysis when finished.'],
] as const

const structuredData = {
  '@context': 'https://schema.org',
  '@graph': [
    {
      '@type': 'WebApplication',
      name: 'PolicyCue',
      applicationCategory: 'BusinessApplication',
      operatingSystem: 'Web',
      description: 'An AI policy analyzer for website policies, PDFs, and pasted policy text that provides grounded answers with source attribution.',
    },
    {
      '@type': 'FAQPage',
      mainEntity: faqs.map(([name, text]) => ({
        '@type': 'Question',
        name,
        acceptedAnswer: { '@type': 'Answer', text },
      })),
    },
  ],
}

function Logo() {
  return <span className="brand" aria-label="PolicyCue"><svg className="brand-mark" viewBox="0 0 36 36" aria-hidden="true"><path d="M9 4.5h13l5 5v22H9z"/><path d="M22 4.5v6h5M13.5 16h9M13.5 21h5"/><circle cx="23.5" cy="23.5" r="3.5"/><path d="m26 26 3.2 3.2"/></svg><span>PolicyCue</span></span>
}

function Header({ onAnalyze }: { onAnalyze: () => void }) {
  const [open, setOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16)
    onScroll(); window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])
  const visit = (id: string) => {
    setOpen(false); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    window.setTimeout(() => document.getElementById(id)?.focus({ preventScroll: true }), 350)
  }
  return <header className={`site-header ${scrolled ? 'is-scrolled' : ''}`}><div className="header-inner page-shell">
    <a className="logo-link" href="#top" aria-label="PolicyCue home" onClick={() => setOpen(false)}><Logo /></a>
    <nav className="desktop-nav" aria-label="Primary navigation"><a href="#features">Features</a><a href="#how-it-works">How it works</a><a href="#faq">FAQ</a><button className="button button-small" onClick={onAnalyze}>Analyze</button></nav>
    <button className={`menu-button ${open ? 'is-open' : ''}`} type="button" aria-label={open ? 'Close navigation menu' : 'Open navigation menu'} aria-expanded={open} aria-controls="mobile-navigation" onClick={() => setOpen((value) => !value)}><span/><span/><span/></button>
  </div><AnimatePresence>{open && <motion.nav id="mobile-navigation" className="mobile-nav" aria-label="Mobile navigation" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}><button onClick={() => visit('features')}>Features</button><button onClick={() => visit('how-it-works')}>How it works</button><button onClick={() => visit('faq')}>FAQ</button><button className="button" onClick={() => { setOpen(false); onAnalyze() }}>Analyze policies</button></motion.nav>}</AnimatePresence></header>
}

function friendlyError(error: unknown) {
  if (!(error instanceof ApiError)) return 'Something went wrong. Please try again.'
  if (error.status === 429 && error.retryAfter) return `Too many requests. Please try again in about ${error.retryAfter} seconds.`
  if (error.status === 404) return 'This analysis is no longer available. Start a new analysis to continue.'
  return error.message
}

function LoadingState({ type, onStop }: { type: AnalysisType; onStop: () => void }) {
  const [step, setStep] = useState(0)
  const messages = type === 'website' ? ['Connecting to website…', 'Discovering policy pages…', 'Analyzing available policy content…', 'Preparing grounded retrieval…'] : type === 'pdf' ? ['Reading the PDF…', 'Preparing policy passages…', 'Building the analysis…'] : ['Preparing the text…', 'Building policy passages…', 'Creating the analysis…']
  useEffect(() => { const timer = window.setInterval(() => setStep((value) => Math.min(value + 1, messages.length - 1)), 2200); return () => window.clearInterval(timer) }, [messages.length])
  return <div className="loading-state" role="status" aria-live="polite"><span className="spinner" aria-hidden="true"/><div><strong>{messages[step]}</strong>{type === 'website' && <p>Protected websites can take a few minutes. You can keep this page open while PolicyCue works.</p>}</div><button className="text-button" type="button" onClick={onStop}>Stop waiting</button></div>
}

function Analyzer({ onComplete, initialTab }: { onComplete: (analysis: ActiveAnalysis) => void; initialTab: AnalysisType }) {
  const [tab, setTab] = useState<AnalysisType>(initialTab)
  const [url, setUrl] = useState(''), [file, setFile] = useState<File | null>(null), [title, setTitle] = useState(''), [text, setText] = useState('')
  const [dragging, setDragging] = useState(false), [error, setError] = useState(''), [loading, setLoading] = useState(false)
  const controller = useRef<AbortController | null>(null), fileInput = useRef<HTMLInputElement>(null)
  const usefulCharacters = text.replace(/\s/g, '').length
  const chooseTab = (next: AnalysisType) => { setTab(next); setError('') }
  const onTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault(); let next = index
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length
    if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length
    if (event.key === 'Home') next = 0
    if (event.key === 'End') next = tabs.length - 1
    chooseTab(tabs[next].id); document.getElementById(`tab-${tabs[next].id}`)?.focus()
  }
  const validateFile = (candidate: File | null) => {
    setError(''); if (!candidate) return setFile(null)
    if (candidate.type !== 'application/pdf' && !candidate.name.toLowerCase().endsWith('.pdf')) { setFile(null); return setError('Choose a PDF file.') }
    if (candidate.size > MAX_PDF_SIZE) { setFile(null); return setError('PDF files must be 10 MB or smaller.') }
    setFile(candidate)
  }
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setError('')
    if (tab === 'website') { try { new URL(url) } catch { return setError('Enter a complete http or https URL.') }; if (!/^https?:\/\//i.test(url)) return setError('Enter a complete http or https URL.') }
    if (tab === 'pdf' && !file) return setError('Choose a PDF to analyze.')
    if (tab === 'text' && usefulCharacters < MIN_TEXT_USEFUL_CHARACTERS) return setError('Paste at least 50 non-whitespace characters.')
    if (tab === 'text' && text.length > MAX_TEXT_LENGTH) return setError('Pasted text cannot exceed 100,000 characters.')
    controller.current = new AbortController(); setLoading(true)
    try {
      if (tab === 'website') { const result = await ingestWebsite(url.trim(), controller.current.signal); onComplete({ type: 'website', documentId: result.document_id, identity: result.url, chunks: result.chunks, result }) }
      else if (tab === 'pdf' && file) { const result = await ingestPdf(file, controller.current.signal); onComplete({ type: 'pdf', documentId: result.document_id, identity: result.filename || file.name, chunks: result.chunks, result }) }
      else { const result = await ingestText(text, title, controller.current.signal); onComplete({ type: 'text', documentId: result.document_id, identity: result.title, chunks: result.chunks, result }) }
    } catch (submissionError) {
      if (submissionError instanceof DOMException && submissionError.name === 'AbortError') setError('Stopped waiting for the response. The server may still finish processing this source.')
      else setError(friendlyError(submissionError))
    } finally { controller.current = null; setLoading(false) }
  }
  return <motion.div className="analyzer-card" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .18 }}>
    <div className="tabs" role="tablist" aria-label="Analysis source">{tabs.map((item, index) => <button id={`tab-${item.id}`} key={item.id} type="button" role="tab" aria-selected={tab === item.id} aria-controls={`panel-${item.id}`} tabIndex={tab === item.id ? 0 : -1} onClick={() => chooseTab(item.id)} onKeyDown={(event) => onTabKeyDown(event, index)}>{tab === item.id && <motion.span className="tab-indicator" layoutId="active-tab"/>}<span>{item.label}</span></button>)}</div>
    {loading ? <LoadingState type={tab} onStop={() => controller.current?.abort()}/> : <form onSubmit={submit} noValidate>
      {tab === 'website' && <div id="panel-website" role="tabpanel" aria-labelledby="tab-website" className="form-panel"><label htmlFor="website-url">Website or direct policy URL</label><div className="input-action-row"><input id="website-url" type="url" value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://example.com/privacy" autoComplete="url"/><button className="button" type="submit">Analyze website</button></div><p className="helper">PolicyCue analyzes accessible public policy content on the same website.</p></div>}
      {tab === 'pdf' && <div id="panel-pdf" role="tabpanel" aria-labelledby="tab-pdf" className="form-panel"><label className={`drop-zone ${dragging ? 'is-dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); validateFile(event.dataTransfer.files[0] || null) }}><input aria-label="Choose PDF file" ref={fileInput} type="file" accept="application/pdf,.pdf" onChange={(event: ChangeEvent<HTMLInputElement>) => validateFile(event.target.files?.[0] || null)}/><span className="upload-icon" aria-hidden="true">↑</span><strong>{file ? file.name : 'Drop a policy PDF here'}</strong><span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : 'or choose a file · PDF up to 10 MB'}</span></label>{file && <button className="text-button remove-file" type="button" onClick={() => { setFile(null); if (fileInput.current) fileInput.current.value = '' }}>Remove file</button>}<button className="button full-button" type="submit" disabled={!file}>Analyze PDF</button></div>}
      {tab === 'text' && <div id="panel-text" role="tabpanel" aria-labelledby="tab-text" className="form-panel"><label htmlFor="text-title">Title <span className="optional">Optional</span></label><input id="text-title" value={title} maxLength={255} onChange={(event) => setTitle(event.target.value)} placeholder="e.g. Marketplace policy"/><div className="textarea-label"><label htmlFor="policy-text">Policy text</label><span className={text.length > MAX_TEXT_LENGTH ? 'counter-over' : ''}>{text.length.toLocaleString()} / 100,000</span></div><textarea id="policy-text" value={text} onChange={(event) => setText(event.target.value)} rows={8} placeholder="Paste the policy text you want to analyze…" aria-describedby="text-help"/><p id="text-help" className="helper">At least 50 non-whitespace characters. Text over the limit is never silently truncated.</p><button className="button full-button" type="submit" disabled={text.length > MAX_TEXT_LENGTH}>Analyze text</button></div>}
      {error && <div className="form-error" role="alert">{error}</div>}
    </form>}
  </motion.div>
}

function CoverageNotice({ result, onRecovery }: { result: WebsiteIngestion; onRecovery: (type: AnalysisType) => void }) {
  const messages: Record<CoverageStatus, { title: string; body: string; tone: string }> = {
    policies_found: { title: 'Analysis ready', body: 'PolicyCue found and analyzed relevant policy content.', tone: 'success' },
    partial: { title: 'Partial policy coverage', body: 'PolicyCue analyzed the policy content it could access. Some policy pages or sections may not be included.', tone: 'warning' },
    access_limited: { title: 'Website access was limited', body: "PolicyCue couldn't retrieve enough public policy content from this website.", tone: 'warning' },
    no_policies_found: { title: 'No relevant policies found', body: 'No accessible content matched the supported policy categories.', tone: 'neutral' },
  }
  const notice = messages[result.coverage_status], needsRecovery = result.coverage_status !== 'policies_found'
  return <div className={`coverage-notice ${notice.tone}`} role="status"><div><strong>{notice.title}</strong><p>{notice.body}</p></div>{needsRecovery && <div className="recovery-actions" aria-label="Other ways to analyze"><button type="button" onClick={() => onRecovery('website')}>Direct policy URL</button><button type="button" onClick={() => onRecovery('text')}>Paste text</button><button type="button" onClick={() => onRecovery('pdf')}>Upload PDF</button></div>}</div>
}
function categoryLabel(value: string) { return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) }
function safeExternalUrl(value: string | undefined) {
  if (!value) return null
  try {
    const parsed = new URL(value)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed : null
  } catch { return null }
}
function websiteName(value: string) { return safeExternalUrl(value)?.hostname || 'Website analysis' }

function Sources({ sources }: { sources: SourceAttribution[] }) {
  if (!sources.length) return null
  return <section className="sources" aria-labelledby="sources-heading"><h3 id="sources-heading">Sources</h3><div className="source-list">{sources.map((source, index) => {
    const externalUrl = safeExternalUrl(source.source_url)
    const label = source.title || source.filename || externalUrl?.hostname || 'Analyzed text'
    return <motion.article className="source-card" key={`${source.source_url}-${source.filename}-${index}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}><span className="source-number">{String(index + 1).padStart(2, '0')}</span><div><strong>{label}</strong>{!!source.policy_categories?.length && <p>{source.policy_categories.map(categoryLabel).join(' · ')}</p>}{externalUrl && <a href={externalUrl.href} target="_blank" rel="noopener noreferrer">Open source <span aria-hidden="true">↗</span></a>}</div></motion.article>
  })}</div></section>
}

function Workspace({ analysis, onReset, onRecovery }: { analysis: ActiveAnalysis; onReset: () => void; onRecovery: (type: AnalysisType) => void }) {
  const [question, setQuestion] = useState(''), [answer, setAnswer] = useState<AnswerResponse | null>(null), [asking, setAsking] = useState(false), [deleting, setDeleting] = useState(false), [error, setError] = useState('')
  const website = analysis.type === 'website' ? analysis.result as WebsiteIngestion : null, canAsk = analysis.chunks > 0
  const ask = async (event: FormEvent) => { event.preventDefault(); if (!question.trim() || asking || !canAsk) return; setAsking(true); setError(''); try { setAnswer(await askQuestion(analysis.documentId, question.trim())); setQuestion('') } catch (requestError) { setError(friendlyError(requestError)); if (requestError instanceof ApiError && requestError.status === 404) setAnswer(null) } finally { setAsking(false) } }
  const beginNewAnalysis = async () => { setDeleting(true); setError(''); try { await deleteAnalysis(analysis.documentId); onReset() } catch (requestError) { setError(friendlyError(requestError)); setDeleting(false) } }
  return <motion.section className="workspace" aria-labelledby="workspace-title" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }}>
    <div className="workspace-header"><div className="source-identity"><span className="eyebrow">{analysis.type === 'website' ? 'Website analysis' : analysis.type === 'pdf' ? 'PDF analysis' : 'Text analysis'}</span><h2 id="workspace-title">{analysis.type === 'website' ? websiteName(analysis.identity) : analysis.identity}</h2><p>{analysis.chunks ? `${analysis.chunks.toLocaleString()} searchable passages` : 'No searchable passages were created'}</p></div><div className="workspace-actions"><button className="button button-secondary" type="button" onClick={beginNewAnalysis} disabled={deleting}>{deleting ? 'Removing analysis…' : 'New analysis'}</button></div></div>
    {website && <CoverageNotice result={website} onRecovery={onRecovery}/>} 
    {website && website.accepted_policies.length > 0 && <details className="policy-summary"><summary>{website.accepted_policies.length} accepted policy {website.accepted_policies.length === 1 ? 'page' : 'pages'}</summary><ul>{website.accepted_policies.map((policy) => { const externalUrl = safeExternalUrl(policy.source_url); return <li key={policy.source_url}>{externalUrl ? <a href={externalUrl.href} target="_blank" rel="noopener noreferrer">{policy.policy_categories.map(categoryLabel).join(', ')}</a> : <span>{policy.policy_categories.map(categoryLabel).join(', ')}</span>}<span>{policy.chunks} passages</span></li> })}</ul></details>}
    {canAsk && <div className="question-area"><form className="question-form" onSubmit={ask}><label htmlFor="question">Ask about this analysis</label><div className="question-input-row"><textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} rows={2} maxLength={2000} placeholder="What does this policy say about…?" onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }}/><button className="button ask-button" type="submit" disabled={!question.trim() || asking}>{asking ? 'Asking…' : 'Ask question'}<span aria-hidden="true">→</span></button></div></form><div className="suggestions" aria-label="Suggested questions">{suggestions.map((item) => <button type="button" key={item} onClick={() => setQuestion(item)}>{item}</button>)}</div></div>}
    {error && <div className="form-error workspace-error" role="alert">{error}</div>}
    <AnimatePresence mode="wait">{answer && <motion.div className={`answer-panel ${answer.answer === NO_ANSWER ? 'unsupported' : ''}`} key={answer.answer} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} aria-live="polite"><span className="eyebrow">Grounded answer</span><p className="answer-text">{answer.answer}</p>{answer.answer === NO_ANSWER ? <p className="grounding-note">PolicyCue stays within the analyzed material instead of filling gaps with outside knowledge.</p> : <Sources sources={answer.source_attributions || []}/>}</motion.div>}</AnimatePresence>
  </motion.section>
}

function FAQ() {
  const [open, setOpen] = useState<number | null>(0)
  return <section id="faq" className="section faq-section page-shell" tabIndex={-1}><div className="section-heading"><span className="eyebrow">FAQ</span><h2>Clear answers about the analyzer</h2><p>What PolicyCue does, what it cannot promise, and how to get the best result.</p></div><div className="faq-list">{faqs.map(([question, response], index) => { const expanded = open === index; return <div className="faq-item" key={question}><button type="button" aria-expanded={expanded} aria-controls={`faq-answer-${index}`} onClick={() => setOpen(expanded ? null : index)}><span>{question}</span><span aria-hidden="true">{expanded ? '−' : '+'}</span></button><AnimatePresence initial={false}>{expanded && <motion.div id={`faq-answer-${index}`} className="faq-answer" initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}><p>{response}</p></motion.div>}</AnimatePresence></div> })}</div></section>
}

function MarketingSections() {
  // A small viewport threshold avoids cards remaining hidden until a large part
  // of a short mobile viewport intersects; reduced-motion users receive static content.
  const reduceMotion = useReducedMotion(), reveal = reduceMotion ? {} : { initial: { opacity: 0, y: 18 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true, amount: .05 } }
  return <><section id="features" className="section page-shell" tabIndex={-1}><div className="section-heading"><span className="eyebrow">Focused policy analysis</span><h2>Evidence first, from source to answer</h2><p>PolicyCue is a RAG document analysis tool built for privacy policies, terms of service, and other policy material.</p></div><div className="feature-grid">{features.map(([title, body], index) => <motion.article className="feature-card" key={title} {...reveal} transition={{ delay: reduceMotion ? 0 : index * .04 }}><span className="feature-index">0{index + 1}</span><h3>{title}</h3><p>{body}</p></motion.article>)}</div></section><section id="how-it-works" className="section process-section" tabIndex={-1}><div className="page-shell"><div className="section-heading light"><span className="eyebrow">How it works</span><h2>From policy source to verifiable answer</h2></div><div className="steps">{steps.map(([number, title, body], index) => <motion.article key={number} {...reveal} transition={{ delay: reduceMotion ? 0 : index * .08 }}><span>{number}</span><h3>{title}</h3><p>{body}</p></motion.article>)}</div></div></section><FAQ/></>
}

function App() {
  const [analysis, setAnalysis] = useState<ActiveAnalysis | null>(null), [preferredTab, setPreferredTab] = useState<AnalysisType>('website')
  const analyzerRef = useRef<HTMLElement>(null)
  const scrollToAnalyzer = () => { analyzerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); window.setTimeout(() => document.getElementById('tab-website')?.focus({ preventScroll: true }), 350) }
  const reset = () => { setAnalysis(null); window.setTimeout(scrollToAnalyzer, 0) }
  const recover = (type: AnalysisType) => { setPreferredTab(type); reset() }
  return <div id="top"><script type="application/ld+json">{JSON.stringify(structuredData)}</script><Header onAnalyze={scrollToAnalyzer}/><main><section className="hero-section" aria-labelledby="hero-heading"><div className="hero-copy page-shell"><motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}><span className="hero-kicker"><span/> Ask policies. Get grounded answers.</span><h1 id="hero-heading">Understand policies without reading every page.</h1><p>PolicyCue analyzes website policies, PDFs, and pasted policy text, then answers your questions using evidence from the content it analyzed.</p></motion.div></div><section id="analyze" ref={analyzerRef} className="analyzer-wrap page-shell" tabIndex={-1} aria-label="Policy analyzer">{analysis ? <Workspace analysis={analysis} onReset={reset} onRecovery={recover}/> : <Analyzer initialTab={preferredTab} onComplete={setAnalysis}/>}</section></section><MarketingSections/></main><footer className="site-footer"><div className="page-shell footer-grid"><div><Logo/><p>Ask policies. Get grounded answers.</p></div><nav aria-label="Footer navigation"><a href="#features">Features</a><a href="#how-it-works">How it works</a><a href="#faq">FAQ</a><a href="#analyze">Analyze</a></nav><p className="disclaimer">PolicyCue helps users understand analyzed policy content and does not provide legal advice.</p></div></footer></div>
}
export default App
