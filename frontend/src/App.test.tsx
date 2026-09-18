import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const websiteResult = {
  status: 'success', document_id: '11111111-1111-4111-8111-111111111111', url: 'https://example.com/', chunks: 8,
  policies: { privacy_policy: ['https://example.com/privacy'] },
  accepted_policies: [{ source_url: 'https://example.com/privacy', source_urls: ['https://example.com/privacy'], policy_categories: ['privacy_policy'], content_hash: 'hash', chunks: 8 }],
  warnings: [], skipped: {}, candidates: 1, coverage_status: 'policies_found', fallback_used: 'none', fallback: {}, recovery_options: [],
}
const pdfResult = { status: 'success', document_id: '22222222-2222-4222-8222-222222222222', filename: 'policy.pdf', chunks: 3 }
const textResult = { status: 'success', document_id: '33333333-3333-4333-8333-333333333333', source_type: 'text', title: 'Pasted text', chunks: 2 }

function jsonResponse(payload: unknown, status = 200, headers?: HeadersInit) {
  return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json', ...headers } })
}

async function submitWebsite(result: Record<string, unknown> = websiteResult) {
  const user = userEvent.setup()
  vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(result))
  await user.type(screen.getByLabelText(/website or direct policy url/i), 'https://example.com')
  await user.click(screen.getByRole('button', { name: /analyze website/i }))
  await screen.findByRole('heading', { name: 'example.com' })
  return user
}

describe('PolicyCue application', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
    render(<App />)
  })

  it('renders the Website tab by default and switches between all accessible tabs', async () => {
    const user = userEvent.setup()
    expect(screen.getByRole('tab', { name: 'Website' })).toHaveAttribute('aria-selected', 'true')
    await user.click(screen.getByRole('tab', { name: 'PDF' }))
    expect(screen.getByLabelText('Choose PDF file')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: 'Paste Text' }))
    expect(screen.getByLabelText('Policy text')).toBeInTheDocument()
  })

  it('supports keyboard arrow navigation between tabs', async () => {
    const websiteTab = screen.getByRole('tab', { name: 'Website' })
    websiteTab.focus()
    fireEvent.keyDown(websiteTab, { key: 'ArrowRight' })
    expect(screen.getByRole('tab', { name: 'PDF' })).toHaveFocus()
  })

  it('validates malformed website URLs before making a request', async () => {
    const user = userEvent.setup()
    await user.type(screen.getByLabelText(/website or direct policy url/i), 'example.com')
    await user.click(screen.getByRole('button', { name: /analyze website/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/complete http or https url/i)
    expect(fetch).not.toHaveBeenCalled()
  })

  it('uploads a PDF and transitions to its analysis workspace', async () => {
    const user = userEvent.setup()
    await user.click(screen.getByRole('tab', { name: 'PDF' }))
    await user.upload(screen.getByLabelText('Choose PDF file'), new File(['%PDF'], 'policy.pdf', { type: 'application/pdf' }))
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(pdfResult))
    await user.click(screen.getByRole('button', { name: 'Analyze PDF' }))
    expect(await screen.findByRole('heading', { name: 'policy.pdf' })).toBeInTheDocument()
  })

  it('rejects unsupported and oversized PDF files in the browser', async () => {
    const user = userEvent.setup()
    await user.click(screen.getByRole('tab', { name: 'PDF' }))
    const input = screen.getByLabelText('Choose PDF file')
    fireEvent.change(input, { target: { files: [new File(['text'], 'notes.txt', { type: 'text/plain' })] } })
    expect(screen.getByRole('alert')).toHaveTextContent('Choose a PDF file')
    const oversized = new File(['x'], 'large.pdf', { type: 'application/pdf' })
    Object.defineProperty(oversized, 'size', { value: 10 * 1024 * 1024 + 1 })
    fireEvent.change(input, { target: { files: [oversized] } })
    expect(screen.getByRole('alert')).toHaveTextContent('10 MB or smaller')
  })

  it('enforces the pasted-text minimum and maximum without storing private text', async () => {
    const storage = vi.spyOn(Storage.prototype, 'setItem')
    const user = userEvent.setup()
    await user.click(screen.getByRole('tab', { name: 'Paste Text' }))
    await user.type(screen.getByLabelText('Policy text'), 'too short')
    await user.click(screen.getByRole('button', { name: 'Analyze text' }))
    expect(screen.getByRole('alert')).toHaveTextContent('50 non-whitespace')
    fireEvent.change(screen.getByLabelText('Policy text'), { target: { value: 'x'.repeat(100_001) } })
    expect(screen.getByText('100,001 / 100,000')).toHaveClass('counter-over')
    expect(screen.getByRole('button', { name: 'Analyze text' })).toBeDisabled()
    expect(storage).not.toHaveBeenCalled()
  })

  it('submits valid pasted text and uses the returned title', async () => {
    const user = userEvent.setup()
    await user.click(screen.getByRole('tab', { name: 'Paste Text' }))
    await user.type(screen.getByLabelText('Policy text'), 'This is a sufficiently detailed policy statement describing user responsibilities and account controls.')
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(textResult))
    await user.click(screen.getByRole('button', { name: 'Analyze text' }))
    expect(await screen.findByRole('heading', { name: 'Pasted text' })).toBeInTheDocument()
  })

  it.each([
    ['partial', 'Partial policy coverage'],
    ['access_limited', 'Website access was limited'],
    ['no_policies_found', 'No relevant policies found'],
  ])('renders the %s website coverage state and recovery choices', async (coverage_status, heading) => {
    await submitWebsite({ ...websiteResult, status: coverage_status === 'partial' ? 'success' : 'no_policies_found', coverage_status, chunks: coverage_status === 'partial' ? 2 : 0, accepted_policies: coverage_status === 'partial' ? websiteResult.accepted_policies : [] })
    expect(screen.getByText(heading)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Paste text' })).toBeInTheDocument()
  })

  it.each([
    ['Direct policy URL', 'Website'],
    ['Paste text', 'Paste Text'],
    ['Upload PDF', 'PDF'],
  ])('deletes the active document before switching to %s', async (action, targetTab) => {
    const user = await submitWebsite({ ...websiteResult, coverage_status: 'partial', chunks: 2 })
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ status: 'deleted', document_id: websiteResult.document_id }))

    await user.click(screen.getByRole('button', { name: action }))

    const target = await screen.findByRole('tab', { name: targetTab })
    expect(target).toHaveAttribute('aria-selected', 'true')
    const deleteCalls = vi.mocked(fetch).mock.calls.filter(([url, init]) => String(url).includes(`/documents/${websiteResult.document_id}`) && (init as RequestInit).method === 'DELETE')
    expect(deleteCalls).toHaveLength(1)
    expect(deleteCalls[0][1]).toEqual(expect.objectContaining({ credentials: 'include' }))
    expect(screen.queryByRole('heading', { name: 'example.com' })).not.toBeInTheDocument()
  })

  it('keeps the workspace in place when a recovery deletion fails', async () => {
    const user = await submitWebsite({ ...websiteResult, coverage_status: 'partial', chunks: 2 })
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ detail: 'Document deletion is temporarily unavailable. Please retry later.' }, 503))

    await user.click(screen.getByRole('button', { name: 'Upload PDF' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/temporarily unavailable/i)
    expect(screen.getByRole('heading', { name: 'example.com' })).toBeInTheDocument()
    expect(screen.queryByLabelText('Choose PDF file')).not.toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/documents/'))).toHaveLength(1)
  })

  it('asks a question and displays genuine, safe source attribution', async () => {
    const user = await submitWebsite()
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ answer: 'The service collects account details.', document_found: true, sources: [{ text: 'Account details', chunk_index: 2 }], source_attributions: [{ source_url: 'https://example.com/privacy', filename: '', policy_categories: ['privacy_policy'] }] }))
    await user.type(screen.getByLabelText('Ask about this analysis'), 'What is collected?')
    await user.click(screen.getByRole('button', { name: /ask question/i }))
    expect(await screen.findByText('The service collects account details.')).toBeInTheDocument()
    const source = screen.getByRole('link', { name: /open source/i })
    expect(source).toHaveAttribute('href', 'https://example.com/privacy')
    expect(source).toHaveAttribute('target', '_blank')
    expect(source).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('keeps the exact ingested document for URL questions and never deletes on workspace mount or ask', async () => {
    const user = await submitWebsite()
    expect(vi.mocked(fetch).mock.calls.some(([url, init]) => String(url).includes('/documents/') || (init as RequestInit | undefined)?.method === 'DELETE')).toBe(false)
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ answer: 'The policy applies to account data.', document_found: true, sources: [{ text: 'Account data', chunk_index: 0 }], source_attributions: [] }))
    await user.type(screen.getByLabelText('Ask about this analysis'), 'What does it apply to?')
    await user.click(screen.getByRole('button', { name: /ask question/i }))
    const askCall = vi.mocked(fetch).mock.calls.at(-1)
    if (!askCall) throw new Error('Expected an ask request.')
    const askRequest = askCall[1] as RequestInit
    expect(JSON.parse(String(askRequest.body))).toMatchObject({ document_id: websiteResult.document_id })
    expect(askRequest.credentials).toBe('include')
    expect(screen.getByText('The policy applies to account data.')).toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/documents/'))).toHaveLength(0)
  })

  it('uses the retained PDF document ID when asking', async () => {
    const user = userEvent.setup()
    await user.click(screen.getByRole('tab', { name: 'PDF' }))
    await user.upload(screen.getByLabelText('Choose PDF file'), new File(['%PDF'], 'policy.pdf', { type: 'application/pdf' }))
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(pdfResult))
    await user.click(screen.getByRole('button', { name: 'Analyze PDF' }))
    await screen.findByRole('heading', { name: 'policy.pdf' })
    const noAnswer = 'I could not find that information in the provided document.'
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ answer: noAnswer, document_found: true, sources: [], source_attributions: [] }))
    await user.type(screen.getByLabelText('Ask about this analysis'), 'What is missing?')
    await user.click(screen.getByRole('button', { name: /ask question/i }))
    const askCall = vi.mocked(fetch).mock.calls.at(-1)
    if (!askCall) throw new Error('Expected an ask request.')
    expect(JSON.parse(String((askCall[1] as RequestInit).body))).toMatchObject({ document_id: pdfResult.document_id })
    expect(screen.getByText(noAnswer)).toBeInTheDocument()
  })

  it('uses the retained pasted-text document ID when asking', async () => {
    const user = userEvent.setup()
    await user.click(screen.getByRole('tab', { name: 'Paste Text' }))
    await user.type(screen.getByLabelText('Policy text'), 'This policy explains account deletion requests, data use, and service responsibilities in enough detail to analyze.')
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(textResult))
    await user.click(screen.getByRole('button', { name: 'Analyze text' }))
    await screen.findByRole('heading', { name: 'Pasted text' })
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ answer: 'Users can request account deletion.', document_found: true, sources: [], source_attributions: [] }))
    await user.type(screen.getByLabelText('Ask about this analysis'), 'What can users request?')
    await user.click(screen.getByRole('button', { name: /ask question/i }))
    const askCall = vi.mocked(fetch).mock.calls.at(-1)
    if (!askCall) throw new Error('Expected an ask request.')
    expect(JSON.parse(String((askCall[1] as RequestInit).body))).toMatchObject({ document_id: textResult.document_id })
  })

  it('presents an unsupported answer as a grounded result without fake sources', async () => {
    const user = await submitWebsite()
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ answer: 'I could not find that information in the provided document.', document_found: true, sources: [], source_attributions: [] }))
    await user.type(screen.getByLabelText('Ask about this analysis'), 'What is the weather?')
    await user.click(screen.getByRole('button', { name: /ask question/i }))
    expect(await screen.findByText('I could not find that information in the provided document.')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Sources' })).not.toBeInTheDocument()
    expect(screen.getByText(/stays within the analyzed material/i)).toBeInTheDocument()
  })

  it.each([
    [404, {}, 'no longer available'],
    [429, { 'Retry-After': '12' }, 'about 12 seconds'],
    [503, {}, 'Policy storage failed'],
  ])('shows a sanitized error for HTTP %s', async (status, headers, expected) => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ detail: status === 404 ? 'Document not found.' : status === 429 ? 'Too many requests. Please retry later.' : 'Policy storage failed. Please retry ingestion.' }, status, headers))
    const user = userEvent.setup()
    await user.type(screen.getByLabelText(/website or direct policy url/i), 'https://example.com')
    await user.click(screen.getByRole('button', { name: /analyze website/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(new RegExp(expected, 'i'))
  })

  it('replaces the current analysis only after its owner-scoped deletion succeeds', async () => {
    const user = await submitWebsite()
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ status: 'deleted', document_id: websiteResult.document_id }))
    await user.click(screen.getByRole('button', { name: 'New analysis' }))
    expect(await screen.findByLabelText(/website or direct policy url/i)).toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.at(-1)?.[1]).toEqual(expect.objectContaining({ method: 'DELETE' }))
  })

  it('keeps the active analysis when replacement deletion fails', async () => {
    const user = await submitWebsite()
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ detail: 'Policy deletion failed. Please retry.' }, 503))
    await user.click(screen.getByRole('button', { name: 'New analysis' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Policy deletion failed/i)
    expect(screen.getByRole('heading', { name: 'example.com' })).toBeInTheDocument()
  })

  it('opens the keyboard-accessible mobile menu and closes it after navigation', async () => {
    const user = userEvent.setup()
    const button = screen.getByRole('button', { name: 'Open navigation menu' })
    await user.click(button)
    expect(button).toHaveAttribute('aria-expanded', 'true')
    await user.click(within(screen.getByRole('navigation', { name: 'Mobile navigation' })).getByRole('button', { name: 'Features' }))
    await waitFor(() => expect(screen.queryByRole('navigation', { name: 'Mobile navigation' })).not.toBeInTheDocument())
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled()
  })

  it('uses an accessible FAQ accordion with matching visible answers', async () => {
    const user = userEvent.setup()
    const question = screen.getByRole('button', { name: /what can policycue analyze/i })
    expect(question).toHaveAttribute('aria-expanded', 'false')
    await user.click(question)
    expect(question).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText(/PDF files up to 10 MB/i)).toBeInTheDocument()
  })
})
