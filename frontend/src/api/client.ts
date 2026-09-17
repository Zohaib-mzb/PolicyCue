import type {
  AnswerResponse,
  DeleteResponse,
  PdfIngestion,
  TextIngestion,
  WebsiteIngestion,
} from './types'

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
// Keep the Vite and API origins on localhost in development.  `localhost` and
// `127.0.0.1` are different sites for SameSite cookies, so mixing them loses
// the server-issued anonymous owner session between ingestion and /ask.
export const API_BASE_URL = (configuredBaseUrl || 'http://localhost:8001').replace(/\/$/, '')

export class ApiError extends Error {
  status: number
  retryAfter: number | null

  constructor(message: string, status: number, retryAfter: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.retryAfter = retryAfter
  }
}

function endpoint(path: string) {
  return `${API_BASE_URL}${path}`
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    try {
      return (await response.json()) as T
    } catch {
      throw new ApiError('The server returned an unreadable response.', 502)
    }
  }

  let detail = ''
  try {
    const payload = (await response.json()) as { detail?: unknown }
    if (typeof payload.detail === 'string') detail = payload.detail
  } catch {
    // The safe status-based fallback below handles malformed error bodies.
  }

  const retryHeader = response.headers.get('Retry-After')
  const parsedRetry = retryHeader ? Number.parseInt(retryHeader, 10) : Number.NaN
  throw new ApiError(
    detail || defaultErrorMessage(response.status),
    response.status,
    Number.isFinite(parsedRetry) ? parsedRetry : null,
  )
}

function defaultErrorMessage(status: number) {
  if (status === 404) return 'This analysis is no longer available.'
  if (status === 429) return 'Too many requests. Please wait a moment and try again.'
  if (status === 503) return 'PolicyCue is temporarily unavailable. Please try again.'
  return 'Something went wrong. Please try again.'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  try {
    const response = await fetch(endpoint(path), {
      ...init,
      credentials: 'include',
    })
    return await parseResponse<T>(response)
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('Could not reach PolicyCue. Check your connection and try again.', 0)
  }
}

export function ingestWebsite(url: string, signal?: AbortSignal) {
  return request<WebsiteIngestion>('/api/v1/ingest/url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
    signal,
  })
}

export function getHealth() {
  return request<{ status: 'ok' }>('/health')
}

export function ingestPdf(file: File, signal?: AbortSignal) {
  const body = new FormData()
  body.append('file', file)
  return request<PdfIngestion>('/api/v1/ingest/pdf', { method: 'POST', body, signal })
}

export function ingestText(text: string, title?: string, signal?: AbortSignal) {
  return request<TextIngestion>('/api/v1/ingest/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, title: title?.trim() || null }),
    signal,
  })
}

export function askQuestion(documentId: string, question: string) {
  return request<AnswerResponse>('/api/v1/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId, question, top_k: 5 }),
  })
}

export function deleteAnalysis(documentId: string) {
  return request<DeleteResponse>(`/api/v1/documents/${encodeURIComponent(documentId)}`, {
    method: 'DELETE',
  })
}
