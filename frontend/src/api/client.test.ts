import { beforeEach, describe, expect, it, vi } from 'vitest'
import { askQuestion, deleteAnalysis, getHealth, ingestPdf, ingestText, ingestWebsite, SESSION_HEADER_NAME } from './client'

function jsonResponse(payload: unknown, status = 200, headers?: HeadersInit) {
  return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json', ...headers } })
}

describe('API client', () => {
  beforeEach(() => vi.stubGlobal('fetch', vi.fn().mockImplementation(() => Promise.resolve(jsonResponse({})))))

  it('uses the real URL ingestion route and includes session credentials', async () => {
    await ingestWebsite('https://example.com')
    expect(fetch).toHaveBeenCalledWith(expect.stringMatching(/\/api\/v1\/ingest\/url$/), expect.objectContaining({ method: 'POST', credentials: 'include' }))
    expect(JSON.parse((fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body)).toEqual({ url: 'https://example.com' })
  })

  it('uses the backend health route', async () => {
    await getHealth()
    expect(fetch).toHaveBeenCalledWith(expect.stringMatching(/\/health$/), expect.objectContaining({ credentials: 'include' }))
  })

  it('sends PDF uploads as multipart without setting an incorrect content type', async () => {
    const file = new File(['pdf'], 'policy.pdf', { type: 'application/pdf' })
    await ingestPdf(file)
    const init = (fetch as ReturnType<typeof vi.fn>).mock.calls[0][1]
    expect(init.body).toBeInstanceOf(FormData)
    expect(init.headers).toBeUndefined()
    expect((init.body as FormData).get('file')).toBe(file)
  })

  it('uses the text, ask, and delete contracts exactly', async () => {
    await ingestText('a sufficiently long policy text', 'Policy')
    await askQuestion('doc-1', 'What is allowed?', 'signed-token')
    await deleteAnalysis('doc-1', 'signed-token')
    const calls = (fetch as ReturnType<typeof vi.fn>).mock.calls
    expect(calls[0][0]).toMatch(/\/api\/v1\/ingest\/text$/)
    expect(JSON.parse(calls[0][1].body)).toEqual({ text: 'a sufficiently long policy text', title: 'Policy' })
    expect(JSON.parse(calls[1][1].body)).toEqual({ document_id: 'doc-1', question: 'What is allowed?', top_k: 5 })
    expect(new Headers(calls[1][1].headers).get(SESSION_HEADER_NAME)).toBe('signed-token')
    expect(calls[2][0]).toMatch(/\/api\/v1\/documents\/doc-1$/)
    expect(calls[2][1]).toEqual(expect.objectContaining({ method: 'DELETE', credentials: 'include' }))
    expect(new Headers(calls[2][1].headers).get(SESSION_HEADER_NAME)).toBe('signed-token')
  })
})
