export type CoverageStatus =
  | 'policies_found'
  | 'partial'
  | 'access_limited'
  | 'no_policies_found'

export interface AcceptedPolicy {
  source_url: string
  source_urls: string[]
  policy_categories: string[]
  content_hash: string
  chunks: number
}

export interface WebsiteIngestion {
  status: 'success' | 'no_policies_found'
  document_id: string
  url: string
  chunks: number
  policies: Record<string, string[]>
  accepted_policies: AcceptedPolicy[]
  warnings: string[]
  skipped: Record<string, number>
  candidates: number
  coverage_status: CoverageStatus
  fallback_used: 'none' | 'apify'
  fallback: Record<string, unknown>
  recovery_options: string[]
}

export interface PdfIngestion {
  status: 'success'
  document_id: string
  filename: string | null
  chunks: number
}

export interface TextIngestion {
  status: 'success'
  document_id: string
  source_type: 'text'
  title: string
  chunks: number
}

export interface AnswerSource {
  score: number
  text: string
  document_id: string
  owner_id?: string
  chunk_index: number
  source?: string
  source_url?: string
  source_urls?: string[]
  filename?: string
  policy_categories?: string[]
  source_type?: string
  title?: string
}

export interface SourceAttribution {
  source_url: string
  source_urls?: string[]
  filename: string
  policy_categories: string[]
  title?: string
  source_type?: string
}

export interface AnswerResponse {
  answer: string
  sources: AnswerSource[]
  source_attributions: SourceAttribution[]
  document_found: boolean
}

export interface DeleteResponse {
  status: 'deleted'
  document_id: string
}

export type IngestionResult = WebsiteIngestion | PdfIngestion | TextIngestion

export type AnalysisType = 'website' | 'pdf' | 'text'

export interface ActiveAnalysis {
  type: AnalysisType
  documentId: string
  identity: string
  chunks: number
  result: IngestionResult
}
