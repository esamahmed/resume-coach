import type { AnalyzeResult, ChatResponse } from '../types'

const API_BASE = import.meta.env.VITE_API_URL ?? '/api'
const API_KEY  = import.meta.env.VITE_API_KEY  ?? 'dev-key-change-in-production'

const authHeaders: HeadersInit = { 'X-API-Key': API_KEY }

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail ?? 'Request failed')
  }
  return res.json() as Promise<T>
}

export async function analyzeResume(
  resumeFile: File,
  jdText: string,
): Promise<AnalyzeResult> {
  const form = new FormData()
  form.append('resume_file', resumeFile)
  form.append('jd_text', jdText)

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: authHeaders,
    body: form,
  })
  return handleResponse<AnalyzeResult>(res)
}

export async function sendChat(
  sessionId: string,
  question: string,
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { ...authHeaders, 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question }),
  })
  return handleResponse<ChatResponse>(res)
}
