import type { TabDefinition } from '../types'

export const pct   = (n: number | null | undefined): string =>
  n != null ? `${Math.round(n * 100)}%` : '—'

export const score = (n: number | null | undefined): string =>
  n != null ? `${n}/100` : '—'

export const TABS: TabDefinition[] = [
  { id: 'overview',  label: '📋 Overview'      },
  { id: 'coaching',  label: '🎯 Coaching'       },
  { id: 'ats',       label: '📊 ATS Report'     },
  { id: 'interview', label: '💬 Interview Prep' },
  { id: 'chat',      label: '🤖 Chat'           },
]

export const CHAT_SUGGESTIONS: string[] = [
  'Which bullet should I rewrite first?',
  'Why did I score low on experience?',
  'What skills should I add urgently?',
  'How can I improve my ATS score?',
  "What's my biggest gap for this role?",
]

export const LOADING_STEPS: string[] = [
  'Parsing resume and job description...',
  'Running ATS keyword analysis...',
  'Analysing skills gap...',
  'Generating coaching report...',
  'Building interview questions...',
]

export const QUESTION_TYPE_COLORS: Record<string, string> = {
  TECHNICAL:     '#185fa5',
  BEHAVIORAL:    '#2d7a4f',
  SITUATIONAL:   '#b57c1a',
  SYSTEM_DESIGN: '#7f77dd',
}

export const ACCEPTED_EXTENSIONS = ['pdf', 'docx', 'txt'] as const
export type AcceptedExtension = typeof ACCEPTED_EXTENSIONS[number]
