// ─── API response shapes ──────────────────────────────────────────────────────

export interface SkillsGap {
  missing_required: string[]
  missing_preferred: string[]
  present_and_strong: string[]
}

export interface ExperienceGap {
  years_required: number | null
  years_candidate: number | null
  domain_gaps: string[]
  domain_strengths: string[]
}

export interface EducationGap {
  required: string | null
  candidate_has: string
  gap_exists: boolean
}

export interface GapAnalysis {
  skills_gap: SkillsGap
  experience_gap: ExperienceGap
  education_gap: EducationGap
  strengths: string[]
  critical_gaps: string[]
  overall_fit_score: number
  fit_summary: string
}

export interface SectionScores {
  skills: number
  experience: number
  education: number
  impact_statements: number
  ats_optimization: number
}

export interface Recommendation {
  priority: number
  section: string
  action: string
  detail: string
  example?: string
}

export interface RewrittenBullet {
  original: string
  rewritten: string
  why: string
}

export interface CoachingReport {
  executive_summary: string
  section_scores: SectionScores
  recommendations: Recommendation[]
  rewritten_bullets: RewrittenBullet[]
  missing_keywords_to_add: string[]
  cover_letter_angles: string[]
}

export type ATSLikelihood = 'HIGH' | 'MEDIUM' | 'LOW'

export interface SectionPresence {
  contact_info: boolean
  summary: boolean
  skills: boolean
  experience: boolean
  education: boolean
  certifications: boolean
}

export interface ATSReport {
  ats_pass_likelihood: ATSLikelihood
  ats_pass_likelihood_pct: number
  keyword_density_score: number
  format_score: number
  section_presence: SectionPresence
  top_missing_keywords: string[]
  keyword_stuffing_risk: boolean
  recommended_format_changes: string[]
  ats_friendly_title_suggestion: string
  // merged in from ML fast-pass
  tfidf_ats_score: number
  xgb_hire_probability: number
  composite_score: number
  matched_keywords: string[]
  missing_keywords: string[]
}

export interface InterviewQuestion {
  question: string
  type: 'TECHNICAL' | 'BEHAVIORAL' | 'SITUATIONAL' | 'SYSTEM_DESIGN'
  difficulty: 'EASY' | 'MEDIUM' | 'HARD'
  gap_it_targets: string
  why_they_will_ask: string
  sample_answer_hint: string
  follow_up: string
}

export interface AnalyzeResult {
  session_id: string
  trace_url: string
  tfidf_ats_score: number
  xgb_hire_prob: number
  composite_score: number
  overall_fit_score: number | null
  fit_summary: string
  gap_analysis: GapAnalysis
  coaching_report: CoachingReport
  ats_report: ATSReport
  interview_questions: InterviewQuestion[]
  errors: string[]
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatResponse {
  session_id: string
  answer: string
  chat_history: ChatMessage[]
}

// ─── UI types ─────────────────────────────────────────────────────────────────

export interface TabDefinition {
  id: string
  label: string
}
