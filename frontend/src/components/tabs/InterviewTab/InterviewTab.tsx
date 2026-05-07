import React from 'react'
import styles from './InterviewTab.module.scss'
import { QUESTION_TYPE_COLORS } from '../../../utils/formatters'
import type { AnalyzeResult, InterviewQuestion } from '../../../types'

interface Props {
  result: AnalyzeResult
}

const QuestionCard: React.FC<{ q: InterviewQuestion }> = ({ q }) => (
  <div className={styles['question-card']}>
    <div
      className={styles['question-type']}
      style={{ color: QUESTION_TYPE_COLORS[q.type] ?? 'var(--ink-3)' }}
    >
      {q.type} · {q.difficulty}
    </div>
    <div className={styles['question-text']}>{q.question}</div>
    {q.gap_it_targets && (
      <div className={styles['question-gap']}>🎯 Targets: {q.gap_it_targets}</div>
    )}
    {q.sample_answer_hint && (
      <div className={styles['question-hint']}>
        <strong>Hint:</strong> {q.sample_answer_hint}
      </div>
    )}
  </div>
)

const InterviewTab: React.FC<Props> = ({ result }) => {
  const questions = result.interview_questions ?? []

  return (
    <div className="card">
      <h3>Gap-Targeted Interview Questions</h3>
      <p style={{ fontSize: '0.82rem', color: 'var(--ink-3)', marginBottom: '1rem' }}>
        Questions weighted towards your identified critical gaps — expect these in your interview.
      </p>
      {questions.slice(0, 8).map((q, i) => (
        <QuestionCard key={i} q={q} />
      ))}
    </div>
  )
}

export default InterviewTab
