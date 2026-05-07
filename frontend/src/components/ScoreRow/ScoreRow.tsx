import React from 'react'
import styles from './ScoreRow.module.scss'
import { pct } from '../../utils/formatters'
import type { GapAnalysis, ATSReport } from '../../types'

interface Props {
  gap?: Partial<GapAnalysis>
  ats?: Partial<ATSReport>
}

const ScoreRow: React.FC<Props> = ({ gap = {}, ats = {} }) => (
  <div className={styles['score-row']}>
    <div className={`${styles['score-card']} ${styles['score-card--fit']}`}>
      <div className={styles.label}>Fit Score</div>
      <div className={styles.value}>
        {gap.overall_fit_score ?? '—'}
        <span>/100</span>
      </div>
      <div className={styles.sub}>vs. job requirements</div>
    </div>

    <div className={`${styles['score-card']} ${styles['score-card--ats']}`}>
      <div className={styles.label}>ATS Score</div>
      <div className={styles.value}>{pct(ats.tfidf_ats_score)}</div>
      <div className={styles.sub}>keyword match</div>
    </div>

    <div className={`${styles['score-card']} ${styles['score-card--hire']}`}>
      <div className={styles.label}>Hire Probability</div>
      <div className={styles.value}>{pct(ats.xgb_hire_probability)}</div>
      <div className={styles.sub}>XGBoost estimate</div>
    </div>
  </div>
)

export default ScoreRow
