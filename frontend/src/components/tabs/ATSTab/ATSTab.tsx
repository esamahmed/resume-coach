import React from 'react'
import styles from './ATSTab.module.scss'
import { pct } from '../../../utils/formatters'
import type { AnalyzeResult, ATSLikelihood } from '../../../types'

interface Props {
  result: AnalyzeResult
}

const ATSTab: React.FC<Props> = ({ result }) => {
  const ats        = result.ats_report ?? {}
  const composite  = ats.composite_score  ?? 0
  const likelihood = (ats.ats_pass_likelihood ?? 'MEDIUM') as ATSLikelihood

  const miniScores: Array<{ label: string; val: string }> = [
    { label: 'TF-IDF ATS',       val: pct(ats.tfidf_ats_score)      },
    { label: 'Hire probability',  val: pct(ats.xgb_hire_probability) },
    { label: 'Composite',         val: pct(composite)                },
  ]

  return (
    <>
      <div className="card">
        <h3>ATS Pass Assessment</h3>
        <div className={styles['pass-row']}>
          <span className={`${styles['pass-badge']} ${styles[`pass-badge--${likelihood}`]}`}>
            {likelihood} likelihood
          </span>
          <span className={styles['pass-note']}>Composite score: {pct(composite)}</span>
        </div>

        <div className={styles.meter}>
          <div className={styles['meter-fill']} style={{ width: pct(composite) }} />
        </div>

        <div className={styles['score-grid']}>
          {miniScores.map(({ label, val }) => (
            <div key={label} className={styles['mini-score']}>
              <div className={styles['mini-label']}>{label}</div>
              <div className={styles['mini-value']}>{val}</div>
            </div>
          ))}
        </div>
      </div>

      {(ats.recommended_format_changes?.length ?? 0) > 0 && (
        <div className="card">
          <h3>📝 Format Recommendations</h3>
          <ul style={{ paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            {ats.recommended_format_changes!.map((r, i) => (
              <li key={i} style={{ fontSize: '0.88rem', color: 'var(--ink-2)' }}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {(ats.top_missing_keywords?.length ?? 0) > 0 && (
        <div className="card">
          <h3>🔑 Top Missing ATS Keywords — Add These First</h3>
          <div className="chips">
            {ats.top_missing_keywords!.map((k, i) => (
              <span key={i} className="chip chip--missing">{k}</span>
            ))}
          </div>
        </div>
      )}
    </>
  )
}

export default ATSTab
