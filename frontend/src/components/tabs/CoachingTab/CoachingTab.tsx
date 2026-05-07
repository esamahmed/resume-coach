import React from 'react'
import styles from './CoachingTab.module.scss'
import type { AnalyzeResult, Recommendation, RewrittenBullet } from '../../../types'

interface Props {
  result: AnalyzeResult
}

const RecItem: React.FC<{ rec: Recommendation; index: number }> = ({ rec, index }) => (
  <div className={styles['rec-item']}>
    <div className={styles['rec-priority']}>{rec.priority || index + 1}</div>
    <div className={styles['rec-content']}>
      <h4>{rec.action}</h4>
      <p>{rec.detail}</p>
      {rec.example && <p className={styles.example}>e.g. {rec.example}</p>}
    </div>
  </div>
)

const BulletRewrite: React.FC<{ bullet: RewrittenBullet }> = ({ bullet }) => (
  <div className={styles['bullet-rewrite']}>
    <div className={styles['bullet-original']}>{bullet.original}</div>
    <div className={styles['bullet-improved']}>▶ {bullet.rewritten}</div>
    {bullet.why && <div className={styles['bullet-why']}>{bullet.why}</div>}
  </div>
)

const CoachingTab: React.FC<Props> = ({ result }) => {
  const coaching = result.coaching_report ?? {}
  const recs     = [...(coaching.recommendations ?? [])].sort(
    (a, b) => (b.priority ?? 0) - (a.priority ?? 0),
  )
  const bullets = coaching.rewritten_bullets ?? []

  return (
    <>
      {coaching.executive_summary && (
        <div className="card">
          <h3>Executive Summary</h3>
          <p style={{ fontSize: '0.92rem', color: 'var(--ink-2)', lineHeight: 1.65 }}>
            {coaching.executive_summary}
          </p>
        </div>
      )}

      {recs.length > 0 && (
        <div className="card">
          <h3>🎯 Prioritised Recommendations</h3>
          <div className={styles['rec-list']}>
            {recs.slice(0, 6).map((rec, i) => (
              <RecItem key={i} rec={rec} index={i} />
            ))}
          </div>
        </div>
      )}

      {bullets.length > 0 && (
        <div className="card">
          <h3>✏️ Rewritten Bullets</h3>
          {bullets.slice(0, 4).map((b, i) => (
            <BulletRewrite key={i} bullet={b} />
          ))}
        </div>
      )}
    </>
  )
}

export default CoachingTab
