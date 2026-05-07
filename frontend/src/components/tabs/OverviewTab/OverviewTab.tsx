import React from 'react'
import type { AnalyzeResult } from '../../../types'

interface Props {
  result: AnalyzeResult
}

const OverviewTab: React.FC<Props> = ({ result }) => {
  const gap = result.gap_analysis ?? {}
  const ats = result.ats_report   ?? {}

  return (
    <>
      {gap.fit_summary && (
        <div
          className="card"
          style={{
            background: 'linear-gradient(135deg,#fff8f4,#fff)',
            borderColor: '#fac775',
            fontSize: '0.92rem',
            color: 'var(--ink-2)',
            lineHeight: 1.65,
          }}
        >
          {gap.fit_summary}
        </div>
      )}

      <div className="two-col">
        <div className="card">
          <h3>✅ Strengths</h3>
          <div className="chips">
            {(gap.strengths ?? []).slice(0, 8).map((s, i) => (
              <span key={i} className="chip chip--matched">{s}</span>
            ))}
          </div>
        </div>
        <div className="card">
          <h3>⚠️ Critical Gaps</h3>
          <div className="chips">
            {(gap.critical_gaps ?? []).slice(0, 8).map((g, i) => (
              <span key={i} className="chip chip--critical">{g}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="two-col">
        <div className="card">
          <h3>🔴 Missing Keywords</h3>
          <div className="chips">
            {(ats.missing_keywords ?? []).slice(0, 15).map((k, i) => (
              <span key={i} className="chip chip--missing">{k}</span>
            ))}
          </div>
        </div>
        <div className="card">
          <h3>🟢 Matched Keywords</h3>
          <div className="chips">
            {(ats.matched_keywords ?? []).slice(0, 15).map((k, i) => (
              <span key={i} className="chip chip--matched">{k}</span>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}

export default OverviewTab
