import React, { useState } from 'react'
import styles from './Results.module.scss'
import ScoreRow     from '../ScoreRow/ScoreRow'
import TabNav       from '../TabNav/TabNav'
import OverviewTab  from '../tabs/OverviewTab/OverviewTab'
import CoachingTab  from '../tabs/CoachingTab/CoachingTab'
import ATSTab       from '../tabs/ATSTab/ATSTab'
import InterviewTab from '../tabs/InterviewTab/InterviewTab'
import ChatTab      from '../tabs/ChatTab/ChatTab'
import type { AnalyzeResult } from '../../types'

interface Props {
  result:  AnalyzeResult
  error:   string | null
  onReset: () => void
}

type TabId = 'overview' | 'coaching' | 'ats' | 'interview' | 'chat'

const Results: React.FC<Props> = ({ result, error, onReset }) => {
  const [activeTab, setActiveTab] = useState<TabId>('overview')

  const gap = result.gap_analysis ?? {}
  const ats = result.ats_report   ?? {}

  return (
    <div className={styles.results}>
      <div className={styles['results-header']}>
        <h2>Your Coaching Report</h2>
        <button className={styles['new-btn']} onClick={onReset} type="button">
          ↩ New Analysis
        </button>
      </div>

      {error && <div className="error-box">⚠ {error}</div>}

      <ScoreRow gap={gap} ats={ats} />

      <TabNav active={activeTab} onChange={(id) => setActiveTab(id as TabId)} />

      {activeTab === 'overview'  && <OverviewTab  result={result} />}
      {activeTab === 'coaching'  && <CoachingTab  result={result} />}
      {activeTab === 'ats'       && <ATSTab        result={result} />}
      {activeTab === 'interview' && <InterviewTab  result={result} />}
      {activeTab === 'chat'      && <ChatTab sessionId={result.session_id} />}
    </div>
  )
}

export default Results
