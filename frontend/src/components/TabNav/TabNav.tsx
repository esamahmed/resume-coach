import React from 'react'
import styles from './TabNav.module.scss'
import { TABS } from '../../utils/formatters'

interface Props {
  active:   string
  onChange: (id: string) => void
}

const TabNav: React.FC<Props> = ({ active, onChange }) => (
  <div className={styles.tabs} role="tablist">
    {TABS.map((t) => (
      <button
        key={t.id}
        role="tab"
        aria-selected={active === t.id}
        className={`${styles['tab-btn']} ${active === t.id ? styles.active : ''}`}
        onClick={() => onChange(t.id)}
        type="button"
      >
        {t.label}
      </button>
    ))}
  </div>
)

export default TabNav
