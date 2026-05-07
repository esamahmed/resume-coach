import React from 'react'
import styles from './Header.module.scss'

const Header: React.FC = () => (
  <header className={styles.header}>
    <div className={styles.logo}>
      <div className={styles['logo-mark']}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14,2 14,8 20,8" />
          <line x1="9" y1="13" x2="15" y2="13" />
          <line x1="9" y1="17" x2="13" y2="17" />
        </svg>
      </div>
      <span className={styles['logo-text']}>Resume Coach</span>
    </div>
    <span className={styles.badge}>AI · P5 Capstone</span>
  </header>
)

export default Header
