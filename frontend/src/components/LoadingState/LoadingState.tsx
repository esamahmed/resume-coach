import React from 'react'
import styles from './LoadingState.module.scss'

interface Props {
  message: string
}

const LoadingState: React.FC<Props> = ({ message }) => (
  <div className={styles.loading} role="status" aria-live="polite">
    <div className={styles.spinner} aria-hidden="true" />
    <p className={styles['loading-msg']}>{message}</p>
    <div className={styles.dots} aria-hidden="true">
      <div className={styles.dot} />
      <div className={styles.dot} />
      <div className={styles.dot} />
    </div>
  </div>
)

export default LoadingState
