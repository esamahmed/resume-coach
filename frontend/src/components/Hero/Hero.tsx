import React from 'react'
import styles from './Hero.module.scss'

const Hero: React.FC = () => (
  <div className={styles.hero}>
    <h1>Land the role you <em>actually</em> want</h1>
    <p>
      Upload your resume and a job description. Get a precise gap analysis,
      ATS optimisation, and interview prep in seconds.
    </p>
  </div>
)

export default Hero
