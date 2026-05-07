import React from 'react'
import styles from './UploadSection.module.scss'

interface Props {
  resumeFile:   File | null
  jdText:       string
  dragOver:     boolean
  fileInputRef: React.RefObject<HTMLInputElement>
  onFile:       (file: File | null | undefined) => void
  onDragOver:   (e: React.DragEvent<HTMLDivElement>) => void
  onDragLeave:  () => void
  onDrop:       (e: React.DragEvent<HTMLDivElement>) => void
  onJdChange:   (value: string) => void
  onAnalyze:    () => void
}

const UploadSection: React.FC<Props> = ({
  resumeFile, jdText, dragOver, fileInputRef,
  onFile, onDragOver, onDragLeave, onDrop,
  onJdChange, onAnalyze,
}) => {
  const dropZoneClass = [
    styles['upload-card'],
    dragOver   ? styles['drag-over'] : '',
    resumeFile ? styles['has-file']  : '',
  ].filter(Boolean).join(' ')

  return (
    <>
      <div className={styles['upload-grid']}>
        {/* Resume drop zone */}
        <div
          className={dropZoneClass}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          role="button"
          tabIndex={0}
          aria-label="Upload resume file"
          onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
        >
          <div className={styles['upload-icon']}>{resumeFile ? '✅' : '📄'}</div>
          <h3>{resumeFile ? 'Resume uploaded' : 'Upload your resume'}</h3>
          <p>{resumeFile ? resumeFile.name : 'PDF or DOCX · drag & drop or click'}</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={(e) => onFile(e.target.files?.[0])}
            aria-hidden="true"
          />
        </div>

        {/* JD label card */}
        <div className={`${styles['upload-card']} ${styles['upload-card--static']}`}>
          <div className={styles['upload-icon']}>💼</div>
          <h3>Job description</h3>
          <p>Paste the full JD below</p>
        </div>
      </div>

      {/* Job description textarea */}
      <div className={styles['jd-area']}>
        <label htmlFor="jd-input">Job Description</label>
        <textarea
          id="jd-input"
          placeholder="Paste the full job description here — requirements, responsibilities, tech stack..."
          value={jdText}
          onChange={(e) => onJdChange(e.target.value)}
        />
      </div>

      {/* Analyze CTA */}
      <button
        className={styles['analyze-btn']}
        onClick={onAnalyze}
        disabled={!resumeFile || !jdText.trim()}
        type="button"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        Analyse My Resume
      </button>
    </>
  )
}

export default UploadSection
