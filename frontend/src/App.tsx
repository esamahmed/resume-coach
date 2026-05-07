import React from 'react'
import styles from './App.module.scss'
import './styles/global.scss'

import Header        from './components/Header/Header'
import Hero          from './components/Hero/Hero'
import UploadSection from './components/UploadSection/UploadSection'
import LoadingState  from './components/LoadingState/LoadingState'
import Results       from './components/Results/Results'
import { useAnalyze } from './hooks/useAnalyze'

const App: React.FC = () => {
  const {
    resumeFile, jdText, setJdText,
    loading, loadingMsg,
    result, error,
    dragOver, setDragOver,
    fileInputRef,
    handleFile, handleDrop,
    analyze, reset,
  } = useAnalyze()

  return (
    <div className={styles.app}>
      <Header />

      <main className={styles.main}>
        {!result && !loading && (
          <>
            <Hero />
            {error && <div className="error-box">⚠ {error}</div>}
            <UploadSection
              resumeFile={resumeFile}
              jdText={jdText}
              dragOver={dragOver}
              fileInputRef={fileInputRef}
              onFile={handleFile}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onJdChange={setJdText}
              onAnalyze={analyze}
            />
          </>
        )}

        {loading && <LoadingState message={loadingMsg} />}

        {result && (
          <Results result={result} error={error} onReset={reset} />
        )}
      </main>
    </div>
  )
}

export default App
