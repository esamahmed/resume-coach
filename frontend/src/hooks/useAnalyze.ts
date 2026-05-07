import { useState, useCallback, useRef } from 'react'
import { analyzeResume } from '../api/client'
import { LOADING_STEPS, ACCEPTED_EXTENSIONS } from '../utils/formatters'
import type { AnalyzeResult } from '../types'

interface UseAnalyzeReturn {
  resumeFile:   File | null
  jdText:       string
  setJdText:    (v: string) => void
  loading:      boolean
  loadingMsg:   string
  result:       AnalyzeResult | null
  error:        string | null
  dragOver:     boolean
  setDragOver:  (v: boolean) => void
  fileInputRef: React.RefObject<HTMLInputElement>
  handleFile:   (file: File | null | undefined) => void
  handleDrop:   (e: React.DragEvent<HTMLDivElement>) => void
  analyze:      () => Promise<void>
  reset:        () => void
}

export function useAnalyze(): UseAnalyzeReturn {
  const [resumeFile, setResumeFile] = useState<File | null>(null)
  const [jdText,     setJdText]     = useState<string>('')
  const [loading,    setLoading]    = useState<boolean>(false)
  const [loadingMsg, setLoadingMsg] = useState<string>('')
  const [result,     setResult]     = useState<AnalyzeResult | null>(null)
  const [error,      setError]      = useState<string | null>(null)
  const [dragOver,   setDragOver]   = useState<boolean>(false)
  const fileInputRef                = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((file: File | null | undefined): void => {
    if (!file) return
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (!ACCEPTED_EXTENSIONS.includes(ext as typeof ACCEPTED_EXTENSIONS[number])) {
      setError('Please upload a PDF, DOCX, or TXT file.')
      return
    }
    setResumeFile(file)
    setError(null)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const analyze = useCallback(async (): Promise<void> => {
    if (!resumeFile || !jdText.trim()) {
      setError('Please upload a resume and paste a job description.')
      return
    }

    setLoading(true)
    setError(null)

    let step = 0
    setLoadingMsg(LOADING_STEPS[0])
    const interval = window.setInterval(() => {
      step = (step + 1) % LOADING_STEPS.length
      setLoadingMsg(LOADING_STEPS[step])
    }, 2200)

    try {
      const data = await analyzeResume(resumeFile, jdText)
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'An unexpected error occurred.')
    } finally {
      clearInterval(interval)
      setLoading(false)
    }
  }, [resumeFile, jdText])

  const reset = useCallback((): void => {
    setResult(null)
    setResumeFile(null)
    setJdText('')
    setError(null)
  }, [])

  return {
    resumeFile, jdText, setJdText,
    loading, loadingMsg,
    result, error,
    dragOver, setDragOver,
    fileInputRef,
    handleFile, handleDrop,
    analyze, reset,
  }
}
