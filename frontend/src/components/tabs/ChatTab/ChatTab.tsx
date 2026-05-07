import React, { useState, useRef, useEffect, useCallback } from 'react'
import styles from './ChatTab.module.scss'
import { sendChat } from '../../../api/client'
import { CHAT_SUGGESTIONS } from '../../../utils/formatters'
import type { ChatMessage } from '../../../types'

interface Props {
  sessionId: string
}

const INITIAL_MESSAGE: ChatMessage = {
  role: 'assistant',
  content:
    "Hi! I'm your AI resume coach. Ask me anything about your gap analysis, how to improve specific bullets, what to study before the interview, or how to boost your ATS score. I have full access to your coaching report.",
}

const ChatTab: React.FC<Props> = ({ sessionId }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([INITIAL_MESSAGE])
  const [input,    setInput]    = useState<string>('')
  const [loading,  setLoading]  = useState<boolean>(false)

  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef  = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = useCallback(async (text?: string): Promise<void> => {
    const q = (text ?? input).trim()
    if (!q || loading) return

    setInput('')
    setMessages((m) => [...m, { role: 'user', content: q }])
    setLoading(true)

    try {
      const data = await sendChat(sessionId, q)
      setMessages((m) => [...m, { role: 'assistant', content: data.answer }])
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      setMessages((m) => [...m, { role: 'assistant', content: `Sorry, I encountered an error: ${msg}` }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, sessionId])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send()
    }
  }

  return (
    <div className={styles['chat-wrapper']}>
      {/* Suggestion pills */}
      <div className={styles.suggestions}>
        {CHAT_SUGGESTIONS.map((s) => (
          <button
            key={s}
            className={styles['suggestion-pill']}
            onClick={() => void send(s)}
            type="button"
          >
            {s}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div className={styles.messages} role="log" aria-live="polite">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`${styles.msg} ${m.role === 'user' ? styles['msg--user'] : ''}`}
          >
            <div className={`${styles.avatar} ${m.role === 'user' ? styles['avatar--user'] : styles['avatar--ai']}`}>
              {m.role === 'assistant' ? 'AI' : 'You'}
            </div>
            <div className={`${styles.bubble} ${m.role === 'user' ? styles['bubble--user'] : styles['bubble--ai']}`}>
              {m.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className={styles.msg}>
            <div className={`${styles.avatar} ${styles['avatar--ai']}`}>AI</div>
            <div className={`${styles.bubble} ${styles['bubble--ai']}`}>
              <div className={styles['typing-dots']} aria-label="AI is typing">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input row */}
      <div className={styles['input-row']}>
        <textarea
          ref={inputRef}
          className={styles['chat-input']}
          rows={1}
          placeholder="Ask about your resume, gaps, or how to improve..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          aria-label="Chat message input"
        />
        <button
          className={styles['send-btn']}
          onClick={() => void send()}
          disabled={loading || !input.trim()}
          type="button"
          aria-label="Send message"
        >
          Send
        </button>
      </div>
    </div>
  )
}

export default ChatTab
