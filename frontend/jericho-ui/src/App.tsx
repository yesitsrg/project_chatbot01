// frontend/jericho-ui/src/App.tsx
import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'

type Screen = 'login' | 'chat'

type ChatMessage = {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: any[]
  confidence?: number
  feedback?: 'like' | 'dislike' | null
}

type SessionInfo = {
  session_id: number
  session_name: string
}

type DomainTemplate = {
  id: string
  name: string
  icon: string
  prompt: string
  description: string
}

function App() {
  // ========== LOGIN STATE ==========
  const [screen, setScreen] = useState<Screen>('login')
  const [username, setUsername] = useState<string>('')
  const [password, setPassword] = useState<string>('')
  const [role, setRole] = useState<string | null>(null)
  const [loginError, setLoginError] = useState<string | null>(null)
  const [loginLoading, setLoginLoading] = useState(false)

  // ========== CHAT STATE ==========
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [nextId, setNextId] = useState(1)

  // ========== SESSIONS ==========
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null)
  const [sessionsLoading, setSessionsLoading] = useState(false)

  // ========== UX ENHANCEMENTS STATE ==========
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [showTemplates, setShowTemplates] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [recognition, setRecognition] = useState<any>(null)
  const [feedbackStates, setFeedbackStates] = useState<
    Record<number, 'like' | 'dislike' | null>
  >({})

  // ========== SESSION RENAME/DELETE ==========
  const [sessionMenuOpenId, setSessionMenuOpenId] = useState<number | null>(
    null
  )
  const [showRenameModal, setShowRenameModal] = useState(false)
  const [renameSessionId, setRenameSessionId] = useState<number | null>(null)
  const [renameValue, setRenameValue] = useState('')

  // ========== UPLOAD ==========
  const [showUpload, setShowUpload] = useState(false)
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null)
  const [uploadPrivate, setUploadPrivate] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)
  const [uploadLoading, setUploadLoading] = useState(false)

  // ========== ADMIN ==========
  const [adminView, setAdminView] = useState<'chat' | 'admin'>('chat')
  const [adminStats, setAdminStats] = useState<any | null>(null)
  const [adminStatsLoading, setAdminStatsLoading] = useState(false)
  const [adminTestQuery, setAdminTestQuery] = useState('')
  const [adminTestResponse, setAdminTestResponse] = useState<any | null>(null)
  const [adminTestLoading, setAdminTestLoading] = useState(false)

  // ========== REFS ==========
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  // ========== DOMAIN TEMPLATES ==========
  const DOMAIN_TEMPLATES: DomainTemplate[] = [
    {
      id: 'transcript',
      name: '🎓 Student Transcripts',
      icon: '📊',
      prompt: "What's my GPA?",
      description: 'GPA, courses, progress',
    },
    {
      id: 'payroll',
      name: '💰 Payroll Calendar',
      icon: '💵',
      prompt: "When's my next paycheck?",
      description: 'Pay dates, periods',
    },
    {
      id: 'bor',
      name: '📅 Board Meetings',
      icon: '📋',
      prompt: 'Next Board of Regents meeting?',
      description: 'BOR schedule, dates',
    },
    {
      id: 'policy',
      name: '📚 Policies & Handbook',
      icon: '📖',
      prompt: 'What are the sick leave policies?',
      description: 'HR policies, rules',
    },
  ]

  // ========== VOICE RECOGNITION SETUP ==========
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition =
        (window as any).SpeechRecognition ||
        (window as any).webkitSpeechRecognition
      const recog = new SpeechRecognition()
      recog.continuous = false
      recog.interimResults = true
      recog.lang = 'en-US'

      recog.onstart = () => setIsListening(true)
      recog.onresult = (event: any) => {
        const transcript = Array.from(event.results)
          .map((result: any) => result[0].transcript)
          .join('')
        setInput(transcript)
      }
      recog.onend = () => setIsListening(false)

      setRecognition(recog)
    }
  }, [])

  // ========== AUTO-SCROLL ==========
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  // ========== LOAD SESSIONS ON SCREEN CHANGE ==========
  useEffect(() => {
    if (screen === 'chat') {
      loadSessions()
    }
  }, [screen])

  // ========== LOAD ADMIN STATS ==========
  useEffect(() => {
    if (screen === 'chat' && role === 'admin' && adminView === 'admin') {
      loadAdminStats()
    }
  }, [screen, role, adminView])

  // ========== FUNCTIONS ==========

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoginError(null)
    if (!username.trim() || !password.trim()) {
      setLoginError('Username and password are required.')
      return
    }
    setLoginLoading(true)
    try {
      const resp = await fetch('http://localhost:8000/apiv1/react-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      setRole(data.role)
      setScreen('chat')
    } catch (err) {
      console.error(err)
      setLoginError('Invalid username or password.')
    } finally {
      setLoginLoading(false)
    }
  }

  const loadSessions = async () => {
    setSessionsLoading(true)
    try {
      const resp = await fetch('http://localhost:8000/user_sessions', {
        method: 'GET',
        credentials: 'include',
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      const raw = data.sessions || []

      const list: SessionInfo[] = raw.map((s: any) => ({
        session_id: s.session_id ?? s.sessionid,
        session_name:
          s.session_name ??
          s.sessionname ??
          `Chat ${s.session_id ?? s.sessionid}`,
      }))
      setSessions(list)

      if (list.length > 0) {
        const last = list[list.length - 1]
        setCurrentSessionId(last.session_id)
        await loadHistory(last.session_id)
      } else {
        await handleNewChat()
      }
    } catch (err) {
      console.error(err)
    } finally {
      setSessionsLoading(false)
    }
  }

  const loadHistory = async (sessionId: number) => {
    try {
      const resp = await fetch(
        `http://localhost:8000/history?session_id=${sessionId}`,
        {
          method: 'GET',
          credentials: 'include',
        }
      )
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      const history = data.history || []
      const mapped: ChatMessage[] = []
      let idCounter = 1
      for (const item of history) {
        if (item.question) {
          mapped.push({ id: idCounter++, role: 'user', content: item.question })
        }
        if (item.answer) {
          mapped.push({
            id: idCounter++,
            role: 'assistant',
            content: item.answer,
            feedback: null,
          })
        }
      }
      setMessages(mapped)
      setNextId(idCounter)
    } catch (err) {
      console.error(err)
      setMessages([])
      setNextId(1)
    }
  }

  const handleNewChat = async () => {
    setShowTemplates(true)
    try {
      const resp = await fetch('http://localhost:8000/new_session', {
        method: 'POST',
        credentials: 'include',
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      const sessionId: number = data.session_id ?? data.sessionid

      console.log('[handleNewChat] Created session:', sessionId)

      if (!sessionId) {
        console.error('[handleNewChat] No sessionId returned!', data)
        return
      }

      const newSession: SessionInfo = {
        session_id: sessionId,
        session_name: 'New Chat',
      }
      setSessions((prev) => [...prev, newSession])
      setCurrentSessionId(sessionId)
      setMessages([])
      setNextId(1)
    } catch (err) {
      console.error('[handleNewChat] Error:', err)
    }
  }

  const handleSelectSession = async (sessionId: number) => {
    if (sessionId === currentSessionId) return
    setCurrentSessionId(sessionId)
    setShowTemplates(false)
    await loadHistory(sessionId)
  }

  const openRenameModal = (session: SessionInfo) => {
    setRenameSessionId(session.session_id)
    setRenameValue(session.session_name)
    setShowRenameModal(true)
    setSessionMenuOpenId(null)
  }

  const handleRenameSubmit = async () => {
    if (!renameSessionId) return
    const name = renameValue.trim() || 'New Chat'
    try {
      const form = new FormData()
      form.append('session_id', String(renameSessionId))
      form.append('new_name', name)

      const resp = await fetch('http://localhost:8000/rename_session', {
        method: 'POST',
        body: form,
        credentials: 'include',
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      setSessions((prev) =>
        prev.map((s) =>
          s.session_id === renameSessionId ? { ...s, session_name: name } : s
        )
      )
      setShowRenameModal(false)
    } catch (err) {
      console.error(err)
    }
  }

  const handleDeleteSession = async (sessionId: number) => {
    try {
      const form = new FormData()
      form.append('session_id', String(sessionId))

      const resp = await fetch('http://localhost:8000/delete_session', {
        method: 'POST',
        body: form,
        credentials: 'include',
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
      setSessionMenuOpenId(null)

      if (currentSessionId === sessionId) {
        const remaining = sessions.filter((s) => s.session_id !== sessionId)
        if (remaining.length > 0) {
          const last = remaining[remaining.length - 1]
          setCurrentSessionId(last.session_id)
          await loadHistory(last.session_id)
        } else {
          setCurrentSessionId(null)
          setMessages([])
          setNextId(1)
          await handleNewChat()
        }
      }
    } catch (err) {
      console.error(err)
    }
  }

  const handleSend = async () => {
    const q = input.trim()
    if (!q || !currentSessionId) return
    setInput('')
    setShowTemplates(false)

    const userMessage: ChatMessage = {
      id: nextId,
      role: 'user',
      content: q,
    }
    setNextId(nextId + 1)
    setMessages((prev) => [...prev, userMessage])
    setLoading(true)

    try {
      const resp = await fetch('http://localhost:8000/react-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          sessionid: currentSessionId,
          private: false,
        }),
        credentials: 'include',
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      const assistantMessage: ChatMessage = {
        id: nextId + 1,
        role: 'assistant',
        content: data.answer ?? 'No answer field in response.',
        sources: data.sources || [],
        confidence:
          typeof data.confidence === 'number' ? data.confidence : undefined,
        feedback: null,
      }
      setNextId(nextId + 2)
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      console.error(err)
      const errorMessage: ChatMessage = {
        id: nextId + 1,
        role: 'assistant',
        content: 'Error calling backend. Check server logs.',
      }
      setNextId(nextId + 2)
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const toggleVoice = () => {
    if (isListening && recognition) {
      recognition.stop()
    } else if (recognition) {
      recognition.start()
    }
  }

  const submitFeedback = async (
    messageId: number,
    rating: 'like' | 'dislike'
  ) => {
    if (!currentSessionId) return

    try {
      const form = new FormData()
      form.append('message_id', String(messageId))
      form.append('rating', rating)
      form.append('session_id', String(currentSessionId))

      await fetch('http://localhost:8000/feedback', {
        method: 'POST',
        body: form,
        credentials: 'include',
      })
    } catch (err) {
      console.error('Feedback failed:', err)
    }

    setFeedbackStates((prev) => ({
      ...prev,
      [messageId]: rating,
    }))
  }

  const useTemplate = (template: DomainTemplate) => {
    setInput(template.prompt)
    setShowTemplates(false)
  }

  const handleUpload = async () => {
    if (!uploadFiles || uploadFiles.length === 0) {
      setUploadStatus('Please select at least one file.')
      return
    }
    if (!currentSessionId) {
      setUploadStatus('No active session selected.')
      return
    }
    setUploadLoading(true)
    setUploadStatus(null)
    try {
      const form = new FormData()
      Array.from(uploadFiles).forEach((file) => {
        form.append('files', file)
      })
      form.append('session_id', String(currentSessionId))
      form.append('private', uploadPrivate ? 'true' : 'false')

      const resp = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: form,
        credentials: 'include',
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      const msg =
        data.message ||
        `Processed ${data.processed_files?.length ?? 0} file(s).`
      setUploadStatus(msg)
    } catch (err) {
      console.error(err)
      setUploadStatus('Upload failed. Check server logs.')
    } finally {
      setUploadLoading(false)
    }
  }

  const loadAdminStats = async () => {
    setAdminStatsLoading(true)
    try {
      const resp = await fetch('http://localhost:8000/admin/stats', {
        method: 'GET',
        credentials: 'include',
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setAdminStats(data)
    } catch (err) {
      console.error(err)
      setAdminStats(null)
    } finally {
      setAdminStatsLoading(false)
    }
  }

  const handleAdminTestQuery = async () => {
    const q = adminTestQuery.trim()
    if (!q) return
    setAdminTestLoading(true)
    setAdminTestResponse(null)
    try {
      const resp = await fetch('http://localhost:8000/react-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          sessionid: currentSessionId || 1,
          private: false,
        }),
        credentials: 'include',
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setAdminTestResponse(data)
    } catch (err) {
      console.error(err)
      setAdminTestResponse({ error: 'Request failed. Check logs.' })
    } finally {
      setAdminTestLoading(false)
    }
  }

  const handleLogout = async () => {
    try {
      await fetch('http://localhost:8000/logout', {
        method: 'POST',
        credentials: 'include',
      })
    } catch (e) {
      console.error(e)
    }
    setUsername('')
    setPassword('')
    setRole(null)
    setSessions([])
    setCurrentSessionId(null)
    setMessages([])
    setScreen('login')
  }

  // ========== LOGIN SCREEN ==========
  if (screen === 'login') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900">
        <div className="absolute top-6 left-6 flex items-center gap-3">
          <img
            src="https://www.dinecollege.edu/wp-content/uploads/2024/12/dc_logoFooter.png"
            alt="Diné College"
            className="h-10 rounded-md object-contain"
          />
        </div>

        <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl p-8 w-full max-w-md text-white">
          <div className="mb-6">
            <h1 className="text-3xl font-bold mb-1">Diné College</h1>
            <p className="text-sm text-amber-300 font-semibold">
              Powered by Jericho
            </p>
          </div>

          {loginError && (
            <div className="mb-4 text-sm text-red-200 bg-red-900/40 border border-red-400/60 rounded px-3 py-2">
              {loginError}
            </div>
          )}

          <form className="space-y-4" onSubmit={handleLogin}>
            <div>
              <label className="block text-sm mb-2 font-medium">Username</label>
              <input
                type="text"
                className="w-full rounded-lg border border-slate-400/30 bg-slate-900/60 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400 focus:border-transparent"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm mb-2 font-medium">Password</label>
              <input
                type="password"
                className="w-full rounded-lg border border-slate-400/30 bg-slate-900/60 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400 focus:border-transparent"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <button
              type="submit"
              className="w-full mt-4 rounded-lg bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-slate-900 font-bold py-2.5 text-sm transition disabled:opacity-60"
              disabled={loginLoading}
            >
              {loginLoading ? 'Signing in...' : 'Login'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  // ========== CHAT SCREEN ==========
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-slate-50 to-slate-100">
      {/* HEADER */}
      <header className="h-16 px-6 flex items-center justify-between bg-white border-b border-slate-200 shadow-sm">
        <div className="flex items-center gap-4">
          <button
            className="p-2 hover:bg-slate-100 rounded-lg transition lg:hidden"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <span className="text-2xl">≡</span>
          </button>
          <div className="flex items-center gap-3">
            <img
              src="https://www.dinecollege.edu/wp-content/uploads/2024/12/dc_logoFooter.png"
              alt="Diné College"
              className="h-8 rounded-md object-contain"
            />
            <div>
              <h1 className="font-bold text-slate-900">Diné College</h1>
              <p className="text-xs text-amber-600 font-semibold">
                Powered by Jericho
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-sm">
          <div className="text-right hidden sm:block">
            <p className="font-semibold text-slate-800">
              Welcome, {username || 'User'}!
            </p>
            <p className="text-xs text-slate-500">
              {role === 'admin' ? '👑 Admin' : 'User'}
            </p>
          </div>
          <button
            className="px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 text-xs font-medium transition"
            onClick={handleLogout}
          >
            Logout
          </button>
        </div>
      </header>

      <main className="flex-1 flex overflow-hidden">
        {/* SIDEBAR */}
        <aside
          className={`fixed lg:relative w-64 h-[calc(100vh-4rem)] bg-white border-r border-slate-200 p-4 flex flex-col overflow-y-auto z-40 transition-transform lg:translate-x-0 ${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          <button
            className="w-full mb-4 rounded-lg bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-slate-900 font-bold py-2.5 text-sm transition shadow-md"
            onClick={handleNewChat}
            disabled={sessionsLoading}
          >
            ✨ New Chat
          </button>

          {role === 'admin' && (
            <button
              className={`w-full mb-4 rounded-lg text-sm font-semibold py-2 transition ${
                adminView === 'admin'
                  ? 'bg-slate-900 text-amber-300'
                  : 'bg-slate-100 text-slate-800 hover:bg-slate-200'
              }`}
              onClick={() =>
                setAdminView(adminView === 'admin' ? 'chat' : 'admin')
              }
            >
              {adminView === 'admin' ? '← Back to Chat' : '⚙️ Admin Console'}
            </button>
          )}

          <div className="text-xs text-slate-500 px-1 mb-3 font-semibold">
            CONVERSATION HISTORY
          </div>

          <div className="flex-1 overflow-y-auto space-y-2">
            {sessions.length === 0 && !sessionsLoading && (
              <div className="text-xs text-slate-400 px-2 py-4 text-center">
                No chats yet. Click "New Chat" to start!
              </div>
            )}
            {sessions.map((s) => (
              <div
                key={s.session_id}
                className={`flex items-center rounded-lg overflow-hidden transition ${
                  s.session_id === currentSessionId
                    ? 'bg-gradient-to-r from-amber-500 to-amber-400 text-slate-900'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                <button
                  className="flex-1 text-left px-3 py-2 text-sm truncate"
                  onClick={() => handleSelectSession(s.session_id)}
                >
                  {s.session_name || `Chat ${s.session_id}`}
                </button>
                <button
                  className="px-2 text-xs hover:opacity-70 transition"
                  onClick={() =>
                    setSessionMenuOpenId(
                      sessionMenuOpenId === s.session_id ? null : s.session_id
                    )
                  }
                >
                  ⋮
                </button>
                {sessionMenuOpenId === s.session_id && (
                  <div className="absolute mt-8 bg-white border border-slate-200 rounded-lg shadow-lg z-50 text-xs text-slate-700 w-32">
                    <button
                      className="block w-full px-3 py-2 hover:bg-slate-100 text-left"
                      onClick={() => openRenameModal(s)}
                    >
                      ✏️ Rename
                    </button>
                    <button
                      className="block w-full px-3 py-2 hover:bg-red-50 text-left text-red-600"
                      onClick={() => handleDeleteSession(s.session_id)}
                    >
                      🗑️ Delete
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </aside>

        {/* MAIN CONTENT */}
        <section className="flex-1 flex flex-col overflow-hidden">
          {adminView === 'chat' || role !== 'admin' ? (
            <>
              {/* CHAT MESSAGES */}
              <div className="flex-1 overflow-y-auto p-6 bg-gradient-to-b from-slate-50 to-slate-100">
                <div className="max-w-3xl mx-auto space-y-4">
                  {messages.length === 0 && !showTemplates && (
                    <div className="text-center py-12">
                      <p className="text-slate-500 mb-4">
                        Start a conversation about Diné College
                      </p>
                      <button
                        className="text-amber-600 hover:text-amber-700 text-sm font-semibold"
                        onClick={() => setShowTemplates(true)}
                      >
                        Browse Templates →
                      </button>
                    </div>
                  )}

                  {/* TEMPLATES MODAL */}
                  {showTemplates && messages.length === 0 && (
                    <div className="bg-white rounded-xl border border-slate-200 shadow-lg p-6 mb-6">
                      <h3 className="font-bold text-slate-900 mb-4">
                        🎯 Quick Start Templates
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {DOMAIN_TEMPLATES.map((tmpl) => (
                          <button
                            key={tmpl.id}
                            className="p-4 rounded-lg border border-slate-200 hover:border-amber-400 hover:bg-amber-50 text-left transition"
                            onClick={() => useTemplate(tmpl)}
                          >
                            <div className="font-semibold text-slate-900 text-sm mb-1">
                              {tmpl.name}
                            </div>
                            <div className="text-xs text-slate-600">
                              {tmpl.description}
                            </div>
                            <div className="text-xs text-amber-600 mt-2 italic">
                              "{tmpl.prompt}"
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {messages.map((m) => (
                    <div
                      key={m.id}
                      className={`flex ${
                        m.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      <div
                        className={`max-w-4xl rounded-xl px-5 py-4 text-sm shadow-md ${
                          m.role === 'user'
                            ? 'bg-gradient-to-r from-amber-500 to-amber-400 text-slate-900 font-medium'
                            : 'bg-white text-slate-800 border border-slate-200'
                        }`}
                      >
                        {/* MESSAGE CONTENT - ENHANCED WITH TABLE SUPPORT */}
                        {m.role === 'user' ? (
                          m.content
                        ) : (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            rehypePlugins={[rehypeRaw]}
                            components={{
                              h3: ({ node, ...props }) => (
                                <h3
                                  className="text-lg font-bold mt-4 mb-2 text-slate-900"
                                  {...props}
                                />
                              ),
                              p: ({ node, ...props }) => (
                                <p
                                  className="mb-3 leading-relaxed"
                                  {...props}
                                />
                              ),
                              ul: ({ node, ...props }) => (
                                <ul
                                  className="list-disc ml-5 mb-3 space-y-1"
                                  {...props}
                                />
                              ),
                              ol: ({ node, ...props }) => (
                                <ol
                                  className="list-decimal ml-5 mb-3 space-y-1"
                                  {...props}
                                />
                              ),
                              li: ({ node, ...props }) => (
                                <li
                                  className="text-sm leading-relaxed"
                                  {...props}
                                />
                              ),
                              strong: ({ node, ...props }) => (
                                <strong
                                  className="font-bold text-slate-900"
                                  {...props}
                                />
                              ),
                              code: ({ node, inline, ...props }: any) =>
                                inline ? (
                                  <code
                                    className="bg-slate-100 px-1.5 py-0.5 rounded text-xs font-mono text-red-600"
                                    {...props}
                                  />
                                ) : (
                                  <code
                                    className="block bg-slate-900 text-emerald-100 p-3 rounded-lg text-xs font-mono overflow-x-auto my-3"
                                    {...props}
                                  />
                                ),
                              blockquote: ({ node, ...props }) => (
                                <blockquote
                                  className="border-l-4 border-amber-400 pl-4 italic text-slate-600 my-3"
                                  {...props}
                                />
                              ),

                              // ====== ENHANCED TABLE COMPONENTS ======
                              table: ({ node, ...props }) => (
                                <div className="overflow-x-auto my-4 rounded-lg border border-slate-300 shadow-sm">
                                  <table
                                    className="min-w-full border-collapse text-xs"
                                    {...props}
                                  />
                                </div>
                              ),
                              thead: ({ node, ...props }) => (
                                <thead
                                  className="bg-gradient-to-r from-slate-100 to-slate-50 sticky top-0"
                                  {...props}
                                />
                              ),
                              tbody: ({ node, ...props }) => (
                                <tbody
                                  className="bg-white divide-y divide-slate-200"
                                  {...props}
                                />
                              ),
                              tr: ({ node, ...props }) => (
                                <tr
                                  className="hover:bg-amber-50 transition-colors"
                                  {...props}
                                />
                              ),
                              th: ({ node, ...props }) => (
                                <th
                                  className="border-b-2 border-slate-300 px-4 py-2.5 text-left text-xs font-bold text-slate-800 uppercase tracking-wide"
                                  {...props}
                                />
                              ),
                              td: ({ node, ...props }) => (
                                <td
                                  className="border-b border-slate-200 px-4 py-2.5 text-slate-700 whitespace-nowrap"
                                  {...props}
                                />
                              ),
                            }}
                          >
                            {m.content}
                          </ReactMarkdown>
                        )}

                        {m.role === 'assistant' && (
                          <>
                            {/* FEEDBACK BUTTONS */}
                            <div className="mt-4 pt-4 border-t border-slate-200 flex items-center gap-2">
                              <span className="text-xs text-slate-500">
                                Was this helpful?
                              </span>
                              <button
                                className={`px-2 py-1 rounded text-xs transition ${
                                  feedbackStates[m.id] === 'like'
                                    ? 'bg-green-100 text-green-700'
                                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                }`}
                                onClick={() => submitFeedback(m.id, 'like')}
                              >
                                👍 Like
                              </button>
                              <button
                                className={`px-2 py-1 rounded text-xs transition ${
                                  feedbackStates[m.id] === 'dislike'
                                    ? 'bg-red-100 text-red-700'
                                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                }`}
                                onClick={() => submitFeedback(m.id, 'dislike')}
                              >
                                👎 Dislike
                              </button>
                            </div>

                            {/* SOURCES & CONFIDENCE */}
                            {m.sources && m.sources.length > 0 && (
                              <div className="mt-3 pt-3 border-t border-slate-200">
                                <div className="text-xs font-semibold text-slate-600 mb-2">
                                  📎 Sources
                                </div>
                                <ul className="text-xs text-slate-600 space-y-1">
                                  {m.sources
                                    .slice(0, 4)
                                    .map((s: any, idx: number) => (
                                      <li key={idx}>
                                        • {s.filename || s.title || 'Source'}{' '}
                                        {s.page && `(p. ${s.page})`}
                                      </li>
                                    ))}
                                </ul>
                              </div>
                            )}

                            {typeof m.confidence === 'number' && (
                              <div className="mt-2 text-xs text-slate-500">
                                ⭐ Confidence: {Math.round(m.confidence * 100)}%
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  ))}

                  {loading && (
                    <div className="flex justify-start">
                      <div className="inline-flex items-center gap-2 px-5 py-4 rounded-xl bg-white text-xs text-slate-600 shadow-md border border-slate-200">
                        <div className="flex gap-1">
                          <span className="h-2 w-2 rounded-full bg-amber-400 animate-bounce"></span>
                          <span
                            className="h-2 w-2 rounded-full bg-amber-400 animate-bounce"
                            style={{ animationDelay: '0.2s' }}
                          ></span>
                          <span
                            className="h-2 w-2 rounded-full bg-amber-400 animate-bounce"
                            style={{ animationDelay: '0.4s' }}
                          ></span>
                        </div>
                        <span>Jericho is thinking...</span>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              </div>

              {/* INPUT AREA */}
              <div className="border-t border-slate-200 bg-white p-4 shadow-lg">
                <div className="max-w-3xl mx-auto flex items-end gap-3">
                  {/* UPLOAD BUTTON */}
                  <button
                    className="p-3 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 transition disabled:opacity-50"
                    onClick={() => {
                      setShowUpload(true)
                      setUploadFiles(null)
                      setUploadStatus(null)
                    }}
                    disabled={!currentSessionId}
                    title="Upload documents"
                  >
                    📎
                  </button>

                  {/* VOICE BUTTON */}
                  <button
                    className={`p-3 rounded-lg transition disabled:opacity-50 ${
                      isListening
                        ? 'bg-red-500 text-white'
                        : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                    }`}
                    onClick={toggleVoice}
                    disabled={!currentSessionId}
                    title={isListening ? 'Stop listening' : 'Start voice input'}
                  >
                    🎤
                  </button>

                  {/* TEXT INPUT */}
                  <textarea
                    className="flex-1 rounded-lg border border-slate-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400 focus:border-transparent resize-none"
                    rows={1}
                    placeholder={
                      currentSessionId
                        ? 'Ask about transcripts, payroll, board meetings, or policies...'
                        : 'Waiting for session...'
                    }
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        if (!loading && input.trim()) handleSend()
                      }
                    }}
                    disabled={!currentSessionId}
                  />

                  {/* SEND BUTTON */}
                  <button
                    className="p-3 rounded-lg bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-slate-900 font-bold transition disabled:opacity-50"
                    disabled={loading || !input.trim() || !currentSessionId}
                    onClick={handleSend}
                    title="Send message"
                  >
                    {loading ? '⏳' : '➤'}
                  </button>
                </div>
              </div>
            </>
          ) : (
            // ADMIN VIEW
            <div className="flex-1 overflow-y-auto p-6 bg-gradient-to-b from-slate-50 to-slate-100">
              <div className="max-w-4xl mx-auto space-y-6">
                <h2 className="text-2xl font-bold text-slate-900">
                  ⚙️ Admin Console
                </h2>

                {/* STATS */}
                <div className="bg-white rounded-xl border border-slate-200 shadow-md p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-slate-900">
                      📊 RAG Statistics
                    </h3>
                    <button
                      className="text-xs px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 transition"
                      onClick={loadAdminStats}
                      disabled={adminStatsLoading}
                    >
                      {adminStatsLoading ? '⏳ Refreshing...' : '🔄 Refresh'}
                    </button>
                  </div>
                  {adminStats && (
                    <div className="text-sm text-slate-700 space-y-2">
                      <div>
                        <span className="font-semibold">Total Documents:</span>{' '}
                        {adminStats.documents?.total ?? 'N/A'}
                      </div>
                      {adminStats.documents?.by_type && (
                        <ul className="text-xs text-slate-600 space-y-1 ml-4">
                          {Object.entries(
                            adminStats.documents.by_type as Record<
                              string,
                              number
                            >
                          ).map(([ext, count]) => (
                            <li key={ext}>
                              • {ext}: {count}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>

                {/* TEST QUERY */}
                <div className="bg-white rounded-xl border border-slate-200 shadow-md p-6">
                  <h3 className="text-lg font-semibold text-slate-900 mb-3">
                    🧪 Test Orchestrator
                  </h3>
                  <p className="text-xs text-slate-600 mb-4">
                    Send test queries and inspect tools, sources, and
                    confidence.
                  </p>
                  <div className="flex gap-2 mb-4">
                    <input
                      type="text"
                      className="flex-1 rounded-lg border border-slate-300 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                      placeholder="E.g. What is the check date for Pay period 3?"
                      value={adminTestQuery}
                      onChange={(e) => setAdminTestQuery(e.target.value)}
                    />
                    <button
                      className="px-4 py-2.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-900 font-bold text-sm transition disabled:opacity-50"
                      disabled={adminTestLoading || !adminTestQuery.trim()}
                      onClick={handleAdminTestQuery}
                    >
                      {adminTestLoading ? '⏳' : 'Run'}
                    </button>
                  </div>
                  {adminTestResponse && (
                    <pre className="mt-4 text-xs bg-slate-900 text-emerald-100 rounded-lg p-4 overflow-auto max-h-72 font-mono">
                      {JSON.stringify(adminTestResponse, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* UPLOAD MODAL */}
      {showUpload && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg p-6 mx-4">
            <h2 className="text-xl font-bold mb-2 text-slate-900">
              📤 Upload Documents
            </h2>
            <p className="text-xs text-slate-600 mb-4">
              Files will be parsed by Jericho and added to your knowledge base.
            </p>

            <div className="space-y-4">
              <div className="border-2 border-dashed border-slate-300 rounded-lg p-6 hover:border-amber-400 transition">
                <input
                  type="file"
                  multiple
                  className="block w-full text-sm"
                  onChange={(e) => setUploadFiles(e.target.files)}
                />
              </div>

              <label className="flex items-center gap-3 p-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition cursor-pointer">
                <input
                  type="checkbox"
                  className="rounded border-slate-300"
                  checked={uploadPrivate}
                  onChange={(e) => setUploadPrivate(e.target.checked)}
                />
                <div>
                  <div className="text-sm font-medium text-slate-900">
                    Private Upload
                  </div>
                  <div className="text-xs text-slate-600">
                    Only visible in this session, not shared
                  </div>
                </div>
              </label>

              {uploadStatus && (
                <div className="text-xs text-slate-700 bg-slate-100 rounded-lg px-4 py-3">
                  {uploadStatus}
                </div>
              )}
            </div>

            <div className="mt-6 flex justify-end gap-3">
              <button
                className="px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 text-sm font-medium transition"
                onClick={() => setShowUpload(false)}
                disabled={uploadLoading}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 rounded-lg bg-gradient-to-r from-amber-500 to-amber-400 text-slate-900 font-bold text-sm transition disabled:opacity-50"
                onClick={handleUpload}
                disabled={uploadLoading}
              >
                {uploadLoading ? '⏳ Uploading...' : '✓ Upload'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* RENAME MODAL */}
      {showRenameModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-sm p-6 mx-4">
            <h2 className="text-lg font-bold mb-4 text-slate-900">
              ✏️ Rename Chat
            </h2>
            <input
              type="text"
              className="w-full border border-slate-300 rounded-lg px-4 py-2.5 text-sm mb-4 focus:outline-none focus:ring-2 focus:ring-amber-400"
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
            />
            <div className="flex justify-end gap-3">
              <button
                className="px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 text-sm font-medium transition"
                onClick={() => setShowRenameModal(false)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-900 font-bold text-sm transition"
                onClick={handleRenameSubmit}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
