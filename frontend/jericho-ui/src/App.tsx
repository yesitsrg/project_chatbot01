import React, { useState, useRef, useEffect } from 'react'

type Screen = 'login' | 'chat'

type ChatMessage = {
  id: number
  role: 'user' | 'assistant'
  content: string
}

function App() {
  const [screen, setScreen] = useState<Screen>('login')
  const [username, setUsername] = useState<string>('')
  const [password, setPassword] = useState<string>('')
  const [role, setRole] = useState<string | null>(null)
  const [loginError, setLoginError] = useState<string | null>(null)
  const [loginLoading, setLoginLoading] = useState(false)

  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [nextId, setNextId] = useState(1)

  // Upload state
  const [showUpload, setShowUpload] = useState(false)
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null)
  const [uploadPrivate, setUploadPrivate] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)
  const [uploadLoading, setUploadLoading] = useState(false)

  // Auto-scroll anchor
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

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
        credentials: 'include', // important for cookies
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

  const handleSend = async () => {
    const q = input.trim()
    if (!q) return
    setInput('')

    const userMessage: ChatMessage = {
      id: nextId,
      role: 'user',
      content: q,
    }
    setNextId(nextId + 1)
    setMessages((prev) => [...prev, userMessage])
    setLoading(true)

    try {
      const resp = await fetch('http://localhost:8000/api/v1/react-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, sessionid: 1, private: false }),
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      const assistantMessage: ChatMessage = {
        id: nextId + 1,
        role: 'assistant',
        content: data.answer ?? 'No answer field in response.',
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

  const handleUpload = async () => {
    if (!uploadFiles || uploadFiles.length === 0) {
      setUploadStatus('Please select at least one file.')
      return
    }
    setUploadLoading(true)
    setUploadStatus(null)
    try {
      const form = new FormData()
      // backend expects: files (one or more), session_id, private
      Array.from(uploadFiles).forEach((file) => {
        form.append('files', file)
      })
      form.append('session_id', '1')
      form.append('private', uploadPrivate ? 'true' : 'false')

      const resp = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: form,
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
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

  if (screen === 'login') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Top-left logos */}
        <div className="absolute top-6 left-6 flex items-center gap-3">
          <img
            src="https://www.dinecollege.edu/wp-content/uploads/2024/12/dc_logoFooter.png"
            alt="Diné College"
            className="h-10 rounded-md object-contain"
          />
          <img
            src="http://localhost:8000/static/jericho_image.jpg"
            alt="Jericho"
            className="h-8 object-contain"
          />
        </div>

        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl shadow-2xl p-8 w-full max-w-md text-white">
          <h1 className="text-2xl font-semibold mb-2">
            Diné College Assistant
          </h1>
          <p className="text-sm text-slate-200 mb-4">
            Sign in with your Jericho account. Default admin:{' '}
            <span className="font-mono text-amber-300">admin / admin123</span>
          </p>

          {loginError && (
            <div className="mb-4 text-sm text-red-300 bg-red-900/40 border border-red-400/60 rounded px-3 py-2">
              {loginError}
            </div>
          )}

          <form className="space-y-4" onSubmit={handleLogin}>
            <div>
              <label className="block text-sm mb-1">Username</label>
              <input
                type="text"
                className="w-full rounded-md border border-slate-500 bg-slate-900/60 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Password</label>
              <input
                type="password"
                className="w-full rounded-md border border-slate-500 bg-slate-900/60 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <button
              type="submit"
              className="w-full mt-2 rounded-md bg-amber-500 hover:bg-amber-400 text-slate-900 font-semibold py-2 text-sm transition disabled:opacity-60"
              disabled={loginLoading}
            >
              {loginLoading ? 'Signing in...' : 'Login'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  // Chat screen
  return (
    <div className="min-h-screen flex flex-col bg-slate-100">
      <header className="h-14 px-4 flex items-center justify-between bg-white border-b">
        <div className="flex items-center gap-3">
          <img
            src="https://www.dinecollege.edu/wp-content/uploads/2024/12/dc_logoFooter.png"
            alt="Diné College"
            className="h-8 rounded-md object-contain"
          />
          <img
            src="http://localhost:8000/static/jericho_image.jpg"
            alt="Jericho"
            className="h-7 object-contain"
          />
          <span className="font-semibold text-slate-800">
            Diné College Assistant (Jericho)
          </span>
        </div>
        <div className="flex items-center gap-3 text-sm text-slate-600">
          <span>{username || 'User'}</span>
          {role === 'admin' && (
            <span className="px-2 py-0.5 rounded-full bg-slate-900 text-amber-300 text-xs">
              Admin
            </span>
          )}
        </div>
      </header>
      <main className="flex-1 flex">
        <aside className="w-64 border-r bg-white p-3">
          <button className="w-full mb-3 rounded-md bg-amber-500 hover:bg-amber-400 text-sm font-semibold py-2 text-slate-900">
            New chat
          </button>
          <div className="text-xs text-slate-500 px-1">Chats</div>
          <div className="mt-2 text-xs text-slate-400">
            (Chat history UI coming next)
          </div>
        </aside>
        <section className="flex-1 flex flex-col">
          <div className="flex-1 p-4 overflow-auto bg-slate-50">
            <div className="max-w-3xl mx-auto space-y-3">
              {messages.length === 0 && (
                <div className="text-slate-500 text-sm">
                  Start a conversation by asking a question about Diné College.
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
                    className={`max-w-xl px-4 py-2 rounded-lg text-sm whitespace-pre-wrap shadow-sm ${
                      m.role === 'user'
                        ? 'bg-amber-500 text-slate-900'
                        : 'bg-white text-slate-800'
                    }`}
                  >
                    {m.content}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex justify-start">
                  <div className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white text-xs text-slate-500 shadow-sm">
                    <span className="inline-block h-2 w-2 rounded-full bg-amber-400 animate-bounce"></span>
                    <span>Jericho is generating an answer…</span>
                  </div>
                </div>
              )}

              {/* anchor for auto-scroll */}
              <div ref={messagesEndRef} />
            </div>
          </div>
          <div className="border-t bg-white p-3 flex items-center gap-2">
            <button
              className="rounded-md border border-slate-300 text-sm px-3 py-2 text-slate-700 hover:bg-slate-100"
              onClick={() => {
                setShowUpload(true)
                setUploadFiles(null)
                setUploadStatus(null)
              }}
            >
              Upload
            </button>
            <textarea
              className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
              rows={1}
              placeholder="Ask a question about Diné College…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  if (!loading) handleSend()
                }
              }}
            />
            <button
              className="rounded-md bg-amber-500 hover:bg-amber-400 text-sm font-semibold px-4 py-2 text-slate-900 disabled:opacity-60"
              disabled={loading || !input.trim()}
              onClick={handleSend}
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </section>
      </main>

      {/* Upload modal */}
      {showUpload && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-lg p-6">
            <h2 className="text-lg font-semibold mb-3">Upload documents</h2>
            <p className="text-xs text-slate-500 mb-3">
              Files will be parsed by Jericho and added to the knowledge base.
            </p>
            <div className="space-y-3">
              <div>
                <input
                  type="file"
                  multiple
                  className="block w-full text-sm text-slate-700"
                  onChange={(e) => setUploadFiles(e.target.files)}
                />
              </div>
              <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                <input
                  type="checkbox"
                  className="rounded border-slate-300"
                  checked={uploadPrivate}
                  onChange={(e) => setUploadPrivate(e.target.checked)}
                />
                <span>Private upload (visible only to you)</span>
              </label>
              {uploadStatus && (
                <div className="text-xs text-slate-600 bg-slate-100 rounded px-3 py-2">
                  {uploadStatus}
                </div>
              )}
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button
                className="px-3 py-2 text-sm rounded-md border border-slate-300 text-slate-700 hover:bg-slate-100"
                onClick={() => {
                  setShowUpload(false)
                }}
                disabled={uploadLoading}
              >
                Close
              </button>
              <button
                className="px-4 py-2 text-sm rounded-md bg-amber-500 text-slate-900 font-semibold hover:bg-amber-400 disabled:opacity-60"
                onClick={handleUpload}
                disabled={uploadLoading}
              >
                {uploadLoading ? 'Uploading...' : 'Upload'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
