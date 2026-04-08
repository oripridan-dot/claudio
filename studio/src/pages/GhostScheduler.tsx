import React, { useState } from 'react'
import { Calendar, Clock, Music2, Sliders, Play, Pause, Ghost } from 'lucide-react'

interface GhostProfile {
  id: string
  musicianName: string
  instrument: string
  ghostHash: string
  sessionsHosted: number
  tokensEarned: number
}

interface ScheduledSession {
  id: string
  ghostId: string
  scheduledAt: string
  tempo: number
  genre: string
  reactivity: number   // 0–100
  status: 'scheduled' | 'running' | 'completed'
}

const DEMO_GHOSTS: GhostProfile[] = [
  {
    id: 'g_001',
    musicianName: 'Studio Ghost (You)',
    instrument: 'Bass',
    ghostHash: 'a3f9c12d84eb6a75',
    sessionsHosted: 47,
    tokensEarned: 2.341,
  },
]

const DEMO_SESSIONS: ScheduledSession[] = [
  {
    id: 's_001',
    ghostId: 'g_001',
    scheduledAt: '2026-03-14T10:00',
    tempo: 98,
    genre: 'Neo Soul',
    reactivity: 75,
    status: 'scheduled',
  },
  {
    id: 's_002',
    ghostId: 'g_001',
    scheduledAt: '2026-03-13T22:00',
    tempo: 120,
    genre: 'Funk',
    reactivity: 90,
    status: 'completed',
  },
]

const GENRES = ['Blues', 'Funk', 'Jazz', 'Neo Soul', 'R&B', 'Rock', 'Electronic', 'Ambient']

export default function GhostScheduler() {
  const [ghosts]  = useState<GhostProfile[]>(DEMO_GHOSTS)
  const [sessions, setSessions] = useState<ScheduledSession[]>(DEMO_SESSIONS)
  const [form, setForm] = useState({
    scheduledAt: '',
    tempo: 120,
    genre: 'Funk',
    reactivity: 70,
  })

  const scheduleSession = () => {
    if (!form.scheduledAt) return
    const newSession: ScheduledSession = {
      id: `s_${Date.now().toString(36)}`,
      ghostId: ghosts[0].id,
      scheduledAt: form.scheduledAt,
      tempo: form.tempo,
      genre: form.genre,
      reactivity: form.reactivity,
      status: 'scheduled',
    }
    setSessions(ss => [newSession, ...ss])
    setForm(f => ({ ...f, scheduledAt: '' }))
  }

  const statusColour = (s: ScheduledSession['status']) =>
    s === 'running' ? 'text-claudio-safe' :
    s === 'completed' ? 'text-claudio-muted' :
    'text-claudio-accent'

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-claudio-text">Ghost Session Scheduler</h2>
        <p className="text-sm text-claudio-muted mt-1">
          Let your digital twin jam while you sleep. Earn tokens. Own the rights.
        </p>
      </div>

      {/* Ghost Profiles */}
      <div className="mb-6 space-y-3">
        {ghosts.map(g => (
          <div key={g.id} className="bg-claudio-surface border border-claudio-border rounded-xl p-5 flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-claudio-accent/20 flex items-center justify-center">
              <Ghost size={18} className="text-claudio-accent" />
            </div>
            <div className="flex-1">
              <div className="text-sm font-semibold text-claudio-text">{g.musicianName}</div>
              <div className="text-xs text-claudio-muted">{g.instrument} · <span className="font-mono">{g.ghostHash}</span></div>
            </div>
            <div className="text-right">
              <div className="text-sm font-bold text-claudio-accent">{g.tokensEarned.toFixed(3)} tokens</div>
              <div className="text-xs text-claudio-muted">{g.sessionsHosted} sessions hosted</div>
            </div>
          </div>
        ))}
      </div>

      {/* Schedule Form */}
      <div className="bg-claudio-surface border border-claudio-border rounded-xl p-6 mb-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-claudio-muted mb-4">
          New Async Session
        </h3>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <label className="block">
            <span className="text-xs text-claudio-muted">Scheduled At</span>
            <input
              data-testid="schedule-datetime"
              type="datetime-local"
              value={form.scheduledAt}
              onChange={e => setForm(f => ({ ...f, scheduledAt: e.target.value }))}
              className="mt-1 w-full bg-claudio-panel border border-claudio-border rounded-lg px-3 py-2 text-sm text-claudio-text focus:outline-none focus:border-claudio-accent"
            />
          </label>
          <label className="block">
            <span className="text-xs text-claudio-muted">Tempo (BPM)</span>
            <input
              type="number" min={40} max={300}
              value={form.tempo}
              onChange={e => setForm(f => ({ ...f, tempo: +e.target.value }))}
              className="mt-1 w-full bg-claudio-panel border border-claudio-border rounded-lg px-3 py-2 text-sm text-claudio-text focus:outline-none focus:border-claudio-accent"
            />
          </label>
          <label className="block">
            <span className="text-xs text-claudio-muted">Genre</span>
            <select
              value={form.genre}
              onChange={e => setForm(f => ({ ...f, genre: e.target.value }))}
              className="mt-1 w-full bg-claudio-panel border border-claudio-border rounded-lg px-3 py-2 text-sm text-claudio-text focus:outline-none focus:border-claudio-accent"
            >
              {GENRES.map(g => <option key={g}>{g}</option>)}
            </select>
          </label>
          <label className="block">
            <span className="text-xs text-claudio-muted">Ghost Reactivity {form.reactivity}%</span>
            <input
              type="range" min={0} max={100}
              value={form.reactivity}
              onChange={e => setForm(f => ({ ...f, reactivity: +e.target.value }))}
              className="mt-2 w-full accent-amber-400"
            />
          </label>
        </div>
        <button
          data-testid="schedule-session-btn"
          onClick={scheduleSession}
          className="w-full flex items-center justify-center gap-2 bg-claudio-accent text-black font-bold py-2.5 rounded-lg hover:bg-claudio-gold transition-colors text-sm"
        >
          <Calendar size={14} />
          Schedule Ghost Session
        </button>
      </div>

      {/* Session List */}
      <div className="space-y-2">
        {sessions.map(s => (
          <div key={s.id} className="bg-claudio-surface border border-claudio-border rounded-lg px-4 py-3 flex items-center gap-4">
            <div className={`w-2 h-2 rounded-full ${s.status === 'running' ? 'bg-green-400 animate-pulse' : s.status === 'completed' ? 'bg-claudio-border' : 'bg-claudio-accent'}`} />
            <div className="flex-1 text-xs">
              <span className="text-claudio-text font-medium">{s.genre}</span>
              <span className="text-claudio-muted ml-2">{s.tempo} bpm · reactivity {s.reactivity}%</span>
            </div>
            <div className="text-xs text-claudio-muted font-mono">
              {new Date(s.scheduledAt).toLocaleString()}
            </div>
            <span className={`text-xs font-medium ${statusColour(s.status)}`}>
              {s.status}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
