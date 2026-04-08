import React, { useState } from 'react'
import { Users, Lock, Mic2, DollarSign, Play, Plus, Radio } from 'lucide-react'

type Tier = 'observer' | 'student' | 'band'

interface VenueConfig {
  name: string
  bpm: number
  key: string
  maxBandMembers: number
  ticketPrices: Record<Tier, number>
  hwVerified: boolean
  isLive: boolean
}

const TIERS: { id: Tier; label: string; desc: string; icon: React.ElementType }[] = [
  { id: 'observer', label: 'Observer',    desc: 'Listen-only binaural stream',              icon: Radio },
  { id: 'student',  label: 'Student',     desc: 'Voice/chat — no instrument injection',     icon: Mic2  },
  { id: 'band',     label: 'Band Member', desc: 'Full zero-latency bidirectional audio',    icon: Users },
]

const DEFAULT_VENUE: VenueConfig = {
  name: 'My Room',
  bpm: 120,
  key: 'C Major',
  maxBandMembers: 4,
  ticketPrices: { observer: 0, student: 5, band: 25 },
  hwVerified: false,
  isLive: false,
}

export default function VenueManager() {
  const [venue, setVenue] = useState<VenueConfig>(DEFAULT_VENUE)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

  const launchVenue = () => {
    const sid = `session_${Date.now().toString(36)}`
    setActiveSessionId(sid)
    setVenue(v => ({ ...v, isLive: true }))
  }

  const closeVenue = () => {
    setActiveSessionId(null)
    setVenue(v => ({ ...v, isLive: false }))
  }

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-claudio-text">Virtual Venue Manager</h2>
          <p className="text-sm text-claudio-muted mt-1">
            Configure your room, set ticket tiers, and open the doors.
          </p>
        </div>
        {venue.isLive && (
          <div className="flex items-center gap-2 bg-red-900/30 border border-red-500/40 px-3 py-1.5 rounded-full">
            <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
            <span className="text-xs text-red-300 font-medium uppercase tracking-wider">
              Live — {activeSessionId?.slice(0, 12)}
            </span>
          </div>
        )}
      </div>

      {/* Room Config */}
      <div className="bg-claudio-surface border border-claudio-border rounded-xl p-6 mb-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-claudio-muted mb-4">
          Room Configuration
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <label className="block">
            <span className="text-xs text-claudio-muted">Venue Name</span>
            <input
              data-testid="venue-name"
              value={venue.name}
              onChange={e => setVenue(v => ({ ...v, name: e.target.value }))}
              className="mt-1 w-full bg-claudio-panel border border-claudio-border rounded-lg px-3 py-2 text-sm text-claudio-text focus:outline-none focus:border-claudio-accent"
            />
          </label>
          <label className="block">
            <span className="text-xs text-claudio-muted">BPM</span>
            <input
              data-testid="venue-bpm"
              type="number"
              min={40} max={300}
              value={venue.bpm}
              onChange={e => setVenue(v => ({ ...v, bpm: +e.target.value }))}
              className="mt-1 w-full bg-claudio-panel border border-claudio-border rounded-lg px-3 py-2 text-sm text-claudio-text focus:outline-none focus:border-claudio-accent"
            />
          </label>
          <label className="block">
            <span className="text-xs text-claudio-muted">Key / Mode</span>
            <input
              value={venue.key}
              onChange={e => setVenue(v => ({ ...v, key: e.target.value }))}
              className="mt-1 w-full bg-claudio-panel border border-claudio-border rounded-lg px-3 py-2 text-sm text-claudio-text focus:outline-none focus:border-claudio-accent"
            />
          </label>
          <label className="block">
            <span className="text-xs text-claudio-muted">Max Band Members</span>
            <input
              type="number" min={1} max={12}
              value={venue.maxBandMembers}
              onChange={e => setVenue(v => ({ ...v, maxBandMembers: +e.target.value }))}
              className="mt-1 w-full bg-claudio-panel border border-claudio-border rounded-lg px-3 py-2 text-sm text-claudio-text focus:outline-none focus:border-claudio-accent"
            />
          </label>
        </div>
      </div>

      {/* Ticketing Tiers */}
      <div className="bg-claudio-surface border border-claudio-border rounded-xl p-6 mb-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-claudio-muted mb-4">
          Ticket Tiers
        </h3>
        <div className="space-y-3">
          {TIERS.map(({ id, label, desc, icon: Icon }) => (
            <div key={id} className="flex items-center gap-4 bg-claudio-panel rounded-lg px-4 py-3">
              <Icon size={16} className="text-claudio-accent flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-claudio-text">{label}</div>
                <div className="text-xs text-claudio-muted">{desc}</div>
              </div>
              <div className="flex items-center gap-1">
                <DollarSign size={13} className="text-claudio-muted" />
                <input
                  data-testid={`price-${id}`}
                  type="number"
                  min={0}
                  value={venue.ticketPrices[id]}
                  onChange={e =>
                    setVenue(v => ({
                      ...v,
                      ticketPrices: { ...v.ticketPrices, [id]: +e.target.value },
                    }))
                  }
                  className="w-20 bg-claudio-surface border border-claudio-border rounded px-2 py-1 text-sm text-claudio-text text-right focus:outline-none focus:border-claudio-accent"
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Hardware Badge */}
      <div className="bg-claudio-surface border border-claudio-border rounded-xl p-6 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Lock size={16} className={venue.hwVerified ? 'text-claudio-safe' : 'text-claudio-muted'} />
            <div>
              <div className="text-sm font-medium">
                {venue.hwVerified ? 'Certified Console Masterclass' : 'Hardware Unverified'}
              </div>
              <div className="text-xs text-claudio-muted">
                {venue.hwVerified
                  ? 'Genuine 96 kHz FPGA-processed signal — Allen & Heath verified'
                  : 'Verify your console for the certified badge'}
              </div>
            </div>
          </div>
          <button
            data-testid="verify-hw-btn"
            onClick={() => setVenue(v => ({ ...v, hwVerified: !v.hwVerified }))}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
              venue.hwVerified
                ? 'border-claudio-safe/40 text-claudio-safe hover:bg-claudio-safe/10'
                : 'border-claudio-border text-claudio-muted hover:border-claudio-accent hover:text-claudio-accent'
            }`}
          >
            {venue.hwVerified ? 'Verified ✓' : 'Verify Hardware'}
          </button>
        </div>
      </div>

      {/* Launch */}
      {!venue.isLive ? (
        <button
          data-testid="launch-venue-btn"
          onClick={launchVenue}
          className="w-full flex items-center justify-center gap-2 bg-claudio-accent text-black font-bold py-3 rounded-xl hover:bg-claudio-gold transition-colors"
        >
          <Play size={16} fill="currentColor" />
          Open the Doors
        </button>
      ) : (
        <button
          onClick={closeVenue}
          className="w-full flex items-center justify-center gap-2 bg-red-900/30 border border-red-500/40 text-red-300 font-semibold py-3 rounded-xl hover:bg-red-900/50 transition-colors"
        >
          Close Venue
        </button>
      )}
    </div>
  )
}
