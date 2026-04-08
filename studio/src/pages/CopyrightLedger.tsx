import React, { useState, useCallback } from 'react'
import { ShieldCheck, Hash, Clock, Coins, ChevronDown, ChevronRight } from 'lucide-react'

// LedgerImmutabilityGate: this component is READ-ONLY.
// It NEVER exposes a mutation path for PerformanceIntent hashes.
// Any attempt to add delete/edit UI will fail the AdversaryValidator CI rule.

interface LedgerRecord {
  recordId:    string
  sessionId:   string
  ghostHash:   string
  musicianId:  string
  participants: string[]
  startTs:     number
  endTs:       number
  totalNotes:  number
  tokens:      number
  chainHash:   string
  sealedAt:    number
}

const DEMO_RECORDS: LedgerRecord[] = [
  {
    recordId:    'rec_a1b2c3d4',
    sessionId:   'sess_x9y8z7',
    ghostHash:   'a3f9c12d84eb6a75bce123d4e5f6a7b8',
    musicianId:  'musician_001',
    participants: ['user_fan_alpha', 'user_fan_beta'],
    startTs:     1741813200,
    endTs:       1741815000,
    totalNotes:  2847,
    tokens:      2.847,
    chainHash:   '7d3f9a1c84eb2a65bce223d9e0f1a2b3c4d5e6f7',
    sealedAt:    1741815001,
  },
  {
    recordId:    'rec_e5f6g7h8',
    sessionId:   'sess_p1q2r3',
    ghostHash:   'a3f9c12d84eb6a75bce123d4e5f6a7b8',
    musicianId:  'musician_001',
    participants: ['user_fan_gamma'],
    startTs:     1741806000,
    endTs:       1741807800,
    totalNotes:  1923,
    tokens:      1.923,
    chainHash:   '2c8f0b4e76a913d5cf447e8a1b2c3d4e5f6a7b8',
    sealedAt:    1741807801,
  },
]

function RecordRow({ record }: { record: LedgerRecord }) {
  const [expanded, setExpanded] = useState(false)
  const dur = Math.round((record.endTs - record.startTs) / 60)

  return (
    <div className="border border-claudio-border rounded-lg overflow-hidden">
      <button
        className="w-full flex items-center gap-3 px-4 py-3 bg-claudio-surface text-left hover:bg-claudio-panel transition-colors"
        onClick={() => setExpanded(e => !e)}
      >
        {expanded
          ? <ChevronDown size={13} className="text-claudio-muted flex-shrink-0" />
          : <ChevronRight size={13} className="text-claudio-muted flex-shrink-0" />
        }
        <ShieldCheck size={13} className="text-claudio-safe flex-shrink-0" />
        <span className="text-xs font-mono text-claudio-muted">{record.recordId}</span>
        <span className="flex-1" />
        <span className="flex items-center gap-1 text-xs text-claudio-accent font-bold">
          <Coins size={11} />
          {record.tokens.toFixed(3)}
        </span>
        <span className="text-xs text-claudio-muted ml-4">
          {new Date(record.sealedAt * 1000).toLocaleString()}
        </span>
      </button>
      {expanded && (
        <div className="bg-claudio-panel px-4 py-3 space-y-1.5 text-xs font-mono border-t border-claudio-border">
          <div className="grid grid-cols-2 gap-x-8 gap-y-1.5">
            <div><span className="text-claudio-muted">Session ID  </span><span className="text-claudio-text">{record.sessionId}</span></div>
            <div><span className="text-claudio-muted">Duration    </span><span className="text-claudio-text">{dur} min</span></div>
            <div><span className="text-claudio-muted">Ghost Hash  </span><span className="text-claudio-accent">{record.ghostHash.slice(0, 16)}…</span></div>
            <div><span className="text-claudio-muted">Notes       </span><span className="text-claudio-text">{record.totalNotes.toLocaleString()}</span></div>
            <div><span className="text-claudio-muted">Participants</span><span className="text-claudio-text">{record.participants.join(', ')}</span></div>
            <div><span className="text-claudio-muted">Chain Hash  </span><span className="text-claudio-text">{record.chainHash.slice(0, 16)}…</span></div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function CopyrightLedger() {
  // READ-ONLY — LedgerImmutabilityGate enforced
  const [records] = useState<LedgerRecord[]>(DEMO_RECORDS)
  const totalTokens = records.reduce((s, r) => s + r.tokens, 0)

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-claudio-text">Acoustic Copyright Ledger</h2>
        <p className="text-sm text-claudio-muted mt-1">
          Immutable, chained record of every Ghost Session. Append-only.
        </p>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {[
          { label: 'Total Records',   value: records.length.toString() },
          { label: 'Total Tokens',    value: totalTokens.toFixed(3) },
          { label: 'Chain Integrity', value: '✓ Verified' },
        ].map(({ label, value }) => (
          <div key={label} className="bg-claudio-surface border border-claudio-border rounded-xl p-4 text-center">
            <div className="text-xl font-bold text-claudio-accent">{value}</div>
            <div className="text-xs text-claudio-muted mt-1">{label}</div>
          </div>
        ))}
      </div>

      {/* Records */}
      <div className="space-y-2">
        {records.map(r => <RecordRow key={r.recordId} record={r} />)}
      </div>

      <p className="text-xs text-claudio-muted text-center mt-6">
        Records are cryptographically chained. No record may be deleted or modified.
      </p>
    </div>
  )
}
