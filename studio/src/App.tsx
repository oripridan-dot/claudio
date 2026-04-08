import React, { useState } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import VenueManager from './pages/VenueManager'
import ToneVault from './pages/ToneVault'
import GhostScheduler from './pages/GhostScheduler'
import CopyrightLedger from './pages/CopyrightLedger'
import HardwareAuth from './pages/HardwareAuth'
import {
  Music, Store, Ghost, BookOpen, Cpu, ChevronRight
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/',          icon: Music,     label: 'Venue'         },
  { to: '/vault',     icon: Store,     label: 'Tone Vault'    },
  { to: '/ghost',     icon: Ghost,     label: 'Ghost Sessions'},
  { to: '/ledger',    icon: BookOpen,  label: 'Ledger'        },
  { to: '/hardware',  icon: Cpu,       label: 'Hardware'      },
]

export default function App() {
  return (
    <div className="flex h-screen bg-claudio-bg text-claudio-text font-studio overflow-hidden">
      {/* Sidebar */}
      <nav className="w-56 flex-shrink-0 bg-claudio-surface border-r border-claudio-border flex flex-col">
        <div className="p-5 border-b border-claudio-border">
          <h1 className="text-xl font-bold tracking-widest text-claudio-accent uppercase">
            Claudio
          </h1>
          <p className="text-xs text-claudio-muted mt-0.5 tracking-wide">
            Creator Studio
          </p>
        </div>
        <div className="flex-1 py-4 space-y-0.5">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
                  isActive
                    ? 'bg-claudio-border text-claudio-accent font-medium'
                    : 'text-claudio-muted hover:text-claudio-text hover:bg-claudio-panel'
                }`
              }
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </div>
        <div className="p-4 border-t border-claudio-border">
          <div className="text-xs text-claudio-muted text-center">
            Claudio Studio v0.1.0
          </div>
        </div>
      </nav>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/"         element={<VenueManager />} />
          <Route path="/vault"    element={<ToneVault />} />
          <Route path="/ghost"    element={<GhostScheduler />} />
          <Route path="/ledger"   element={<CopyrightLedger />} />
          <Route path="/hardware" element={<HardwareAuth />} />
        </Routes>
      </main>
    </div>
  )
}
