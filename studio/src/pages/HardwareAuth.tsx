import React, { useState } from 'react'
import { Cpu, Wifi, CheckCircle, XCircle, RefreshCw, Shield } from 'lucide-react'

interface HardwareDevice {
  id: string
  model: string
  manufacturer: string
  serial: string
  fwVersion: string
  networkIp: string
  sampleRate: number
  status: 'verified' | 'unverified' | 'scanning' | 'failed'
  certBadge: string | null
}

const DEMO_DEVICES: HardwareDevice[] = [
  {
    id: 'hw_001',
    model: 'dLive S5000',
    manufacturer: 'Allen & Heath',
    serial: 'AH-S5-2024-001',
    fwVersion: 'V1.90',
    networkIp: '192.168.1.100',
    sampleRate: 96000,
    status: 'verified',
    certBadge: 'Certified Console Masterclass — 96 kHz FPGA',
  },
  {
    id: 'hw_002',
    model: 'SQ-6',
    manufacturer: 'Allen & Heath',
    serial: 'AH-SQ6-2025-042',
    fwVersion: 'V1.7.2',
    networkIp: '192.168.1.101',
    sampleRate: 96000,
    status: 'unverified',
    certBadge: null,
  },
]

const STATUS_CONFIG = {
  verified:   { icon: CheckCircle, colour: 'text-claudio-safe',   label: 'Verified'  },
  unverified: { icon: XCircle,     colour: 'text-claudio-muted',  label: 'Unverified'},
  scanning:   { icon: RefreshCw,   colour: 'text-claudio-accent', label: 'Scanning…' },
  failed:     { icon: XCircle,     colour: 'text-claudio-danger', label: 'Failed'    },
}

export default function HardwareAuth() {
  const [devices, setDevices] = useState<HardwareDevice[]>(DEMO_DEVICES)

  const scan = (id: string) => {
    setDevices(ds => ds.map(d => d.id === id ? { ...d, status: 'scanning' } : d))
    setTimeout(() => {
      setDevices(ds => ds.map(d =>
        d.id === id
          ? {
              ...d,
              status: 'verified',
              certBadge: `Certified Console Masterclass — ${d.sampleRate / 1000} kHz FPGA`,
            }
          : d
      ))
    }, 2000)
  }

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-claudio-text">Hardware Authentication</h2>
        <p className="text-sm text-claudio-muted mt-1">
          Verify consoles and interfaces. Certified gear badges your session for premium listeners.
        </p>
      </div>

      <div className="space-y-4">
        {devices.map(device => {
          const cfg = STATUS_CONFIG[device.status]
          const Icon = cfg.icon
          return (
            <div
              key={device.id}
              data-testid={`hw-card-${device.id}`}
              className="bg-claudio-surface border border-claudio-border rounded-xl p-6"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-claudio-panel flex items-center justify-center">
                    <Cpu size={18} className="text-claudio-accent" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-claudio-text">
                      {device.manufacturer} {device.model}
                    </div>
                    <div className="text-xs text-claudio-muted">
                      Serial: {device.serial} · FW {device.fwVersion}
                    </div>
                  </div>
                </div>
                <div className={`flex items-center gap-1.5 text-xs font-medium ${cfg.colour}`}>
                  <Icon size={13} className={device.status === 'scanning' ? 'animate-spin' : ''} />
                  {cfg.label}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs mb-4">
                <div className="bg-claudio-panel rounded-lg px-3 py-2">
                  <span className="text-claudio-muted">IP  </span>
                  <span className="text-claudio-text font-mono">{device.networkIp}</span>
                </div>
                <div className="bg-claudio-panel rounded-lg px-3 py-2">
                  <span className="text-claudio-muted">Sample Rate  </span>
                  <span className="text-claudio-text font-mono">{device.sampleRate / 1000} kHz</span>
                </div>
              </div>

              {device.certBadge && (
                <div className="flex items-center gap-2 bg-claudio-safe/10 border border-claudio-safe/30 rounded-lg px-3 py-2 mb-4">
                  <Shield size={13} className="text-claudio-safe flex-shrink-0" />
                  <span className="text-xs text-claudio-safe">{device.certBadge}</span>
                </div>
              )}

              {device.status !== 'verified' && (
                <button
                  data-testid={`verify-btn-${device.id}`}
                  onClick={() => scan(device.id)}
                  disabled={device.status === 'scanning'}
                  className="w-full flex items-center justify-center gap-2 bg-claudio-accent text-black font-bold py-2 rounded-lg hover:bg-claudio-gold transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Wifi size={13} />
                  {device.status === 'scanning' ? 'Authenticating…' : 'Verify Hardware'}
                </button>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
