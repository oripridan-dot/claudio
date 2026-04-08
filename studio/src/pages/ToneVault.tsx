import React, { useState } from 'react'
import { ShoppingBag, Star, Lock, Download, Plus } from 'lucide-react'

interface ToneProfile {
  id: string
  name: string
  gear: string
  artist: string
  tags: string[]
  rating: number
  downloads: number
  price: number
  owned: boolean
  certified: boolean
}

const DEMO_PROFILES: ToneProfile[] = [
  {
    id: 'tp_001',
    name: 'Neve 1073 + LA-2A',
    gear: 'Neve 1073 preamp → LA-2A optical compressor',
    artist: 'Studio A',
    tags: ['vintage', 'warm', 'vocal'],
    rating: 4.9,
    downloads: 1840,
    price: 12,
    owned: true,
    certified: true,
  },
  {
    id: 'tp_002',
    name: 'Fender Tweed + Spring',
    gear: 'Fender 5E3 Deluxe → Accutronics spring reverb',
    artist: 'Analog Shelf',
    tags: ['guitar', 'tweed', 'sag'],
    rating: 4.7,
    downloads: 962,
    price: 8,
    owned: false,
    certified: true,
  },
  {
    id: 'tp_003',
    name: 'SSL G-Bus Glue',
    gear: 'SSL G-Bus stereo buss compressor',
    artist: 'Mix Lab',
    tags: ['buss', 'glue', 'punch'],
    rating: 4.8,
    downloads: 3107,
    price: 15,
    owned: false,
    certified: false,
  },
  {
    id: 'tp_004',
    name: 'AKG C12 + V72',
    gear: 'AKG C12 tube mic → Telefunken V72 preamp',
    artist: 'Tone Archivist',
    tags: ['vintage mic', 'tube', 'air'],
    rating: 4.6,
    downloads: 741,
    price: 18,
    owned: false,
    certified: true,
  },
]

export default function ToneVault() {
  const [profiles, setProfiles] = useState<ToneProfile[]>(DEMO_PROFILES)
  const [activeTag, setActiveTag] = useState<string | null>(null)
  const [licensed, setLicensed] = useState<string | null>(null)

  const allTags = Array.from(new Set(profiles.flatMap(p => p.tags)))

  const filtered = activeTag
    ? profiles.filter(p => p.tags.includes(activeTag))
    : profiles

  const license = (id: string) => {
    setProfiles(ps => ps.map(p => p.id === id ? { ...p, owned: true } : p))
    setLicensed(id)
    setTimeout(() => setLicensed(null), 2000)
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-claudio-text">Tone Vault</h2>
        <p className="text-sm text-claudio-muted mt-1">
          License certified Neural DDSP gear profiles. Earn from your own.
        </p>
      </div>

      {/* Tag filter */}
      <div className="flex flex-wrap gap-2 mb-6">
        <button
          onClick={() => setActiveTag(null)}
          className={`px-3 py-1 text-xs rounded-full border transition-colors ${
            !activeTag
              ? 'border-claudio-accent text-claudio-accent bg-claudio-accent/10'
              : 'border-claudio-border text-claudio-muted hover:border-claudio-accent/40'
          }`}
        >
          All
        </button>
        {allTags.map(tag => (
          <button
            key={tag}
            onClick={() => setActiveTag(tag === activeTag ? null : tag)}
            className={`px-3 py-1 text-xs rounded-full border transition-colors ${
              activeTag === tag
                ? 'border-claudio-accent text-claudio-accent bg-claudio-accent/10'
                : 'border-claudio-border text-claudio-muted hover:border-claudio-accent/40'
            }`}
          >
            {tag}
          </button>
        ))}
      </div>

      {/* Grid */}
      <div className="grid grid-cols-2 gap-4">
        {filtered.map(profile => (
          <div
            key={profile.id}
            data-testid={`tone-card-${profile.id}`}
            className="bg-claudio-surface border border-claudio-border rounded-xl p-5 flex flex-col gap-3"
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-claudio-text">{profile.name}</span>
                  {profile.certified && (
                    <span className="text-[10px] bg-claudio-accent/20 text-claudio-accent px-1.5 py-0.5 rounded-full font-medium">
                      CERTIFIED
                    </span>
                  )}
                </div>
                <div className="text-xs text-claudio-muted mt-0.5">{profile.gear}</div>
              </div>
            </div>

            <div className="flex items-center gap-3 text-xs text-claudio-muted">
              <span className="flex items-center gap-1">
                <Star size={10} className="text-claudio-accent" fill="currentColor" />
                {profile.rating}
              </span>
              <span className="flex items-center gap-1">
                <Download size={10} />
                {profile.downloads.toLocaleString()}
              </span>
            </div>

            <div className="flex flex-wrap gap-1">
              {profile.tags.map(tag => (
                <span key={tag} className="text-[10px] bg-claudio-panel px-2 py-0.5 rounded text-claudio-muted">
                  {tag}
                </span>
              ))}
            </div>

            <div className="mt-auto flex items-center justify-between">
              <span className="text-sm font-bold text-claudio-accent">
                {profile.price === 0 ? 'Free' : `$${profile.price}`}
              </span>
              {profile.owned ? (
                <span
                  data-testid={`owned-${profile.id}`}
                  className="flex items-center gap-1.5 text-xs text-claudio-safe"
                >
                  <Lock size={11} />
                  In Your Vault
                </span>
              ) : (
                <button
                  data-testid={`license-btn-${profile.id}`}
                  onClick={() => license(profile.id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-claudio-accent text-black font-semibold rounded-lg hover:bg-claudio-gold transition-colors"
                >
                  {licensed === profile.id ? '✓ Licensed!' : (
                    <><ShoppingBag size={11} /> License</>
                  )}
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
