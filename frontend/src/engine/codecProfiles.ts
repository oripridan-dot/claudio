/**
 * codecProfiles.ts — Bluetooth codec latency + quality profiles
 *
 * Typical total round-trip latency (encode + transmit + decode) observed on
 * Android / iOS / macOS hardware.  Values represent the *additional* scheduling
 * offset the AudioEngine should pre-buffer to keep MIDI/click playback in sync.
 *
 * Sources: aptX white-paper, LDAC Sony technical spec, SBC RFC-3016 model.
 */

export interface CodecProfile {
  id:           string;
  label:        string;
  latencyMs:    number;   // nominal add-on latency (ms)
  latencyMinMs: number;   // best-case
  latencyMaxMs: number;   // worst-case / SBC Adaptive
  bitrate:      string;   // informational
  lossy:        boolean;
  color:        string;   // accent for the UI badge
  description:  string;
}

export const CODEC_PROFILES: CodecProfile[] = [
  {
    id: 'sbc',
    label: 'SBC',
    latencyMs: 170,
    latencyMinMs: 100,
    latencyMaxMs: 250,
    bitrate: '345 kbps',
    lossy: true,
    color: '#ef4444',
    description: 'Mandatory BT baseline. High latency, adequate quality.',
  },
  {
    id: 'aac',
    label: 'AAC',
    latencyMs: 120,
    latencyMinMs: 80,
    latencyMaxMs: 160,
    bitrate: '256 kbps',
    lossy: true,
    color: '#f59e0b',
    description: 'Apple default. Lower latency than SBC, good transparency.',
  },
  {
    id: 'aptx',
    label: 'aptX',
    latencyMs: 40,
    latencyMinMs: 32,
    latencyMaxMs: 60,
    bitrate: '384 kbps',
    lossy: true,
    color: '#22c55e',
    description: 'Qualcomm aptX. Near-CD quality, significantly lower latency.',
  },
  {
    id: 'aptx_hd',
    label: 'aptX HD',
    latencyMs: 50,
    latencyMinMs: 40,
    latencyMaxMs: 80,
    bitrate: '576 kbps',
    lossy: true,
    color: '#0ea5e9',
    description: 'Higher resolution variant. Slightly more latency than aptX.',
  },
  {
    id: 'aptx_ll',
    label: 'aptX LL',
    latencyMs: 16,
    latencyMinMs: 10,
    latencyMaxMs: 22,
    bitrate: '384 kbps',
    lossy: true,
    color: '#00ff88',
    description: 'Low-Latency aptX. Closest to wired monitoring over BT.',
  },
  {
    id: 'ldac',
    label: 'LDAC',
    latencyMs: 90,
    latencyMinMs: 70,
    latencyMaxMs: 130,
    bitrate: '990 kbps',
    lossy: false,
    color: '#6366f1',
    description: 'Sony LDAC. Highest fidelity (near-lossless). Mod. latency.',
  },
  {
    id: 'lc3',
    label: 'LC3 (LE Audio)',
    latencyMs: 20,
    latencyMinMs: 10,
    latencyMaxMs: 40,
    bitrate: '320 kbps',
    lossy: false,
    color: '#c084fc',
    description: 'Bluetooth LE Audio codec. Very low latency + high quality.',
  },
  {
    id: 'wired',
    label: 'Wired / USB',
    latencyMs: 0,
    latencyMinMs: 0,
    latencyMaxMs: 2,
    bitrate: 'PCM',
    lossy: false,
    color: '#94a3b8',
    description: 'No Bluetooth path. Reference mode — zero compensation.',
  },
];

export const DEFAULT_CODEC_ID = 'wired';

export function getCodecById(id: string): CodecProfile {
  return CODEC_PROFILES.find(c => c.id === id) ?? CODEC_PROFILES[CODEC_PROFILES.length - 1];
}
