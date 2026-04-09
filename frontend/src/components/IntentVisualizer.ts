/**
 * IntentVisualizer.ts — Premium canvas-based visualizers for intent data
 *
 * Renders real-time pitch contours, loudness waveforms, and spectrum
 * with professional-grade aesthetics (glows, gradients, grid overlays).
 */
import type { IntentFrame } from '../engine/IntentEngine';

export const PITCH_HISTORY_SIZE = 200;
export const LOUDNESS_HISTORY_SIZE = 200;

// ─── Pitch to Note ──────────────────────────────────────────────────────────

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export function freqToNote(hz: number): string {
  if (hz <= 0) return '—';
  const midi = 12 * Math.log2(hz / 440) + 69;
  const note = Math.round(midi);
  const name = NOTE_NAMES[note % 12];
  const octave = Math.floor(note / 12) - 1;
  const cents = Math.round((midi - note) * 100);
  const sign = cents >= 0 ? '+' : '';
  return `${name}${octave} ${sign}${cents}¢`;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function freqToY(freq: number, h: number): number {
  return h - (Math.log2(freq / 32) / Math.log2(4186 / 32)) * h;
}

// ─── Pitch Contour ──────────────────────────────────────────────────────────

export function drawPitchHistory(
  ctx: CanvasRenderingContext2D,
  history: IntentFrame[],
  w: number,
  h: number,
  color: string,
  label: string,
): void {
  ctx.clearRect(0, 0, w, h);

  // Dark gradient background
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, '#0a0a14');
  bg.addColorStop(0.5, '#0d0d18');
  bg.addColorStop(1, '#080810');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // Grid lines with note labels
  const gridFreqs = [65, 131, 262, 523, 1047, 2093];
  const gridNotes = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7'];
  ctx.lineWidth = 1;
  ctx.font = '9px JetBrains Mono, monospace';
  for (let i = 0; i < gridFreqs.length; i++) {
    const y = freqToY(gridFreqs[i], h);
    // Subtle grid line
    ctx.strokeStyle = '#141428';
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    // Note label
    ctx.fillStyle = '#2a2a44';
    ctx.fillText(gridNotes[i], 4, y + 3);
  }

  if (history.length < 2) {
    // Empty state label
    ctx.fillStyle = '#222';
    ctx.font = '12px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for audio...', w / 2, h / 2);
    ctx.textAlign = 'start';
    return;
  }

  // Draw confidence heatmap behind pitch line
  for (let i = 0; i < history.length; i++) {
    const f = history[i];
    if (f.f0Hz <= 0 || f.confidence < 0.3) continue;
    const x = (i / PITCH_HISTORY_SIZE) * w;
    const y = freqToY(f.f0Hz, h);
    const alpha = Math.min(0.15, f.confidence * 0.2);
    ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0');
    ctx.fillRect(x - 1, y - 8, 3, 16);
  }

  // Main pitch contour with glow
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < history.length; i++) {
    const f = history[i];
    if (f.f0Hz <= 0 || f.confidence < 0.3) { started = false; continue; }
    const x = (i / PITCH_HISTORY_SIZE) * w;
    const y = freqToY(f.f0Hz, h);
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw onset markers
  ctx.shadowBlur = 0;
  for (let i = 0; i < history.length; i++) {
    if (history[i].isOnset) {
      const x = (i / PITCH_HISTORY_SIZE) * w;
      ctx.strokeStyle = '#ffaa0088';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
  }

  // Label badge
  ctx.shadowBlur = 0;
  const lw = ctx.measureText(label).width + 12;
  ctx.fillStyle = color + '20';
  ctx.beginPath();
  ctx.roundRect(w - lw - 8, 6, lw, 18, 4);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.font = 'bold 10px JetBrains Mono, monospace';
  ctx.fillText(label, w - lw - 2, 18);
}

// ─── Loudness Waveform ──────────────────────────────────────────────────────

export function drawLoudnessHistory(
  ctx: CanvasRenderingContext2D,
  history: IntentFrame[],
  w: number,
  h: number,
  color: string,
): void {
  ctx.clearRect(0, 0, w, h);
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, '#0a0a12');
  bg.addColorStop(1, '#08080e');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  if (history.length < 2) return;

  // Mirrored waveform style (like a DAW meter)
  const midY = h / 2;
  ctx.beginPath();
  ctx.moveTo(0, midY);
  // Upper half
  for (let i = 0; i < history.length; i++) {
    const x = (i / LOUDNESS_HISTORY_SIZE) * w;
    const amp = history[i].loudnessNorm * midY * 0.9;
    ctx.lineTo(x, midY - amp);
  }
  // Lower half (mirror)
  for (let i = history.length - 1; i >= 0; i--) {
    const x = (i / LOUDNESS_HISTORY_SIZE) * w;
    const amp = history[i].loudnessNorm * midY * 0.9;
    ctx.lineTo(x, midY + amp);
  }
  ctx.closePath();

  // Fill with gradient
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + '50');
  grad.addColorStop(0.4, color + '30');
  grad.addColorStop(0.5, color + '10');
  grad.addColorStop(0.6, color + '30');
  grad.addColorStop(1, color + '50');
  ctx.fillStyle = grad;
  ctx.fill();

  // Center line
  ctx.strokeStyle = color + '30';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(w, midY);
  ctx.stroke();

  // Edge glow line
  ctx.strokeStyle = color + '80';
  ctx.lineWidth = 1;
  ctx.shadowColor = color;
  ctx.shadowBlur = 4;
  ctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = (i / LOUDNESS_HISTORY_SIZE) * w;
    const amp = history[i].loudnessNorm * midY * 0.9;
    if (i === 0) ctx.moveTo(x, midY - amp);
    else ctx.lineTo(x, midY - amp);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

// ─── Styles ─────────────────────────────────────────────────────────────────

export const collabStyles: Record<string, React.CSSProperties> = {
  container: {
    width: '100vw', height: '100vh',
    background: 'linear-gradient(155deg, #06060c 0%, #0c0c18 40%, #08081a 70%, #050510 100%)',
    display: 'flex', flexDirection: 'column',
    fontFamily: "'Inter', 'JetBrains Mono', system-ui, sans-serif",
    color: '#ccc', overflow: 'hidden',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '14px 24px', borderBottom: '1px solid rgba(255,255,255,0.04)',
    background: 'linear-gradient(180deg, rgba(15,15,28,0.95) 0%, rgba(10,10,20,0.9) 100%)',
    backdropFilter: 'blur(20px)',
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: '14px' },
  title: { fontSize: '20px', fontWeight: 700, margin: 0, color: '#f0f0ff', letterSpacing: '0.5px' },
  titleAccent: { color: '#00ff88', fontSize: '24px', fontWeight: 800 },
  titleSub: { color: '#555', fontWeight: 400, fontSize: '15px', marginLeft: '2px' },
  statusDot: { width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0 },
  statusText: { fontSize: '12px', color: '#777', fontFamily: "'JetBrains Mono', monospace" },
  disconnectBtn: {
    padding: '7px 18px', border: '1px solid rgba(255,68,68,0.4)', borderRadius: '8px',
    background: 'rgba(255,68,68,0.08)', color: '#ff5555', cursor: 'pointer',
    fontSize: '12px', fontWeight: 500, transition: 'all 0.15s',
  },
  connectPanel: { flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  connectCard: {
    background: 'linear-gradient(145deg, rgba(18,18,34,0.95), rgba(14,14,26,0.98))',
    border: '1px solid rgba(255,255,255,0.06)', borderRadius: '20px',
    padding: '44px 40px', width: '400px',
    boxShadow: '0 24px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04)',
    backdropFilter: 'blur(30px)',
  },
  connectTitle: {
    margin: '0 0 28px 0', color: '#e8e8ff', fontSize: '20px', fontWeight: 600,
    textAlign: 'center' as const, letterSpacing: '0.3px',
  },
  inputGroup: { marginBottom: '18px' },
  label: {
    display: 'block', fontSize: '10px', color: '#555', marginBottom: '7px',
    textTransform: 'uppercase' as const, letterSpacing: '1.5px', fontWeight: 600,
  },
  input: {
    width: '100%', padding: '12px 16px', background: 'rgba(8,8,18,0.8)',
    border: '1px solid rgba(255,255,255,0.06)', borderRadius: '10px',
    color: '#e0e0f0', fontSize: '14px', outline: 'none', boxSizing: 'border-box' as const,
    transition: 'border-color 0.2s, box-shadow 0.2s',
    fontFamily: "'JetBrains Mono', monospace",
  },
  btnRow: { display: 'flex', gap: '12px', marginTop: '28px' },
  primaryBtn: {
    flex: 1, padding: '13px', border: 'none', borderRadius: '10px',
    background: 'linear-gradient(135deg, #00ff88, #00dd77, #00bb66)', color: '#000',
    fontWeight: 700, fontSize: '13px', cursor: 'pointer', letterSpacing: '0.3px',
    boxShadow: '0 4px 20px rgba(0,255,136,0.2)',
    transition: 'transform 0.1s, box-shadow 0.15s',
  },
  secondaryBtn: {
    flex: 1, padding: '13px', border: '1px solid rgba(0,255,136,0.2)', borderRadius: '10px',
    background: 'rgba(0,255,136,0.04)', color: '#00ff88',
    fontWeight: 600, fontSize: '13px', cursor: 'pointer', transition: 'all 0.15s',
  },
  dashboard: { flex: 1, display: 'flex', overflow: 'hidden' },
  sidebar: {
    width: '230px', borderRight: '1px solid rgba(255,255,255,0.03)', padding: '16px',
    display: 'flex', flexDirection: 'column', gap: '14px', overflowY: 'auto' as const,
    background: 'rgba(8,8,16,0.6)',
  },
  sideSection: {
    background: 'linear-gradient(145deg, rgba(18,18,32,0.7), rgba(14,14,24,0.5))',
    border: '1px solid rgba(255,255,255,0.04)', borderRadius: '12px', padding: '14px',
  },
  sideTitle: {
    margin: '0 0 10px 0', fontSize: '10px', color: '#555', fontWeight: 600,
    textTransform: 'uppercase' as const, letterSpacing: '1.5px',
  },
  peerCard: {
    padding: '10px', background: 'rgba(0,255,136,0.03)',
    border: '1px solid rgba(0,255,136,0.08)', borderRadius: '8px', marginBottom: '6px',
  },
  peerName: { fontSize: '13px', fontWeight: 600, color: '#e0e0f0' },
  peerMeta: { fontSize: '10px', color: '#555', marginTop: '3px', fontFamily: "'JetBrains Mono', monospace" },
  peerPackets: { fontSize: '9px', color: 'rgba(0,255,136,0.5)', marginTop: '2px', fontFamily: "'JetBrains Mono', monospace" },
  metricRow: { display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid rgba(255,255,255,0.02)' },
  metricLabel: { fontSize: '10px', color: '#555', fontFamily: "'JetBrains Mono', monospace" },
  metricValue: { fontSize: '10px', color: '#00ff88', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" },
  captureBtn: {
    width: '100%', padding: '12px', border: 'none', borderRadius: '10px',
    background: 'linear-gradient(135deg, #00ff88, #00dd77)', color: '#000',
    fontWeight: 700, fontSize: '12px', cursor: 'pointer',
    boxShadow: '0 4px 16px rgba(0,255,136,0.15)',
  },
  stopBtn: {
    width: '100%', padding: '12px', border: '1px solid rgba(255,68,68,0.3)', borderRadius: '10px',
    background: 'rgba(255,68,68,0.06)', color: '#ff5555',
    fontWeight: 600, fontSize: '12px', cursor: 'pointer',
  },
  mainArea: { flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px', overflowY: 'auto' as const },
  frameRow: { display: 'flex', gap: '10px' },
  frameBox: {
    flex: 1, background: 'linear-gradient(145deg, rgba(18,18,32,0.6), rgba(12,12,22,0.4))',
    border: '1px solid rgba(255,255,255,0.04)', borderRadius: '12px',
    padding: '16px 20px', textAlign: 'center' as const,
  },
  frameLabel: {
    fontSize: '9px', color: '#00ff88', textTransform: 'uppercase' as const,
    letterSpacing: '3px', marginBottom: '6px', fontWeight: 600,
  },
  frameValue: {
    fontSize: '30px', fontWeight: 700, color: '#00ff88',
    fontVariantNumeric: 'tabular-nums', fontFamily: "'JetBrains Mono', monospace",
    textShadow: '0 0 20px rgba(0,255,136,0.3)',
  },
  frameHz: { fontSize: '11px', color: 'rgba(0,255,136,0.4)', marginTop: '4px', fontFamily: "'JetBrains Mono', monospace" },
  canvasRow: { display: 'flex', gap: '10px' },
  canvasBox: {
    flex: 1, background: 'linear-gradient(145deg, rgba(14,14,24,0.6), rgba(10,10,18,0.4))',
    border: '1px solid rgba(255,255,255,0.03)', borderRadius: '12px',
    padding: '6px', overflow: 'hidden',
  },
  canvasLabel: {
    fontSize: '9px', color: '#00ff88', textTransform: 'uppercase' as const,
    letterSpacing: '1.5px', marginBottom: '4px', paddingLeft: '6px', fontWeight: 600,
  },
  canvas: { width: '100%', height: '100%', borderRadius: '8px' },
  statsRow: { display: 'flex', gap: '10px' },
  statCard: {
    flex: 1, background: 'linear-gradient(145deg, rgba(18,18,32,0.5), rgba(12,12,22,0.3))',
    border: '1px solid rgba(255,255,255,0.03)', borderRadius: '12px', padding: '14px',
  },
  statLabel: {
    fontSize: '9px', color: '#555', textTransform: 'uppercase' as const,
    letterSpacing: '1.5px', marginBottom: '8px', fontWeight: 600,
  },
  statBar: { height: '4px', background: 'rgba(255,255,255,0.04)', borderRadius: '2px', overflow: 'hidden' },
  statFill: { height: '100%', borderRadius: '2px', transition: 'width 0.1s', boxShadow: '0 0 8px rgba(0,255,136,0.3)' },
  statValue: { fontSize: '14px', fontWeight: 600, color: '#e0e0f0', fontFamily: "'JetBrains Mono', monospace" },
  onsetDot: {
    width: '22px', height: '22px', borderRadius: '50%', margin: '4px auto',
    transition: 'all 0.08s', border: '2px solid rgba(255,170,0,0.2)',
  },
};
