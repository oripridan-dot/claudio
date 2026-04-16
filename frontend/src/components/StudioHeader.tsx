import React from 'react';

const A = '#00ff88';
const B = '#0088ff';
const W = '#ff8844';

interface Props {
  ready: boolean;
  sampleRate: number | undefined;
  activeNote: string | null;
  grReduction: number;
  micOn: boolean;
  oscOn: boolean;
  spatialOn: boolean;
  toggleMic: () => void;
  toggleOscManual: () => void;
  toggleSpatial: () => void;
}

export default function StudioHeader({
  ready, sampleRate, activeNote, grReduction,
  micOn, oscOn, spatialOn,
  toggleMic, toggleOscManual, toggleSpatial
}: Props) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '6px 14px', borderBottom: '1px solid #1e1e2a',
      background: '#0e0e18', flexShrink: 0,
    }}>
      <span style={{ fontSize: 18, fontWeight: 900, color: A, letterSpacing: -1 }}>■ CLAUDIO</span>
      <span style={{ fontSize: 9, color: '#303048', letterSpacing: 2 }}>v0.2.0 · NEURAL DSP + HOLOGRAPHIC SPATIAL</span>

      <div style={{ flex: 1 }} />

      {activeNote && (
        <span style={{
          fontSize: 13, fontWeight: 700, color: A,
          background: `${A}11`, padding: '2px 10px', borderRadius: 4,
          border: `1px solid ${A}44`,
        }}>♪ {activeNote}</span>
      )}

      <span style={{ fontSize: 10, color: ready ? A : '#505070' }}>
        {ready ? `● ${sampleRate || 48000}Hz` : '○ IDLE'}
      </span>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 9, color: '#505070' }}>GR</span>
        <div style={{ width: 48, height: 5, background: '#1e1e2a', borderRadius: 3 }}>
          <div style={{
            width: `${Math.min(100, Math.abs(grReduction) * 5)}%`,
            height: '100%', background: W, borderRadius: 3,
            transition: 'width 0.06s',
          }} />
        </div>
        <span style={{ fontSize: 9, color: W, minWidth: 32 }}>{grReduction.toFixed(1)}dB</span>
      </div>

      <button onClick={toggleMic} style={{
        padding: '3px 10px', borderRadius: 4,
        border: `1px solid ${micOn ? B : '#252535'}`,
        background: micOn ? `${B}22` : 'transparent',
        color: micOn ? B : '#505070', fontSize: 10, cursor: 'pointer',
      }}>
        {micOn ? '● MIC' : '○ MIC'}
      </button>

      <button onClick={toggleOscManual} style={{
        padding: '3px 10px', borderRadius: 4,
        border: `1px solid ${oscOn ? A : '#252535'}`,
        background: oscOn ? `${A}22` : 'transparent',
        color: oscOn ? A : '#505070', fontSize: 10, cursor: 'pointer',
      }}>
        {oscOn ? '● OSC' : '○ OSC'}
      </button>

      <button onClick={toggleSpatial} style={{
        padding: '3px 10px', borderRadius: 4,
        border: `1px solid ${spatialOn ? '#aa44ff' : '#252535'}`,
        background: spatialOn ? '#aa44ff22' : 'transparent',
        color: spatialOn ? '#aa44ff' : '#505070', fontSize: 10, cursor: 'pointer',
      }}>
        {spatialOn ? '◈ SPATIAL' : '◇ SPATIAL'}
      </button>
    </div>
  );
}
