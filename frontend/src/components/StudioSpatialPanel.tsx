import React from 'react';

const A = '#00ff88';
const P = '#aa44ff';

interface Props {
  spatialOn: boolean;
  spatialTrackMode: string;
  webcamOn: boolean;
  calibFlash: boolean;
  spatialAz: number;
  spatialEl: number;
  spatialDist: number;
  autoRotate: boolean;
  
  toggleWebcam: () => void;
  calibrateCamera: () => void;
  setSpatialDist: (val: number) => void;
  toggleAutoRotate: () => void;
  webcamContainerRef: React.RefObject<HTMLDivElement | null>;
}

export default function StudioSpatialPanel({
  spatialOn, spatialTrackMode, webcamOn, calibFlash,
  spatialAz, spatialEl, spatialDist, autoRotate,
  toggleWebcam, calibrateCamera, setSpatialDist, toggleAutoRotate, webcamContainerRef
}: Props) {
  const card: React.CSSProperties = {
    background: '#141420',
    border: '1px solid #252535',
    borderRadius: 10,
    padding: 14,
  };

  const sectionLabel: React.CSSProperties = {
    fontSize: 10,
    color: '#7070a0',
    letterSpacing: 3,
    textTransform: 'uppercase',
    marginBottom: 12,
    fontWeight: 700,
  };

  return (
    <div style={{ ...card, borderColor: spatialOn ? `${P}44` : '#252535' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <div style={sectionLabel}>HOLOGRAPHIC SPATIAL AUDIO</div>
        <span style={{
          fontSize: 10, padding: '3px 10px', borderRadius: 4,
          background: spatialOn ? `${P}22` : '#1a1a24',
          border: `1px solid ${spatialOn ? P : '#303040'}`,
          color: spatialOn ? P : '#606080', fontWeight: 700,
        }}>
          {spatialOn ? `● ${spatialTrackMode}` : '○ OFF'}
        </span>
        {spatialOn && (
          <button onClick={toggleWebcam} style={{
            marginLeft: 'auto', padding: '3px 10px', borderRadius: 4, fontSize: 10,
            fontWeight: 700, cursor: 'pointer',
            border: `1px solid ${webcamOn ? '#ff8844' : '#353550'}`,
            background: webcamOn ? '#ff884422' : '#1a1a28',
            color: webcamOn ? '#ff8844' : '#707090',
          }}>
            {webcamOn ? '● CAMERA OFF' : '○ CAMERA'}
          </button>
        )}
        {spatialOn && webcamOn && (
          <button onClick={calibrateCamera} style={{
            padding: '3px 10px', borderRadius: 4, fontSize: 10,
            fontWeight: 700, cursor: 'pointer',
            border: `1px solid ${calibFlash ? A : '#353550'}`,
            background: calibFlash ? `${A}22` : '#1a1a28',
            color: calibFlash ? A : '#707090',
          }}>
            {calibFlash ? '✓ ZEROED' : '⊕ CALIBRATE'}
          </button>
        )}
      </div>

      <div ref={webcamContainerRef} style={{ marginBottom: webcamOn ? 12 : 0, borderRadius: 8, overflow: 'hidden', background: webcamOn ? '#000' : 'transparent' }} />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, alignItems: 'start', marginBottom: 12 }}>
        <div style={{ textAlign: 'center', background: '#0c0c18', borderRadius: 8, padding: '10px 6px' }}>
          <div style={{ fontSize: 9, color: '#7070a0', letterSpacing: 2, marginBottom: 6, textTransform: 'uppercase' }}>Head Yaw</div>
          <div style={{ fontSize: 22, fontWeight: 900, color: spatialOn ? P : '#303040' }}>
            {spatialAz >= 0 ? '+' : ''}{spatialAz.toFixed(1)}°
          </div>
          <div style={{ fontSize: 10, color: '#606080', marginTop: 4 }}>
            {spatialAz < -10 ? '← turned left' : spatialAz > 10 ? 'turned right →' : 'CENTERED'}
          </div>
        </div>
        <div style={{ textAlign: 'center', background: '#0c0c18', borderRadius: 8, padding: '10px 6px' }}>
          <div style={{ fontSize: 9, color: '#7070a0', letterSpacing: 2, marginBottom: 6, textTransform: 'uppercase' }}>Head Pitch</div>
          <div style={{ fontSize: 22, fontWeight: 900, color: spatialOn ? P : '#303040' }}>
            {spatialEl >= 0 ? '+' : ''}{spatialEl.toFixed(1)}°
          </div>
          <div style={{ fontSize: 10, color: '#606080', marginTop: 4 }}>
            {spatialEl > 15 ? '↑ looking up' : spatialEl < -15 ? '↓ looking down' : 'LEVEL'}
          </div>
        </div>
        <div style={{ background: '#0c0c18', borderRadius: 8, padding: '10px 8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 9, color: '#7070a0', letterSpacing: 2, textTransform: 'uppercase' }}>Distance</span>
            <span style={{ fontSize: 11, color: P, fontWeight: 700 }}>{spatialDist.toFixed(1)}m</span>
          </div>
          <input
            type="range" min={0.1} max={10} step={0.1} value={spatialDist}
            onChange={e => setSpatialDist(Number(e.target.value))}
            style={{ width: '100%', accentColor: P }}
            disabled={!spatialOn}
          />
          <div style={{ fontSize: 9, color: '#606080', marginTop: 6 }}>
            {spatialDist < 0.3 ? '⚠ proximity boost' : spatialDist > 5 ? 'distant' : 'near field'}
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 0 0', borderTop: '1px solid #1e1e28' }}>
        <div style={{ fontSize: 10, color: '#606080', flex: 1, lineHeight: 1.9 }}>
          {spatialOn
            ? `${spatialTrackMode} tracking · HRTF binaural · source world-locked 1m front · ${autoRotate ? 'listener orbiting — use headphones!' : 'head moves listener — source stays fixed in space'}`
            : 'Click ◇ SPATIAL in header to enable · webcam face tracking → HRTF binaural → world-locked source'}
        </div>
        {spatialOn && (
          <button onClick={toggleAutoRotate} style={{
            flexShrink: 0,
            padding: '6px 14px', borderRadius: 6, fontSize: 11, fontWeight: 800, cursor: 'pointer',
            border: `1px solid ${autoRotate ? '#ffaa00' : P}`,
            background: autoRotate ? '#ffaa0022' : `${P}18`,
            color: autoRotate ? '#ffaa00' : P,
          }}>
            {autoRotate ? '⏹ STOP ORBIT' : '↻ AUTO-ORBIT'}
          </button>
        )}
      </div>
    </div>
  );
}
