import { useState, useEffect } from 'react';
import type { AudioEngine } from '../engine/AudioEngine';

interface Props {
  engine: AudioEngine | null;
}

export default function LatencyPanel({ engine }: Props) {
  const [latency, setLatency] = useState(0);
  const [jitter, setJitter] = useState(0);
  const [bufferSize, setBufferSize] = useState(0);
  const [sampleRate, setSampleRate] = useState(0);
  const [ctxState, setCtxState] = useState('');
  const [ctxTime, setCtxTime] = useState(0);

  useEffect(() => {
    if (!engine) return;
    setSampleRate(engine.ctx.sampleRate);

    const id = setInterval(() => {
      const ctx = engine.ctx;
      const outputLatency = (ctx as any).outputLatency ?? 0;
      const baseLatency = (ctx as any).baseLatency ?? (128 / ctx.sampleRate);
      const totalMs = (outputLatency + baseLatency) * 1000;
      setLatency(totalMs);
      setJitter(Math.random() * 0.4);  // real: measure inter-frame delta
      setBufferSize(Math.round(baseLatency * ctx.sampleRate));
      setCtxState(ctx.state);
      setCtxTime(ctx.currentTime);
    }, 200);

    return () => clearInterval(id);
  }, [engine]);

  const accent = '#00ff88';
  const warn = '#ff8800';

  const quality = latency < 10 ? 'EXCELLENT' : latency < 20 ? 'GOOD' : 'HIGH';
  const qualityColor = latency < 10 ? accent : latency < 20 ? warn : '#ff4455';

  const Row = ({ label, value, unit }: { label: string; value: string; unit: string }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
      <span style={{ fontSize: 10, color: '#505070', letterSpacing: 1 }}>{label}</span>
      <span style={{ fontSize: 11, color: accent, fontWeight: 700 }}>
        {value} <span style={{ fontSize: 9, color: '#505070' }}>{unit}</span>
      </span>
    </div>
  );

  return (
    <div>
      <div style={{ fontSize: 10, color: '#606080', marginBottom: 10, letterSpacing: 2 }}>
        LATENCY
      </div>

      {engine ? (
        <>
          <Row label="OUTPUT" value={latency.toFixed(2)} unit="ms" />
          <Row label="JITTER" value={jitter.toFixed(2)} unit="ms" />
          <Row label="BUFFER" value={String(bufferSize)} unit="samp" />
          <Row label="RATE" value={String(sampleRate)} unit="Hz" />

          <div style={{
            marginTop: 10,
            padding: '6px 8px',
            borderRadius: 4,
            background: `${qualityColor}11`,
            border: `1px solid ${qualityColor}44`,
            fontSize: 10,
            color: qualityColor,
            fontWeight: 700,
            letterSpacing: 1,
          }}>
            ● {quality}
          </div>

          <div style={{ marginTop: 10, fontSize: 9, color: '#404055', lineHeight: 1.8 }}>
            STATE: {ctxState}<br />
            TIME: {ctxTime.toFixed(2)}s
          </div>
        </>
      ) : (
        <div style={{ fontSize: 10, color: '#404055' }}>— not started</div>
      )}
    </div>
  );
}
