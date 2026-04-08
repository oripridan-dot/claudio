import { useState, useEffect, useCallback } from 'react';
import type { AudioEngine } from '../engine/AudioEngine';
import type { CodecProfile } from '../engine/codecProfiles';

interface Props {
  engine:       AudioEngine;
  codec:        CodecProfile;
  onLatencyMs:  (ms: number) => void;
}

/**
 * BTLatencyPanel — Bluetooth latency compensation controls.
 *
 * The panel lets the user:
 *  1. Choose between "Auto" (use codec nominal value) and "Manual" mode.
 *  2. In Manual mode: drag a slider from 0–300 ms.
 *  3. Hit "Calibrate" to fire a 1 kHz click and measure the round-trip
 *     using the Web Audio clock, then auto-set the compensation value.
 *
 * The compensation is achieved via a pure DelayNode splice in AudioEngine
 * (addBTDelay / removeBTDelay), so the latency is absorbed into the
 * scheduling timeline without audible artefacts.
 */
export default function BTLatencyPanel({ engine, codec, onLatencyMs }: Props) {
  const [mode, setMode] = useState<'auto' | 'manual'>('auto');
  const [manualMs, setManualMs] = useState(0);
  const [calibrating, setCalibrating] = useState(false);
  const [measured, setMeasured] = useState<number | null>(null);

  const compensationMs = mode === 'auto' ? codec.latencyMs : manualMs;

  // Push compensation into AudioEngine whenever it changes
  useEffect(() => {
    engine.setBTLatencyCompensation(compensationMs / 1000);
    onLatencyMs(compensationMs);
  }, [compensationMs, engine, onLatencyMs]);

  // Reset to codec nominal when codec changes
  useEffect(() => {
    if (mode === 'auto') setManualMs(codec.latencyMs);
  }, [codec, mode]);

  // ── Calibration: fire click → detect echo via analyser ──────────────────
  const runCalibration = useCallback(async () => {
    setCalibrating(true);
    setMeasured(null);

    const ctx = engine.ctx;
    if (ctx.state === 'suspended') await ctx.resume();

    const bufSize = ctx.sampleRate * 0.4; // 400 ms window
    const recBuf = ctx.createBuffer(1, bufSize, ctx.sampleRate);

    // 1 kHz click source
    const clickBuf = ctx.createBuffer(1, 512, ctx.sampleRate);
    const ch = clickBuf.getChannelData(0);
    for (let i = 0; i < 512; i++) ch[i] = Math.sin((2 * Math.PI * 1000 * i) / ctx.sampleRate) * (1 - i / 512);

    // Recording: mic → analyser
    let micStream: MediaStream | null = null;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch {
      setCalibrating(false);
      return;
    }

    const micSrc = ctx.createMediaStreamSource(micStream);
    const recAnalyser = ctx.createAnalyser();
    recAnalyser.fftSize = 2048;
    micSrc.connect(recAnalyser);

    const clickSrc = ctx.createBufferSource();
    clickSrc.buffer = clickBuf;
    clickSrc.connect(ctx.destination);

    const startTime = ctx.currentTime;
    clickSrc.start(startTime);

    // Poll analyser for the echo peak over the next 350 ms
    let found = false;
    const poll = setInterval(() => {
      const data = new Float32Array(recAnalyser.fftSize);
      recAnalyser.getFloatTimeDomainData(data);
      const peak = Math.max(...data.map(Math.abs));
      if (peak > 0.05 && !found) {
        found = true;
        const roundTripMs = Math.round((ctx.currentTime - startTime) * 1000);
        setMeasured(roundTripMs);
        setManualMs(roundTripMs);
        setMode('manual');
        clearInterval(poll);
        micStream?.getTracks().forEach(t => t.stop());
        micSrc.disconnect();
        setCalibrating(false);
      }
    }, 10);

    // Safety timeout: stop after 350 ms regardless
    setTimeout(() => {
      if (!found) {
        clearInterval(poll);
        micStream?.getTracks().forEach(t => t.stop());
        micSrc.disconnect();
        setCalibrating(false);
      }
    }, 350);
  }, [engine]);

  const pct = Math.round((compensationMs / 300) * 100);

  return (
    <div className="flex flex-col gap-3 p-3 rounded-lg bg-claudio-card border border-claudio-border">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-widest text-claudio-muted">BT Latency Comp</span>
        <div className="flex gap-1">
          {(['auto', 'manual'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={[
                'text-[10px] px-2 py-0.5 rounded transition-colors',
                mode === m ? 'bg-claudio-accent/20 text-claudio-accent' : 'text-claudio-muted hover:text-claudio-text',
              ].join(' ')}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Value display */}
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold tabular-nums" style={{ color: compensationMs === 0 ? '#606080' : '#00ff88' }}>
          {compensationMs}
        </span>
        <span className="text-claudio-muted text-xs">ms offset</span>
        {measured !== null && (
          <span className="ml-auto text-[10px] text-indigo-400">measured: {measured}ms</span>
        )}
      </div>

      {/* Slider (manual mode only) */}
      <input
        type="range"
        min={0}
        max={300}
        step={1}
        value={compensationMs}
        disabled={mode === 'auto'}
        onChange={(e) => {
          setMode('manual');
          setManualMs(Number(e.target.value));
        }}
        className="w-full"
        style={{ opacity: mode === 'auto' ? 0.4 : 1 }}
      />

      {/* Range labels */}
      <div className="flex justify-between text-[9px] text-claudio-muted -mt-2">
        <span>0 ms</span>
        <span>150 ms</span>
        <span>300 ms</span>
      </div>

      {/* Codec hint */}
      <div className="text-[10px] text-claudio-muted leading-snug">
        <span
          className="font-semibold mr-1"
          style={{ color: codec.color }}
        >
          {codec.label}
        </span>
        nominal: {codec.latencyMinMs}–{codec.latencyMaxMs} ms
      </div>

      {/* Calibrate button */}
      <button
        onClick={runCalibration}
        disabled={calibrating}
        className={[
          'text-xs px-3 py-1.5 rounded border transition-all',
          calibrating
            ? 'border-yellow-500/40 text-yellow-400 animate-pulse cursor-not-allowed'
            : 'border-claudio-border text-claudio-muted hover:border-claudio-accent hover:text-claudio-accent',
        ].join(' ')}
      >
        {calibrating ? '⟳ calibrating…' : '⊕ auto-calibrate (mic required)'}
      </button>
    </div>
  );
}
