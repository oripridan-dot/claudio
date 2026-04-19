import React from 'react';
import type { OscillatorType } from '../../engine/AudioEngine';

interface OscillatorBentoProps {
  oscType: OscillatorType;
  oscFreq: number;
  setOscType: (t: OscillatorType) => void;
  setOscFreq: (f: number) => void;
}

export default function OscillatorBento({ oscType, oscFreq, setOscType, setOscFreq }: OscillatorBentoProps) {
  const types: OscillatorType[] = ['sine', 'square', 'sawtooth', 'triangle'];

  return (
    <div className="sota-bento p-6 flex flex-col gap-6">
      <div>
        <div className="sota-label">Oscillator Node</div>
        <div className="flex gap-2">
          {types.map(t => (
            <button
              key={t}
              onClick={() => setOscType(t)}
              className={`sota-button flex-1 py-3 text-xs font-bold tracking-wider ${
                oscType === t ? 'active' : ''
              }`}
            >
              {t.slice(0, 3).toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="flex justify-between items-baseline mb-3">
          <span className="sota-label">Frequency Sweep</span>
          <span className="text-xl font-bold tracking-tight text-[var(--accent-primary)]">
            {oscFreq.toFixed(1)} <span className="text-sm opacity-50 font-normal">Hz</span>
          </span>
        </div>
        
        <input
          type="range"
          min={20}
          max={4000}
          step={1}
          value={oscFreq}
          onChange={e => setOscFreq(Number(e.target.value))}
          className="sota-slider w-full"
        />
        
        <div className="flex justify-between text-[10px] text-white/30 font-bold mt-2 tracking-widest">
          <span>20</span>
          <span>4000</span>
        </div>
      </div>

      <div className="mt-auto px-4 py-3 bg-white/5 rounded-xl border border-white/5">
        <div className="sota-label mb-1 opacity-70">MIDI Binding</div>
        <div className="text-xs text-white/50 leading-relaxed font-medium">
          Z S X D C V G B H N J M ,
        </div>
      </div>
    </div>
  );
}
