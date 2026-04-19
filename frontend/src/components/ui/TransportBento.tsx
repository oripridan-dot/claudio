import React from 'react';
import CodecSelector from '../CodecSelector';
import BTLatencyPanel from '../BTLatencyPanel';
import type { CodecProfile } from '../../engine/codecProfiles';
import type { AudioEngine } from '../../engine/AudioEngine';

interface TransportBentoProps {
  codec: CodecProfile;
  setCodec: (c: CodecProfile) => void;
  ready: boolean;
  engineRef: React.MutableRefObject<AudioEngine | null>;
  btLatencyMs: number;
  setBtLatencyMs: (ms: number) => void;
}

export default function TransportBento({
  codec, setCodec, ready, engineRef, btLatencyMs, setBtLatencyMs
}: TransportBentoProps) {
  return (
    <div className="sota-bento p-6 flex flex-col gap-6">
      <div>
        <div className="sota-label">Audio Transport</div>
        
        <div className="bg-white/5 rounded-xl border border-white/10 p-4">
          <CodecSelector selected={codec.id} onSelect={setCodec} />
        </div>
      </div>

      {ready && engineRef.current ? (
        <div>
          <div className="flex justify-between items-baseline mb-3">
            <span className="sota-label">Phase Alignment</span>
            <span className="text-xl font-bold tracking-tight text-[var(--accent-secondary)]">
              {btLatencyMs} <span className="text-sm opacity-50 font-normal">ms</span>
            </span>
          </div>
          
          <div className="bg-white/5 rounded-xl border border-white/10 p-4">
            <BTLatencyPanel
              engine={engineRef.current}
              codec={codec}
              onLatencyMs={setBtLatencyMs}
            />
          </div>
        </div>
      ) : (
        <div className="flex-1 border border-white/5 rounded-xl border-dashed flex items-center justify-center opacity-30">
          <span className="sota-label m-0">Awaiting Context</span>
        </div>
      )}
    </div>
  );
}
