import React from 'react';
import SpectrumAnalyzer from '../SpectrumAnalyzer';
import WaveformViewer from '../WaveformViewer';
import LatencyPanel from '../LatencyPanel';
import type { AudioEngine } from '../../engine/AudioEngine';

interface VisualizerBentoProps {
  ready: boolean;
  engineRef: React.MutableRefObject<AudioEngine | null>;
}

export default function VisualizerBento({ ready, engineRef }: VisualizerBentoProps) {
  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Scope Container */}
      <div className="sota-bento flex-1 flex flex-col relative overflow-hidden group">
        <div className="absolute top-4 left-4 z-10 opactiy-50 transition-opacity group-hover:opacity-100">
          <div className="sota-label">Oscilloscope</div>
        </div>
        
        <div className="flex-1 w-full h-full pt-10">
          {ready && engineRef.current ? (
            <WaveformViewer engine={engineRef.current} height={160} color="#00ff88" />
          ) : (
            <div className="w-full h-full flex items-center justify-center border-white/5 border border-dashed rounded-xl m-4 w-[calc(100%-2rem)]">
              <span className="text-white/20 text-xs tracking-widest uppercase">Awaiting Context</span>
            </div>
          )}
        </div>
      </div>

      {/* Spectrum & Telemetry Container */}
      <div className="grid grid-cols-3 gap-4 h-48">
        <div className="col-span-2 sota-bento relative flex flex-col overflow-hidden group">
          <div className="absolute top-4 left-4 z-10 opactiy-50 transition-opacity group-hover:opacity-100">
            <div className="sota-label">Spectral Analysis</div>
          </div>
          
          <div className="flex-1 w-full h-full pt-10 px-4 pb-4">
            {ready && engineRef.current ? (
              <SpectrumAnalyzer engine={engineRef.current} height={120} />
            ) : (
              <div className="w-full h-full flex items-center justify-center border-white/5 border border-dashed rounded-xl">
                <span className="text-white/20 text-xs tracking-widest uppercase">Awaiting Context</span>
              </div>
            )}
          </div>
        </div>

        <div className="col-span-1 sota-bento p-6 flex flex-col">
          <div className="sota-label">Telemetry</div>
          <div className="flex-1 flex items-center bg-white/5 rounded-xl px-4 border border-white/10 mt-2">
            <LatencyPanel engine={engineRef.current} />
          </div>
        </div>
      </div>
    </div>
  );
}
