import React from 'react';
import StudioHeader from "./StudioHeader";
import OscillatorBento from './ui/OscillatorBento';
import TransportBento from './ui/TransportBento';
import VisualizerBento from './ui/VisualizerBento';
import EffectsBento from './ui/EffectsBento';
import type { OscillatorType } from '../engine/AudioEngine';
import type { CodecProfile } from '../engine/codecProfiles';
import { AudioEngine } from '../engine/AudioEngine';

interface DumbStudioVisualsProps {
  engineRef: React.MutableRefObject<AudioEngine | null>;
  ready: boolean;
  started: boolean;
  micOn: boolean;
  oscOn: boolean;
  oscType: OscillatorType;
  oscFreq: number;
  activeNote: string | null;
  grReduction: number;
  spatialOn: boolean;
  spatialAz: number;
  spatialEl: number;
  spatialDist: number;
  spatialTrackMode: string;
  autoRotate: boolean;
  webcamOn: boolean;
  webcamContainerRef: React.MutableRefObject<HTMLDivElement | null>;
  codec: CodecProfile;
  btLatencyMs: number;
  calibFlash: boolean;
  launch: () => void;
  toggleMic: () => void;
  toggleOscManual: () => void;
  toggleSpatial: () => void;
  setOscType: (t: OscillatorType) => void;
  setOscFreq: (f: number) => void;
  setCodec: (c: CodecProfile) => void;
  setBtLatencyMs: (ms: number) => void;
  toggleWebcam: () => void;
  calibrateCamera: () => void;
  setSpatialDist: (d: number) => void;
  toggleAutoRotate: () => void;
}

export default function DumbStudioVisuals(props: DumbStudioVisualsProps) {
  const {
    engineRef, ready, started, micOn, oscOn, oscType, oscFreq, activeNote,
    grReduction, spatialOn, spatialAz, spatialEl, spatialDist, spatialTrackMode,
    autoRotate, webcamOn, webcamContainerRef, codec, btLatencyMs, calibFlash,
    launch, toggleMic, toggleOscManual, toggleSpatial, setOscType, setOscFreq,
    setCodec, setBtLatencyMs, toggleWebcam, calibrateCamera, setSpatialDist,
    toggleAutoRotate
  } = props;

  if (!started) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-[var(--deep-space)]">
        <div className="text-center mb-16 space-y-4">
          <div className="sota-title">CLAUDIO</div>
          <div className="text-white/40 tracking-widest text-sm uppercase">Neural DSP Studio</div>
        </div>

        <button onClick={launch} className="sota-button px-12 py-5 text-lg font-bold tracking-[0.3em] uppercase text-[var(--accent-primary)] shadow-[0_0_64px_rgba(0,255,136,0.2)] hover:shadow-[0_0_128px_rgba(0,255,136,0.4)]">
          ▶ Launch Engine
        </button>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-[var(--deep-space)] overflow-hidden">
      {/* SOTA Header Layer */}
      <div className="z-50 backdrop-blur-3xl bg-white/[0.01] border-b border-white/5 shadow-2xl">
        <StudioHeader 
          ready={ready} 
          sampleRate={engineRef.current?.ctx.sampleRate}
          activeNote={activeNote}
          grReduction={grReduction}
          micOn={micOn}
          oscOn={oscOn}
          spatialOn={spatialOn}
          toggleMic={toggleMic}
          toggleOscManual={toggleOscManual}
          toggleSpatial={toggleSpatial}
        />
      </div>

      {/* Main SOTA CSS Grid Assembly */}
      <div className="flex-1 overflow-y-auto p-4 md:p-8">
        <div className="max-w-[1920px] mx-auto h-full gap-6 flex flex-col xl:flex-row">
          
          {/* Left Column: Tonal Generators & Transport */}
          <div className="flex flex-col gap-6 w-full xl:w-[320px] xl:shrink-0">
            <OscillatorBento
              oscType={oscType}
              oscFreq={oscFreq}
              setOscType={setOscType}
              setOscFreq={setOscFreq}
            />
            <TransportBento
              codec={codec}
              setCodec={setCodec}
              ready={ready}
              engineRef={engineRef}
              btLatencyMs={btLatencyMs}
              setBtLatencyMs={setBtLatencyMs}
            />
          </div>

          {/* Center Column: Expansive Visualizers */}
          <div className="flex-1 min-w-[400px]">
            <VisualizerBento ready={ready} engineRef={engineRef} />
          </div>

          {/* Right Column: Spatial & FX Chains */}
          <div className="w-full xl:w-auto xl:max-w-[400px] flex shrink-0">
            <EffectsBento
              engineRef={engineRef}
              ready={ready}
              grReduction={grReduction}
              spatialOn={spatialOn}
              spatialAz={spatialAz}
              spatialEl={spatialEl}
              spatialDist={spatialDist}
              spatialTrackMode={spatialTrackMode}
              autoRotate={autoRotate}
              webcamOn={webcamOn}
              webcamContainerRef={webcamContainerRef}
              calibFlash={calibFlash}
              toggleWebcam={toggleWebcam}
              calibrateCamera={calibrateCamera}
              setSpatialDist={setSpatialDist}
              toggleAutoRotate={toggleAutoRotate}
            />
          </div>
          
        </div>
      </div>
    </div>
  );
}
