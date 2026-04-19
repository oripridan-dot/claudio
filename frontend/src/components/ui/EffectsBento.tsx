import React from 'react';
import StudioEffectsChain from '../StudioEffectsChain';
import AudioFilePlayer from '../AudioFilePlayer';
import MidiPanel from '../MidiPanel';
import StudioSpatialPanel from '../StudioSpatialPanel';
import type { AudioEngine } from '../../engine/AudioEngine';

interface EffectsBentoProps {
  engineRef: React.MutableRefObject<AudioEngine | null>;
  ready: boolean;
  grReduction: number;
  spatialOn: boolean;
  spatialAz: number;
  spatialEl: number;
  spatialDist: number;
  spatialTrackMode: string;
  autoRotate: boolean;
  webcamOn: boolean;
  webcamContainerRef: React.MutableRefObject<HTMLDivElement | null>;
  calibFlash: boolean;
  toggleWebcam: () => void;
  calibrateCamera: () => void;
  setSpatialDist: (d: number) => void;
  toggleAutoRotate: () => void;
}

export default function EffectsBento(props: EffectsBentoProps) {
  const {
    engineRef, ready, grReduction, spatialOn, spatialAz, spatialEl,
    spatialDist, spatialTrackMode, autoRotate, webcamOn, webcamContainerRef,
    calibFlash, toggleWebcam, calibrateCamera, setSpatialDist, toggleAutoRotate
  } = props;

  return (
    <div className="sota-bento p-6 flex flex-col gap-6 h-full overflow-y-auto w-full max-w-[400px]">
      <div>
        <div className="sota-label">Global FX Chain</div>
        <div className="bg-white/5 rounded-2xl border border-white/5 overflow-hidden">
          <StudioEffectsChain engine={engineRef.current} ready={ready} grReduction={grReduction} />
        </div>
      </div>

      <div>
        <div className="sota-label">Spatial Computing</div>
        <div className="bg-white/5 rounded-2xl border border-white/5 overflow-hidden">
          <StudioSpatialPanel
            spatialOn={spatialOn}
            spatialTrackMode={spatialTrackMode}
            webcamOn={webcamOn}
            calibFlash={calibFlash}
            spatialAz={spatialAz}
            spatialEl={spatialEl}
            spatialDist={spatialDist}
            autoRotate={autoRotate}
            toggleWebcam={toggleWebcam}
            calibrateCamera={calibrateCamera}
            setSpatialDist={setSpatialDist}
            toggleAutoRotate={toggleAutoRotate}
            webcamContainerRef={webcamContainerRef}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white/5 rounded-2xl border border-white/5 overflow-hidden p-2">
          <div className="sota-label px-2 pt-2">File I/O</div>
          <AudioFilePlayer engine={engineRef.current} />
        </div>
        <div className="bg-white/5 rounded-2xl border border-white/5 overflow-hidden p-2">
          <div className="sota-label px-2 pt-2">MIDI Interface</div>
          <MidiPanel />
        </div>
      </div>
    </div>
  );
}
