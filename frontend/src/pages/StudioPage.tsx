import React, { useCallback, useEffect, useRef, useState } from 'react';
import AudioFilePlayer from '../components/AudioFilePlayer';
import BTLatencyPanel from '../components/BTLatencyPanel';
import CodecSelector from '../components/CodecSelector';
import WaveformViewer from '../components/WaveformViewer';
import StudioHeader from "../components/StudioHeader";
import StudioEffectsChain from '../components/StudioEffectsChain';
import type { OscillatorType } from '../engine/AudioEngine';
import { AudioEngine } from '../engine/AudioEngine';
import '../engine/AudioEngineExtensions'; // side-effect: adds setBTLatencyCompensation
import type { CodecProfile } from '../engine/codecProfiles';
import { DEFAULT_CODEC_ID, getCodecById } from '../engine/codecProfiles';
import { HeadTracker } from '../spatial/HeadTracker';
import { HeadTracker } from '../spatial/HeadTracker';
import { SpatialEngine } from '../spatial/SpatialEngine';
import DumbStudioVisuals from '../components/DumbStudioVisuals';

// Keyboard → MIDI note map (C4 = 60)
const KEY_NOTES: Record<string, number> = {
  z: 60, s: 61, x: 62, d: 63, c: 64, v: 65,
  g: 66, b: 67, h: 68, n: 69, j: 70, m: 71, ',': 72,
};

const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const noteName = (n: number) => `${NOTE_NAMES[n % 12]}${Math.floor(n / 12) - 1}`;

const A = '#00ff88';   // primary accent
const B = '#0088ff';   // secondary accent
const W = '#ff8844';   // warning/warm
const R = '#ff4466';   // distortion accent
const P = '#aa44ff';   // spatial/purple

export default function StudioPage() {
  const engineRef = useRef<AudioEngine | null>(null);
  const [ready, setReady] = useState(false);
  const [started, setStarted] = useState(false);
  const [micOn, setMicOn] = useState(false);
  const [oscOn, setOscOn] = useState(false);
  const [oscType, setOscType] = useState<OscillatorType>('sine');
  const [oscFreq, setOscFreq] = useState(440);
  const [activeNote, setActiveNote] = useState<string | null>(null);

  // Effects
  const [reverb, setReverb] = useState(0.15);
  const [reverbDecay, setReverbDecay] = useState(2.0);
  const [delay, setDelay] = useState(0.0);
  const [delayTime, setDelayTime] = useState(0.33);
  const [delayFb, setDelayFb] = useState(0.35);
  const [distortion, setDistortion] = useState(0.0);
  const [compThresh, setCompThresh] = useState(-24);
  const [compRatio, setCompRatio] = useState(4);
  const [masterVol, setMasterVol] = useState(0.75);

  // Meters
  const [grReduction, setGrReduction] = useState(0);

  // Noise Reduction
  const [nrEnabled,   setNrEnabled]   = useState(false);
  const [nrThreshold, setNrThreshold] = useState(-40);
  const [hpFreq,      setHpFreq]      = useState(80);
  const [lpFreq,      setLpFreq]      = useState(16000);

  // Studio EQ
  const [eqLow,       setEqLow]       = useState(0);
  const [eqMid,       setEqMid]       = useState(0);
  const [eqMidFreq,   setEqMidFreq]   = useState(1000);
  const [eqHigh,      setEqHigh]      = useState(0);

  // Camera calibration flash
  const [calibFlash, setCalibFlash] = useState(false);

  const activeKeys  = useRef(new Set<string>());
  const spatialRef   = useRef<SpatialEngine | null>(null);
  const trackerRef   = useRef<HeadTracker  | null>(null);

  // Spatial state
  const [spatialOn,        setSpatialOn]        = useState(false);
  const [spatialAz,        setSpatialAz]        = useState(0);
  const [spatialEl,        setSpatialEl]        = useState(0);
  const [spatialDist,      setSpatialDist]      = useState(1.0);
  const [spatialTrackMode, setSpatialTrackMode] = useState('—');
  const [autoRotate,       setAutoRotate]       = useState(false);
  const [webcamOn,         setWebcamOn]         = useState(false);
  const webcamContainerRef = useRef<HTMLDivElement | null>(null);

  // Bluetooth / Codec state
  const [codec,      setCodec]      = useState<CodecProfile>(() => getCodecById(DEFAULT_CODEC_ID));
  const [btLatencyMs, setBtLatencyMs] = useState(0);

  // ── Engine Initialization ──

  // Lazy engine init (requires user gesture for AudioContext)
  const launch = useCallback(async () => {
    if (engineRef.current) return;
    const eng = new AudioEngine();
    engineRef.current = eng;
    spatialRef.current = new SpatialEngine(eng.ctx);
    setReady(true);
    setStarted(true);
  }, []);


  // Spatial distance → reposition source
  useEffect(() => {
    if (!spatialOn || !spatialRef.current) return;
    // Source is always directly in front at 0° azimuth — only distance changes.
    // Head tracking rotates the LISTENER, not the source.
    spatialRef.current.setPosition(0, 0, -spatialDist);
  }, [spatialDist, spatialOn]);

  // Computer keyboard → notes
  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      if (!engineRef.current || e.repeat) return;
      const midi = KEY_NOTES[e.key.toLowerCase()];
      if (midi !== undefined) {
        activeKeys.current.add(e.key);
        const freq = 440 * Math.pow(2, (midi - 69) / 12);
        engineRef.current.startOscillator(oscType, freq);
        setOscOn(true);
        setOscFreq(freq);
        setActiveNote(noteName(midi));
      }
    };
    const onUp = (e: KeyboardEvent) => {
      if (!KEY_NOTES[e.key.toLowerCase()]) return;
      activeKeys.current.delete(e.key);
      if (activeKeys.current.size === 0) {
        engineRef.current?.stopOscillator();
        setOscOn(false);
        setActiveNote(null);
      }
    };
    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    return () => {
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup', onUp);
    };
  }, [oscType]);

  // Compressor gain-reduction metering
  useEffect(() => {
    if (!ready) return;
    const id = setInterval(() => {
      setGrReduction(engineRef.current?.compressorReduction ?? 0);
    }, 80);
    return () => clearInterval(id);
  }, [ready]);

  const toggleMic = async () => {
    if (!ready) await launch();
    const eng = engineRef.current!;
    if (micOn) { eng.disableMic(); setMicOn(false); }
    else {
      try { await eng.enableMic(); setMicOn(true); }
      catch { alert('Microphone access denied'); }
    }
  };

  const toggleOscManual = () => {
    if (!engineRef.current) return;
    if (oscOn) { engineRef.current.stopOscillator(); setOscOn(false); }
    else { engineRef.current.startOscillator(oscType, oscFreq); setOscOn(true); }
  };

  const toggleSpatial = async () => {
    const eng     = engineRef.current;
    const spatial = spatialRef.current;
    if (!eng || !spatial) return;
    if (spatialOn) {
      if (webcamContainerRef.current) webcamContainerRef.current.innerHTML = '';
      trackerRef.current?.stop();
      trackerRef.current = null;
      spatial.resetListener();
      eng.restoreDirectOutput(spatial.pannerNode);
      try { spatial.outputNode.disconnect(eng.ctx.destination); } catch (_) {}
      setSpatialOn(false);
      setSpatialTrackMode('—');
      setAutoRotate(false);
      setWebcamOn(false);
    } else {
      // masterGain → hrtfPanner → stereoPanner → destination
      eng.routeOutputThrough(spatial.pannerNode);
      spatial.outputNode.connect(eng.ctx.destination);
      // Place source directly in front — head tracking only rotates the listener.
      spatial.setPosition(0, 0, -spatialDist);
      const tracker = new HeadTracker();
      trackerRef.current = tracker;
      await tracker.start(q => {
        spatial.updateOrientation(q);
        setSpatialAz(+(spatial.currentAzimuth  * 180 / Math.PI).toFixed(1));
        setSpatialEl(+(spatial.currentElevation * 180 / Math.PI).toFixed(1));
      });
      setSpatialTrackMode(tracker.trackingMode.toUpperCase());
      setSpatialOn(true);
    }
  };

  const toggleWebcam = async () => {
    const tracker = trackerRef.current;
    if (!tracker || !spatialOn) return;
    if (webcamOn) {
      tracker.stop();
      if (webcamContainerRef.current) webcamContainerRef.current.innerHTML = '';
      // restart without webcam
      const spatial = spatialRef.current!;
      await tracker.start(q => {
        spatial.updateOrientation(q);
        setSpatialAz(+(spatial.currentAzimuth * 180 / Math.PI).toFixed(1));
        setSpatialEl(+(spatial.currentElevation * 180 / Math.PI).toFixed(1));
      });
      setSpatialTrackMode(tracker.trackingMode.toUpperCase());
      setWebcamOn(false);
    } else {
      try {
        const spatial = spatialRef.current!;
        await tracker.startWebcam(q => {
          spatial.updateOrientation(q);
          setSpatialAz(+(spatial.currentAzimuth * 180 / Math.PI).toFixed(1));
          setSpatialEl(+(spatial.currentElevation * 180 / Math.PI).toFixed(1));
        });
        if (tracker.webcamVideoEl && webcamContainerRef.current) {
          const vid = tracker.webcamVideoEl;
          vid.style.cssText = 'width:100%;border-radius:6px;display:block;background:#000;';
          webcamContainerRef.current.innerHTML = '';
          webcamContainerRef.current.appendChild(vid);
        }
        setSpatialTrackMode('WEBCAM');
        setWebcamOn(true);
      } catch (e) {
        alert('Camera permission denied or unavailable.');
      }
    }
  };

  const toggleAutoRotate = () => {
    const tracker = trackerRef.current;
    if (!tracker || !spatialOn) return;
    if (autoRotate) {
      tracker.stopAutoRotate();
      setAutoRotate(false);
      setSpatialTrackMode(tracker.trackingMode.toUpperCase());
    } else {
      tracker.startAutoRotate(25);
      setAutoRotate(true);
      setSpatialTrackMode('AUTO');
    }
  };

  const calibrateCamera = () => {
    trackerRef.current?.calibrateNow();
    setCalibFlash(true);
    setTimeout(() => setCalibFlash(false), 1500);
  };

  const card: React.CSSProperties = {
    background: '#141420',
    border: '1px solid #252535',
    borderRadius: 10,
    padding: 14,
  };

  return (
    <DumbStudioVisuals
      engineRef={engineRef}
      ready={ready}
      started={started}
      micOn={micOn}
      oscOn={oscOn}
      oscType={oscType}
      oscFreq={oscFreq}
      activeNote={activeNote}
      grReduction={grReduction}
      spatialOn={spatialOn}
      spatialAz={spatialAz}
      spatialEl={spatialEl}
      spatialDist={spatialDist}
      spatialTrackMode={spatialTrackMode}
      autoRotate={autoRotate}
      webcamOn={webcamOn}
      webcamContainerRef={webcamContainerRef}
      codec={codec}
      btLatencyMs={btLatencyMs}
      calibFlash={calibFlash}
      launch={launch}
      toggleMic={toggleMic}
      toggleOscManual={toggleOscManual}
      toggleSpatial={toggleSpatial}
      setOscType={setOscType}
      setOscFreq={setOscFreq}
      setCodec={setCodec}
      setBtLatencyMs={setBtLatencyMs}
      toggleWebcam={toggleWebcam}
      calibrateCamera={calibrateCamera}
      setSpatialDist={setSpatialDist}
      toggleAutoRotate={toggleAutoRotate}
    />
  );
}
