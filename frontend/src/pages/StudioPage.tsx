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

  const sectionLabel: React.CSSProperties = {
    fontSize: 10,
    color: '#7070a0',
    letterSpacing: 3,
    textTransform: 'uppercase' as const,
    marginBottom: 12,
    fontWeight: 700,
  };

  // ── Splash ──────────────────────────────────────────────────────────────
  if (!started) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', height: '100vh', background: '#08080d',
      }}>
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <div style={{
            fontSize: 56, fontWeight: 900, color: A, letterSpacing: -3,
            fontFamily: 'monospace', marginBottom: 8,
          }}>
            CLAUDIO
          </div>
          <div style={{ color: '#404058', fontSize: 13, letterSpacing: 1 }}>
            Neural DSP Studio · zero-perceptual-latency musical collaboration
          </div>
        </div>

        <button
          onClick={launch}
          style={{
            background: A, color: '#08080d', fontWeight: 900,
            padding: '16px 48px', borderRadius: 8, fontSize: 15,
            border: 'none', cursor: 'pointer', letterSpacing: 3,
            fontFamily: 'monospace',
            boxShadow: `0 0 32px ${A}44`,
          }}
        >
          ▶  LAUNCH STUDIO
        </button>

        <div style={{ color: '#303048', fontSize: 11, marginTop: 24, textAlign: 'center', lineHeight: 2 }}>
          Z S X D C V G B H N J M · keyboard notes (C4–C5)<br />
          Click LAUNCH to initialise the AudioContext
        </div>
      </div>
    );
  }

  // ── Studio ──────────────────────────────────────────────────────────────
  return (
    <div style={{
      height: '100vh', display: 'flex', flexDirection: 'column',
      background: '#08080d', fontFamily: 'monospace', overflow: 'hidden',
    }}>

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

      {/* ── Spectrum analyser ── */}
      <div style={{
        padding: '4px 8px', borderBottom: '1px solid #1e1e2a',
        background: '#08080d', flexShrink: 0,
      }}>
        {ready && <SpectrumAnalyzer engine={engineRef.current!} height={130} />}
        {!ready && (
          <div style={{ height: 130, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ color: '#252535', fontSize: 11 }}>spectrum analyser — click LAUNCH</span>
          </div>
        )}
      </div>

      {/* ── Main body ── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* Left column */}
        <div style={{
          width: 240, flexShrink: 0,
          borderRight: '1px solid #1e1e2a',
          display: 'flex', flexDirection: 'column',
          background: '#0c0c16',
          overflowY: 'auto',
        }}>
          {/* Oscillator */}
          <div style={{ padding: 10, borderBottom: '1px solid #1e1e2a' }}>
            <div style={sectionLabel}>OSCILLATOR</div>

            {/* Wave selector */}
            <div style={{ display: 'flex', gap: 4, marginBottom: 10 }}>
              {(['sine','square','sawtooth','triangle'] as OscillatorType[]).map(t => (
                <button key={t} onClick={() => {
                  setOscType(t);
                  engineRef.current?.setOscillatorType(t);
                }} style={{
                  flex: 1, padding: '4px 0', borderRadius: 4, fontSize: 9,
                  border: `1px solid ${oscType === t ? A : '#252535'}`,
                  background: oscType === t ? `${A}18` : '#0a0a14',
                  color: oscType === t ? A : '#505070', cursor: 'pointer',
                }}>
                  {t.slice(0, 3).toUpperCase()}
                </button>
              ))}
            </div>

            {/* Frequency */}
            <div style={{ marginBottom: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 9, color: '#505070', letterSpacing: 1 }}>FREQUENCY</span>
                <span style={{ fontSize: 10, color: A, fontWeight: 700 }}>{oscFreq.toFixed(1)} Hz</span>
              </div>
              <input
                type="range" min={20} max={4000} step={1} value={oscFreq}
                onChange={e => {
                  const f = Number(e.target.value);
                  setOscFreq(f);
                  engineRef.current?.setOscillatorFrequency(f);
                }}
                style={{ width: '100%', accentColor: A }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#404055' }}>
                <span>20</span><span>4000</span>
              </div>
            </div>

            <div style={{ fontSize: 9, color: '#404055', lineHeight: 1.7 }}>
              Keyboard: Z S X D C V G B H N J M ,
            </div>
          </div>

          {/* Waveform */}
          <div style={{ padding: 8, borderBottom: '1px solid #1e1e2a' }}>
            <div style={sectionLabel}>OSCILLOSCOPE</div>
            {ready
              ? <WaveformViewer engine={engineRef.current!} height={72} color={A} />
              : <div style={{ height: 72, background: '#08080d', borderRadius: 4 }} />}
          </div>

          {/* Latency */}
          <div style={{ padding: 10, borderBottom: '1px solid #1e1e2a' }}>
            <div style={sectionLabel}>SYSTEM LATENCY</div>
            <LatencyPanel engine={engineRef.current} />
          </div>

          {/* Bluetooth Codec */}
          <div style={{ padding: 10, borderBottom: '1px solid #1e1e2a' }}>
            <div style={sectionLabel}>BLUETOOTH CODEC</div>
            <CodecSelector selected={codec.id} onSelect={c => setCodec(c)} />
          </div>

          {/* BT Latency Compensation */}
          {ready && engineRef.current && (
            <div style={{ padding: 10, flex: 1 }}>
              <div style={sectionLabel}>BT LATENCY COMP</div>
              <BTLatencyPanel
                engine={engineRef.current}
                codec={codec}
                onLatencyMs={ms => setBtLatencyMs(ms)}
              />
              <div style={{ marginTop: 8, fontSize: 10, color: A }}>
                Compensation: {btLatencyMs}ms
              </div>
            </div>
          )}
          {(!ready || !engineRef.current) && <div style={{ flex: 1 }} />}
        </div>

        {/* Right column: effects + MIDI */}
        <div style={{
          flex: 1, padding: 12, overflowY: 'auto',
          display: 'flex', flexDirection: 'column', gap: 10,
        }}>

          <StudioEffectsChain engine={engineRef.current} ready={ready} grReduction={grReduction} />

          {/* Audio File Player */}
          <AudioFilePlayer engine={engineRef.current} />

          {/* MIDI panel */}
          <MidiPanel />

          {/* SPATIAL panel */}
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
    </div>
  );
}
