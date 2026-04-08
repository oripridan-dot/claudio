import React, { useCallback, useEffect, useRef, useState } from 'react';
import AudioFilePlayer from '../components/AudioFilePlayer';
import BTLatencyPanel from '../components/BTLatencyPanel';
import CodecSelector from '../components/CodecSelector';
import InstrumentDetector from '../components/InstrumentDetector';
import Knob from '../components/Knob';
import LatencyPanel from '../components/LatencyPanel';
import MentorCard from '../components/MentorCard';
import MidiPanel from '../components/MidiPanel';
import PhaseMeter from '../components/PhaseMeter';
import RoadmapOverlay from '../components/RoadmapOverlay';
import RoomScannerPanel from '../components/RoomScannerPanel';
import SpectrumAnalyzer from '../components/SpectrumAnalyzer';
import SweetSpotHUD from '../components/SweetSpotHUD';
import WaveformViewer from '../components/WaveformViewer';
import type { OscillatorType } from '../engine/AudioEngine';
import { AudioEngine } from '../engine/AudioEngine';
import '../engine/AudioEngineExtensions'; // side-effect: adds setBTLatencyCompensation
import type { CodecProfile } from '../engine/codecProfiles';
import { DEFAULT_CODEC_ID, getCodecById } from '../engine/codecProfiles';
import { useClaudioSocket } from '../hooks/useClaudioSocket';
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

  // ── Claudio Intelligence State ──
  const { send, lastMessage, connected } = useClaudioSocket('ws://localhost:8420/ws/session');

  const [mentorTip, setMentorTip] = useState<{
    mentor_name: string; mentor_photo: string; specialty: string;
    quote: string; physical_action: string; location: string; date: string;
  } | null>(null);
  const [phaseData, setPhaseData] = useState<{
    correlation: number; offset_samples: number; polarity_ok: boolean;
  } | null>(null);
  const [roomScan, setRoomScan] = useState<{
    rt60: number; modes: { freq: number; q: number }[];
    flutter_detected: boolean; bass_buildup_db: number;
    advice: string[];
  } | null>(null);
  const [sweetSpot, setSweetSpot] = useState<{
    left_delay_ms: number; right_delay_ms: number;
    left_gain_db: number; right_gain_db: number;
    listener_x: number; listener_y: number;
    mode: string;
  } | null>(null);
  const [instrumentDetection, setInstrumentDetection] = useState<{
    family: string; confidence: number; pickup_type: string;
    model_guess: string; model_confidence: number; coaching_hints: string[];
  } | null>(null);
  const [roadmap, setRoadmap] = useState<{
    current_phase: string; phases: {
      id: string; name: string; status: string;
      items: { key: string; label: string; completed: boolean }[];
    }[];
  } | null>(null);
  const [mentorDismissed, setMentorDismissed] = useState(false);

  // ── Process incoming Claudio messages ──
  useEffect(() => {
    if (!lastMessage) return;
    const msg = lastMessage as Record<string, unknown>;
    const t = msg.type as string | undefined;
    if (t === 'instrument_detection') setInstrumentDetection(msg.data as typeof instrumentDetection);
    if (t === 'mentor_tip') { setMentorTip(msg.data as typeof mentorTip); setMentorDismissed(false); }
    if (t === 'phase_report') setPhaseData(msg.data as typeof phaseData);
    if (t === 'room_scan_result') setRoomScan(msg.data as typeof roomScan);
    if (t === 'sweet_spot_update') setSweetSpot(msg.data as typeof sweetSpot);
    if (t === 'roadmap_update') setRoadmap(msg.data as typeof roadmap);
  }, [lastMessage]);

  // Lazy engine init (requires user gesture for AudioContext)
  const launch = useCallback(async () => {
    if (engineRef.current) return;
    const eng = new AudioEngine();
    engineRef.current = eng;
    spatialRef.current = new SpatialEngine(eng.ctx);
    setReady(true);
    setStarted(true);
  }, []);

  // Sync effects → engine
  useEffect(() => { engineRef.current?.setReverb(reverb); }, [reverb]);
  useEffect(() => { engineRef.current?.setReverbDecay(reverbDecay); }, [reverbDecay]);
  useEffect(() => { engineRef.current?.setDelay(delay); }, [delay]);
  useEffect(() => { engineRef.current?.setDelayTime(delayTime); }, [delayTime]);
  useEffect(() => { engineRef.current?.setDelayFeedback(delayFb); }, [delayFb]);
  useEffect(() => { engineRef.current?.setDistortion(distortion); }, [distortion]);
  useEffect(() => { engineRef.current?.setCompThreshold(compThresh); }, [compThresh]);
  useEffect(() => { engineRef.current?.setCompRatio(compRatio); }, [compRatio]);
  useEffect(() => { engineRef.current?.setMasterGain(masterVol); }, [masterVol]);

  // Noise Reduction
  useEffect(() => { engineRef.current?.setNoiseGate(nrEnabled); }, [nrEnabled]);
  useEffect(() => { engineRef.current?.setNoiseGateThreshold(nrThreshold); }, [nrThreshold]);
  useEffect(() => { engineRef.current?.setHPFreq(hpFreq); }, [hpFreq]);
  useEffect(() => { engineRef.current?.setLPFreq(lpFreq); }, [lpFreq]);

  // Studio EQ
  useEffect(() => { engineRef.current?.setEqLow(eqLow); }, [eqLow]);
  useEffect(() => { engineRef.current?.setEqMid(eqMid); }, [eqMid]);
  useEffect(() => { engineRef.current?.setEqMidFreq(eqMidFreq); }, [eqMidFreq]);
  useEffect(() => { engineRef.current?.setEqHigh(eqHigh); }, [eqHigh]);

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

      {/* ── Header bar ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '6px 14px', borderBottom: '1px solid #1e1e2a',
        background: '#0e0e18', flexShrink: 0,
      }}>
        <span style={{ fontSize: 18, fontWeight: 900, color: A, letterSpacing: -1 }}>■ CLAUDIO</span>
        <span style={{ fontSize: 9, color: '#303048', letterSpacing: 2 }}>v0.2.0 · NEURAL DSP + HOLOGRAPHIC SPATIAL</span>

        <div style={{ flex: 1 }} />

        {/* Active note indicator */}
        {activeNote && (
          <span style={{
            fontSize: 13, fontWeight: 700, color: A,
            background: `${A}11`, padding: '2px 10px', borderRadius: 4,
            border: `1px solid ${A}44`,
          }}>♪ {activeNote}</span>
        )}

        {/* Engine state */}
        <span style={{ fontSize: 10, color: ready ? A : '#505070' }}>
          {ready ? `● ${engineRef.current?.ctx.sampleRate}Hz` : '○ IDLE'}
        </span>

        {/* GR meter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 9, color: '#505070' }}>GR</span>
          <div style={{ width: 48, height: 5, background: '#1e1e2a', borderRadius: 3 }}>
            <div style={{
              width: `${Math.min(100, Math.abs(grReduction) * 5)}%`,
              height: '100%', background: W, borderRadius: 3,
              transition: 'width 0.06s',
            }} />
          </div>
          <span style={{ fontSize: 9, color: W, minWidth: 32 }}>{grReduction.toFixed(1)}dB</span>
        </div>

        {/* Mic button */}
        <button onClick={toggleMic} style={{
          padding: '3px 10px', borderRadius: 4,
          border: `1px solid ${micOn ? B : '#252535'}`,
          background: micOn ? `${B}22` : 'transparent',
          color: micOn ? B : '#505070', fontSize: 10, cursor: 'pointer',
        }}>
          {micOn ? '● MIC' : '○ MIC'}
        </button>

        {/* OSC button */}
        <button onClick={toggleOscManual} style={{
          padding: '3px 10px', borderRadius: 4,
          border: `1px solid ${oscOn ? A : '#252535'}`,
          background: oscOn ? `${A}22` : 'transparent',
          color: oscOn ? A : '#505070', fontSize: 10, cursor: 'pointer',
        }}>
          {oscOn ? '● OSC' : '○ OSC'}
        </button>

        {/* SPATIAL button */}
        <button onClick={toggleSpatial} style={{
          padding: '3px 10px', borderRadius: 4,
          border: `1px solid ${spatialOn ? '#aa44ff' : '#252535'}`,
          background: spatialOn ? '#aa44ff22' : 'transparent',
          color: spatialOn ? '#aa44ff' : '#505070', fontSize: 10, cursor: 'pointer',
        }}>
          {spatialOn ? '◈ SPATIAL' : '◇ SPATIAL'}
        </button>
      </div>

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

          {/* Effects row */}
          <div style={sectionLabel}>EFFECTS CHAIN</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>

            {/* Reverb */}
            <div style={{ ...card, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <div style={{ fontSize: 9, color: '#505070', letterSpacing: 2 }}>REVERB</div>
              <Knob value={reverb} min={0} max={1} label="Mix" onChange={setReverb} color={A} />
              <Knob value={reverbDecay} min={0.2} max={6} label="Decay" unit="s" decimals={1}
                    onChange={setReverbDecay} color={A} />
            </div>

            {/* Delay */}
            <div style={{ ...card, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <div style={{ fontSize: 9, color: '#505070', letterSpacing: 2 }}>DELAY</div>
              <Knob value={delay} min={0} max={1} label="Mix" onChange={setDelay} color={B} />
              <Knob value={delayTime} min={0.05} max={1} label="Time" unit="s" onChange={setDelayTime} color={B} />
              <Knob value={delayFb} min={0} max={0.95} label="Fdbk" onChange={setDelayFb} color={B} />
            </div>

            {/* Distortion */}
            <div style={{ ...card, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <div style={{ fontSize: 9, color: '#505070', letterSpacing: 2 }}>DISTORTION</div>
              <Knob value={distortion} min={0} max={1} label="Drive" onChange={setDistortion} color={R} size={64} />
              <div style={{ fontSize: 9, color: '#404055', textAlign: 'center', lineHeight: 1.5 }}>
                WaveShaper<br />4x oversample
              </div>
            </div>

            {/* Compressor */}
            <div style={{ ...card, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <div style={{ fontSize: 9, color: '#505070', letterSpacing: 2 }}>COMPRESSOR</div>
              <Knob value={compThresh} min={-60} max={0} label="Thresh" unit="dB" decimals={0}
                    onChange={setCompThresh} color={W} />
              <Knob value={compRatio} min={1} max={20} label="Ratio" unit=":1" decimals={1}
                    onChange={setCompRatio} color={W} />
            </div>
          </div>

          {/* Master volume */}
          <div style={{
            ...card, display: 'flex', alignItems: 'center', gap: 16,
          }}>
            <Knob value={masterVol} min={0} max={1} label="Master" size={64}
                  onChange={setMasterVol} color={A} />
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 9, color: '#505070', letterSpacing: 2, marginBottom: 8 }}>
                MASTER VOLUME
              </div>
              <input
                type="range" min={0} max={1} step={0.01} value={masterVol}
                onChange={e => setMasterVol(Number(e.target.value))}
                style={{ width: '100%', accentColor: A }}
              />
              <div style={{ fontSize: 11, color: A, marginTop: 4, fontWeight: 700 }}>
                {(masterVol * 100).toFixed(0)}% {masterVol > 0.9 ? '⚠ hot' : ''}
              </div>
            </div>

            {/* Signal chain legend */}
            <div style={{ fontSize: 9, color: '#404055', lineHeight: 1.8, minWidth: 160 }}>
              INPUT → DIST → COMP<br />
              → DELAY → REVERB<br />
              → MASTER → OUT
            </div>
          </div>

          {/* Noise Reduction + Studio EQ */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>

            {/* Noise Reduction */}
            <div style={{ ...card }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                <div style={{ ...sectionLabel, marginBottom: 0 }}>NOISE REDUCTION</div>
                <button onClick={() => setNrEnabled(v => !v)} style={{
                  marginLeft: 'auto', padding: '2px 9px', borderRadius: 4, fontSize: 10,
                  fontWeight: 700, cursor: 'pointer',
                  border: `1px solid ${nrEnabled ? A : '#353550'}`,
                  background: nrEnabled ? `${A}22` : '#1a1a28',
                  color: nrEnabled ? A : '#707090',
                }}>
                  {nrEnabled ? '● ON' : '○ OFF'}
                </button>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: 10 }}>
                <Knob value={nrThreshold} min={-80} max={-10} label="Gate" unit="dB" decimals={0}
                      onChange={setNrThreshold} color={nrEnabled ? A : '#404055'} size={52} />
                <Knob value={hpFreq} min={20} max={800} label="HP Cut" unit="Hz" decimals={0}
                      onChange={setHpFreq} color={B} size={52} />
                <Knob value={lpFreq} min={4000} max={20000} label="LP Cut" unit="Hz" decimals={0}
                      onChange={setLpFreq} color={B} size={52} />
              </div>
              <div style={{ fontSize: 9, color: '#404055', lineHeight: 1.7 }}>
                Gate · threshold/attack/release<br />
                HP removes rumble · LP removes hiss<br />
                {ready && engineRef.current ? (
                  nrEnabled
                    ? (engineRef.current.gateOpen ? <span style={{ color: A }}>▶ gate OPEN</span> : <span style={{ color: '#ff4466' }}>■ gate CLOSED</span>)
                    : <span>gate bypass</span>
                ) : null}
              </div>
            </div>

            {/* Studio EQ */}
            <div style={{ ...card }}>
              <div style={{ ...sectionLabel, marginBottom: 12 }}>STUDIO EQ — 3 BAND</div>
              <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: 10 }}>
                <Knob value={eqLow} min={-12} max={12} label="Low" unit="dB" decimals={1}
                      onChange={setEqLow} color={'#44aaff'} size={52} />
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                  <Knob value={eqMid} min={-12} max={12} label="Mid" unit="dB" decimals={1}
                        onChange={setEqMid} color={'#ffaa00'} size={52} />
                  <input
                    type="range" min={200} max={8000} step={10} value={eqMidFreq}
                    onChange={e => setEqMidFreq(Number(e.target.value))}
                    style={{ width: 52, accentColor: '#ffaa00' }}
                    title={`Mid centre: ${eqMidFreq}Hz`}
                  />
                  <span style={{ fontSize: 8, color: '#606070' }}>{eqMidFreq >= 1000 ? (eqMidFreq/1000).toFixed(1)+'k' : eqMidFreq}Hz</span>
                </div>
                <Knob value={eqHigh} min={-12} max={12} label="High" unit="dB" decimals={1}
                      onChange={setEqHigh} color={'#ff6688'} size={52} />
              </div>
              <div style={{ fontSize: 9, color: '#404055', lineHeight: 1.7 }}>
                Low shelf 200Hz · Mid peak {eqMidFreq >= 1000 ? (eqMidFreq/1000).toFixed(1)+'k' : eqMidFreq}Hz<br />
                High shelf 8kHz · ±12dB per band<br />
                Placed pre-distortion in signal chain
              </div>
            </div>
          </div>

          {/* Audio File Player */}
          <AudioFilePlayer engine={engineRef.current} />

          {/* MIDI panel */}
          <MidiPanel />

          {/* SPATIAL panel */}
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

            {/* Webcam preview — only shown when webcam is active */}
            <div ref={webcamContainerRef} style={{ marginBottom: webcamOn ? 12 : 0, borderRadius: 8, overflow: 'hidden', background: webcamOn ? '#000' : 'transparent' }} />

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, alignItems: 'start', marginBottom: 12 }}>
              {/* Azimuth */}
              <div style={{ textAlign: 'center', background: '#0c0c18', borderRadius: 8, padding: '10px 6px' }}>
                <div style={{ fontSize: 9, color: '#7070a0', letterSpacing: 2, marginBottom: 6, textTransform: 'uppercase' }}>Head Yaw</div>
                <div style={{ fontSize: 22, fontWeight: 900, color: spatialOn ? P : '#303040' }}>
                  {spatialAz >= 0 ? '+' : ''}{spatialAz.toFixed(1)}°
                </div>
                <div style={{ fontSize: 10, color: '#606080', marginTop: 4 }}>
                  {spatialAz < -10 ? '← turned left' : spatialAz > 10 ? 'turned right →' : 'CENTERED'}
                </div>
              </div>
              {/* Elevation */}
              <div style={{ textAlign: 'center', background: '#0c0c18', borderRadius: 8, padding: '10px 6px' }}>
                <div style={{ fontSize: 9, color: '#7070a0', letterSpacing: 2, marginBottom: 6, textTransform: 'uppercase' }}>Head Pitch</div>
                <div style={{ fontSize: 22, fontWeight: 900, color: spatialOn ? P : '#303040' }}>
                  {spatialEl >= 0 ? '+' : ''}{spatialEl.toFixed(1)}°
                </div>
                <div style={{ fontSize: 10, color: '#606080', marginTop: 4 }}>
                  {spatialEl > 15 ? '↑ looking up' : spatialEl < -15 ? '↓ looking down' : 'LEVEL'}
                </div>
              </div>
              {/* Distance */}
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

          {/* ── CLAUDIO INTELLIGENCE PANELS ── */}
          <div style={sectionLabel}>CLAUDIO INTELLIGENCE {connected && <span style={{ color: A }}>● LIVE</span>}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {/* Instrument Detector */}
            <InstrumentDetector detection={instrumentDetection} />

            {/* Phase Meter */}
            <PhaseMeter
              correlation={phaseData?.correlation ?? 0}
              offsetSamples={phaseData?.offset_samples ?? 0}
              polarityOk={phaseData?.polarity_ok ?? true}
              onFlipPolarity={() => send({ type: 'flip_polarity' })}
            />
          </div>

          {/* Room Scanner */}
          <RoomScannerPanel
            rt60={roomScan?.rt60 ?? null}
            modes={roomScan?.modes ?? []}
            flutterDetected={roomScan?.flutter_detected ?? false}
            bassBuildupDb={roomScan?.bass_buildup_db ?? 0}
            advice={roomScan?.advice ?? []}
            onScanClap={() => send({ type: 'room_scan_clap' })}
          />

          {/* Sweet Spot */}
          <SweetSpotHUD
            leftDelayMs={sweetSpot?.left_delay_ms ?? 0}
            rightDelayMs={sweetSpot?.right_delay_ms ?? 0}
            leftGainDb={sweetSpot?.left_gain_db ?? 0}
            rightGainDb={sweetSpot?.right_gain_db ?? 0}
            listenerX={sweetSpot?.listener_x ?? 0}
            listenerY={sweetSpot?.listener_y ?? 0}
            mode={sweetSpot?.mode ?? 'FOCUS_ENGINEER'}
            onModeChange={(mode) => send({ type: 'set_sweet_spot_mode', mode })}
          />
        </div>
      </div>

      {/* ── Mentor Card Overlay ── */}
      {mentorTip && !mentorDismissed && (
        <MentorCard
          mentorName={mentorTip.mentor_name}
          mentorPhoto={mentorTip.mentor_photo}
          specialty={mentorTip.specialty}
          quote={mentorTip.quote}
          physicalAction={mentorTip.physical_action}
          location={mentorTip.location}
          date={mentorTip.date}
          onDismiss={() => setMentorDismissed(true)}
        />
      )}

      {/* ── Roadmap Bar ── */}
      <RoadmapOverlay
        currentPhase={roadmap?.current_phase ?? 'setup'}
        phases={roadmap?.phases ?? []}
        onCompleteItem={(phaseId, itemKey) =>
          send({ type: 'roadmap_action', action: 'complete_item', phase_id: phaseId, item_key: itemKey })
        }
      />
    </div>
  );
}
