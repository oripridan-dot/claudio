import React, { useEffect, useState } from 'react';
import type { AudioEngine } from '../engine/AudioEngine';
import Knob from './Knob';

const A = '#00ff88';
const B = '#0088ff';
const W = '#ff8844';
const R = '#ff4466';

interface Props {
  engine: AudioEngine | null;
  ready: boolean;
  grReduction: number;
}

export default function StudioEffectsChain({ engine, ready, grReduction }: Props) {
  const [reverb, setReverb] = useState(0.15);
  const [reverbDecay, setReverbDecay] = useState(2.0);
  const [delay, setDelay] = useState(0.0);
  const [delayTime, setDelayTime] = useState(0.33);
  const [delayFb, setDelayFb] = useState(0.35);
  const [distortion, setDistortion] = useState(0.0);
  const [compThresh, setCompThresh] = useState(-24);
  const [compRatio, setCompRatio] = useState(4);
  const [masterVol, setMasterVol] = useState(0.75);

  const [nrEnabled,   setNrEnabled]   = useState(false);
  const [nrThreshold, setNrThreshold] = useState(-40);
  const [hpFreq,      setHpFreq]      = useState(80);
  const [lpFreq,      setLpFreq]      = useState(16000);

  const [eqLow,       setEqLow]       = useState(0);
  const [eqMid,       setEqMid]       = useState(0);
  const [eqMidFreq,   setEqMidFreq]   = useState(1000);
  const [eqHigh,      setEqHigh]      = useState(0);

  useEffect(() => { engine?.setReverb(reverb); }, [engine, reverb]);
  useEffect(() => { engine?.setReverbDecay(reverbDecay); }, [engine, reverbDecay]);
  useEffect(() => { engine?.setDelay(delay); }, [engine, delay]);
  useEffect(() => { engine?.setDelayTime(delayTime); }, [engine, delayTime]);
  useEffect(() => { engine?.setDelayFeedback(delayFb); }, [engine, delayFb]);
  useEffect(() => { engine?.setDistortion(distortion); }, [engine, distortion]);
  useEffect(() => { engine?.setCompThreshold(compThresh); }, [engine, compThresh]);
  useEffect(() => { engine?.setCompRatio(compRatio); }, [engine, compRatio]);
  useEffect(() => { engine?.setMasterGain(masterVol); }, [engine, masterVol]);

  useEffect(() => { engine?.setNoiseGate(nrEnabled); }, [engine, nrEnabled]);
  useEffect(() => { engine?.setNoiseGateThreshold(nrThreshold); }, [engine, nrThreshold]);
  useEffect(() => { engine?.setHPFreq(hpFreq); }, [engine, hpFreq]);
  useEffect(() => { engine?.setLPFreq(lpFreq); }, [engine, lpFreq]);

  useEffect(() => { engine?.setEqLow(eqLow); }, [engine, eqLow]);
  useEffect(() => { engine?.setEqMid(eqMid); }, [engine, eqMid]);
  useEffect(() => { engine?.setEqMidFreq(eqMidFreq); }, [engine, eqMidFreq]);
  useEffect(() => { engine?.setEqHigh(eqHigh); }, [engine, eqHigh]);

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
    textTransform: 'uppercase',
    marginBottom: 12,
    fontWeight: 700,
  };

  return (
    <>
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
            {ready && engine ? (
              nrEnabled
                ? ((engine as any).gateOpen ? <span style={{ color: A }}>▶ gate OPEN</span> : <span style={{ color: '#ff4466' }}>■ gate CLOSED</span>)
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
    </>
  );
}
