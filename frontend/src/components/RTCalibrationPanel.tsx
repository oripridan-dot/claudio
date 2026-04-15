import React, { useEffect, useState, useRef } from 'react';
import type { RTCalibrationEngine, CalibrationParams, CalibrationMetrics } from '../engine/RTCalibrationEngine';

export function RTCalibrationPanel({ engine }: { engine: RTCalibrationEngine | null }) {
  const [params, setParams] = useState<CalibrationParams>({
    fftSize: 2048,
    gainCompStrength: 0.5,
    eqCompStrength: 0.5,
    smoothingMs: 50,
  });

  const [metrics, setMetrics] = useState<CalibrationMetrics>({
    coherence: 0,
    inputMagnitudeDb: -80,
    outputMagnitudeDb: -80,
    magnitudeDeltaDb: 0,
    inputCentroidHz: 0,
    outputCentroidHz: 0,
    freqDeltaHz: 0,
    compensationGainLvl: 1.0,
    compensationFilterFreq: 1000,
  });

  const animRef = useRef<number>(0);

  useEffect(() => {
    if (!engine) return;
    
    // Sync initial params
    setParams({ ...engine.params });

    const loop = () => {
      const m = engine.updateMetrics();
      if (m) setMetrics({ ...m });
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);

    return () => cancelAnimationFrame(animRef.current);
  }, [engine]);

  const handleParamChange = (key: keyof CalibrationParams, val: number) => {
    if (!engine) return;
    const newParams = { ...params, [key]: val };
    setParams(newParams);
    engine.setParams(newParams);
  };

  if (!engine) return null;

  return (
    <div style={panelStyles.container}>
      <h3 style={panelStyles.title}>🎛 RT Calibration & Auto-Compensate</h3>
      
      <div style={panelStyles.grid}>
        {/* Meters Section */}
        <div style={panelStyles.section}>
          <div style={panelStyles.sectionTitle}>Live Telemetry</div>
          
          <div style={panelStyles.meterBox}>
            <div style={panelStyles.meterLabel}>Coherence</div>
            <div style={panelStyles.barBg}>
              <div style={{ ...panelStyles.barFill, width: `${(metrics.coherence || 0) * 100}%`, background: '#00ff88' }} />
            </div>
            <div style={panelStyles.meterValue}>{(metrics.coherence * 100).toFixed(1)}%</div>
          </div>

          <div style={panelStyles.meterBox}>
            <div style={panelStyles.meterLabel}>Mag Delta (dB)</div>
            <div style={{...panelStyles.barBg, display: 'flex', justifyContent: 'center'}}>
              {/* Center-aligned delta bar (-12dB to +12dB mapping) */}
              <div style={{ display: 'flex', width: '100%', alignItems: 'center' }}>
                 <div style={{ flex: 1, display: 'flex', flexDirection: 'row-reverse' }}>
                   {metrics.magnitudeDeltaDb < 0 && (
                     <div style={{ height: '4px', background: '#ff4444', width: `${Math.min(100, Math.abs(metrics.magnitudeDeltaDb) / 12 * 100)}%` }}/>
                   )}
                 </div>
                 <div style={{ width: '2px', height: '10px', background: '#fff' }} />
                 <div style={{ flex: 1, display: 'flex' }}>
                   {metrics.magnitudeDeltaDb > 0 && (
                      <div style={{ height: '4px', background: '#00ff88', width: `${Math.min(100, metrics.magnitudeDeltaDb / 12 * 100)}%` }}/>
                   )}
                 </div>
              </div>
            </div>
            <div style={panelStyles.meterValue}>{metrics.magnitudeDeltaDb.toFixed(2)} dB</div>
          </div>

          <div style={panelStyles.meterBox}>
            <div style={panelStyles.meterLabel}>Freq Delta (Hz)</div>
            <div style={{...panelStyles.barBg, display: 'flex', justifyContent: 'center'}}>
              <div style={{ display: 'flex', width: '100%', alignItems: 'center' }}>
                 <div style={{ flex: 1, display: 'flex', flexDirection: 'row-reverse' }}>
                   {metrics.freqDeltaHz < 0 && (
                     <div style={{ height: '4px', background: '#ffaa00', width: `${Math.min(100, Math.abs(metrics.freqDeltaHz) / 3000 * 100)}%` }}/>
                   )}
                 </div>
                 <div style={{ width: '2px', height: '10px', background: '#fff' }} />
                 <div style={{ flex: 1, display: 'flex' }}>
                   {metrics.freqDeltaHz > 0 && (
                      <div style={{ height: '4px', background: '#44ccff', width: `${Math.min(100, metrics.freqDeltaHz / 3000 * 100)}%` }}/>
                   )}
                 </div>
              </div>
            </div>
            <div style={panelStyles.meterValue}>{metrics.freqDeltaHz.toFixed(0)} Hz</div>
          </div>
        </div>

        {/* Controls Section */}
        <div style={panelStyles.section}>
          <div style={panelStyles.sectionTitle}>Engine Parameters</div>
          
          <div style={panelStyles.controlRow}>
            <label style={panelStyles.controlLabel}>Gain Match Strength: {(params.gainCompStrength * 100).toFixed(0)}%</label>
            <input type="range" min="0" max="1" step="0.05" value={params.gainCompStrength} onChange={e => handleParamChange('gainCompStrength', parseFloat(e.target.value))} style={panelStyles.slider} />
          </div>

          <div style={panelStyles.controlRow}>
            <label style={panelStyles.controlLabel}>EQ Target Strength: {(params.eqCompStrength * 100).toFixed(0)}%</label>
            <input type="range" min="0" max="1" step="0.05" value={params.eqCompStrength} onChange={e => handleParamChange('eqCompStrength', parseFloat(e.target.value))} style={panelStyles.slider} />
          </div>

          <div style={panelStyles.controlRow}>
            <label style={panelStyles.controlLabel}>Smoothing: {params.smoothingMs} ms</label>
            <input type="range" min="10" max="500" step="10" value={params.smoothingMs} onChange={e => handleParamChange('smoothingMs', parseFloat(e.target.value))} style={panelStyles.slider} />
          </div>

          <div style={panelStyles.controlRow}>
             <label style={panelStyles.controlLabel}>Analysis Resolution (FFT Bins)</label>
             <select value={params.fftSize} onChange={e => handleParamChange('fftSize', parseInt(e.target.value))} style={panelStyles.select}>
               <option value={1024}>1024 Bins (≈23ms)</option>
               <option value={2048}>2048 Bins (≈46ms)</option>
               <option value={4096}>4096 Bins (≈93ms)</option>
             </select>
          </div>
        </div>
      </div>
      
      {/* Active Compensation Feedback */}
      <div style={panelStyles.footer}>
         <div style={panelStyles.footerItem}>
           <span>Auto-Gain Filter:</span>
           <span style={{ color: '#00ff88', marginLeft: '8px' }}>{metrics.compensationGainLvl.toFixed(3)}x</span>
         </div>
         <div style={panelStyles.footerItem}>
           <span>Mid/High Contour:</span>
           <span style={{ color: '#44ccff', marginLeft: '8px' }}>Tracking @ {metrics.compensationFilterFreq.toFixed(0)}Hz</span>
         </div>
      </div>
    </div>
  );
}

const panelStyles: Record<string, React.CSSProperties> = {
  container: {
    background: 'linear-gradient(145deg, rgba(18,18,34,0.8), rgba(14,14,26,0.6))',
    border: '1px solid rgba(255,255,255,0.06)', 
    borderRadius: '16px',
    padding: '20px',
    marginTop: '14px',
    boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
    backdropFilter: 'blur(30px)',
  },
  title: {
    margin: '0 0 16px 0',
    fontSize: '13px',
    color: '#00ff88',
    textTransform: 'uppercase',
    letterSpacing: '2px',
    fontWeight: 700,
    textShadow: '0 0 10px rgba(0,255,136,0.3)'
  },
  grid: {
    display: 'flex',
    gap: '20px',
  },
  section: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: '12px'
  },
  sectionTitle: {
    fontSize: '10px',
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '1px',
    borderBottom: '1px solid #333',
    paddingBottom: '6px'
  },
  meterBox: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px'
  },
  meterLabel: {
    fontSize: '10px',
    color: '#ccc',
    fontFamily: "'JetBrains Mono', monospace"
  },
  barBg: {
    height: '6px',
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '3px',
    overflow: 'hidden'
  },
  barFill: {
    height: '100%',
    transition: 'width 0.1s linear'
  },
  meterValue: {
    fontSize: '10px',
    color: '#00ff88',
    textAlign: 'right',
    fontFamily: "'JetBrains Mono', monospace"
  },
  controlRow: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px'
  },
  controlLabel: {
    fontSize: '10px',
    color: '#ccc',
    fontFamily: "'JetBrains Mono', monospace"
  },
  slider: {
    width: '100%',
    cursor: 'pointer',
    accentColor: '#00ff88'
  },
  select: {
    background: 'rgba(0,0,0,0.4)',
    border: '1px solid rgba(255,255,255,0.1)',
    color: '#fff',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '11px',
    fontFamily: "'JetBrains Mono', monospace",
    cursor: 'pointer'
  },
  footer: {
    marginTop: '16px',
    paddingTop: '12px',
    borderTop: '1px solid rgba(255,255,255,0.05)',
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '11px',
    color: '#888',
    fontFamily: "'JetBrains Mono', monospace"
  },
  footerItem: {
    display: 'flex',
    alignItems: 'center'
  }
};
