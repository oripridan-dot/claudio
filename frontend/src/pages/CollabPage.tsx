import { useState, useEffect, useRef, useCallback } from 'react';
import {
  IntentEngine,
  type IntentFrame,
  type PeerInfo,
  type CollabMetrics,
} from '../engine/IntentEngine';

import {
  PITCH_HISTORY_SIZE,
  freqToNote,
  drawPitchHistory,
  drawLoudnessHistory,
  drawSpatialArena,
  collabStyles as styles,
} from '../components/IntentVisualizer';
import { RTCalibrationEngine } from '../engine/RTCalibrationEngine';
import { RTCalibrationPanel } from '../components/RTCalibrationPanel';
import { GeometricFracture, type Critique } from '../components/GeometricFracture';

// ─── Constants ──────────────────────────────────────────────────────────────

const SERVER_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';


// ─── CollabPage Component ───────────────────────────────────────────────────

export default function CollabPage() {
  const [engine] = useState(() => new IntentEngine());
  const [connected, setConnected] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [ddspMode, setDdspMode] = useState(false);
  const [loopbackMode, setLoopbackMode] = useState(false);
  const [roomId, setRoomId] = useState('');
  const [inputRoom, setInputRoom] = useState('');
  const [userName, setUserName] = useState('Musician');
  const [peers, setPeers] = useState<PeerInfo[]>([]);
  const [metrics, setMetrics] = useState<CollabMetrics | null>(null);
  const [telemetry, setTelemetry] = useState({ jitter: 0, loss: 0 });

  const [localFrame, setLocalFrame] = useState<IntentFrame | null>(null);
  const [remoteFrame, setRemoteFrame] = useState<IntentFrame | null>(null);



  // RT Calibration Component
  const [calibrationEngine, setCalibrationEngine] = useState<RTCalibrationEngine | null>(null);

  // Geometric Validation State
  const [activeCritiques, setActiveCritiques] = useState<Critique[]>([]);

  const localHistoryRef = useRef<IntentFrame[]>([]);
  const remoteHistoryRef = useRef<IntentFrame[]>([]);

  const localPitchRef = useRef<HTMLCanvasElement>(null);
  const remotePitchRef = useRef<HTMLCanvasElement>(null);
  const localLoudRef = useRef<HTMLCanvasElement>(null);
  const remoteLoudRef = useRef<HTMLCanvasElement>(null);
  const arenaRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  // Use a ref for peers so the animation loop can access the latest state without closure issues
  const peersRef = useRef<PeerInfo[]>([]);
  useEffect(() => { peersRef.current = peers; }, [peers]);

  useEffect(() => {
    const draw = () => {
      const lp = localPitchRef.current;
      const rp = remotePitchRef.current;
      const ll = localLoudRef.current;
      const rl = remoteLoudRef.current;
      const arena = arenaRef.current;
      if (lp) { const c = lp.getContext('2d'); if (c) drawPitchHistory(c, localHistoryRef.current, lp.width, lp.height, '#00ff88', 'LOCAL'); }
      if (rp) { const c = rp.getContext('2d'); if (c) drawPitchHistory(c, remoteHistoryRef.current, rp.width, rp.height, '#ff6644', 'REMOTE'); }
      if (ll) { const c = ll.getContext('2d'); if (c) drawLoudnessHistory(c, localHistoryRef.current, ll.width, ll.height, '#00ff88'); }
      if (rl) { const c = rl.getContext('2d'); if (c) drawLoudnessHistory(c, remoteHistoryRef.current, rl.width, rl.height, '#ff6644'); }
      if (arena) { const c = arena.getContext('2d'); if (c) drawSpatialArena(c, peersRef.current, arena.width, arena.height); }
      animRef.current = requestAnimationFrame(draw);
    };
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  useEffect(() => {
    engine.onLocalIntent = (frame) => {
      setLocalFrame(frame);
      localHistoryRef.current.push(frame);
      if (localHistoryRef.current.length > PITCH_HISTORY_SIZE) localHistoryRef.current.shift();
    };
    engine.onRemoteIntent = (frame) => {
      setRemoteFrame(frame);
      remoteHistoryRef.current.push(frame);
      if (remoteHistoryRef.current.length > PITCH_HISTORY_SIZE) remoteHistoryRef.current.shift();
    };
    engine.onPeersUpdated = setPeers;
    engine.onMetrics = setMetrics;
    engine.onConnectionChange = setConnected;
    engine.onCritique = (critiques: Critique[]) => {
      setActiveCritiques(critiques);
      // Auto-clear visual feedback after 1.5 seconds to bounce back to cohesive state if the musician fixed their performance
      setTimeout(() => setActiveCritiques([]), 1500);
    };
  }, [engine]);

  useEffect(() => {
    if (!connected) return;
    const interval = setInterval(() => {
      engine.requestMetrics();
      setTelemetry({ jitter: engine.jitterMs, loss: engine.packetLossPercent });
    }, 1000);
    return () => clearInterval(interval);
  }, [connected, engine]);

  const handleCreateRoom = useCallback(async () => {
    const res = await fetch(`${SERVER_URL}/api/collab/create`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: userName })
    });
    const data = await res.json();
    setRoomId(data.room_id);
    setInputRoom(data.room_id);
    await engine.connectToRoom(SERVER_URL, data.room_id, userName);
  }, [engine, userName]);

  const handleJoinRoom = useCallback(async () => {
    if (!inputRoom) return;
    setRoomId(inputRoom);
    await engine.connectToRoom(SERVER_URL, inputRoom, userName);
  }, [engine, inputRoom, userName]);

  const handleStartCapture = useCallback(async () => {
    await engine.startCapture();
    setCapturing(true);

    const ctx = engine.getAudioContext();
    if (ctx) {
      if (calibrationEngine) calibrationEngine.destroy();
      const calib = new RTCalibrationEngine(ctx);
      const micSrc = engine.getInputAnalyser();
      if (micSrc) calib.connectInputSource(micSrc);

      const compNode = calib.getCompensationInputNode();
      engine.redirectOutput(compNode);
      
      calib.connectOutputTap(compNode);
      calib.getFinalOutputNode().connect(ctx.destination);

      setCalibrationEngine(calib);
    }
  }, [engine, resynthActive, calibrationEngine]);

  const handleStopCapture = useCallback(() => {
    engine.stopCapture();
    setCapturing(false);
    calibrationEngine?.destroy();
    setCalibrationEngine(null);
  }, [engine, resynthActive, calibrationEngine]);

  const handleDisconnect = useCallback(() => {
    engine.disconnect();
    engine.stopCapture();
    setCapturing(false);
    setConnected(false);
    setRoomId('');
    setPeers([]);
    setMetrics(null);
    localHistoryRef.current = [];
    remoteHistoryRef.current = [];
    calibrationEngine?.destroy();
    setCalibrationEngine(null);
  }, [engine, calibrationEngine]);



  return (
    <div style={styles.container}>
      {import.meta.env.VITE_APP_ENV === 'teaching' && connected && (
        <GeometricFracture critiques={activeCritiques} />
      )}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <h1 style={styles.title}>
            <span style={styles.titleAccent}>C</span>laudio
            <span style={styles.titleSub}> Collab</span>
          </h1>
          <div style={{
            ...styles.statusDot,
            backgroundColor: connected ? '#00ff88' : '#ff4444',
            boxShadow: connected ? '0 0 8px #00ff88' : '0 0 8px #ff4444',
          }} />
          <span style={styles.statusText}>
            {connected ? `Room: ${roomId}` : 'Disconnected'}
          </span>
          {connected && (
             <>
               <label style={{ marginLeft: '20px', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: '#ccc', fontSize: '0.9rem' }}>
                  <input type="checkbox" checked={ddspMode} onChange={e => {
                     setDdspMode(e.target.checked);
                     engine.setDDSPMode(e.target.checked);
                  }} />
                  High-Fidelity DDSP Mode (Server-Side)
               </label>
               <label style={{ marginLeft: '20px', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: '#00ff88', fontSize: '0.9rem' }}>
                  <input type="checkbox" checked={loopbackMode} onChange={e => {
                     setLoopbackMode(e.target.checked);
                     engine.setLocalLoopback(e.target.checked);
                  }} />
                  Local Neural Loopback
               </label>
             </>
          )}
        </div>
        {connected && (
          <button style={styles.disconnectBtn} onClick={handleDisconnect}>Disconnect</button>
        )}
      </header>

      {!connected && (
        <div style={styles.connectPanel}>
          <div style={styles.connectCard}>
            <h2 style={styles.connectTitle}>Join a Session</h2>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Your Name</label>
              <input id="input-user-name" style={styles.input} value={userName}
                onChange={e => setUserName(e.target.value)} placeholder="Musician" />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Room Code</label>
              <input id="input-room-code" style={styles.input} value={inputRoom}
                onChange={e => setInputRoom(e.target.value)} placeholder="e.g. a1b2c3d4" />
            </div>
            <div style={styles.btnRow}>
              <button id="btn-create-room" style={styles.primaryBtn} onClick={handleCreateRoom}>Create Room</button>
              <button id="btn-join-room" style={styles.secondaryBtn} onClick={handleJoinRoom}>Join Room</button>
            </div>
          </div>
        </div>
      )}

      {connected && (
        <div style={styles.dashboard}>
          <aside style={styles.sidebar}>
            <div style={styles.sideSection}>
              <h3 style={styles.sideTitle}>Peers ({peers.length})</h3>
              {peers.map(p => (
                <div key={p.peer_id} style={styles.peerCard}>
                  <div style={styles.peerName}>{p.display_name}</div>
                  <div style={styles.peerMeta}>{p.instrument} · {p.latency_ms}ms</div>
                  <div style={styles.peerPackets}>{p.packets_sent.toLocaleString()} pkts</div>
                </div>
              ))}
            </div>
            {metrics && (
              <div style={styles.sideSection}>
                <h3 style={styles.sideTitle}>Network</h3>
                <div style={styles.metricRow}><span style={styles.metricLabel}>Bandwidth</span><span style={styles.metricValue}>{metrics.bandwidth_kbps.toFixed(1)} KB/s</span></div>
                <div style={styles.metricRow}>
                  <span style={styles.metricLabel}>Latency / Jitter</span>
                  <span style={styles.metricValue}>
                    {metrics.avg_latency_ms} ms /{' '}
                    <span style={{ color: telemetry.jitter < 20 ? '#00ff88' : telemetry.jitter < 50 ? '#ffaa00' : '#ff4444' }}>
                      {telemetry.jitter.toFixed(1)} ms
                    </span>
                  </span>
                </div>
                <div style={styles.metricRow}>
                  <span style={styles.metricLabel}>Packet Loss</span>
                  <span style={{ ...styles.metricValue, color: telemetry.loss < 1 ? '#00ff88' : telemetry.loss < 5 ? '#ffaa00' : '#ff4444' }}>
                    {telemetry.loss.toFixed(1)}%
                  </span>
                </div>
                <div style={styles.metricRow}><span style={styles.metricLabel}>Packets</span><span style={styles.metricValue}>{metrics.total_packets.toLocaleString()}</span></div>
                <div style={styles.metricRow}><span style={styles.metricLabel}>Data</span><span style={styles.metricValue}>{(metrics.bytes_transmitted / 1024).toFixed(1)} KB</span></div>
              </div>
            )}
            <div style={styles.sideSection}>
              {!capturing
                ? <button id="btn-start-capture" style={styles.captureBtn} onClick={handleStartCapture}>Start Capture</button>
                : <button id="btn-stop-capture" style={styles.stopBtn} onClick={handleStopCapture}>Stop Capture</button>
              }
            </div>



          </aside>

          <main style={styles.mainArea}>
            {calibrationEngine && <RTCalibrationPanel engine={calibrationEngine} />}
            <div style={styles.arenaBox}>
              <canvas ref={arenaRef} width={800} height={280} style={styles.canvas} />
            </div>
            <div style={styles.frameRow}>
              <div style={styles.frameBox}>
                <div style={styles.frameLabel}>LOCAL PITCH</div>
                <div style={styles.frameValue}>{localFrame ? freqToNote(localFrame.f0Hz) : '—'}</div>
                <div style={styles.frameHz}>{localFrame && localFrame.f0Hz > 0 ? `${localFrame.f0Hz.toFixed(1)} Hz` : ''}</div>
              </div>
              <div style={styles.frameBox}>
                <div style={{ ...styles.frameLabel, color: '#ff6644' }}>REMOTE PITCH</div>
                <div style={{ ...styles.frameValue, color: '#ff6644' }}>{remoteFrame ? freqToNote(remoteFrame.f0Hz) : '—'}</div>
                <div style={{ ...styles.frameHz, color: '#ff664488' }}>{remoteFrame && remoteFrame.f0Hz > 0 ? `${remoteFrame.f0Hz.toFixed(1)} Hz` : ''}</div>
              </div>
            </div>
            <div style={styles.canvasRow}>
              <div style={styles.canvasBox}><canvas ref={localPitchRef} width={560} height={180} style={styles.canvas} /></div>
              <div style={styles.canvasBox}><canvas ref={remotePitchRef} width={560} height={180} style={styles.canvas} /></div>
            </div>
            <div style={styles.canvasRow}>
              <div style={styles.canvasBox}>
                <div style={styles.canvasLabel}>Local Loudness</div>
                <canvas ref={localLoudRef} width={560} height={60} style={styles.canvas} />
              </div>
              <div style={styles.canvasBox}>
                <div style={{ ...styles.canvasLabel, color: '#ff6644' }}>Remote Loudness</div>
                <canvas ref={remoteLoudRef} width={560} height={60} style={styles.canvas} />
              </div>
            </div>
            <div style={styles.statsRow}>
              <div style={styles.statCard}>
                <div style={styles.statLabel}>Confidence</div>
                <div style={styles.statBar}>
                  <div style={{ ...styles.statFill, width: `${(localFrame?.confidence ?? 0) * 100}%`, backgroundColor: '#00ff88' }} />
                </div>
              </div>
              <div style={styles.statCard}>
                <div style={styles.statLabel}>Onset</div>
                <div style={{
                  ...styles.onsetDot,
                  backgroundColor: localFrame?.isOnset ? '#ffaa00' : '#222',
                  boxShadow: localFrame?.isOnset ? '0 0 12px #ffaa00' : 'none',
                }} />
              </div>
              <div style={styles.statCard}>
                <div style={styles.statLabel}>Centroid</div>
                <div style={styles.statValue}>{localFrame ? `${Math.round(localFrame.spectralCentroid)} Hz` : '—'}</div>
              </div>
              <div style={styles.statCard}>
                <div style={styles.statLabel}>RMS</div>
                <div style={styles.statValue}>{localFrame ? localFrame.rmsEnergy.toFixed(4) : '—'}</div>
              </div>
            </div>
          </main>
        </div>
      )}
    </div>
  );
}
