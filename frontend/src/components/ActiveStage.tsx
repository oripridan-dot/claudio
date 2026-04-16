import React, { useEffect, useRef } from 'react';
import { type PeerInfo } from '../engine/IntentEngine';
import { drawSpatialArena } from './IntentVisualizer';
import { uiStyles as styles } from './styles';

interface ActiveStageProps {
  peers: PeerInfo[];
  localPitch?: string;
  isCapturing: boolean;
  roomId: string;
}

export default function ActiveStage({ peers, localPitch, isCapturing, roomId }: ActiveStageProps) {
  const arenaRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  
  // Use a ref so the animation loop always has the latest without closure capture issues
  const peersRef = useRef<PeerInfo[]>([]);
  useEffect(() => { peersRef.current = peers; }, [peers]);

  useEffect(() => {
    const draw = () => {
      const arena = arenaRef.current;
      if (arena) {
        const c = arena.getContext('2d');
        if (c) {
          drawSpatialArena(c, peersRef.current, arena.width, arena.height);
        }
      }
      animRef.current = requestAnimationFrame(draw);
    };
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '24px', gap: '24px' }}>
      {/* Top Status Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            ...styles.statusDot, 
            backgroundColor: isCapturing ? '#ff3333' : '#00ff88',
            boxShadow: isCapturing ? '0 0 12px #ff3333' : '0 0 12px #00ff88',
            animation: isCapturing ? 'pulse 2s infinite' : 'none'
          }} />
          <span style={{ fontSize: '13px', fontWeight: 600, color: '#fff', textTransform: 'uppercase', letterSpacing: '1px' }}>
            {isCapturing ? 'LIVE' : 'CONNECTED'}
          </span>
          <style>
            {`
              @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.4; }
                100% { opacity: 1; }
              }
            `}
          </style>
        </div>
        <div style={{ fontSize: '12px', color: '#888', fontFamily: "'JetBrains Mono', monospace" }}>
          ROOM: <span style={{ color: '#fff' }}>{roomId}</span> · PEERS: <span style={{ color: '#fff' }}>{peers.length}</span>
        </div>
      </div>

      {/* Main Arena */}
      <div style={{ ...styles.arenaBox, position: 'relative' }}>
         <canvas ref={arenaRef} width={800} height={500} style={{ width: '100%', height: '100%', borderRadius: '12px' }} />
         {localPitch && (
            <div style={{ 
               position: 'absolute', bottom: '24px', left: '24px', 
               background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)',
               padding: '12px 20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)'
            }}>
               <div style={{ fontSize: '10px', color: '#00ff88', letterSpacing: '2px', fontWeight: 700, marginBottom: '4px' }}>CURRENT PITCH</div>
               <div style={{ fontSize: '32px', color: '#fff', fontWeight: 800, fontFamily: "'JetBrains Mono', monospace" }}>{localPitch}</div>
            </div>
         )}
      </div>

      {/* Minimal Peer List */}
      <div style={{ display: 'flex', gap: '12px', overflowX: 'auto', paddingBottom: '8px' }}>
         {peers.map(p => (
            <div key={p.peer_id} style={{ 
               background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', 
               borderRadius: '12px', padding: '12px 16px', minWidth: '140px' 
            }}>
               <div style={{ fontSize: '13px', fontWeight: 600, color: '#fff', marginBottom: '4px' }}>{p.display_name}</div>
               <div style={{ fontSize: '11px', color: '#888', fontFamily: "'JetBrains Mono', monospace" }}>
                  {p.latency_ms}ms · {p.instrument}
               </div>
            </div>
         ))}
      </div>
    </div>
  );
}
