import { useState, useEffect, useCallback } from 'react';
import { IntentEngine, type IntentFrame, type PeerInfo } from '../engine/IntentEngine';
import { freqToNote } from '../components/IntentVisualizer';
import { uiStyles as styles } from '../components/styles';
import JoinCard from '../components/JoinCard';
import ActiveStage from '../components/ActiveStage';

const SERVER_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function CollabPage() {
  const [engine] = useState(() => new IntentEngine());
  const [connected, setConnected] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [roomId, setRoomId] = useState('');
  const [inputRoom, setInputRoom] = useState('');
  const [userName, setUserName] = useState('Musician');
  const [peers, setPeers] = useState<PeerInfo[]>([]);

  const [localFrame, setLocalFrame] = useState<IntentFrame | null>(null);

  useEffect(() => {
    engine.onLocalIntent = setLocalFrame;
    engine.onPeersUpdated = setPeers;
    engine.onConnectionChange = setConnected;
    // v4.0: Opus audio is the default. DDSP is emergency-only.
    // No setDDSPMode, no setLocalLoopback.
  }, [engine]);

  const handleStartCaptureAsync = useCallback(async () => {
    await engine.startCapture();
    setCapturing(true);
  }, [engine]);

  const handleCreateRoom = useCallback(async () => {
    try {
      const res = await fetch(`${SERVER_URL}/api/collab/create`, { 
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: userName })
      });
      const data = await res.json();
      setRoomId(data.room_id);
      await engine.connectToRoom(SERVER_URL, data.room_id, userName);
      await handleStartCaptureAsync();
    } catch (e) {
      console.error('Failed to create room:', e);
    }
  }, [engine, userName, handleStartCaptureAsync]);

  const handleJoinRoom = useCallback(async () => {
    if (!inputRoom) return;
    try {
      setRoomId(inputRoom);
      await engine.connectToRoom(SERVER_URL, inputRoom, userName);
      await handleStartCaptureAsync();
    } catch (e) {
      console.error('Failed to join room:', e);
    }
  }, [engine, inputRoom, userName, handleStartCaptureAsync]);

  const handleDisconnect = useCallback(() => {
    engine.disconnect();
    engine.stopCapture();
    setCapturing(false);
    setConnected(false);
    setRoomId('');
    setPeers([]);
  }, [engine]);

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h1 style={styles.title}>
            <span style={styles.titleAccent}>C</span>laudio
            <span style={styles.titleSub}> Studio</span>
          </h1>
        </div>
        {connected && (
          <button style={styles.disconnectBtn} onClick={handleDisconnect}>Leave Studio</button>
        )}
      </header>

      {!connected ? (
        <JoinCard 
          userName={userName} setUserName={setUserName}
          inputRoom={inputRoom} setInputRoom={setInputRoom}
          onJoin={handleJoinRoom} onCreate={handleCreateRoom}
        />
      ) : (
        <ActiveStage 
          peers={peers} 
          roomId={roomId}
          isCapturing={capturing}
          localPitch={localFrame ? freqToNote(localFrame.f0Hz) : undefined}
          localRms={localFrame?.rmsEnergy || 0}
        />
      )}
    </div>
  );
}
