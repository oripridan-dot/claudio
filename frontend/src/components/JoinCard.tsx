import React from 'react';
import { uiStyles as styles } from './styles';

interface JoinCardProps {
  userName: string;
  setUserName: (val: string) => void;
  inputRoom: string;
  setInputRoom: (val: string) => void;
  onJoin: () => void;
  onCreate: () => void;
}

export default function JoinCard({
  userName,
  setUserName,
  inputRoom,
  setInputRoom,
  onJoin,
  onCreate
}: JoinCardProps) {
  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1, position: 'relative' }}>
      {/* Background glow sphere for visual effect */}
      <div style={{ position: 'absolute', width: '500px', height: '500px', background: 'radial-gradient(circle, rgba(0,255,136,0.05) 0%, rgba(0,0,0,0) 70%)', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', pointerEvents: 'none' }} />
      
      <div style={{ ...styles.glassCard, width: '420px', padding: '40px' }}>
        <h2 style={{ margin: '0 0 32px 0', color: '#fff', fontSize: '24px', fontWeight: 700, textAlign: 'center', letterSpacing: '0.5px' }}>
          Enter Studio
        </h2>
        <div style={{ marginBottom: '24px' }}>
          <label style={styles.label}>Your Name</label>
          <input style={styles.input} value={userName}
            onChange={e => setUserName(e.target.value)} placeholder="Musician" />
        </div>
        <div style={{ marginBottom: '24px' }}>
          <label style={styles.label}>Room Code (Optional)</label>
          <input style={styles.input} value={inputRoom}
            onChange={e => setInputRoom(e.target.value)} placeholder="Leave blank to create new" />
        </div>
        <button 
          style={styles.primaryBtn} 
          onClick={inputRoom.trim() !== '' ? onJoin : onCreate}
        >
          {inputRoom.trim() !== '' ? 'Join Room' : 'Start New Session'}
        </button>
      </div>
    </div>
  );
}
