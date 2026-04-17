import React from 'react';
import { uiStyles as styles } from './styles';

interface JoinCardProps {
  userName: string;
  setUserName: (val: string) => void;
  inputRoom: string;
  setInputRoom: (val: string) => void;
  instrument: string;
  setInstrument: (val: string) => void;
  environment: string;
  setEnvironment: (val: string) => void;
  onJoin: () => void;
  onCreate: () => void;
}

export default function JoinCard({
  userName,
  setUserName,
  inputRoom,
  setInputRoom,
  instrument,
  setInstrument,
  environment,
  setEnvironment,
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
        <div style={{ marginBottom: '24px' }}>
            <label style={styles.label}>Target Instrument Model</label>
            <select style={{...styles.input, background: 'rgba(255,255,255,0.03)', appearance: 'none', color: '#fff'}} value={instrument} onChange={e => setInstrument(e.target.value)}>
                <option value="/models/ddsp_model.onnx">Universal Hybrid (ddsp_model.onnx)</option>
                <option value="/models/ddsp_acoustic.onnx">Acoustic Guitar (Placeholder)</option>
            </select>
        </div>
        <div style={{ marginBottom: '36px' }}>
            <label style={styles.label}>Acoustic Environment</label>
            <select style={{...styles.input, background: 'rgba(255,255,255,0.03)', appearance: 'none', color: '#fff'}} value={environment} onChange={e => setEnvironment(e.target.value)}>
                <option value="Studio_A">Rigid Studio A</option>
                <option value="EMT_Plate">EMT 140 Vintage Plate</option>
                <option value="Cathedral">Cathedral</option>
                <option value="Claudio_Ambient">Claudio Ambient Wash</option>
            </select>
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
