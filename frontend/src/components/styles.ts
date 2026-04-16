export const uiStyles: Record<string, React.CSSProperties> = {
  container: {
    width: '100vw', height: '100vh',
    background: 'linear-gradient(155deg, #020205 0%, #080812 40%, #04040a 70%, #000000 100%)',
    display: 'flex', flexDirection: 'column',
    fontFamily: "'Inter', 'JetBrains Mono', system-ui, sans-serif",
    color: '#e0e0f0', overflow: 'hidden',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '16px 32px', borderBottom: '1px solid rgba(255,255,255,0.03)',
    background: 'rgba(10, 10, 16, 0.6)',
    backdropFilter: 'blur(20px)',
    boxShadow: '0 4px 30px rgba(0, 0, 0, 0.5)',
    zIndex: 10,
  },
  title: { fontSize: '22px', fontWeight: 800, margin: 0, color: '#fff', letterSpacing: '0.5px' },
  titleAccent: { color: '#00ff88', fontSize: '26px' },
  titleSub: { color: '#888', fontWeight: 400, fontSize: '15px', marginLeft: '4px' },
  
  statusDot: { width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0 },
  statusText: { fontSize: '13px', color: '#888', fontFamily: "'JetBrains Mono', monospace" },

  glassCard: {
    background: 'rgba(20, 20, 32, 0.4)',
    border: '1px solid rgba(255,255,255,0.05)', borderRadius: '16px',
    boxShadow: '0 16px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)',
    backdropFilter: 'blur(24px)',
    padding: '30px',
  },

  primaryBtn: {
    padding: '14px 24px', border: 'none', borderRadius: '12px',
    background: 'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)', color: '#000',
    fontWeight: 700, fontSize: '14px', cursor: 'pointer', letterSpacing: '0.5px',
    boxShadow: '0 4px 20px rgba(0,255,136,0.3)',
    transition: 'transform 0.15s ease, box-shadow 0.15s ease',
    width: '100%',
  },
  secondaryBtn: {
    padding: '14px 24px', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px',
    background: 'rgba(255,255,255,0.03)', color: '#fff',
    fontWeight: 600, fontSize: '14px', cursor: 'pointer', transition: 'all 0.15s',
    width: '100%',
  },

  disconnectBtn: {
    padding: '8px 20px', border: '1px solid rgba(255,68,68,0.4)', borderRadius: '8px',
    background: 'rgba(255,68,68,0.08)', color: '#ff5555', cursor: 'pointer',
    fontSize: '13px', fontWeight: 600, transition: 'all 0.15s',
  },

  label: {
    display: 'block', fontSize: '11px', color: '#777', marginBottom: '8px',
    textTransform: 'uppercase' as const, letterSpacing: '1.5px', fontWeight: 600,
  },
  input: {
    width: '100%', padding: '14px 18px', background: 'rgba(0,0,0,0.4)',
    border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px',
    color: '#fff', fontSize: '15px', outline: 'none', boxSizing: 'border-box' as const,
    transition: 'border-color 0.2s, box-shadow 0.2s',
    fontFamily: "'JetBrains Mono', monospace",
  },
  mainArea: { flex: 1, padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' as const },
  arenaBox: {
    background: 'rgba(10, 10, 18, 0.4)',
    border: '1px solid rgba(255,255,255,0.06)', borderRadius: '20px',
    padding: '16px', overflow: 'hidden', flex: 1, 
    minHeight: '400px', boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
    display: 'flex', justifyContent: 'center', alignItems: 'center'
  },
};
