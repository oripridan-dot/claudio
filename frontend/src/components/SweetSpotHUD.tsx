
interface SweetSpotData {
  left_delay_ms: number;
  right_delay_ms: number;
  left_gain_db: number;
  right_gain_db: number;
  phantom_center_offset_deg: number;
  listener_offset_m: { x: number; y: number; z: number };
  mode: string;
  correction_active: boolean;
}

interface Props {
  data: SweetSpotData | null;
  mode: string;
  onModeChange: (mode: string) => void;
  active: boolean;
  onToggle: () => void;
}

const modes = [
  { id: 'focus_engineer', label: 'ENGINEER', icon: '◎' },
  { id: 'focus_couch', label: 'COUCH', icon: '◉' },
  { id: 'wide_compromise', label: 'WIDE', icon: '◈' },
  { id: 'dynamic_follow', label: 'FOLLOW', icon: '◍' },
];

const P = '#aa44ff';

/**
 * SweetSpotHUD — dynamic speaker calibration overlay.
 * Shows listener position, delay/gain corrections, and mode selection.
 */
export default function SweetSpotHUD({ data, mode, onModeChange, active, onToggle }: Props) {
  return (
    <div
      style={{
        background: '#0c0c18',
        borderRadius: 10,
        padding: 14,
        border: `1px solid ${active ? `${P}44` : '#252535'}`,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 12,
        }}
      >
        <span
          style={{
            fontSize: 9,
            color: '#7070a0',
            letterSpacing: 2,
            fontWeight: 700,
          }}
        >
          DYNAMIC SWEET SPOT
        </span>
        <button
          onClick={onToggle}
          style={{
            padding: '2px 9px',
            borderRadius: 4,
            fontSize: 10,
            fontWeight: 700,
            cursor: 'pointer',
            border: `1px solid ${active ? P : '#353550'}`,
            background: active ? `${P}22` : '#1a1a28',
            color: active ? P : '#707090',
            fontFamily: 'monospace',
          }}
        >
          {active ? '● ON' : '○ OFF'}
        </button>
      </div>

      {/* Mode selector */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 4,
          marginBottom: 12,
        }}
      >
        {modes.map((m) => (
          <button
            key={m.id}
            onClick={() => onModeChange(m.id)}
            style={{
              padding: '6px 2px',
              borderRadius: 6,
              fontSize: 8,
              fontWeight: 700,
              cursor: 'pointer',
              border: `1px solid ${mode === m.id ? P : '#252535'}`,
              background: mode === m.id ? `${P}22` : '#141420',
              color: mode === m.id ? P : '#505070',
              fontFamily: 'monospace',
              letterSpacing: 1,
              textAlign: 'center',
            }}
          >
            {m.icon}
            <br />
            {m.label}
          </button>
        ))}
      </div>

      {/* Correction display */}
      {data && active && (
        <>
          {/* Visual: listener position dot on a speaker diagram */}
          <div
            style={{
              position: 'relative',
              height: 100,
              background: '#141420',
              borderRadius: 8,
              marginBottom: 10,
              overflow: 'hidden',
            }}
          >
            {/* Speaker L */}
            <div
              style={{
                position: 'absolute',
                left: 10,
                top: 10,
                width: 14,
                height: 14,
                borderRadius: 3,
                background: P,
                fontSize: 7,
                color: '#000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 900,
              }}
            >
              L
            </div>
            {/* Speaker R */}
            <div
              style={{
                position: 'absolute',
                right: 10,
                top: 10,
                width: 14,
                height: 14,
                borderRadius: 3,
                background: P,
                fontSize: 7,
                color: '#000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 900,
              }}
            >
              R
            </div>

            {/* Listener dot */}
            <div
              style={{
                position: 'absolute',
                left: `calc(50% + ${Math.min(40, data.listener_offset_m.x * 40)}%)`,
                bottom: `${Math.max(10, Math.min(80, 50 + data.listener_offset_m.z * 20))}%`,
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: data.correction_active ? '#00ff88' : '#ff4466',
                boxShadow: `0 0 12px ${data.correction_active ? '#00ff88' : '#ff4466'}`,
                transform: 'translate(-50%, 50%)',
                transition: 'all 0.15s ease-out',
              }}
            />

            {/* Center marker */}
            <div
              style={{
                position: 'absolute',
                left: '50%',
                bottom: '50%',
                width: 6,
                height: 6,
                borderRadius: '50%',
                border: '1px dashed #303048',
                transform: 'translate(-50%, 50%)',
              }}
            />
          </div>

          {/* Correction values */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 6,
              fontSize: 10,
            }}
          >
            <div
              style={{
                background: '#141420',
                borderRadius: 6,
                padding: '6px 8px',
                textAlign: 'center',
              }}
            >
              <div style={{ color: '#7070a0', fontSize: 8, letterSpacing: 1 }}>
                L DELAY
              </div>
              <div style={{ color: P, fontWeight: 700 }}>
                {data.left_delay_ms.toFixed(2)}ms
              </div>
            </div>
            <div
              style={{
                background: '#141420',
                borderRadius: 6,
                padding: '6px 8px',
                textAlign: 'center',
              }}
            >
              <div style={{ color: '#7070a0', fontSize: 8, letterSpacing: 1 }}>
                R DELAY
              </div>
              <div style={{ color: P, fontWeight: 700 }}>
                {data.right_delay_ms.toFixed(2)}ms
              </div>
            </div>
            <div
              style={{
                background: '#141420',
                borderRadius: 6,
                padding: '6px 8px',
                textAlign: 'center',
              }}
            >
              <div style={{ color: '#7070a0', fontSize: 8, letterSpacing: 1 }}>
                L GAIN
              </div>
              <div style={{ color: P, fontWeight: 700 }}>
                {data.left_gain_db > 0 ? '+' : ''}
                {data.left_gain_db.toFixed(1)}dB
              </div>
            </div>
            <div
              style={{
                background: '#141420',
                borderRadius: 6,
                padding: '6px 8px',
                textAlign: 'center',
              }}
            >
              <div style={{ color: '#7070a0', fontSize: 8, letterSpacing: 1 }}>
                R GAIN
              </div>
              <div style={{ color: P, fontWeight: 700 }}>
                {data.right_gain_db > 0 ? '+' : ''}
                {data.right_gain_db.toFixed(1)}dB
              </div>
            </div>
          </div>

          {/* Phantom center offset */}
          <div
            style={{
              marginTop: 8,
              fontSize: 10,
              color: '#808098',
              textAlign: 'center',
            }}
          >
            Phantom center offset:{' '}
            <strong
              style={{
                color:
                  Math.abs(data.phantom_center_offset_deg) < 5 ? '#00ff88' : '#ff8844',
              }}
            >
              {data.phantom_center_offset_deg > 0 ? '+' : ''}
              {data.phantom_center_offset_deg.toFixed(1)}°
            </strong>
          </div>
        </>
      )}

      {!active && (
        <div
          style={{
            fontSize: 10,
            color: '#404058',
            textAlign: 'center',
            padding: '10px 0',
            lineHeight: 1.7,
          }}
        >
          Enable to track your head position
          <br />
          and auto-correct stereo imaging
        </div>
      )}
    </div>
  );
}
