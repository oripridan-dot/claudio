
interface RoomScanData {
  rt60_ms: number;
  rt60_category: string;
  room_modes: Array<{
    frequency_hz: number;
    magnitude_db: number;
    likely_dimension: string;
    treatment_advice: string;
  }>;
  flutter_echo_detected: boolean;
  bass_buildup_detected: boolean;
  overall_quality: string;
  acoustic_advice: Array<{
    category: string;
    description: string;
    action: string;
  }>;
}

interface Props {
  scanData: RoomScanData | null;
  scanning: boolean;
  onStartScan: () => void;
}

const qualityColors: Record<string, string> = {
  excellent: '#00ff88',
  good: '#88cc44',
  fair: '#ffaa00',
  poor: '#ff8844',
  critical: '#ff4466',
};

/**
 * RoomScannerPanel — visual room acoustics analysis.
 * Shows RT60, room modes, flutter echo, bass buildup, and correction advice.
 */
export default function RoomScannerPanel({ scanData, scanning, onStartScan }: Props) {
  const color = qualityColors[scanData?.overall_quality ?? 'fair'] ?? '#505070';

  return (
    <div
      style={{
        background: '#0c0c18',
        borderRadius: 10,
        padding: 14,
        border: '1px solid #252535',
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
          ROOM SCANNER
        </span>
        {scanData && (
          <span
            style={{
              fontSize: 10,
              color,
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: 4,
              background: `${color}22`,
              textTransform: 'uppercase',
            }}
          >
            {scanData.overall_quality}
          </span>
        )}
      </div>

      {/* Scan button */}
      {!scanData && (
        <button
          onClick={onStartScan}
          disabled={scanning}
          style={{
            width: '100%',
            padding: '12px 0',
            background: scanning ? '#1a1a28' : '#00ff8822',
            border: `1px solid ${scanning ? '#353550' : '#00ff8866'}`,
            borderRadius: 8,
            color: scanning ? '#505070' : '#00ff88',
            fontWeight: 800,
            fontSize: 12,
            cursor: scanning ? 'default' : 'pointer',
            fontFamily: 'monospace',
            letterSpacing: 2,
          }}
        >
          {scanning ? '● SCANNING...' : '◉ SCAN ROOM (CLAP)'}
        </button>
      )}

      {scanData && (
        <>
          {/* RT60 */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 10,
              marginBottom: 12,
            }}
          >
            <div
              style={{
                background: '#141420',
                borderRadius: 8,
                padding: '10px 12px',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  fontSize: 9,
                  color: '#7070a0',
                  letterSpacing: 2,
                  marginBottom: 4,
                }}
              >
                RT60
              </div>
              <div style={{ fontSize: 20, fontWeight: 900, color }}>
                {scanData.rt60_ms.toFixed(0)}
                <span style={{ fontSize: 10, color: '#505070' }}>ms</span>
              </div>
              <div
                style={{
                  fontSize: 9,
                  color: '#505070',
                  marginTop: 2,
                  textTransform: 'capitalize',
                }}
              >
                {scanData.rt60_category}
              </div>
            </div>
            <div
              style={{
                background: '#141420',
                borderRadius: 8,
                padding: '10px 12px',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  fontSize: 9,
                  color: '#7070a0',
                  letterSpacing: 2,
                  marginBottom: 4,
                }}
              >
                ISSUES
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 10 }}>
                <span
                  style={{
                    color: scanData.flutter_echo_detected ? '#ff8844' : '#00ff8866',
                  }}
                >
                  {scanData.flutter_echo_detected ? '⚠ Flutter Echo' : '✓ No Flutter'}
                </span>
                <span
                  style={{
                    color: scanData.bass_buildup_detected ? '#ff8844' : '#00ff8866',
                  }}
                >
                  {scanData.bass_buildup_detected ? '⚠ Bass Buildup' : '✓ Bass OK'}
                </span>
              </div>
            </div>
          </div>

          {/* Room modes */}
          {scanData.room_modes.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div
                style={{
                  fontSize: 9,
                  color: '#7070a0',
                  letterSpacing: 2,
                  marginBottom: 6,
                }}
              >
                ROOM MODES
              </div>
              {scanData.room_modes.slice(0, 4).map((mode, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '4px 8px',
                    borderRadius: 4,
                    background: i % 2 === 0 ? '#141420' : 'transparent',
                    fontSize: 10,
                    color: '#808098',
                    marginBottom: 2,
                  }}
                >
                  <span style={{ color: '#ffaa00' }}>
                    {mode.frequency_hz.toFixed(0)}Hz
                  </span>
                  <span>{mode.magnitude_db.toFixed(1)}dB</span>
                  <span style={{ color: '#505070' }}>{mode.likely_dimension}</span>
                </div>
              ))}
            </div>
          )}

          {/* Advice */}
          {scanData.acoustic_advice.length > 0 && (
            <div>
              <div
                style={{
                  fontSize: 9,
                  color: '#7070a0',
                  letterSpacing: 2,
                  marginBottom: 6,
                }}
              >
                CORRECTIONS
              </div>
              {scanData.acoustic_advice.map((adv, i) => (
                <div
                  key={i}
                  style={{
                    background: '#ff884411',
                    border: '1px solid #ff884433',
                    borderRadius: 6,
                    padding: '8px 10px',
                    marginBottom: 6,
                  }}
                >
                  <div
                    style={{
                      fontSize: 10,
                      color: '#ff8844',
                      fontWeight: 700,
                      marginBottom: 4,
                      textTransform: 'capitalize',
                    }}
                  >
                    {adv.category.replace('_', ' ')}
                  </div>
                  <div style={{ fontSize: 10, color: '#c0c0d8', lineHeight: 1.5 }}>
                    {adv.action}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Re-scan button */}
          <button
            onClick={onStartScan}
            style={{
              width: '100%',
              padding: '8px 0',
              marginTop: 8,
              background: 'transparent',
              border: '1px solid #353550',
              borderRadius: 6,
              color: '#505070',
              fontWeight: 700,
              fontSize: 10,
              cursor: 'pointer',
              fontFamily: 'monospace',
            }}
          >
            ↻ RE-SCAN
          </button>
        </>
      )}
    </div>
  );
}
