
interface Props {
  correlation: number;   // -1 to +1
  severity: string;      // aligned | acceptable | warning | critical
  phaseAngle: number;
  needsFlip: boolean;
  offsetMs: number;
  onFlipPhase?: () => void;
}

const sevColors: Record<string, string> = {
  aligned: '#00ff88',
  acceptable: '#88cc44',
  warning: '#ff8844',
  critical: '#ff4466',
};

/**
 * PhaseMeter — real-time phase correlation display.
 * Shows a horizontal meter from -1 (out of phase) to +1 (in phase).
 */
export default function PhaseMeter({
  correlation,
  severity,
  phaseAngle,
  needsFlip,
  offsetMs,
  onFlipPhase,
}: Props) {
  const color = sevColors[severity] ?? '#505070';
  // Map correlation (-1..+1) to percentage (0..100)
  const pct = ((correlation + 1) / 2) * 100;

  return (
    <div
      style={{
        background: '#0c0c18',
        borderRadius: 10,
        padding: 12,
        border: `1px solid ${needsFlip ? '#ff446644' : '#252535'}`,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 10,
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
          PHASE CORRELATION
        </span>
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
          {severity}
        </span>
      </div>

      {/* Meter bar */}
      <div
        style={{
          position: 'relative',
          height: 16,
          background: '#141420',
          borderRadius: 8,
          overflow: 'hidden',
          marginBottom: 8,
        }}
      >
        {/* Center line */}
        <div
          style={{
            position: 'absolute',
            left: '50%',
            top: 0,
            bottom: 0,
            width: 1,
            background: '#303048',
          }}
        />
        {/* Indicator */}
        <div
          style={{
            position: 'absolute',
            left: `${pct}%`,
            top: 2,
            bottom: 2,
            width: 4,
            borderRadius: 2,
            background: color,
            transform: 'translateX(-50%)',
            boxShadow: `0 0 8px ${color}`,
            transition: 'left 0.1s ease-out',
          }}
        />
        {/* Labels */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: '0 4px',
            fontSize: 8,
            color: '#404058',
            lineHeight: '16px',
          }}
        >
          <span>-1 (OUT)</span>
          <span>0</span>
          <span>+1 (IN)</span>
        </div>
      </div>

      {/* Values */}
      <div
        style={{
          display: 'flex',
          gap: 16,
          justifyContent: 'space-between',
          fontSize: 10,
          color: '#808098',
        }}
      >
        <span>
          r = <strong style={{ color }}>{correlation.toFixed(3)}</strong>
        </span>
        <span>Angle: {phaseAngle.toFixed(1)}°</span>
        <span>Offset: {offsetMs.toFixed(2)}ms</span>
      </div>

      {/* Phase flip button */}
      {needsFlip && (
        <button
          onClick={onFlipPhase}
          style={{
            marginTop: 10,
            width: '100%',
            padding: '8px 0',
            background: '#ff446622',
            border: '1px solid #ff446666',
            borderRadius: 6,
            color: '#ff4466',
            fontWeight: 800,
            fontSize: 11,
            cursor: 'pointer',
            fontFamily: 'monospace',
            letterSpacing: 1,
          }}
        >
          ⚠ Ø FLIP PHASE — Polarity Inversion Detected
        </button>
      )}
    </div>
  );
}
