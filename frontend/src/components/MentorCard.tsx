
interface MentorTip {
  tip_id: string;
  trigger: string;
  phase: string;
  mentor: {
    name: string;
    title: string;
    photo_asset: string;
    notable_works: string[];
    era: string;
    specialty: string;
  };
  quote: string;
  context_location: string;
  context_date: string;
  physical_action: string;
  ui_action: string;
  severity: string;
}

interface Props {
  tip: MentorTip | null;
  onDismiss: () => void;
  visible: boolean;
}

const severityColors: Record<string, string> = {
  critical: '#ff4466',
  warning: '#ff8844',
  tip: '#00ff88',
  info: '#0088ff',
};

/**
 * MentorCard — Frosted-glass overlay showing a pro-tip "Mentorship Moment".
 * Displays the engineer's name, title, quote, studio/date context,
 * and a concrete physical action the user should take.
 */
export default function MentorCard({ tip, onDismiss, visible }: Props) {
  if (!tip || !visible) return null;

  const accent = severityColors[tip.severity] ?? '#00ff88';

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        width: 420,
        maxHeight: '60vh',
        background: 'rgba(14, 14, 24, 0.92)',
        backdropFilter: 'blur(20px)',
        border: `1px solid ${accent}44`,
        borderRadius: 16,
        padding: 0,
        zIndex: 1000,
        boxShadow: `0 8px 48px ${accent}22, 0 0 0 1px ${accent}11`,
        animation: 'slideInRight 0.4s ease-out',
        fontFamily: 'monospace',
        overflow: 'hidden',
      }}
    >
      {/* Severity bar */}
      <div
        style={{
          height: 3,
          background: `linear-gradient(90deg, ${accent}, transparent)`,
        }}
      />

      <div style={{ padding: '16px 20px' }}>
        {/* Header: Mentor info */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            marginBottom: 14,
          }}
        >
          {/* Mentor photo placeholder (circular) */}
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: '50%',
              background: `${accent}22`,
              border: `2px solid ${accent}66`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 22,
              color: accent,
              fontWeight: 900,
              flexShrink: 0,
            }}
          >
            {tip.mentor.name
              .split(' ')
              .map((n) => n[0])
              .join('')}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 14, fontWeight: 800, color: '#e0e0f0' }}>
              {tip.mentor.name}
            </div>
            <div style={{ fontSize: 10, color: '#7070a0', marginTop: 2 }}>
              {tip.mentor.title}
            </div>
            <div style={{ fontSize: 9, color: '#505070', marginTop: 1 }}>
              {tip.mentor.notable_works?.slice(0, 2).join(' · ')}
            </div>
          </div>
          <button
            onClick={onDismiss}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#505070',
              fontSize: 18,
              cursor: 'pointer',
              padding: 4,
            }}
          >
            ✕
          </button>
        </div>

        {/* Quote */}
        <div
          style={{
            background: '#0a0a16',
            borderRadius: 10,
            padding: '12px 14px',
            marginBottom: 12,
            borderLeft: `3px solid ${accent}`,
          }}
        >
          <div
            style={{
              fontSize: 12,
              color: '#c0c0d8',
              lineHeight: 1.7,
              fontStyle: 'italic',
            }}
          >
            "{tip.quote}"
          </div>
          <div
            style={{
              fontSize: 9,
              color: '#505070',
              marginTop: 8,
              textAlign: 'right',
            }}
          >
            — {tip.context_location}, {tip.context_date}
          </div>
        </div>

        {/* Physical action */}
        <div
          style={{
            background: `${accent}11`,
            border: `1px solid ${accent}33`,
            borderRadius: 8,
            padding: '10px 12px',
          }}
        >
          <div
            style={{
              fontSize: 9,
              color: accent,
              letterSpacing: 2,
              fontWeight: 700,
              marginBottom: 6,
            }}
          >
            ▸ ACTION
          </div>
          <div style={{ fontSize: 11, color: '#d0d0e0', lineHeight: 1.6 }}>
            {tip.physical_action}
          </div>
        </div>

        {/* Phase badge */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: 12,
            alignItems: 'center',
          }}
        >
          <span
            style={{
              fontSize: 9,
              color: '#505070',
              letterSpacing: 1,
              textTransform: 'uppercase',
            }}
          >
            {tip.phase?.replace('_', ' ')} · {tip.trigger?.replace('_', ' ')}
          </span>
          <span
            style={{
              fontSize: 9,
              color: accent,
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: 4,
              background: `${accent}22`,
              border: `1px solid ${accent}44`,
              textTransform: 'uppercase',
            }}
          >
            {tip.severity}
          </span>
        </div>
      </div>
    </div>
  );
}
