
interface InstrumentData {
  family: string;
  confidence: number;
  pickup_type: string;
  model_guess: string;
  model_confidence: number;
  coaching_hints: string[];
}

interface Props {
  detection: InstrumentData | null;
}

const familyLabels: Record<string, { label: string; icon: string; color: string }> = {
  guitar_electric: { label: 'Electric Guitar', icon: '🎸', color: '#ff8844' },
  guitar_acoustic: { label: 'Acoustic Guitar', icon: '🎸', color: '#ffaa00' },
  bass_electric: { label: 'Electric Bass', icon: '🎸', color: '#0088ff' },
  bass_acoustic: { label: 'Acoustic Bass', icon: '🎸', color: '#00aaff' },
  drums_kick: { label: 'Kick Drum', icon: '🥁', color: '#ff4466' },
  drums_snare: { label: 'Snare Drum', icon: '🥁', color: '#ff6688' },
  drums_hihat: { label: 'Hi-Hat', icon: '🥁', color: '#ffaa88' },
  drums_cymbal: { label: 'Cymbal', icon: '🥁', color: '#ffcc88' },
  drums_tom: { label: 'Tom', icon: '🥁', color: '#ff8866' },
  vocal_male: { label: 'Male Vocal', icon: '🎤', color: '#aa44ff' },
  vocal_female: { label: 'Female Vocal', icon: '🎤', color: '#cc66ff' },
  keys_piano: { label: 'Piano', icon: '🎹', color: '#44aaff' },
  keys_synth: { label: 'Synthesizer', icon: '🎹', color: '#44ffaa' },
  keys_organ: { label: 'Organ', icon: '🎹', color: '#66aacc' },
  brass: { label: 'Brass', icon: '🎺', color: '#ffcc00' },
  woodwind: { label: 'Woodwind', icon: '🎷', color: '#88cc44' },
  strings: { label: 'Strings', icon: '🎻', color: '#cc8844' },
  unknown: { label: 'Unknown', icon: '?', color: '#505070' },
};

const pickupLabels: Record<string, string> = {
  single_coil: 'Single Coil',
  humbucker: 'Humbucker',
  p90: 'P-90',
  piezo: 'Piezo',
  active: 'Active',
  unknown: '',
};

/**
 * InstrumentDetector — real-time instrument classification display.
 * Shows detected instrument family, pickup type, model guess, and coaching hints.
 */
export default function InstrumentDetector({ detection }: Props) {
  if (!detection) return null;

  const info = familyLabels[detection.family] ?? familyLabels.unknown;
  const confPct = (detection.confidence * 100).toFixed(0);

  return (
    <div
      style={{
        background: '#0c0c18',
        borderRadius: 10,
        padding: 12,
        border: `1px solid ${info.color}44`,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
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
          INSTRUMENT DETECTION
        </span>
      </div>

      {/* Detection result */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          marginBottom: 10,
        }}
      >
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: 8,
            background: `${info.color}22`,
            border: `1px solid ${info.color}44`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 20,
          }}
        >
          {info.icon}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 800, color: info.color }}>
            {info.label}
          </div>
          {detection.model_guess && (
            <div style={{ fontSize: 10, color: '#c0c0d8', marginTop: 2 }}>
              {detection.model_guess}
            </div>
          )}
          {detection.pickup_type && detection.pickup_type !== 'unknown' && (
            <div style={{ fontSize: 9, color: '#808098', marginTop: 1 }}>
              {pickupLabels[detection.pickup_type] ?? detection.pickup_type}
            </div>
          )}
        </div>
        {/* Confidence gauge */}
        <div style={{ textAlign: 'center' }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: '50%',
              background: `conic-gradient(${info.color} ${confPct}%, #1e1e2a ${confPct}%)`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <div
              style={{
                width: 30,
                height: 30,
                borderRadius: '50%',
                background: '#0c0c18',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 10,
                fontWeight: 900,
                color: info.color,
              }}
            >
              {confPct}
            </div>
          </div>
          <div style={{ fontSize: 7, color: '#505070', marginTop: 2 }}>CONF</div>
        </div>
      </div>

      {/* Coaching hints */}
      {detection.coaching_hints.length > 0 && (
        <div>
          {detection.coaching_hints.map((hint, i) => (
            <div
              key={i}
              style={{
                fontSize: 10,
                color: '#b0b0c8',
                lineHeight: 1.5,
                padding: '4px 8px',
                borderRadius: 4,
                background: '#141420',
                marginBottom: 3,
              }}
            >
              ▸ {hint}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
