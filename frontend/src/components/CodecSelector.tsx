import { CODEC_PROFILES, getCodecById } from '../engine/codecProfiles';
import type { CodecProfile } from '../engine/codecProfiles';

interface Props {
  selected: string;
  onSelect: (codec: CodecProfile) => void;
}

/**
 * CodecSelector — visual codec picker with latency + quality badges.
 * Lives in the right-hand sidebar of StudioPage.
 */
export default function CodecSelector({ selected, onSelect }: Props) {
  return (
    <div className="flex flex-col gap-1">
      <p className="text-[10px] uppercase tracking-widest text-claudio-muted mb-1">Output Codec</p>
      {CODEC_PROFILES.map((c) => {
        const active = c.id === selected;
        return (
          <button
            key={c.id}
            onClick={() => onSelect(c)}
            className={[
              'flex items-center gap-2 px-3 py-1.5 rounded text-left transition-all text-xs',
              active
                ? 'bg-claudio-card ring-1'
                : 'hover:bg-claudio-surface opacity-60 hover:opacity-90',
            ].join(' ')}
            style={active ? { ringColor: c.color } : {}}
          >
            {/* Colour dot */}
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: c.color }}
            />
            {/* Codec name */}
            <span className="flex-1 font-semibold" style={{ color: active ? c.color : '#a0a0c0' }}>
              {c.label}
            </span>
            {/* Latency badge */}
            <span className="text-[10px] px-1.5 py-0.5 rounded" style={{
              background: active ? c.color + '22' : '#ffffff10',
              color: active ? c.color : '#606080',
            }}>
              {c.latencyMs === 0 ? '—' : `~${c.latencyMs}ms`}
            </span>
            {/* Quality badge */}
            {!c.lossy && (
              <span className="text-[10px] px-1 rounded bg-indigo-900/40 text-indigo-400">
                lossless
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
