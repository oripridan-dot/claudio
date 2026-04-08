
interface ChecklistItem {
  item_id: string;
  label: string;
  description: string;
  is_automated: boolean;
  completed: boolean;
  mentor_tip_id: string;
}

interface PhaseConfig {
  phase: string;
  title: string;
  subtitle: string;
  status: string;
  checklist: ChecklistItem[];
  completion_percentage: number;
}

interface RoadmapState {
  current_phase: string;
  phases: PhaseConfig[];
  overall_progress: number;
}

interface Props {
  roadmap: RoadmapState | null;
  onAdvancePhase: () => void;
  onCompleteItem: (itemId: string) => void;
  expanded: boolean;
  onToggle: () => void;
}

const phaseIcons: Record<string, string> = {
  setup: '◎',
  tracking: '●',
  mixing: '◈',
  mastering: '◍',
};

const statusColors: Record<string, string> = {
  locked: '#303048',
  active: '#00ff88',
  completed: '#0088ff',
  skipped: '#505070',
};

/**
 * RoadmapOverlay — progressive disclosure roadmap showing user's journey
 * from setup to mastering release.
 */
export default function RoadmapOverlay({
  roadmap,
  onAdvancePhase,
  onCompleteItem,
  expanded,
  onToggle,
}: Props) {
  if (!roadmap) return null;

  // Compact bar
  if (!expanded) {
    return (
      <div
        onClick={onToggle}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '6px 14px',
          background: '#0e0e18',
          borderTop: '1px solid #1e1e2a',
          cursor: 'pointer',
          fontFamily: 'monospace',
          userSelect: 'none',
        }}
      >
        <span style={{ fontSize: 9, color: '#7070a0', letterSpacing: 2, fontWeight: 700 }}>
          ROADMAP
        </span>
        {/* Mini phase pills */}
        {roadmap.phases.map((ph) => {
          const col = statusColors[ph.status] ?? '#303048';
          return (
            <div
              key={ph.phase}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                padding: '2px 8px',
                borderRadius: 4,
                background: `${col}22`,
                border: `1px solid ${col}44`,
              }}
            >
              <span style={{ fontSize: 9, color: col }}>
                {phaseIcons[ph.phase] ?? '○'}
              </span>
              <span style={{ fontSize: 8, color: col, fontWeight: 700, textTransform: 'uppercase' }}>
                {ph.phase}
              </span>
              {ph.completion_percentage > 0 && ph.completion_percentage < 100 && (
                <span style={{ fontSize: 8, color: '#505070' }}>
                  {ph.completion_percentage.toFixed(0)}%
                </span>
              )}
            </div>
          );
        })}
        <div style={{ flex: 1 }} />
        {/* Overall progress bar */}
        <div style={{ width: 80, height: 4, background: '#1e1e2a', borderRadius: 2 }}>
          <div
            style={{
              width: `${roadmap.overall_progress}%`,
              height: '100%',
              background: '#00ff88',
              borderRadius: 2,
              transition: 'width 0.3s',
            }}
          />
        </div>
        <span style={{ fontSize: 9, color: '#00ff88', fontWeight: 700 }}>
          {roadmap.overall_progress.toFixed(0)}%
        </span>
        <span style={{ fontSize: 10, color: '#505070' }}>▲</span>
      </div>
    );
  }

  // Expanded view
  return (
    <div
      style={{
        background: '#0e0e18',
        borderTop: '1px solid #1e1e2a',
        maxHeight: '40vh',
        overflowY: 'auto',
        fontFamily: 'monospace',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '8px 14px',
          borderBottom: '1px solid #1e1e2a',
        }}
      >
        <span style={{ fontSize: 9, color: '#7070a0', letterSpacing: 2, fontWeight: 700 }}>
          PRODUCTION ROADMAP
        </span>
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 12, color: '#00ff88', fontWeight: 900 }}>
          {roadmap.overall_progress.toFixed(0)}%
        </span>
        <button
          onClick={onToggle}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#505070',
            fontSize: 14,
            cursor: 'pointer',
            padding: 4,
          }}
        >
          ▼
        </button>
      </div>

      {/* Phase cards */}
      <div style={{ padding: '8px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {roadmap.phases.map((phase) => {
          const col = statusColors[phase.status] ?? '#303048';
          const isActive = phase.status === 'active';

          return (
            <div
              key={phase.phase}
              style={{
                background: isActive ? '#141420' : '#0c0c16',
                borderRadius: 10,
                border: `1px solid ${isActive ? `${col}44` : '#1e1e2a'}`,
                padding: 12,
                opacity: phase.status === 'locked' ? 0.5 : 1,
              }}
            >
              {/* Phase header */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: 14, color: col }}>
                  {phaseIcons[phase.phase] ?? '○'}
                </span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, fontWeight: 800, color: '#d0d0e0' }}>
                    {phase.title}
                  </div>
                  <div style={{ fontSize: 9, color: '#505070' }}>{phase.subtitle}</div>
                </div>
                <div
                  style={{
                    fontSize: 10,
                    color: col,
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: 4,
                    background: `${col}22`,
                    textTransform: 'uppercase',
                  }}
                >
                  {phase.status}
                </div>
                <span style={{ fontSize: 11, color: col, fontWeight: 700 }}>
                  {phase.completion_percentage.toFixed(0)}%
                </span>
              </div>

              {/* Checklist */}
              {(isActive || phase.status === 'completed') && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {phase.checklist.map((item) => (
                    <div
                      key={item.item_id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        padding: '4px 8px',
                        borderRadius: 4,
                        background: item.completed ? '#00ff8811' : '#141420',
                      }}
                    >
                      {/* Checkbox */}
                      <button
                        onClick={() => !item.completed && onCompleteItem(item.item_id)}
                        disabled={item.completed || item.is_automated}
                        style={{
                          width: 16,
                          height: 16,
                          borderRadius: 4,
                          background: item.completed ? '#00ff8833' : 'transparent',
                          border: `1px solid ${item.completed ? '#00ff88' : '#353550'}`,
                          color: item.completed ? '#00ff88' : '#353550',
                          fontSize: 10,
                          cursor: item.completed || item.is_automated ? 'default' : 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          padding: 0,
                        }}
                      >
                        {item.completed ? '✓' : ''}
                      </button>
                      <div style={{ flex: 1 }}>
                        <div
                          style={{
                            fontSize: 10,
                            color: item.completed ? '#00ff8888' : '#c0c0d8',
                            fontWeight: item.completed ? 400 : 600,
                            textDecoration: item.completed ? 'line-through' : 'none',
                          }}
                        >
                          {item.label}
                        </div>
                      </div>
                      {item.is_automated && !item.completed && (
                        <span style={{ fontSize: 8, color: '#505070' }}>AUTO</span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Advance button */}
              {isActive && phase.completion_percentage >= 100 && (
                <button
                  onClick={onAdvancePhase}
                  style={{
                    marginTop: 8,
                    width: '100%',
                    padding: '8px 0',
                    background: '#00ff8822',
                    border: '1px solid #00ff8866',
                    borderRadius: 6,
                    color: '#00ff88',
                    fontWeight: 800,
                    fontSize: 11,
                    cursor: 'pointer',
                    fontFamily: 'monospace',
                    letterSpacing: 2,
                  }}
                >
                  ▶ ADVANCE TO NEXT PHASE
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
