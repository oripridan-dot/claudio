import { useState, useEffect } from 'react';

interface MidiDevice { name: string; manufacturer: string; }
interface NoteEvent { note: number; velocity: number; channel: number; }

const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const noteName = (n: number) => `${NOTE_NAMES[n % 12]}${Math.floor(n / 12) - 1}`;
const isBlack = (n: number) => [1, 3, 6, 8, 10].includes(n % 12);

export default function MidiPanel() {
  const [supported, setSupported] = useState<boolean | null>(null);
  const [devices, setDevices] = useState<MidiDevice[]>([]);
  const [lastNote, setLastNote] = useState<NoteEvent | null>(null);
  const [activeNotes, setActiveNotes] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (!navigator.requestMIDIAccess) {
      setSupported(false);
      return;
    }
    setSupported(true);

    navigator.requestMIDIAccess({ sysex: false }).then(access => {
      const sync = () => {
        const devs: MidiDevice[] = [];
        access.inputs.forEach(input => {
          devs.push({
            name: input.name ?? 'MIDI Input',
            manufacturer: input.manufacturer ?? '',
          });
          input.onmidimessage = (e: MIDIMessageEvent) => {
            const [status, note, vel] = e.data;
            const type = status & 0xf0;
            const ch = status & 0x0f;
            if (type === 0x90 && vel > 0) {
              setLastNote({ note, velocity: vel, channel: ch });
              setActiveNotes(prev => new Set([...prev, note]));
            } else if (type === 0x80 || (type === 0x90 && vel === 0)) {
              setActiveNotes(prev => { const s = new Set(prev); s.delete(note); return s; });
            }
          };
        });
        setDevices(devs);
      };
      sync();
      access.onstatechange = sync;
    }).catch(() => setSupported(false));
  }, []);

  const ACCENT = '#0088ff';
  // 37 keys starting from C3 (MIDI 48) to C6 (MIDI 84)
  const allKeys = Array.from({ length: 37 }, (_, i) => i + 48);
  const whiteKeys = allKeys.filter(n => !isBlack(n));

  return (
    <div style={{
      background: '#1a1a24',
      borderRadius: 8,
      padding: 12,
      border: '1px solid #2a2a3a',
    }}>
      <div style={{ fontSize: 10, color: '#606080', marginBottom: 8, letterSpacing: 2 }}>MIDI</div>

      {supported === false && (
        <div style={{ fontSize: 10, color: '#505070' }}>
          Web MIDI not available — use keyboard (Z–M for C4–C5)
        </div>
      )}

      {supported === true && devices.length === 0 && (
        <div style={{ fontSize: 10, color: '#505070' }}>
          No MIDI device · keyboard: Z S X D C V G B H N J M ,
        </div>
      )}

      {devices.map((d, i) => (
        <div key={i} style={{ fontSize: 10, color: ACCENT, marginBottom: 4 }}>
          ● {d.name}{d.manufacturer ? ` · ${d.manufacturer}` : ''}
        </div>
      ))}

      {lastNote && (
        <div style={{ fontSize: 10, color: ACCENT, marginBottom: 8 }}>
          Note: {noteName(lastNote.note)} · vel {lastNote.velocity} · ch {lastNote.channel + 1}
        </div>
      )}

      {/* Mini keyboard */}
      <div style={{ position: 'relative', height: 36, display: 'flex', gap: 1, marginTop: 8 }}>
        {whiteKeys.map((k, i) => {
          const active = activeNotes.has(k);
          return (
            <div
              key={k}
              title={noteName(k)}
              style={{
                flex: 1,
                height: 36,
                background: active ? ACCENT : '#d8d8e0',
                border: '1px solid #444',
                borderRadius: '0 0 3px 3px',
                transition: 'background 0.04s',
                position: 'relative',
              }}
            >
              {/* Black key overlay — simplified */}
              {i < whiteKeys.length - 1 && isBlack(whiteKeys[i] + 1) && (
                <div style={{
                  position: 'absolute',
                  top: 0,
                  right: -4,
                  width: 7,
                  height: 22,
                  background: activeNotes.has(whiteKeys[i] + 1) ? ACCENT : '#1a1a2a',
                  borderRadius: '0 0 2px 2px',
                  zIndex: 2,
                  border: `1px solid ${activeNotes.has(whiteKeys[i] + 1) ? ACCENT : '#333'}`,
                }} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
