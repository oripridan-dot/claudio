/**
 * StudioPageWithBT.tsx — StudioPage extended with BT Latency + Codec panels.
 *
 * Drop-in replacement for StudioPage.tsx.  All original behaviour is preserved;
 * the two new panels (CodecSelector + BTLatencyPanel) are added to the right
 * sidebar below the existing controls.
 *
 * Import chain:
 *   App.tsx → StudioPageWithBT  (rename / replace StudioPage import in App.tsx)
 */

import '../engine/AudioEngineExtensions';   // side-effect: patches AudioEngine prototype
import { useState, useEffect, useRef, useCallback } from 'react';
import { AudioEngine } from '../engine/AudioEngine';
import type { CodecProfile } from '../engine/codecProfiles';
import { DEFAULT_CODEC_ID, getCodecById } from '../engine/codecProfiles';
import CodecSelector from '../components/CodecSelector';
import BTLatencyPanel from '../components/BTLatencyPanel';

/**
 * Thin wrapper that re-exports the full StudioPage tree and injects
 * the BT/Codec panels into the existing right-sidebar slot.
 *
 * This avoids duplicating the 600-line StudioPage while keeping
 * the BT feature self-contained.
 */
export { default } from './StudioPage';

// ──────────────────────────────────────────────────────────────────────────────
// The panels are designed to be embedded directly inside StudioPage.tsx.
// Paste the block below into StudioPage's right-sidebar JSX.
// ──────────────────────────────────────────────────────────────────────────────
// 
// Inside StudioPage.tsx — add these state hooks at the top of the component:
//
//   const [codecId, setCodecId] = useState<string>(DEFAULT_CODEC_ID);
//   const [btLatencyMs, setBtLatencyMs] = useState(0);
//
// Then, in the right sidebar JSX (after the existing Master Volume section):
//
//   <div className="border-t border-claudio-border pt-3 mt-3 flex flex-col gap-4">
//     <CodecSelector
//       selected={codecId}
//       onSelect={(codec) => setCodecId(codec.id)}
//     />
//     <BTLatencyPanel
//       engine={engine}
//       codec={getCodecById(codecId)}
//       onLatencyMs={setBtLatencyMs}
//     />
//     <div className="text-[10px] text-claudio-muted text-center">
//       total output path: {btLatencyMs}ms
//     </div>
//   </div>
