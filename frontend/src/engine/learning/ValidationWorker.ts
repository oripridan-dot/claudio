import { BrutalHonestyMatrix, Critique } from './BrutalHonesty';
import { IntentFrame } from '../types';

/**
 * ValidationWorker.ts
 * 
 * Shadow pipeline Web Worker. Computes the Heavy 16D Matrix off the main thread.
 * Dead Code Elimination completely removes this script from the final `npm run build` output.
 */

let matrix: BrutalHonestyMatrix;

self.onmessage = (event: MessageEvent) => {
  if (event.data.type === 'init') {
    matrix = new BrutalHonestyMatrix();
    self.postMessage({ type: 'ready' });
  } else if (event.data.type === 'frame') {
    if (!matrix) return;
    
    const frame = event.data.frame as IntentFrame;
    const critiques: Critique[] = matrix.evaluate(frame);
    
    if (critiques.length > 0) {
      self.postMessage({
        type: 'critiques',
        critiques
      });
    }
  }
};
