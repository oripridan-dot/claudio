import { IntentFrame } from '../types';

/**
 * BrutalHonesty.ts 
 * 
 * The 16D mathematical matrix validating real-time audio against theoretical physical maxims.
 * Enforces Rule 0. Pure, local, and brutally honest heuristic checks.
 * Completely stripped out of the production build.
 */

// Theoretical bounds for healthy play
const BOUNDS = {
  PITCH_DRIFT_BPM_TOLERANCE: 15, // Cents drift allowable
  LATENCY_MAX_TOLERANCE: 12, // ms
  MAX_TIMBRAL_FATIGUE_DELTA: 0.15, // Change in centroid over time indicating fatigue
  RMS_MIN_CONSISTENCY: 0.05
};

export interface Critique {
  passed: boolean;
  message: string;
  metric: string;
  severity: "low" | "medium" | "high";
  delta: number;
}

interface ValidationState {
  baselineCentroid: number;
  pitchHistory: number[];
  frameCounter: number;
}

export class BrutalHonestyMatrix {
  private state: ValidationState = {
    baselineCentroid: 0,
    pitchHistory: [],
    frameCounter: 0,
  };

  public evaluate(frame: IntentFrame): Critique[] {
    const critiques: Critique[] = [];
    this.state.frameCounter++;

    // Track baseline for fatigue checks
    if (this.state.frameCounter === 10) {
      this.state.baselineCentroid = frame.spectralCentroid;
    }

    // 1. Transient / Onset strength heuristics
    if (frame.isOnset) {
      if (frame.onsetStrength < BOUNDS.RMS_MIN_CONSISTENCY) {
        critiques.push({
          passed: false,
          message: "Transient hesitation detected. Attack lacks conviction.",
          metric: "onsetStrength",
          severity: "high",
          delta: frame.onsetStrength - BOUNDS.RMS_MIN_CONSISTENCY
        });
      }
    }

    // 2. Pitch / Confidence heuristics
    if (frame.confidence > 0.8) {
      this.state.pitchHistory.push(frame.f0Hz);
      if (this.state.pitchHistory.length > 30) this.state.pitchHistory.shift();
      
      // Simple drift check over last 30 valid frames (approx 250ms at 120Hz)
      if (this.state.pitchHistory.length === 30) {
        const drift = Math.abs(this.state.pitchHistory[0] - frame.f0Hz);
        if (drift > BOUNDS.PITCH_DRIFT_BPM_TOLERANCE) {
          critiques.push({
            passed: false,
            message: `Pitch drift of ${drift.toFixed(1)}Hz detected. Intonation is slipping.`,
            metric: "pitchDrift",
            severity: "medium",
            delta: drift
          });
        }
      }
    } else if (frame.loudnessNorm > 0.2) {
      critiques.push({
        passed: false,
        message: "Loud output but low pitch confidence. Check breath control or bow pressure.",
        metric: "confidence",
        severity: "medium",
        delta: 0.8 - frame.confidence
      });
    }

    // 3. Timbral Fatigue (Centroid deviation over long phrases)
    if (this.state.frameCounter > 100) {
      const centroidShift = Math.abs((frame.spectralCentroid - this.state.baselineCentroid) / this.state.baselineCentroid);
      if (centroidShift > BOUNDS.MAX_TIMBRAL_FATIGUE_DELTA) {
        critiques.push({
          passed: false,
          message: "Spectral density dropping. Harmonic richness lost—indicating physical fatigue.",
          metric: "timbralFatigue",
          severity: "low",
          delta: centroidShift
        });
      }
    }

    return critiques;
  }
}
