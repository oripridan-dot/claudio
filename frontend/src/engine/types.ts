export interface IntentFrame {
  timestamp: number;
  f0Hz: number;
  confidence: number;
  loudnessDb: number;
  loudnessNorm: number;
  spectralCentroid: number;
  isOnset: boolean;
  onsetStrength: number;
  rmsEnergy: number;
  mfcc: number[]; // 13 coefficients — timbre fingerprint
}

export interface PeerInfo {
  peer_id: string;
  display_name: string;
  role: string;
  instrument: string;
  packets_sent: number;
  latency_ms: number;
}

export interface CollabMetrics {
  peer_count: number;
  total_packets: number;
  bytes_transmitted: number;
  avg_latency_ms: number;
  uptime_seconds: number;
  bandwidth_kbps: number;
}

export type IntentCallback = (frame: IntentFrame) => void;
export type PeerCallback = (peers: PeerInfo[]) => void;
export type MetricsCallback = (metrics: CollabMetrics) => void;
