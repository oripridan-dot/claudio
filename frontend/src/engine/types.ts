export interface PivotFrame {
  type: 'pivot';
  seq: number;
  timestamp: number;
  f0Hz: number;
  confidence: number;
  loudnessDb: number;
  loudnessNorm: number;
  spectralCentroid: number;
  isOnset: boolean;
  onsetStrength: number;
  rmsEnergy: number;
}

export interface DeltaFrame {
  type: 'delta';
  seq: number;
  ref_seq: number;
  timestamp: number; // reference to the pivot block if async
  melBands: Float32Array;
}

export type NetworkPacket = PivotFrame | DeltaFrame;

export type IntentFrame = PivotFrame & { 
  mfcc?: number[]; // Deprecated compatibility field
  melBands?: Float32Array; // New 64-dim field
  peerId?: string; // Auto-discovery routing
};

export interface PeerInfo {
  peer_id: string;
  display_name: string;
  role: string;
  instrument: string;
  packets_sent: number;
  latency_ms: number;
  rmsEnergy?: number;
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
