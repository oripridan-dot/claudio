import numpy as np
import pytest
from claudio.intent.intent_encoder import IntentEncoder, ArticulationMode

def generate_tone(freq, duration_ms, sr=44100, amplitude=0.5):
    t = np.linspace(0, duration_ms / 1000, int(sr * duration_ms / 1000))
    return amplitude * np.sin(2 * np.pi * freq * t)

def test_articulation_staccato():
    sr = 44100
    encoder = IntentEncoder(sample_rate=sr)
    
    # 1. Generate Staccato burst: 10ms burst then sharp decay
    # Prepend zeros to allow onset detection to see a jump
    silence = np.zeros(int(sr * 0.05))
    t_decay = 0.04
    burst = generate_tone(440, 10, sr)
    decay = burst[-1] * np.exp(-100 * np.linspace(0, t_decay, int(sr * t_decay)))
    audio = np.concatenate([silence, burst, decay, np.zeros(int(sr * 0.1))])
    
    frames = encoder.encode_block(audio)
    
    # Check if STACCATO was detected shortly after onset
    articulations = [f.articulation_mode for f in frames if f.articulation_mode == ArticulationMode.STACCATO]
    assert len(articulations) > 0, "Failed to detect Staccato articulation for sharp decay"

def test_articulation_legato():
    sr = 44100
    encoder = IntentEncoder(sample_rate=sr)
    
    # 2. Generate Legato burst: 100ms sustain
    silence = np.zeros(int(sr * 0.05))
    audio = np.concatenate([silence, generate_tone(440, 100, sr), np.zeros(int(sr * 0.1))])
    
    frames = encoder.encode_block(audio)
    
    # Check if LEGATO was detected
    articulations = [f.articulation_mode for f in frames if f.articulation_mode == ArticulationMode.LEGATO]
    assert len(articulations) > 0, "Failed to detect Legato articulation for sustained tone"

def test_articulation_neutral():
    sr = 44100
    encoder = IntentEncoder(sample_rate=sr)
    
    # 3. Silence / Noise should be Neutral
    audio = np.random.normal(0, 0.001, int(sr * 0.2))
    frames = encoder.encode_block(audio)
    
    for f in frames:
        assert f.articulation_mode == ArticulationMode.NEUTRAL
