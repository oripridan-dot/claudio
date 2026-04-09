"""
Claudio Realtime Engine (Opportunist Refactor).

Replaces the scattered realtime_*.py boilerplates into a single, cohesive
RealtimeEngine state machine that encapsulates PyAudio streams, routing, 
and DSP profiles.
"""
import logging
import pyaudio
import numpy as np

logger = logging.getLogger("Claudio.RealtimeEngine")

class AudioProfile:
    MIC_PASSTHROUGH = "passthrough"
    HIFI_OUTPUT = "hifi"
    CAPTURE = "capture"
    BENCHMARK = "benchmark"
    TEST_TONE = "test_tone"

class RealtimeEngine:
    def __init__(self, profile: str = AudioProfile.MIC_PASSTHROUGH, rate: int = 48000, chunk: int = 1024):
        self.profile = profile
        self.rate = rate
        self.chunk = chunk
        self.p_audio = pyaudio.PyAudio()
        self.stream = None
        self.active = False
        self._captured_data = []

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Unified audio pumping callback routing based on profile."""
        out_data = in_data # Default passthrough

        if self.profile == AudioProfile.TEST_TONE:
            t = np.arange(frame_count) / self.rate
            tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
            # converting float32 to int16 for playback if necessary, assuming float32
            out_data = tone.astype(np.float32).tobytes()
            
        elif self.profile == AudioProfile.CAPTURE:
            self._captured_data.append(in_data)

        # In a full DSP implementation, hooks for `realtime_intelligence` or `realtime_benchmark`
        # would tap into the in_data array here.
        
        return (out_data, pyaudio.paContinue)

    def start(self):
        """Initializes and starts the audio stream safely."""
        logger.info(f"Starting RealtimeEngine [Profile: {self.profile}]")
        self.stream = self.p_audio.open(
            format=pyaudio.paFloat32,
            channels=1 if self.profile == AudioProfile.MIC_PASSTHROUGH else 2,
            rate=self.rate,
            input=True if self.profile in (AudioProfile.MIC_PASSTHROUGH, AudioProfile.CAPTURE) else False,
            output=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
        self.active = True

    def stop(self):
        """Safely tears down the hardware connections."""
        if not self.active: return
        logger.info("Stopping RealtimeEngine...")
        self.stream.stop_stream()
        self.stream.close()
        self.p_audio.terminate()
        self.active = False
        
    def get_captured_audio(self):
        return b''.join(self._captured_data)
