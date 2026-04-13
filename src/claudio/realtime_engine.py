"""
Claudio Realtime Engine (Opportunist Refactor).

Replaces the scattered realtime_*.py boilerplates into a single, cohesive
RealtimeEngine state machine that encapsulates PyAudio streams, routing, 
and DSP profiles.
"""
import logging
import pyaudio
import numpy as np

from .audio_ingestion import preprocess_audio_data
from .audio_analysis import calculate_snr, calculate_thd, analyze_transient_response, calculate_phase_coherence, calculate_inter_channel_correlation

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
        self._analysis_results = {}


    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Unified audio pumping callback routing based on profile."""
        out_data = in_data # Default passthrough

        if self.profile == AudioProfile.TEST_TONE:
            t = np.arange(frame_count) / self.rate
            tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
            # converting float32 to int16 for playback if necessary, assuming float32
            out_data = tone.astype(np.float32).tobytes()
            
        elif self.profile == AudioProfile.CAPTURE:
            # Pre-process captured data for consistency and quality
            processed_in_data = preprocess_audio_data(np.frombuffer(in_data, dtype=np.float32))
            self._captured_data.append(processed_in_data.tobytes())

            if self.profile == AudioProfile.BENCHMARK:
                # Perform real-time analysis for benchmarking
                # Ensure processed_in_data is a 2D array for stereo analysis if needed
                # For now, assuming mono or handling appropriately within analysis functions
                mono_data = processed_in_data.flatten() # Simple conversion for initial analysis

                # Clarity & Detail Metrics
                self._analysis_results['snr'] = calculate_snr(mono_data)
                self._analysis_results['thd'] = calculate_thd(mono_data)
                self._analysis_results['transient_response'] = analyze_transient_response(mono_data)

                # Spatial Accuracy & Immersion Metrics (requires stereo input)
                if processed_in_data.ndim == 2 and processed_in_data.shape[1] == 2:
                    left_channel = processed_in_data[:, 0]
                    right_channel = processed_in_data[:, 1]
                    self._analysis_results['phase_coherence'] = calculate_phase_coherence(left_channel, right_channel)
                    self._analysis_results['inter_channel_correlation'] = calculate_inter_channel_correlation(left_channel, right_channel)
                else:
                    self._analysis_results['phase_coherence'] = 'N/A (mono)'
                    self._analysis_results['inter_channel_correlation'] = 'N/A (mono)'

                logger.debug(f"Benchmark Analysis: {self._analysis_results}")


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
