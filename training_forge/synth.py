import torch
import torch.nn as nn
import torch.nn.functional as F

class HarmonicSynth(nn.Module):
    """
    Differentiable Harmonic Oscillator Bank
    Input:
        f0: Tensor (batch, time, 1) - Fundamental frequency in Hz
        amplitudes: Tensor (batch, time, 1) - Overall loudness mask
        harmonic_distribution: Tensor (batch, time, n_harm) - Normed harmonic weights
    Output:
        audio: Tensor (batch, time_samples)
    """
    def __init__(self, sample_rate=48000, n_harmonics=60, frame_rate=250):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.hop_length = int(sample_rate / frame_rate) # e.g. 192 samples at 48kHz

    def forward(self, f0, amplitudes, harmonic_distribution):
        batch, time_frames, _ = f0.shape
        
        # Upsample frame-rate envelopes to block audio-rate
        # We need f0 and harmonic amplitudes at the audio sample rate
        audio_len = time_frames * self.hop_length
        f0_up = F.interpolate(f0.transpose(1, 2), size=audio_len, mode='linear', align_corners=False).transpose(1, 2)
        amp_up = F.interpolate(amplitudes.transpose(1, 2), size=audio_len, mode='linear', align_corners=False).transpose(1, 2)
        harm_up = F.interpolate(harmonic_distribution.transpose(1, 2), size=audio_len, mode='linear', align_corners=False).transpose(1, 2)
        
        # Build anti-aliasing mask -> zeroes out harmonics above Nyquist (24000)
        # Shape: (1, 1, n_harmonics)
        harmonic_multipliers = torch.arange(1, self.n_harmonics + 1, device=f0.device).float().view(1, 1, -1)
        
        f0_freqs = f0_up * harmonic_multipliers # (batch, audio_len, n_harmonics)
        aa_mask = (f0_freqs < (self.sample_rate / 2)).float()
        
        # Phase Accumulator: integral of 2 * pi * f(t) / fs
        # (batch, audio_len, n_harmonics)
        d_phase = 2.0 * torch.pi * f0_freqs / self.sample_rate
        phase = torch.cumsum(d_phase, dim=1)
        
        # Generate sine waves
        sines = torch.sin(phase)
        
        # Scale by amplitudes (batch, audio_len, n_harmonics)
        # Note: harm_up already defines relative amplitudes; amp_up defines global envelope
        signal = sines * harm_up * aa_mask
        
        # Sum harmonics to get mono output (batch, audio_len)
        mono = torch.sum(signal, dim=-1) * amp_up.squeeze(-1)
        
        return mono

class FilteredNoise(nn.Module):
    """
    Differentiable Filtered Noise Generator.
    Models breath and transient unpitched sound.
    """
    def __init__(self, sample_rate=48000, n_noise=65, frame_rate=250):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_noise = n_noise
        self.hop_length = int(sample_rate / frame_rate)
        
    def forward(self, noise_envelope):
        # noise_envelope: (batch, frames, n_noise) -> Frequency band magnitudes
        batch, time_frames, n_noise = noise_envelope.shape
        audio_len = time_frames * self.hop_length
        
        # Generate uniform white noise [-1, 1]
        white_noise = torch.rand(batch, audio_len, device=noise_envelope.device) * 2.0 - 1.0
        
        # This is a simplification logic for DDSP style filtered noise:
        # We usually take the STFT of white noise, multiply by the envelope, and iSTFT back.
        # Alternatively, we design FIR filters. STFT is more differentiable.
        
        # For simplicity of this minimal architectural forge, we will just return scaled white noise
        # in a complete implementation you'd use torch.stft and multiply.
        
        # To maintain isolation constraint without complex dependencies, 
        # we approx it by upsampling the mean envelope scale
        mean_scale = torch.mean(noise_envelope, dim=-1, keepdim=True)
        scale_up = F.interpolate(mean_scale.transpose(1, 2), size=audio_len, mode='linear', align_corners=False).squeeze(1)
        
        return white_noise * scale_up * 0.1 # scaled down to prevent blowouts

class DDSPSynth(nn.Module):
    def __init__(self, sample_rate=48000, frame_rate=250):
        super().__init__()
        self.harmonic = HarmonicSynth(sample_rate, 60, frame_rate)
        self.noise = FilteredNoise(sample_rate, 65, frame_rate)
        
    def forward(self, f0, amplitudes, harmonics, noise):
        # f0: (b, t, 1), amplitudes: (b, t, 1), harmonics: (b, t, 60), noise: (b, t, 65)
        h_audio = self.harmonic(f0, amplitudes, harmonics)
        n_audio = self.noise(noise)
        return h_audio + n_audio
