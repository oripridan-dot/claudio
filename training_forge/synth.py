import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import scipy.io.wavfile as wav


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
        self.hop_length = int(sample_rate / frame_rate)  # e.g. 192 samples at 48kHz

    def forward(self, f0, amplitudes, harmonic_distribution):
        batch, time_frames, _ = f0.shape

        # Upsample frame-rate envelopes to audio-rate
        audio_len = time_frames * self.hop_length
        # linear mode — bicubic requires 4D spatial tensors, our tensors are 3D
        # linear at 192x upsample is perceptually accurate for harmonic synthesis
        f0_up = F.interpolate(f0.transpose(1, 2), size=audio_len, mode="linear", align_corners=False).transpose(1, 2)
        amp_up = F.interpolate(
            amplitudes.transpose(1, 2), size=audio_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        harm_up = F.interpolate(
            harmonic_distribution.transpose(1, 2), size=audio_len, mode="linear", align_corners=False
        ).transpose(1, 2)

        # Build anti-aliasing mask -> zeroes out harmonics above Nyquist (24000)
        # Shape: (1, 1, n_harmonics)
        harmonic_multipliers = torch.arange(1, self.n_harmonics + 1, device=f0.device).float().view(1, 1, -1)

        f0_freqs = f0_up * harmonic_multipliers  # (batch, audio_len, n_harmonics)
        # 2% safety margin below Nyquist to suppress aliasing noise spikes
        aa_mask = (f0_freqs < (self.sample_rate * 0.49)).float()

        # Phase Accumulator: integral of 2 * pi * f(t) / fs
        # (batch, audio_len, n_harmonics)
        d_phase = 2.0 * torch.pi * f0_freqs / self.sample_rate
        phase = torch.cumsum(d_phase, dim=1)

        # Generate sine waves
        sines = torch.sin(phase)

        # Zero out harmonics if f0 is 0 (unvoiced) to prevent DC offset rumble
        unvoiced_mask = (f0_up > 0).float()

        # Scale by amplitudes (batch, audio_len, n_harmonics)
        # Note: harm_up already defines relative amplitudes; amp_up defines global envelope
        signal = sines * harm_up * aa_mask * unvoiced_mask

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

        # STFT overlap-add construction for differentiable noise filtering
        # Since our hop size is 192, we need an n_fft >= 192 to prevent dropouts.
        # We will pad the 65-bin envelope to 129 bins (which represents n_fft=256),
        # placing silence in the uppermost frequencies (12kHz - 24kHz), which is fine for noise.
        n_fft = 256
        target_bins = (n_fft // 2) + 1  # 129

        if n_noise < target_bins:
            env_pad = F.pad(noise_envelope, (0, target_bins - n_noise))
        else:
            env_pad = noise_envelope[..., :target_bins]

        # Generate complex spectrum with random phase for the noise
        phases = torch.rand(*env_pad.shape, device=env_pad.device) * 2 * torch.pi
        complex_noise = env_pad * torch.exp(1j * phases)

        # Inverse real-FFT to get noise frames (batch, frames, 256)
        noise_frames = torch.fft.irfft(complex_noise, n=n_fft, dim=-1)

        # Apply Hann window to eliminate transient clicking during overlap-add
        window = torch.hann_window(n_fft, device=noise_frames.device).view(1, 1, -1)
        noise_frames = noise_frames * window

        # Overlap-add the frames using nn.Fold
        # Transpose to (batch, 256, frames)
        noise_frames = noise_frames.transpose(1, 2)

        expected_len = time_frames * self.hop_length + n_fft - self.hop_length
        fold = nn.Fold(output_size=(1, expected_len), kernel_size=(1, n_fft), stride=(1, self.hop_length))

        noise_out = fold(noise_frames).squeeze(1).squeeze(1)

        # Trim back to exactly audio_len
        return noise_out[:, :audio_len] * 0.1


class TrainableReverb(nn.Module):
    """
    Differentiable Reverb with a learnable Impulse Response (IR).
    Uses FFT convolution for efficient training.
    """

    def __init__(self, sample_rate=48000, reverb_length=48000, init_ir_path=None):
        super().__init__()
        if init_ir_path and os.path.exists(init_ir_path):
            print(f"Loading acoustic seed IR: {init_ir_path}")
            sr, ir_audio = wav.read(init_ir_path)
            if ir_audio.dtype == np.int16:
                ir_audio = ir_audio.astype(np.float32) / 32768.0
            if len(ir_audio) > reverb_length:
                ir_audio = ir_audio[:reverb_length]
            elif len(ir_audio) < reverb_length:
                ir_audio = np.pad(ir_audio, (0, reverb_length - len(ir_audio)))
            # Multiply by a small scaling factor if needed, but algorithmic IRs are usually pre-normalized.
            # We scale down slightly so it doesn't overwhelm the initial training phase
            self.ir = nn.Parameter(torch.from_numpy(ir_audio).float() * 0.1)
        else:
            # Initialize IR very quiet to prevent initial explosion
            self.ir = nn.Parameter(torch.randn(reverb_length) * 1e-4)

        # Exponential decay envelope prevents IR from blowing up and forces it to act like a room
        t = torch.linspace(0, 1, reverb_length)
        # Drops to ~ -60dB at 1.0 seconds
        decay = torch.exp(-6.9 * t)
        self.register_buffer("decay_envelope", decay)

    def forward(self, audio):
        # audio: (batch, a_len)
        batch, a_len = audio.shape
        # Apply envelope to IR
        ir_applied = self.ir * self.decay_envelope
        ir_len = ir_applied.shape[0]

        # Power of 2 padding for fast FFT
        n_fft = a_len + ir_len - 1
        n_fft = 2 ** int(torch.ceil(torch.log2(torch.tensor(n_fft, dtype=torch.float32))))

        # Convolution via FFT
        A = torch.fft.rfft(audio, n=n_fft)
        B = torch.fft.rfft(ir_applied, n=n_fft).unsqueeze(0)  # broadcast across batch

        # Inverse FFT
        convolved = torch.fft.irfft(A * B, n=n_fft)

        # Trim to original length for cycle matching against ground truth
        return convolved[:, :a_len]


class DDSPSynth(nn.Module):
    def __init__(self, sample_rate=48000, frame_rate=250, init_ir_path=None):
        super().__init__()
        self.harmonic = HarmonicSynth(sample_rate, 60, frame_rate)
        self.noise = FilteredNoise(sample_rate, 65, frame_rate)
        self.reverb = TrainableReverb(sample_rate, sample_rate, init_ir_path=init_ir_path)  # 1 sec IR
        self.hop_length = int(sample_rate / frame_rate)

    def forward(self, f0, amplitudes, harmonics, noise, reverb_mix=None, f0_residual=None, voiced_mask=None):
        # f0: (b, t, 1), amplitudes: (b, t, 1), harmonics: (b, t, 60), noise: (b, t, 65)
        # Apply pitch residual if provided (pitch detune correction)
        if f0_residual is not None:
            f0 = f0 * (1.0 + f0_residual)

        # reverb_mix: (b, t, 1) -> proportion of wet signal
        h_audio = self.harmonic(f0, amplitudes, harmonics)

        # Apply V/UV gating to eliminate harmonic DC rumble during consonants
        if voiced_mask is not None:
            audio_len = f0.shape[1] * self.hop_length
            v_up = F.interpolate(
                voiced_mask.transpose(1, 2), size=audio_len, mode="linear", align_corners=False
            ).transpose(1, 2)
            h_audio = h_audio * v_up.squeeze(-1)

        n_audio = self.noise(noise)
        dry_audio = h_audio + n_audio

        if reverb_mix is None:
            # Default to 0.1 if not provided (compat fallback)
            reverb_mix = torch.full((dry_audio.shape[0], f0.shape[1], 1), 0.1, device=dry_audio.device)

        # Upsample reverb_mix to audio rate
        batch, time_frames, _ = reverb_mix.shape
        audio_len = time_frames * self.hop_length
        mix_up = F.interpolate(
            reverb_mix.transpose(1, 2), size=audio_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        mix_up = mix_up.squeeze(-1)  # (batch, audio_len)

        wet_audio = self.reverb(dry_audio)

        # Mix dry and wet based on reverb_mix
        output = (1.0 - mix_up) * dry_audio + mix_up * wet_audio
        return output
