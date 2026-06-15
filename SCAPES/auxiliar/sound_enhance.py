import torch
import torchaudio.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal

def _format_audio_plot(ax, title, sample_rate):
    """Helper function to apply standard logarithmic audio ticks to plots."""
    ax.set_xscale('log')
    ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    
    # Filter out ticks that exceed the Nyquist frequency
    nyquist = sample_rate / 2.0
    valid_ticks = [t for t in ticks if t <= nyquist]
    valid_labels = labels[:len(valid_ticks)]
    
    ax.set_xticks(valid_ticks)
    ax.set_xticklabels(valid_labels)
    ax.set_xlim([20, min(20000, nyquist)])
    
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

class ArtifactKiller(torch.nn.Module):
    def __init__(self, sample_rate=44100, highpass_cutoff=None, lowpass_cutoff=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        
        self.filters = [
            (298.0,  3.0),       
            (596.0,  3.0),       
            (894.0,  3.0),       
            (1192.0, 3.0),       
            (1000.0, 0.5)        
        ]

    def forward(self, x):
        if self.highpass_cutoff is not None:
            x = F.highpass_biquad(x, self.sample_rate, self.highpass_cutoff)
        for freq, q in self.filters:
            x = F.bandreject_biquad(x, self.sample_rate, central_freq=freq, Q=q)
        if self.lowpass_cutoff is not None:
            x = F.lowpass_biquad(x, self.sample_rate, self.lowpass_cutoff)
        return x
        
    def plot_frequency_response(self):
        sos = []
        nyquist = self.sample_rate / 2
        
        if self.highpass_cutoff is not None:
            b, a = signal.butter(2, self.highpass_cutoff, btype='high', fs=self.sample_rate)
            sos.append([b[0], b[1], b[2], a[0], a[1], a[2]])
            
        for freq, q in self.filters:
            w0 = freq / nyquist
            b, a = signal.iirnotch(w0, q)
            sos.append([b[0], b[1], b[2], a[0], a[1], a[2]])
            
        if self.lowpass_cutoff is not None:
            b, a = signal.butter(2, self.lowpass_cutoff, btype='low', fs=self.sample_rate)
            sos.append([b[0], b[1], b[2], a[0], a[1], a[2]])
            
        sos = np.array(sos)
        w, h = signal.sosfreqz(sos, worN=8192, fs=self.sample_rate)
        db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(w, db, color='#1f77b4', linewidth=1.5)
        
        for freq, _ in self.filters:
            ax.axvline(freq, color='red', linestyle=':', alpha=0.4)
            
        if self.highpass_cutoff:
            ax.axvline(self.highpass_cutoff, color='green', linestyle='--', alpha=0.6, label='HPF/LPF Cutoff')
        if self.lowpass_cutoff:
            ax.axvline(self.lowpass_cutoff, color='green', linestyle='--', alpha=0.6)
            
        ax.set_ylim([-40, 2])
        if self.highpass_cutoff or self.lowpass_cutoff:
            ax.legend()
            
        _format_audio_plot(ax, 'Cascaded HPF + Notches + LPF - Frequency Response', self.sample_rate)
        plt.tight_layout()
        plt.show()

import torch
import torchaudio.functional as F
import matplotlib.pyplot as plt
import numpy as np

class LushReverb(torch.nn.Module):
    """
    A CPU-expensive, high-quality synthetic convolution reverb.
    Simulates a lush acoustic space by splitting synthetic noise into 
    frequency bands and applying different decay rates (highs decay faster).
    """
    def __init__(self, sample_rate=48000, tail_length_sec=4.0, mix=0.25, predelay_ms=20.0, damping=1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.mix = mix
        
        ir_samples = int(sample_rate * tail_length_sec)
        predelay_samples = int(sample_rate * (predelay_ms / 1000.0))
        
        # 1. Generate Stereo Noise Base
        noise = torch.randn(2, ir_samples)
        
        # 2. Split into Frequency Bands
        # Real rooms absorb high frequencies much faster than low frequencies.
        low_noise = F.lowpass_biquad(noise, sample_rate, 600.0)
        mid_noise = F.highpass_biquad(F.lowpass_biquad(noise, sample_rate, 4000.0), sample_rate, 600.0)
        high_noise = F.highpass_biquad(noise, sample_rate, 4000.0)
        
        # 3. Create Multi-Band Decay Envelopes
        time = torch.linspace(0, 1, ir_samples)
        
        # Lows ring out longest, highs die out fastest (scaled by damping)
        env_low = torch.exp(-time * 4.0) 
        env_mid = torch.exp(-time * 6.0)
        env_high = torch.exp(-time * (6.0 + (damping * 8.0)))
        
        # 4. Apply Envelopes and Recombine
        ir = (low_noise * env_low) + (mid_noise * env_mid) + (high_noise * env_high)
        
        # 5. Apply Pre-delay (shift the IR forward and pad the start with silence)
        ir = torch.nn.functional.pad(ir, (predelay_samples, 0))[..., :ir_samples]
        
        # 6. FIX: Peak Normalization (L-infinity norm)
        # We divide by the absolute maximum peak so the IR scales exactly to [-1.0, 1.0]
        ir = ir / (torch.max(torch.abs(ir)) + 1e-8)
        
        # Scale down internally to a sensible nominal level so 'mix' behaves predictably
        ir = ir * 0.5 
        
        self.register_buffer('ir', ir)

    def forward(self, x):
        # Heavy FFT Convolution
        reverb_sig = F.fftconvolve(x, self.ir, mode="full")
        # Trim the tail to match input length
        reverb_sig = reverb_sig[..., :x.shape[-1]]
        
        # Wet/Dry Mix
        return x + (reverb_sig * self.mix)
        
    def plot_frequency_response(self):
        """Plots the spectrum of the newly synthesized lush IR."""
        ir_np = self.ir.cpu().numpy()
        n_fft = ir_np.shape[1]
        
        freqs = np.fft.rfftfreq(n_fft, d=1/self.sample_rate)
        mag_L = np.abs(np.fft.rfft(ir_np[0]))
        mag_R = np.abs(np.fft.rfft(ir_np[1]))
        
        mag_avg = (mag_L + mag_R) / 2.0
        db = 20 * np.log10(mag_avg + 1e-10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, db, color='teal', alpha=0.7, linewidth=1)
        
        max_db = np.max(db)
        ax.set_ylim([-80, max_db + 5]) 
        
        # Assuming you still have your _format_audio_plot helper available in the scope
        # _format_audio_plot(ax, 'Lush Reverb - Multi-Band Synthetic Tail Spectrum', self.sample_rate)
        
        ax.set_xscale('log')
        ax.set_title('Lush Reverb - Multi-Band Synthetic Tail Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

class StereoWidenerEQ(torch.nn.Module):
    def __init__(self, sample_rate=44100, base_freq=50.0, gain_db=4.0, q_factor=1.5, num_bands=9):
        super().__init__()
        self.sample_rate = sample_rate
        self.gain_db = gain_db
        self.q_factor = q_factor
        self.freqs = [base_freq * (2 ** i) for i in range(num_bands)]
        
    def forward(self, x):
        if x.shape[-2] != 2:
            raise ValueError("StereoWidenerEQ requires a 2-channel (stereo) input.")
            
        left = x[..., 0:1, :]
        right = x[..., 1:2, :]
        
        for i, freq in enumerate(self.freqs):
            if freq >= self.sample_rate / 2.0:
                break
            if i % 2 == 0:
                l_gain, r_gain = self.gain_db, -self.gain_db
            else:
                l_gain, r_gain = -self.gain_db, self.gain_db
                
            left = F.equalizer_biquad(left, self.sample_rate, center_freq=freq, gain=l_gain, Q=self.q_factor)
            right = F.equalizer_biquad(right, self.sample_rate, center_freq=freq, gain=r_gain, Q=self.q_factor)
            
        return torch.cat([left, right], dim=-2)

    def plot_frequency_response(self):
        n_fft = 16384  
        impulse = torch.zeros(1, 2, n_fft)
        impulse[0, 0, 0] = 1.0
        impulse[0, 1, 0] = 1.0
        
        with torch.no_grad():
            out = self.forward(impulse).squeeze(0).numpy()
            
        freqs = np.fft.rfftfreq(n_fft, d=1/self.sample_rate)
        mag_L = 20 * np.log10(np.abs(np.fft.rfft(out[0])) + 1e-10)
        mag_R = 20 * np.log10(np.abs(np.fft.rfft(out[1])) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, mag_L, label='Left Channel', color='#1f77b4', alpha=0.8, linewidth=2)
        ax.plot(freqs, mag_R, label='Right Channel', color='#ff7f0e', alpha=0.8, linewidth=2, linestyle='--')
        
        ax.set_ylim([-self.gain_db - 2, self.gain_db + 2])
        for freq in self.freqs:
            if freq < self.sample_rate / 2.0:
                ax.axvline(freq, color='gray', linestyle=':', alpha=0.3)
                
        ax.legend()
        _format_audio_plot(ax, 'Stereo Widener EQ - Alternating Peaks/Dips', self.sample_rate)
        plt.tight_layout()
        plt.show()

class SinusoidSaturator(torch.nn.Module):
    """
    Applies a smooth sinusoidal soft-clip, modeling an analog console or 
    Ableton's 'Analog Clip' saturator curve.
    """
    def __init__(self, drive=1.0):
        super().__init__()
        self.drive = drive
        
    def forward(self, x):
        # Multiply by drive
        driven = x * self.drive
        # Clamp to [-1.0, 1.0] so the sine wave doesn't "fold" inward on extreme peaks
        clamped = torch.clamp(driven, -1.0, 1.0)
        # Apply the sine soft-clip: f(x) = sin(x * pi/2)
        return torch.sin(clamped * (math.pi / 2.0))

class StereoEnergyBalancer(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        if x.shape[-2] != 2:
            return x 
            
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        left_rms = rms[..., 0:1, :]
        right_rms = rms[..., 1:2, :]
        
        target_rms = (left_rms + right_rms) / 2.0
        
        gain_l = target_rms / left_rms
        gain_r = target_rms / right_rms
        
        gains = torch.cat([gain_l, gain_r], dim=-2)
        return x * gains

import torch

class AudioMasteringChain(torch.nn.Module):
    def __init__(
        self, 
        sample_rate=48000, 
        global_mix=0.5,  # 1.0 = 100% processed, 0.0 = completely dry
        chain_order=['reverb', 'artifact_killer', 'stereo_widener', 'balancer'], 
        # Individual Effect Parameters
        drive=1.0,
        ak_hp=100.0, 
        ak_lp=10000.0,
        widener_gain=3.0,
        rev_tail=0.1, 
        rev_mix=0.25, 
        rev_predelay=10.0
    ):
        super().__init__()
        
        self.global_mix = global_mix
        
        # Default routing if none is provided
        if chain_order is None:
            self.chain_order = [
                'artifact_killer', 'stereo_widener', 'saturator', 'reverb', 'balancer'
            ]
        else:
            self.chain_order = chain_order

        # nn.ModuleDict ensures PyTorch properly tracks all sub-modules and their buffers
        self.effects = torch.nn.ModuleDict({
            'artifact_killer': ArtifactKiller(
                sample_rate=sample_rate, highpass_cutoff=ak_hp, lowpass_cutoff=ak_lp
            ),
            'stereo_widener': StereoWidenerEQ(
                sample_rate=sample_rate, gain_db=widener_gain
            ),
            'saturator': SinusoidSaturator(
                drive=drive
            ),
            'reverb': LushReverb(
                sample_rate=sample_rate, tail_length_sec=rev_tail, mix=rev_mix, predelay_ms=rev_predelay
            ),
            'balancer': StereoEnergyBalancer()
        })
        
    def forward(self, x):
        # 1. Save a copy of the dry signal for the global mix later
        dry_x = x.clone()
        
        # 2. Route the audio through only the specified effects in the exact order requested
        processed_x = x
        for effect_name in self.chain_order:
            if effect_name in self.effects:
                processed_x = self.effects[effect_name](processed_x)
            else:
                print(f"Warning: Effect '{effect_name}' not found. Skipping.")
                
        # 3. Apply Global Wet/Dry Mix (Linear Interpolation)
        mixed_x = (dry_x * (1.0 - self.global_mix)) + (processed_x * self.global_mix)
        
        # 4. Final safety normalization (Peak Limit)
        mixed_x = mixed_x / torch.max(torch.abs(mixed_x) + 1e-8)
        
        return mixed_x