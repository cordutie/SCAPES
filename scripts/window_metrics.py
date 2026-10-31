"""Estimators of the window study: coherence, level at the joins, join-masked MSS, the two-way window proposal (design D10, D25).

Pure torch / numpy; the only SCAPES import is the window family. Signals are [ch, N] (torch) unless said otherwise.
A join is the crossfade interval of two neighbouring atoms: [k*H + C - H - X, k*H + C - H) for k = 1..K-1.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SCAPES.data.geometry import WINDOWS, atom_count, crossfade_halves  # noqa: E402

N_BINS = 16
CENTRE_BINS = range(4, 12)        # u in [1/4, 3/4)
EDGE_BINS = (0, 1, 14, 15)
JOIN_FFT_SIZES = (1024, 4096)     # at 48 kHz, hop = size / 4
TOL_DB = 0.5
MSS_EPS = 1e-7                    # as eval_metrics.LOG_EPS


# ─── segmentation and overlap-add ───

# [ch, N] -> [K, ch, C]: atom k is x[:, k*H : k*H + C], K = atom_count(N, C, H)
def segment(x: torch.Tensor, C: int, H: int) -> torch.Tensor:
    K = atom_count(x.shape[-1], C, H)
    if K == 0:
        return x.new_zeros(0, x.shape[0], C)
    return torch.stack([x[:, k * H: k * H + C] for k in range(K)])


# Window of one atom [C] float64: zeros(C - H - X), rise, ones(H - X), fall
def window(C: int, H: int, X: int, name: str) -> torch.Tensor:
    rise, fall = (t.double() for t in crossfade_halves(X, name))
    return torch.cat([torch.zeros(C - H - X, dtype=torch.float64), rise, torch.ones(H - X, dtype=torch.float64), fall])


# [K, ch, C] -> [ch, (K-1)*H + C]: atom k, times w, added at k*H
def overlap_add(y: torch.Tensor, H: int, w: torch.Tensor) -> torch.Tensor:
    K, ch, C = y.shape
    out = torch.zeros(ch, (K - 1) * H + C, dtype=y.dtype)
    for k in range(K):
        out[:, k * H: k * H + C] += y[k] * w.to(y.dtype)
    return out


# Start sample of the crossfade interval of every join
def join_starts(K: int, C: int, H: int, X: int) -> list:
    return [k * H + C - H - X for k in range(1, K)]


# ─── level at the joins ───

# Power of sig in each of n_bins equal bins of the X samples after every start: summed over channels, joins and samples
def bin_powers(sig: torch.Tensor, starts: list, X: int, n_bins: int = N_BINS) -> torch.Tensor:
    sig = sig.double()
    per_sample = torch.zeros(X, dtype=torch.float64)
    for s in starts:
        per_sample += (sig[:, s: s + X] ** 2).sum(dim=0)
    bins = (torch.arange(X) * n_bins) // X
    out = torch.zeros(n_bins, dtype=torch.float64)
    out.index_add_(0, bins, per_sample)
    return out


# Codec gain: energy of the decoded atoms over the energy of the source atoms (whole, unwindowed)
def atom_gain(y: torch.Tensor, xs: torch.Tensor) -> float:
    return float((y.double() ** 2).sum() / (xs.double() ** 2).sum())


# Level in dB of the output against the source scaled by the codec gain, per bin
def level_db(P_y, P_x, G: float) -> torch.Tensor:
    return 10 * torch.log10(torch.as_tensor(P_y, dtype=torch.float64) / (G * torch.as_tensor(P_x, dtype=torch.float64)))


# delta: worst bin deviation, centre: mean level of the central bins (signed), edge: mean level of the outer bins
def level_summary(level: torch.Tensor) -> dict:
    level = torch.as_tensor(level, dtype=torch.float64)
    return {"delta": float(level.abs().max()),
            "centre": float(level[list(CENTRE_BINS)].mean()),
            "edge": float(level[list(EDGE_BINS)].mean())}


# (sum ab, sum a^2, sum b^2) over all channels and samples; rho = sab / sqrt(saa * sbb)
def pooled_coherence(a: torch.Tensor, b: torch.Tensor) -> tuple:
    a, b = a.double(), b.double()
    return float((a * b).sum()), float((a * a).sum()), float((b * b).sum())


# ─── join-masked MSS ───

# STFT frames (hop n_fft/4, no centre padding) whose centre lies in an interval and whose whole extent lies in region
def join_frame_mask(n_samples_48k: int, n_fft: int, intervals_48k: list, region: tuple) -> np.ndarray:
    hop = n_fft // 4
    n_frames = 1 + (n_samples_48k - n_fft) // hop if n_samples_48k >= n_fft else 0
    begin = np.arange(n_frames) * hop
    centre = begin + n_fft // 2
    mask = np.zeros(n_frames, dtype=bool)
    for a, b in intervals_48k:
        mask |= (centre >= a) & (centre < b)
    return mask & (begin >= region[0]) & (begin + n_fft <= region[1])


# |STFT| [frames, n_fft/2+1], Hann window, hop n_fft/4, no centre padding (as eval_metrics.stft_magnitude)
def _stft_magnitude(x: np.ndarray, n_fft: int) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    win = torch.hann_window(n_fft, dtype=torch.float64)
    return torch.stft(t, n_fft, hop_length=n_fft // 4, window=win, center=False, return_complex=True).abs().T


# Mean over the FFT sizes of the log-magnitude L1 over the masked frames only (nan if no size has a masked frame)
def join_mss(x48: np.ndarray, y48: np.ndarray, intervals_48k: list, region: tuple, fft_sizes=JOIN_FFT_SIZES) -> float:
    n = min(len(x48), len(y48))
    values = []
    for n_fft in fft_sizes:
        mask = join_frame_mask(n, n_fft, intervals_48k, region)
        if not mask.any():
            continue
        s, s2 = _stft_magnitude(x48[:n], n_fft)[mask], _stft_magnitude(y48[:n], n_fft)[mask]
        values.append((torch.log(s + MSS_EPS) - torch.log(s2 + MSS_EPS)).abs().mean().item())
    return float(np.mean(values)) if values else float("nan")


# ─── proposal ───

# worst, mss, mss_seed_range: dicts keyed by window. Rule of design.md (proposal rule), plus H1.
def propose_window(worst: dict, mss: dict, mss_seed_range: dict, tol_db: float = TOL_DB) -> dict:
    h1 = {"supported": bool(worst["equal_power"] < worst["hann"] - tol_db),
          "margin_db": worst["hann"] - worst["equal_power"]}
    if abs(worst["hann"] - worst["equal_power"]) > tol_db:
        win = min(WINDOWS, key=lambda w: worst[w])
        return {"window": win, "by": "level", "h1": h1}
    if abs(mss["hann"] - mss["equal_power"]) > max(mss_seed_range.values()):
        win = min(WINDOWS, key=lambda w: mss[w])
        return {"window": win, "by": "mss", "h1": h1}
    return {"window": None, "by": "no metric preference", "h1": h1}
