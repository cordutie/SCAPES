"""Metrics of the teacher-forced evaluation: big-window MSS, KAD with bootstrap, and the signal helpers they share.

No SCAPES or codec import here, so the known-answer tests run in a second (tests_eval/test_eval_metrics.py).
All audio is mono numpy float arrays at 48 kHz unless said otherwise.
"""
import numpy as np
import torch

SR = 48000
MSS_FFT_SIZES = (32768, 65536, 131072)   # 0.68 s, 1.37 s, 2.73 s at 48 kHz
EXCLUDE_SECONDS = 0.4                    # head excluded from every metric (the atoms path is silent for its first ~0.2 s)
LOG_EPS = 1e-7


# ─── signal helpers ───

# [C, N] -> [N] mean of the channels
def to_mono(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else x.mean(axis=0)


# Resample [C, N] (torch or numpy) from sr to 48 kHz with torchaudio's sinc resampler
def resample_to_48k(x, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sr == SR:
        return x
    import torchaudio
    return torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()


# Lag (samples) of y relative to x that maximises their cross-correlation, searched in [-max_lag, max_lag].
# Positive lag = y is x delayed.
def estimate_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 4800) -> int:
    n = min(len(x), len(y))
    x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    cross = np.fft.irfft(np.conj(np.fft.rfft(x, size)) * np.fft.rfft(y, size), size)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    values = np.concatenate([cross[:max_lag + 1], cross[size - max_lag:]])
    return int(lags[int(np.argmax(values))])


# White Gaussian noise with the same RMS as `x` (n samples, default len(x))
def white_noise_like(x: np.ndarray, seed: int, n: int = None) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2)))
    return np.random.RandomState(seed).randn(len(x) if n is None else n) * rms


# ─── MSS ───

# |STFT| [frames, n_fft/2+1], Hann window, hop n_fft/4, no centre padding
def stft_magnitude(x: np.ndarray, n_fft: int) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    window = torch.hann_window(n_fft, dtype=torch.float64)
    return torch.stft(t, n_fft, hop_length=n_fft // 4, window=window, center=False, return_complex=True).abs().T


# Per FFT size: log-magnitude L1 (natural log) and spectral convergence of y against the reference x
def mss_scales(x: np.ndarray, y: np.ndarray, fft_sizes=MSS_FFT_SIZES, eps: float = LOG_EPS) -> dict:
    n = min(len(x), len(y))
    out = {}
    for n_fft in fft_sizes:
        if n < n_fft:
            raise ValueError(f"signal of {n} samples is shorter than the FFT size {n_fft}")
        s, s2 = stft_magnitude(x[:n], n_fft), stft_magnitude(y[:n], n_fft)
        l1 = (torch.log(s + eps) - torch.log(s2 + eps)).abs().mean().item()
        sc = (torch.linalg.norm(s - s2) / torch.linalg.norm(s)).item()
        out[n_fft] = {"log_l1": l1, "spec_conv": sc}
    return out


# Adds the mean over scales
def mss(x: np.ndarray, y: np.ndarray, fft_sizes=MSS_FFT_SIZES, eps: float = LOG_EPS) -> dict:
    scales = mss_scales(x, y, fft_sizes, eps)
    return {"scales": scales,
            "mean_log_l1": float(np.mean([v["log_l1"] for v in scales.values()])),
            "mean_spec_conv": float(np.mean([v["spec_conv"] for v in scales.values()]))}


# ─── KAD ───

# Median of the pairwise Euclidean distances between the rows of x (i < j): sigma of the paper's convention
def median_pairwise_distance(x: np.ndarray) -> float:
    d = torch.cdist(torch.from_numpy(x), torch.from_numpy(x))
    iu = torch.triu_indices(len(x), len(x), offset=1)
    return float(d[iu[0], iu[1]].median())


# exp(-||a-b||^2 / (2 sigma^2)) for all pairs of rows
def rbf_gram(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    d2 = torch.cdist(torch.from_numpy(a), torch.from_numpy(b)) ** 2
    return torch.exp(-d2 / (2.0 * sigma ** 2)).numpy()


# Unbiased MMD^2 from the Gram matrices kxx [n, n], kyy [m, m], kxy [n, m] (diagonals of kxx and kyy are left out)
def mmd2_unbiased(kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray) -> float:
    n, m = kxx.shape[0], kyy.shape[0]
    within_x = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    within_y = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(within_x + within_y - 2.0 * kxy.mean())


KAD_SCALE = 100.0


# KAD = 100 * unbiased MMD^2, Gaussian RBF kernel, sigma = median pairwise distance of the reference set x
def kad(x: np.ndarray, y: np.ndarray, sigma: float = None) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if sigma is None:
        sigma = median_pairwise_distance(x)
    return KAD_SCALE * mmd2_unbiased(rbf_gram(x, x, sigma), rbf_gram(y, y, sigma), rbf_gram(x, y, sigma))


# KAD of several evaluation sets against one reference, with a clip bootstrap.
# ref [n_items, D]; evals {name: [n_items, D]}; clip_of [n_items] = clip index of every item (clip level: arange).
# sigma is the median pairwise distance of the full reference and stays fixed in the resamples.
# Each resample draws clips with replacement (same draw for every set, so differences between sets are paired) and takes
# all the items of the drawn clips. Duplicated clips count as distinct items, which raises the within-set terms: the
# bootstrap mean sits above the point estimate, so the spread is read as a spread, not as a bias-corrected value.
# Returns {name: {kad, boot_mean, boot_std, ci95: [lo, hi]}}, {"sigma"}, and diffs vs `baseline` (paired bootstrap).
def kad_with_bootstrap(ref, evals: dict, clip_of, n_boot: int = 200, seed: int = 0, baseline: str = None) -> dict:
    ref = np.asarray(ref, dtype=np.float64)
    clip_of = np.asarray(clip_of)
    clips = np.unique(clip_of)
    sigma = median_pairwise_distance(ref)
    kxx = rbf_gram(ref, ref, sigma)
    grams = {}
    for name, y in evals.items():
        y = np.asarray(y, dtype=np.float64)
        grams[name] = (rbf_gram(y, y, sigma), rbf_gram(ref, y, sigma))
    items_of = {c: np.where(clip_of == c)[0] for c in clips}

    rng = np.random.RandomState(seed)
    boot = {name: [] for name in evals}
    for _ in range(n_boot):
        idx = np.concatenate([items_of[c] for c in rng.choice(clips, size=len(clips), replace=True)])
        for name, (kyy, kxy) in grams.items():
            boot[name].append(KAD_SCALE * mmd2_unbiased(kxx[np.ix_(idx, idx)], kyy[np.ix_(idx, idx)], kxy[np.ix_(idx, idx)]))

    out = {"sigma": sigma, "n_boot": n_boot, "sets": {}}
    for name, (kyy, kxy) in grams.items():
        b = np.array(boot[name])
        out["sets"][name] = {
            "kad": KAD_SCALE * mmd2_unbiased(kxx, kyy, kxy),
            "boot_mean": float(b.mean()), "boot_std": float(b.std(ddof=1)),
            "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
        }
    if baseline is not None:
        base = np.array(boot[baseline])
        for name in evals:
            if name == baseline:
                continue
            diff = np.array(boot[name]) - base
            lo, hi = np.percentile(diff, [2.5, 97.5])
            out["sets"][name]["minus_" + baseline] = {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)],
                                                      "contains_zero": bool(lo <= 0.0 <= hi)}
    return out
