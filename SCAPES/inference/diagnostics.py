"""Diagnostics of generated latents and audio (design D-c)."""
import torch


# Seam ratio of a latent sequence z [d, T] cut into blocks of `block` frames, the first block starting at frame `phase`:
# mean norm of the frame-to-frame change across block boundaries divided by the mean norm of the change inside blocks.
# A boundary is the step from frame t-1 to frame t with (t - phase) % block == 0. 1 means no seams.
def seam_ratio(z: torch.Tensor, block: int, phase: int = 0) -> float:
    z = torch.as_tensor(z).double()
    step = (z[:, 1:] - z[:, :-1]).norm(dim=0)                  # step[i] is the change from frame i to frame i + 1
    t = torch.arange(1, z.shape[1])
    is_seam = (t - phase) % block == 0
    if not is_seam.any() or is_seam.all():
        raise ValueError("seam_ratio needs both seam and non-seam steps")
    return float(step[is_seam].mean() / step[~is_seam].mean())


# RMS of a signal in dB relative to ref_rms; -inf for silence, nan if the signal has non-finite values
def _rms_db(x: torch.Tensor, ref_rms: float) -> float:
    rms = float(x.double().pow(2).mean().sqrt())
    if rms != rms:
        return float("nan")
    return float("-inf") if rms == 0.0 else 20.0 * float(torch.log10(torch.tensor(rms / ref_rms)))


# Production checks of one generated clip y [C, N] at sample_rate (S5 ii):
# finite; duration within one unit of target_seconds; RMS (mono mixdown) within [-20, +10] dB of ref_rms over the whole
# clip (rms_ok) and over its first second (rms_first_ok); fraction of samples with |y| >= 1 at most 0.1% (clip_ok).
def production_checks(y: torch.Tensor, sample_rate: int, ref_rms: float, target_seconds: float, unit_seconds: float,
                      rms_range_db=(-20.0, 10.0), max_clipped: float = 0.001) -> dict:
    y = torch.as_tensor(y).float()
    if y.dim() == 1:
        y = y.unsqueeze(0)
    finite = bool(torch.isfinite(y).all())
    duration = y.shape[-1] / sample_rate
    mono = y.mean(dim=0)
    db = _rms_db(mono, ref_rms)
    db_first = _rms_db(mono[:sample_rate], ref_rms)
    lo, hi = rms_range_db
    clipped = float((y.abs() >= 1.0).float().mean())
    out = {
        "finite": finite,
        "duration_ok": bool(abs(duration - target_seconds) <= unit_seconds),
        "rms_ok": bool(finite and lo <= db <= hi),
        "rms_first_ok": bool(finite and lo <= db_first <= hi),
        "clip_ok": bool(clipped <= max_clipped),
        "duration_s": float(duration), "rms_db": db, "rms_first_db": db_first, "clipped_fraction": clipped,
    }
    out["pass"] = bool(out["finite"] and out["duration_ok"] and out["rms_ok"] and out["rms_first_ok"] and out["clip_ok"])
    return out
