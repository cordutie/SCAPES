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
