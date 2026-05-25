"""FlowInference with latent denoising via LocalDenoiser.

Extends FlowInference by applying a learned spectral denoiser to generated atoms
before decoding. The denoiser predicts clean magnitude spectra from CLAP embeddings,
and the final latent is a weighted blend between the generated and denoised spectra.

Usage:
    engine = FlowInferenceDenoiser(
        model_dir="models/microtex",
        denoiser_model_path="models/denoiser/checkpoints/best_denoiser.pt",
        denoise_blend=0.5,
    )
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from SCAPES.inference.FlowInference import FlowInference
from SCAPES.models.factorization import LocalDenoiser


class FlowInferenceDenoiser(FlowInference):
    """FlowInference with optional spectral denoising of generated latents."""

    def __init__(
        self,
        model_dir: str,
        denoiser_model_path: Optional[str] = None,
        denoise_blend: float = 0.5,
        device: Optional[str] = None,
        verbose: bool = False,
        checkpoint: str = "best",
    ):
        super().__init__(
            model_dir=model_dir,
            device=device,
            verbose=verbose,
            checkpoint=checkpoint,
        )

        self.denoise_blend = denoise_blend
        self.denoiser = None

        if denoiser_model_path is not None:
            self._load_denoiser(denoiser_model_path)

    def _load_denoiser(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Denoiser checkpoint not found: {path}")

        self.denoiser = LocalDenoiser(
            clap_dim=1024,
            hidden_dim=512,
            n_channels=128,
            n_freq_bins=25,
        ).to(self.device).eval()

        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.denoiser.load_state_dict(state["model_state_dict"])
        else:
            self.denoiser.load_state_dict(state)

        n = sum(p.numel() for p in self.denoiser.parameters() if p.requires_grad)
        if self.verbose:
            print(f"  Loaded denoiser from {path} ({n:,} params, blend={self.denoise_blend})")

    @torch.no_grad()
    def _denoise_atom(self, atom: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply spectral denoising to a generated atom.

        Args:
            atom: [1, 129, 48] generated atom (128 latent + 1 scale)
            context: [1024] or [1, 1024] CLAP embedding
        Returns:
            [1, 129, 48] denoised atom
        """
        latent = atom[:, :128, :]  # [1, 128, 48]

        if context.dim() == 1:
            context = context.unsqueeze(0)  # [1, 1024]

        # FFT without windowing
        spec = torch.fft.rfft(latent, dim=-1)   # [1, 128, 25] complex
        mag = spec.abs()
        phase = spec.angle()

        # Predict clean magnitude from CLAP
        pred_mag = self.denoiser(context)        # [1, 128, 25]
        pred_mag = pred_mag.view_as(mag)

        # Blend
        blended_mag = (1.0 - self.denoise_blend) * mag + self.denoise_blend * pred_mag

        # Reconstruct
        spec_out = blended_mag * torch.exp(1j * phase)
        latent_out = torch.fft.irfft(spec_out, n=self.atoms_frames, dim=-1)

        return torch.cat([latent_out, atom[:, 128:129, :]], dim=1)

    @torch.no_grad()
    def decode_timeline(self, timeline, output_path=None, method="ola_smooth"):
        """Same as FlowInference.decode_timeline but applies denoising to generated atoms."""
        if self.denoiser is not None:
            # Apply denoising to all generated atoms before decoding
            for t, step in enumerate(timeline):
                gen = step.get("atom_generated")
                if gen is not None:
                    ctx = step.get("context_embedding")
                    if ctx is not None:
                        ctx = ctx.to(self.device)
                        timeline[t]["atom_generated"] = self._denoise_atom(gen, ctx)

        # Delegate to base class for the actual OLA decode
        return super().decode_timeline(timeline, output_path=output_path, method=method)
