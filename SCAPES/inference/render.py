"""Turning generated units into audio: overlap-add for atoms, the stateful streaming decoder for AR (design: Inference)."""
import torch

from SCAPES.data.geometry import Geometry, ola_window


# Overlap-add of K unit audios [C, F*h] placed H*h samples apart, each multiplied by the atom window.
# Returns [C, ((K-1)H + F) h].
def ola_render(unit_audios: list, geom: Geometry, law: str = "hann") -> torch.Tensor:
    h, F, H = geom.hop_length, geom.unit_frames, geom.unit_hop_frames
    K = len(unit_audios)
    first = unit_audios[0]
    window = ola_window(geom, law).to(device=first.device, dtype=first.dtype)
    out = torch.zeros(first.shape[0], ((K - 1) * H + F) * h, device=first.device, dtype=first.dtype)
    for k, audio in enumerate(unit_audios):
        start = k * H * h
        out[:, start:start + F * h] += audio * window
    return out


# V1's scale smoothing (ola_smooth): the scale of each generated atom may rise by 15% or fall by 15% from the previous
# smoothed value, then is mixed 60/40 with it.
_ALPHA_SMOOTH, _MAX_JUMP, _MAX_DROP = 0.6, 1.15, 0.85


# Decodes atoms [1, frame_dim, F] (model units) one by one and overlap-adds them.
# mu, std: latent statistics to undo the standardisation (None = raw latents). smooth[k] False = the atom is given
# (teacher forcing): its raw scale is used and the smoothing state is left alone. CoDiCodec-style stochastic decoders get
# the seed base_seed + k. law: crossfade law of the window (see ola_window). Returns [C, N] on the device of the codec.
@torch.no_grad()
def render_atoms(codec, atoms: list, geom: Geometry, mu=None, std=None, base_seed: int = 0, smooth=None,
                 law: str = "hann") -> torch.Tensor:
    d = geom.latent_dim
    prev_scale = None
    audios = []
    for k, atom in enumerate(atoms):
        z = atom[:, :d, :].float()
        if mu is not None:
            z = z * std.view(1, -1, 1).to(z.device) + mu.view(1, -1, 1).to(z.device)
        scale = None
        if geom.has_scale:
            raw = torch.abs(atom[:, d, :]).mean(dim=-1, keepdim=True)
            if smooth is None or smooth[k]:
                if prev_scale is None:
                    sm = raw
                else:
                    target = torch.clamp(raw, prev_scale * _MAX_DROP, prev_scale * _MAX_JUMP)
                    sm = _ALPHA_SMOOTH * target + (1.0 - _ALPHA_SMOOTH) * prev_scale
                prev_scale = sm
            else:
                sm = raw
            scale = sm.reshape(-1).float()
        audios.append(codec.decode(z, scale, base_seed + k)[0])
    return ola_render(audios, geom, law)


# Streaming decode of AR blocks: the codec's stateful decoder keeps its convolution and LSTM state between blocks.
class StreamRenderer:
    def __init__(self, codec, batch_size: int = 1):
        self.batch_size = batch_size
        self.decoder = codec.streaming_decoder(batch_size)
        self.pieces = []

    # z [batch, d, B]: raw (unstandardised) latents of one block
    @torch.no_grad()
    def push(self, z: torch.Tensor) -> None:
        self.pieces.append(self.decoder.push(z))

    # Flushes the decoder and returns the audio [C, K*B*h] (the first L samples, the decoder latency, are dropped)
    @torch.no_grad()
    def finish(self) -> torch.Tensor:
        self.pieces.append(self.decoder.flush())
        latency = int(self.decoder.latency_samples)
        y = torch.cat(self.pieces, dim=-1)[..., latency:]
        self.pieces = []
        return y[0] if self.batch_size == 1 else y
