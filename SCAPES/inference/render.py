"""Turning generated units into audio: overlap-add for atoms, the stateful streaming decoder for AR (design: Inference)."""
import torch

from SCAPES.data.geometry import Geometry, crossfade_window


# Overlap-add of K unit audios [ch, C] placed H samples apart, each multiplied by the atom window.
# Returns [ch, (K-1) * H + C] with C = geom.atom_samples, H = geom.hop_samples.
def ola_render(unit_audios: list, geom: Geometry) -> torch.Tensor:
    C, step = geom.atom_samples, geom.hop_samples
    K = len(unit_audios)
    first = unit_audios[0]
    window = crossfade_window(geom).to(device=first.device, dtype=first.dtype)
    out = torch.zeros(first.shape[0], (K - 1) * step + C, device=first.device, dtype=first.dtype)
    for k, audio in enumerate(unit_audios):
        start = k * step
        out[:, start:start + C] += audio * window
    return out


# V1's scale smoothing (ola_smooth): the scale of each generated atom may rise by 15% or fall by 15% from the previous
# smoothed value, then is mixed 60/40 with it.
_ALPHA_SMOOTH, _MAX_JUMP, _MAX_DROP = 0.6, 1.15, 0.85


# Decodes atoms [1, frame_dim, F] (model units) one by one and overlap-adds them.
# mu, std: latent statistics to undo the standardisation (None = raw latents). smooth[k] False = the atom is given
# (teacher forcing): its raw scale is used and the smoothing state is left alone. Stochastic decoders get
# the seed base_seed + k.
# Returns [ch, N] on the device of the codec.
@torch.no_grad()
def render_atoms(codec, atoms: list, geom: Geometry, mu=None, std=None, base_seed: int = 0, smooth=None) -> torch.Tensor:
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
    return ola_render(audios, geom)


# Atoms one at a time, for real time. push(atom) decodes the atom and returns the next H new samples [ch, H] (H = geom.hop_samples);
# finish() returns the last X samples, the fade-out of the last atom. Everything pushed and finished, joined, is
# render_atoms(...)[:, C - H - X:]: only the leading C - H - X samples, which no window lets through, are not given.
# The atom k is decoded with the seed base_seed + k and the scale smoothing of render_atoms (smooth=False: given atom).
class AtomStreamRenderer:
    def __init__(self, codec, geom: Geometry, mu=None, std=None, base_seed: int = 0):
        self.codec, self.geom, self.mu, self.std, self.base_seed = codec, geom, mu, std, base_seed
        self.k = 0
        self.prev_scale = None
        self.acc = None                   # [ch, C]: the overlap-add from the start of the last pushed atom
        self.window = None

    @torch.no_grad()
    def push(self, atom: torch.Tensor, smooth: bool = True) -> torch.Tensor:
        g = self.geom
        d = g.latent_dim
        z = atom[:, :d, :].float()
        if self.mu is not None:
            z = z * self.std.view(1, -1, 1).to(z.device) + self.mu.view(1, -1, 1).to(z.device)
        scale = None
        if g.has_scale:
            raw = torch.abs(atom[:, d, :]).mean(dim=-1, keepdim=True)
            if smooth:
                if self.prev_scale is None:
                    sm = raw
                else:
                    target = torch.clamp(raw, self.prev_scale * _MAX_DROP, self.prev_scale * _MAX_JUMP)
                    sm = _ALPHA_SMOOTH * target + (1.0 - _ALPHA_SMOOTH) * self.prev_scale
                self.prev_scale = sm
            else:
                sm = raw
            scale = sm.reshape(-1).float()
        audio = self.codec.decode(z, scale, self.base_seed + self.k)[0]
        self.k += 1
        C, H, X = g.atom_samples, g.hop_samples, g.crossfade_samples
        if self.acc is None:
            self.window = crossfade_window(g).to(device=audio.device, dtype=audio.dtype)
            self.acc = torch.zeros(audio.shape[0], C, device=audio.device, dtype=audio.dtype)
        self.acc += audio * self.window
        out = self.acc[:, C - H - X:C - X].clone()          # nothing later than this atom touches these samples
        self.acc = torch.cat([self.acc[:, H:], torch.zeros_like(self.acc[:, :H])], dim=1)
        return out

    def finish(self) -> torch.Tensor:
        C, X = self.geom.atom_samples, self.geom.crossfade_samples
        tail = self.acc[:, C - X - self.geom.hop_samples:C - self.geom.hop_samples].clone()
        self.acc = None
        return tail


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
