# Overlap-add of atoms

In atoms mode every generated atom is decoded on its own and the atoms are laid on one timeline, each one a hop after the
previous one, then multiplied by a window and summed. This page explains the geometry, the two crossfade laws and the overlap
modes of the stochastic decoders. The figures are drawn from the real geometry code (`scripts/make_overlap_diagrams.py`).

## Geometry

An atom has `F` latent frames, the hop is `H` frames, so consecutive atoms overlap by `O = F - H` frames. A frame is `h`
samples. The window of an atom (`ola_window` in `SCAPES/data/geometry.py`) is, in order:

| part | length (samples) | gain |
|---|---|---|
| leading zeros | `O*h - X` | 0 |
| rise | `X` | 0 to 1 |
| flat | `H*h - X` | 1 |
| fall | `X` | 1 to 0 |

`X` is the crossfade length, at most `min(H*h, O*h)`. The fall of one atom lands exactly on the rise of the next, so only the last
`H*h + X` samples of an atom reach the output: the new audio plus the crossfade into the previous atom. The start and the end of
the whole clip fade in and out, because there is nothing to cross into. A clip of `K` atoms has `((K-1)*H + F) * h` samples.

The hop is also the rate at which the semantic conditioning (one CLAP vector per unit) changes.

## Two crossfade laws

![laws](overlap_add/fig_laws.png)

- **Hann** (`hann`): rise + fall = 1. The amplitudes add up to 1. This is right when both atoms hold the same waveform in the
  overlap (a deterministic decoder, where the regenerated frames agree with the previous atom). With independent audio it dips
  by 3 dB in the middle of the fade.
- **Equal power** (`equal_power`): the square root of the Hann halves, rise^2 + fall^2 = 1. The powers add up to 1. This is right
  when the two atoms are independent decodes of similar latents (a stochastic decoder). With the same waveform in both atoms it
  would bump by 3 dB in the middle of the fade.

The law is chosen by codec (`crossfade_law_for`): equal power for CoDiCodec and Music2Latent, Hann for EnCodec 48k. The
config key `atoms.crossfade_law = "hann" | "equal_power"` overrides it. `ola_window(geom)` without a law is Hann, which is what
the V1 tests check.

## EnCodec 48k (deterministic decoder)

![encodec48k](overlap_add/fig_encodec48k.png)

0.32 s atoms, 0.1 s hop, 0.02 s fade. The overlap is long (220 ms) on purpose: the model regenerates the end of the previous
atom, so that the two decodes agree there and the audio is continuous (the design intent of V1). Only the last 0.12 s of each atom is used. `normal` crossfades
are all this codec needs; the overlap modes below are not available for it (`atoms.hop_seconds` and `atoms.crossfade_seconds`
set the geometry).

## Stochastic decoders: CoDiCodec and Music2Latent

The same latents decode to different waveforms each time (the decoder starts from noise). Regenerating past frames therefore
does not make the audio continuous. These codecs use atoms that overlap the previous one only as much as the crossfade needs,
and the hop follows from `atoms.overlap_mode`. The overlap `O` is a fraction of the atom, rounded to the codec quantum and never
less than one quantum (one frame for Music2Latent, one 8-frame chunk of 0.743 s for CoDiCodec); it never exceeds the hop.

| `atoms.overlap_mode` | overlap | crossfade `X` |
|---|---|---|
| `normal` | one quantum | short fade, `atoms.crossfade_seconds` (default 0.05 s) |
| `strong` | 25 % of the atom | the whole overlap |
| `brute` | 50 % of the atom | the whole overlap |

`atoms.hop_seconds` is derived from the mode and must not be set; `atoms.crossfade_seconds` is only used by `normal`. The mode
is part of the geometry that a model is trained with, so it is chosen in the config before dataprep and training and is stored
in `inference.gin`. Other codecs reject `atoms.overlap_mode`.

### Music2Latent (frame 4096 samples = 93 ms, quantum 1)

![music2latent](overlap_add/fig_music2latent.png)

Template `SCAPES/config/codecs/music2latent_atoms.gin`: 4-frame atoms (0.37 s).

| mode | atom | overlap | hop (conditioning rate) | fade |
|---|---|---|---|---|
| `normal` | 4 frames, 372 ms | 1 frame, 93 ms (25 %) | 3 frames, 279 ms (3.6 Hz) | 50 ms |
| `strong` (template default) | 4 frames | 1 frame, 93 ms (25 %) | 3 frames, 279 ms (3.6 Hz) | 93 ms |
| `brute` | 4 frames | 2 frames, 186 ms (50 %) | 2 frames, 186 ms (5.4 Hz) | 186 ms |

With 8-frame atoms (0.74 s) the three modes give hops of 7, 6 and 4 frames (650, 557 and 372 ms). A conditioning rate of 0.1 s
needs a hop of one frame, which means 2-frame atoms; then every mode has the same 1-frame overlap and the modes only differ in
the fade length.

### CoDiCodec (chunk 0.743 s, quantum 8 frames)

![codicodec](overlap_add/fig_codicodec.png)

The quantum limits the geometry. The trained models use 2-chunk atoms (1.486 s) with a 1-chunk hop, which is a 50 % overlap
with a 50 ms fade (the template `codicodec_atoms.gin` keeps `atoms.hop_seconds` and `atoms.crossfade_seconds`; they now render
with equal power). With 2-chunk atoms every overlap mode gives the same 50 % overlap (they differ in the fade). The modes differ
from 4-chunk atoms (about 2.97 s), where `normal` and `strong` overlap one chunk (25 %, hop 3 chunks = 2.23 s) and `brute`
overlaps two chunks (hop 2 chunks = 1.49 s). Such hops are coarse for conditioning.

## Config summary

```
codec.name = "music2latent"
generation.mode = "atoms"
atoms.seconds = 0.37            # atom, rounded up to whole quanta
atoms.overlap_mode = "strong"   # normal | strong | brute (stochastic decoders only)
# atoms.crossfade_seconds       # only for "normal"
# atoms.crossfade_law = "equal_power"   # hann | equal_power; default by codec
```

Not changed: the crossfade that `audio_extender` (`SCAPES/auxiliar/clap_wrapper.py`) applies when it repeats a short clip
for CLAP. It is linear and part of the conditioning, which the V1 reference tests fix.

Regenerate the figures: `python scripts/make_overlap_diagrams.py`.
