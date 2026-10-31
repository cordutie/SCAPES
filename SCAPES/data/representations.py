"""On-disk store of the representations of a dataset: latents, CLAP and structure arrays (design D9).

<dataset>/representations/<codec>__<encoding>/{rep.json, index.json, latents/, clap/<key>/, structure/<key>/}
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# rep.json fields that must match for a folder to be reused (created and scapes_git are bookkeeping)
_IDENTITY = ("key", "codec", "codec_weights", "sample_rate", "hop_length", "latent_dim", "frame_quantum", "encoding",
             "unit_frames", "unit_hop_frames", "atom_type", "chunk_samples", "hop_samples")


class RepresentationMismatch(RuntimeError):
    """An existing representation folder was built with a different codec or geometry."""


# Frames of the codec -> samples at 48 kHz (CLAP and the raw audio slices live at 48 kHz). Vectorised and rounded
# half-to-even, so that dataprep and the dataset always agree on a start sample.
def frames_to_samples_48k(frames, geom):
    out = np.rint(np.asarray(frames, dtype=np.float64) * geom.hop_length * 48000.0 / geom.sample_rate)
    return out.astype(np.int64)


# Ticks (a codec frame for legacy atoms, the hop for the sample-level types) -> samples at 48 kHz, rounded half-to-even.
def ticks_to_samples_48k(ticks, geom):
    out = np.rint(np.asarray(ticks, dtype=np.float64) * geom.tick_samples * 48000.0 / geom.sample_rate)
    return out.astype(np.int64)


# Start samples (48 kHz) of the grid g * stride_ticks whose window of `window_samples` fits in the file.
def grid_starts_48k(num_samples_48k: int, geom, window_samples: int) -> np.ndarray:
    step = geom.stride_samples * 48000.0 / geom.sample_rate
    guess = int(max(0, (num_samples_48k - window_samples) / step)) + 3
    starts = ticks_to_samples_48k(np.arange(guess) * geom.stride_ticks, geom)
    return starts[starts + window_samples <= num_samples_48k]


def _git_head(path: Path):
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _contiguous(t: torch.Tensor) -> torch.Tensor:
    # clone() so that a view never drags its whole parent storage into the file
    return t.detach().cpu().contiguous().clone()


class RepresentationStore:
    def __init__(self, dataset_path, geom, info):
        self.dataset_path = Path(dataset_path)
        self.raw_dir = self.dataset_path / "raw"
        self.geom = geom
        self.info = info
        self.key = f"{geom.codec}__{geom.encoding}"
        self.root = self.dataset_path / "representations" / self.key

    # rep.json content that identifies this representation
    def _rep_dict(self) -> dict:
        g, info = self.geom, self.info
        rep = {"key": self.key, "codec": g.codec, "codec_weights": info.weights, "sample_rate": g.sample_rate,
               "hop_length": g.hop_length, "latent_dim": g.latent_dim, "frame_quantum": g.frame_quantum,
               "encoding": g.encoding}
        if g.atom_type != "frames":
            rep.update({"atom_type": g.atom_type, "chunk_samples": g.atom_samples, "hop_samples": g.hop_samples,
                        "unit_frames": g.unit_frames})
        elif g.encoding != "whole":
            rep["unit_frames"] = g.unit_frames
            rep["unit_hop_frames"] = g.unit_hop_frames
        return rep

    # Creates rep.json, or verifies that the existing one is the same representation.
    def ensure(self) -> None:
        want = self._rep_dict()
        path = self.root / "rep.json"
        if path.exists():
            have = json.loads(path.read_text())
            diff = {k: (have.get(k), want.get(k)) for k in _IDENTITY if have.get(k) != want.get(k)}
            if diff:
                raise RepresentationMismatch(f"{path} differs from the requested representation (existing, requested): {diff}")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        want["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        want["scapes_git"] = _git_head(Path(__file__).resolve().parent)
        path.write_text(json.dumps(want, indent=2))

    def _rel(self, audio_path) -> Path:
        p = Path(audio_path).resolve()
        try:
            return p.relative_to(self.raw_dir.resolve()).with_suffix(".pt")
        except ValueError:
            raise ValueError(f"{audio_path} is not inside {self.raw_dir}")

    def latent_path(self, audio_path) -> Path:
        return self.root / "latents" / self._rel(audio_path)

    def has_latents(self, audio_path) -> bool:
        return self.latent_path(audio_path).exists()

    # whole: {"latent": [d, T] float32}; per-atom: {"latent": [K, d, F] float16, "scale": [K] float32}
    def save_latents(self, audio_path, latent, scale=None) -> None:
        path = self.latent_path(audio_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.geom.encoding == "whole":
            data = {"latent": _contiguous(latent).float()}
        else:
            data = {"latent": _contiguous(latent).half(), "scale": _contiguous(scale).float()}
        torch.save(data, path)

    def load_latents(self, audio_path) -> dict:
        return torch.load(self.latent_path(audio_path), map_location="cpu", weights_only=True)

    def clap_key(self, clap_version: str, context_seconds: float, random_extension: bool) -> str:
        if self.geom.atom_type != "frames":
            return f"{clap_version}__ctx{context_seconds:g}__rext{int(random_extension)}__stride{self.geom.stride_samples}s"
        return f"{clap_version}__ctx{context_seconds:g}__rext{int(random_extension)}__stride{self.geom.stride_frames}"

    def grid_path(self, kind: str, key: str, audio_path) -> Path:
        if kind not in ("clap", "structure"):
            raise ValueError(f"kind must be 'clap' or 'structure', got {kind!r}")
        return self.root / kind / key / self._rel(audio_path)

    def has_grid_array(self, kind: str, key: str, audio_path) -> bool:
        return self.grid_path(kind, key, audio_path).exists()

    # One contiguous [G, D] float32 array per file on the start grid
    def save_grid_array(self, kind: str, key: str, audio_path, arr) -> None:
        path = self.grid_path(kind, key, audio_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_contiguous(torch.as_tensor(arr)).float(), path)

    def load_grid_array(self, kind: str, key: str, audio_path) -> torch.Tensor:
        return torch.load(self.grid_path(kind, key, audio_path), map_location="cpu", weights_only=True)

    def read_index(self) -> dict:
        path = self.root / "index.json"
        return json.loads(path.read_text()) if path.exists() else {}

    # index.json: stem -> {path (relative to the dataset), duration_s, num_samples_48k, num_frames | num_atoms}.
    # Not rewritten when identical, so that a re-run leaves every file untouched.
    def write_index(self, entries: dict) -> None:
        path = self.root / "index.json"
        if path.exists() and json.loads(path.read_text()) == entries:
            return
        path.write_text(json.dumps(entries, indent=2))
