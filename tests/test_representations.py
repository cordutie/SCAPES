"""Slice B3: representation store (design D9, D8; S4)."""
import json
import os

import pytest
import torch

from conftest import DURATION, TOY_D, TOY_HOP, TOY_RATE, load_toy_dataset, make_toy_dataset, max_abs, prepare_toy, stub_clap

N_TOY = int(DURATION * TOY_RATE)  # samples per file at the toy codec rate


def _all_files(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_create_and_shapes(toy_ar):
    from audio_codec_wrappers import codec_info
    root, store = toy_ar
    info = codec_info("toy_causal")
    assert store.root == root / "representations" / "toy_causal__whole"
    rep = json.loads((store.root / "rep.json").read_text())
    want = {"key": "toy_causal__whole", "codec": "toy_causal", "sample_rate": 8000, "hop_length": 8,
            "latent_dim": 16, "frame_quantum": 1, "encoding": "whole"}
    for k, v in want.items():
        assert rep[k] == v, k
    for stem in ("a", "b"):
        lat = store.load_latents(root / "raw" / f"{stem}.wav")["latent"]
        assert lat.shape == (TOY_D, info.num_frames(N_TOY)) == (16, 10000)
        assert lat.dtype == torch.float32


def test_two_codecs_coexist(tmp_path):
    from SCAPES.pipeline import run_dataprep
    root = tmp_path / "toy"
    make_toy_dataset(root, "autoregressive", config_name="config_ar")
    make_toy_dataset(root, "atoms", config_name="config_atoms", write_wavs=False)
    s1 = run_dataprep(root, config_dir=root / "config_ar", device="cuda", clap_embedder=stub_clap)
    s2 = run_dataprep(root, config_dir=root / "config_atoms", device="cuda", clap_embedder=stub_clap)
    assert s1.root != s2.root
    assert sorted(p.name for p in (root / "representations").iterdir()) == ["toy_causal__whole",
                                                                             "toy_noncausal__atoms-f48-h16"]
    f1, f2 = set(_all_files(s1.root)), set(_all_files(s2.root))
    assert f1 and f2 and not (f1 & f2)
    assert len(load_toy_dataset(root, config_dir=root / "config_ar")) > 0
    assert len(load_toy_dataset(root, config_dir=root / "config_atoms")) > 0


def test_rerun_noop(toy_ar, monkeypatch):
    from audio_codec_wrappers.codecs.toy import ToyCodec
    from SCAPES.pipeline import run_dataprep
    root, store = toy_ar
    before = {p: os.stat(p).st_mtime_ns for p in _all_files(root / "representations")}
    calls = {"encode": 0, "clap": 0}
    orig = ToyCodec.encode

    def spy_encode(self, *a, **k):
        calls["encode"] += 1
        return orig(self, *a, **k)

    def spy_clap(audio, sr):
        calls["clap"] += 1
        return stub_clap(audio, sr)

    monkeypatch.setattr(ToyCodec, "encode", spy_encode)
    run_dataprep(root, config_dir=root / "config", device="cuda", clap_embedder=spy_clap)
    after = {p: os.stat(p).st_mtime_ns for p in _all_files(root / "representations")}
    assert calls == {"encode": 0, "clap": 0}
    assert after == before


def test_mismatch_raises(toy_ar):
    from SCAPES.data.representations import RepresentationMismatch
    from SCAPES.pipeline import run_dataprep
    root, store = toy_ar
    p = store.root / "rep.json"
    rep = json.loads(p.read_text())
    rep["hop_length"] = 16
    p.write_text(json.dumps(rep))
    with pytest.raises(RepresentationMismatch):
        run_dataprep(root, config_dir=root / "config", device="cuda", clap_embedder=stub_clap)


def test_whole_file_slicing_exact(toy_ar):
    from audio_codec_wrappers import load_codec
    from audio_codec_wrappers.audio import load_audio
    root, store = toy_ar
    codec = load_codec("toy_causal", device="cuda")
    x, sr = load_audio(root / "raw" / "a.wav")
    x8 = codec.prepare(x, sr)  # [1, 1, 80000]
    lat = store.load_latents(root / "raw" / "a.wav")["latent"]
    for a, b in [(0, 32), (1000, 1096), (9968, 10000)]:
        z = codec.encode(x8[..., a * TOY_HOP: b * TOY_HOP], TOY_RATE).z[0]
        assert max_abs(lat[:, a:b], z) <= 1e-6


def test_per_atom_counts(toy_atoms):
    """K = V1 _compute_atom_counts allowed_count, at the codec rate, with the CLAP window as context."""
    root, store = toy_atoms
    F, H, ctx = 48, 16, int(0.5 * TOY_RATE)
    seg, hop = F * TOY_HOP, H * TOY_HOP
    full = 1 + (N_TOY - seg) // hop
    allowed = min(full, 1 + (N_TOY - ctx) // hop)
    for stem in ("a", "b"):
        d = store.load_latents(root / "raw" / f"{stem}.wav")
        assert d["latent"].shape == (allowed, TOY_D, F)
        assert d["scale"].shape == (allowed,)


def test_contiguous_storage(toy_ar, toy_atoms):
    checked = 0
    for _, store in (toy_ar, toy_atoms):
        for p in _all_files(store.root):
            if p.suffix != ".pt":
                continue
            obj = torch.load(p, map_location="cpu", weights_only=True)
            tensors = obj.values() if isinstance(obj, dict) else [obj]
            for t in tensors:
                if torch.is_tensor(t):
                    assert t.untyped_storage().nbytes() == t.numel() * t.element_size(), p
                    checked += 1
    assert checked >= 4
