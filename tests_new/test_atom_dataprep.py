"""S5, S6: dataprep, representation store and dataset for the atoms of the V3 geometry (toy codecs, stub CLAP, GPU).

Design: Formulation (atoms, Lemma "no time loss"), Dataset indexing, D7, D9. Toy geometries in _atoms_toy.py.
Lives outside tests/ because that folder is locked. Run from SCAPES/: ../venv_SCAPES/bin/python -m pytest tests_new -q
"""
import json
import warnings

import numpy as np
import pytest
import torch

from _atoms_toy import M_TOY, N_TOY, RATE, TOY_GEOM, load_dataset, prepare_new_toy, prepared_audio

KINDS = ["consistent"]


def _geom(config_dir):
    from SCAPES.codecs import get_codec_info
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.geometry import resolve_geometry
    cfg = load_dataprep_config(config_dir / "dataprep.gin")
    return resolve_geometry(cfg, get_codec_info(cfg.codec_name))


@pytest.mark.parametrize("kind", KINDS)
def test_each_segment_encoded_alone(tmp_path, kind):
    from audio_codec_wrappers import load_codec
    from SCAPES.data.geometry import atom_count
    root = tmp_path / "toy"
    cfg_dir, store = prepare_new_toy(root, kind)
    C, H, _ = TOY_GEOM[kind]
    codec = load_codec("toy_noncausal", device="cuda")
    index = store.read_index()
    for fname in ("a.wav", "b.wav"):
        path = root / "raw" / fname
        x = prepared_audio(codec, path)
        assert x.shape[-1] == N_TOY
        K = atom_count(N_TOY, C, H)
        assert index[path.stem]["num_atoms"] == K
        data = store.load_latents(path)
        assert data["latent"].shape == (K, 16, C // 8)
        for k in (0, 1, K - 1):
            enc = codec.encode(x[..., k * H:k * H + C], RATE)
            want = enc.z[0].float().cpu()
            got = data["latent"][k].float()
            assert float((got - want).abs().max()) <= 2e-3 * float(want.abs().max()) + 1e-6, (fname, k)   # float16 store
            assert abs(float(data["scale"][k]) - float(enc.scale.reshape(-1)[0])) <= 1e-6 * float(enc.scale.abs().max()) + 1e-9


def test_two_configs_two_folders(tmp_path):
    from SCAPES.data.representations import RepresentationMismatch, RepresentationStore
    from SCAPES.codecs import get_codec_info
    root = tmp_path / "toy"
    cons_dir, _ = prepare_new_toy(root, "consistent", config_name="config_cons")
    # a second configuration with another conditioning rate: another hop, another folder
    prepare_new_toy(root, "consistent", config_name="config_long", extra_dataprep=["conditioning_rate = 125"],
                    write_wavs=False)
    reps = root / "representations"
    a = reps / "toy_noncausal__atoms-c288-h128"
    assert a.is_dir() and len(list(reps.iterdir())) == 2
    before = {p: p.stat().st_mtime_ns for p in a.rglob("*") if p.is_file()}
    assert before
    prepare_new_toy(root, "consistent", config_name="config_cons", write_wavs=False)
    after = {p: p.stat().st_mtime_ns for p in a.rglob("*") if p.is_file()}
    assert after == before, "a re-run must leave the representation untouched"
    rep = json.loads((a / "rep.json").read_text())
    assert (rep["atom_type"], rep["chunk_samples"], rep["hop_samples"]) == ("consistent", 288, 128)
    rep["hop_samples"] = 129
    (a / "rep.json").write_text(json.dumps(rep))
    g = _geom(cons_dir)
    with pytest.raises(RepresentationMismatch):
        RepresentationStore(root, g, get_codec_info("toy_noncausal")).ensure()


def test_dataset_indices_reference(tmp_path):
    from SCAPES.data.splits import assign_region, region_bounds
    root = tmp_path / "toy"
    cfg_dir, store = prepare_new_toy(root, "consistent")
    ds = load_dataset(root, cfg_dir)
    C, H, _ = TOY_GEOM["consistent"]
    M, ctx48 = M_TOY, 24000                      # context 0.5 s at 48 kHz (TOY_COMMON)
    index = store.read_index()
    want = []
    for fname in sorted(["a.wav", "b.wav"]):
        e = index[fname[:-4]]
        K = e["num_atoms"]
        bounds = region_bounds(e["duration_s"], 0.1, 0.1)
        g = 0
        while int(np.rint(g * H * 48000 / RATE)) + ctx48 <= e["num_samples_48k"]:
            k = g
            g += 1
            if k < M:
                continue
            if k > K - 1:
                break
            span = ((k - M) * H / RATE, max((k * H + C) / RATE, k * H / RATE + 0.5))
            if assign_region(span, bounds) is not None:
                want.append((fname, k))
    got = [(f, int(t)) for f, t in ds.all_indices]
    assert got == want
    lat = store.load_latents(root / "raw" / "a.wav")["latent"].float()
    clap = store.load_grid_array("clap", ds.clap_key, root / "raw" / "a.wav")
    for idx in np.linspace(0, len([w for w in want if w[0] == "a.wav"]) - 1, 5).astype(int):
        item = ds[int(idx)]
        fname, k = ds.all_indices[int(idx)]
        assert fname == "a.wav"
        assert torch.equal(item["memory_buffer_latent"], lat[k - M:k])
        assert torch.equal(item["target_latent"], lat[k])
        assert torch.equal(item["target_semantic"], clap[k])


def test_clap_grid_alignment_ramp(tmp_path):
    root = tmp_path / "toy"
    cfg_dir, store = prepare_new_toy(root, "consistent")
    ds = load_dataset(root, cfg_dir, requested_keys=["index"])
    H = TOY_GEOM["consistent"][1]
    arr = store.load_grid_array("clap", ds.clap_key, root / "raw" / "b.wav")
    G = arr.shape[0]
    step48 = H * 48000 // RATE                                  # 768: the toy hop is 128 samples at 8 kHz
    assert G == 1 + (480000 - 24000) // step48                  # starts whose 0.5 s window fits
    for g in (0, 7, G - 1):
        s = int(np.rint(g * H * 48000 / RATE))
        want = (s + (24000 - 1) / 2) / 48000 / 20               # mean of t/20 over [s, s + 24000)
        assert abs(float(arr[g, 0]) - want) <= 1e-5, g


def test_latent_stats_new_type(tmp_path):
    root = tmp_path / "toy"
    cfg_dir, store = prepare_new_toy(root, "consistent")
    ds = load_dataset(root, cfg_dir)
    train, _ = ds.get_splits()
    mu, std = ds.latent_stats(train)
    C, H, _ = TOY_GEOM["consistent"]
    frames = []
    for fname in ("a.wav", "b.wav"):
        lat = store.load_latents(root / "raw" / fname)["latent"].double()
        for k in range(lat.shape[0]):
            if (k * H + C) / RATE <= 8.0 + 1e-9:                 # train region [0, 8) s of a 10 s file
                frames.append(lat[k])
    allf = torch.cat(frames, dim=1)                             # [16, n_frames]
    assert float((mu.double() - allf.mean(dim=1)).abs().max()) <= 1e-5
    assert float((std.double() - allf.std(dim=1, unbiased=False)).abs().max()) <= 1e-5


def test_no_scale_codec(tmp_path):
    root = tmp_path / "toy"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)              # atoms with a causal codec warn
        cfg_dir, store = prepare_new_toy(root, "consistent", codec="toy_causal")
        ds = load_dataset(root, cfg_dir)
    assert ds.geom.frame_dim == 16 and ds.geom.has_scale is False
    item = ds[0]
    assert "target_scale" not in item and "memory_buffer_scale" not in item
    data = store.load_latents(root / "raw" / "a.wav")
    assert "scale" not in data or torch.equal(data["scale"], torch.ones_like(data["scale"]))
