"""Slice B3: dataset items in both modes (design "Training samples and split", D6)."""
import numpy as np
import torch

from conftest import load_toy_dataset

B, W, S, CTX = 32, 96, 16, 500  # toy AR geometry in frames (1000 Hz): block, ONE past window, stride; CTX = 0.5 s
T_FILE = 10000


def _probe_indices(n):
    g = torch.Generator().manual_seed(0)
    return sorted({0, n // 2, n - 1, *torch.randint(0, n, (20,), generator=g).tolist()})


def test_ar_item(toy_ar):
    root, store = toy_ar
    ds = load_toy_dataset(root, requested_keys=["memory_buffer_latent", "target_latent", "target_semantic", "index"])
    assert ds.has_scale is False and ds.frame_dim == 16
    for idx in _probe_indices(len(ds)):
        fname, t = ds.all_indices[idx]
        u = store.load_latents(root / "raw" / fname)["latent"]
        item = ds[idx]
        assert "memory_buffer_scale" not in item and "target_scale" not in item
        mem = item["memory_buffer_latent"]
        assert mem.shape == (1, 16, W)  # one past window (Esteban review), not M blocks
        assert torch.equal(mem[0], u[:, t - W:t])
        assert torch.equal(item["target_latent"], u[:, t:t + B])
        assert item["target_semantic"].shape == (1024,)


def test_atoms_item(toy_atoms):
    root, store = toy_atoms
    ds = load_toy_dataset(root, requested_keys=["memory_buffer_latent", "target_latent", "memory_buffer_scale",
                                                "target_scale", "index"])
    F, H, M = 48, 16, 1
    assert ds.has_scale is True and ds.frame_dim == 17
    for idx in _probe_indices(len(ds)):
        fname, t = ds.all_indices[idx]
        assert t % H == 0
        k = t // H
        d = store.load_latents(root / "raw" / fname)
        item = ds[idx]
        assert item["memory_buffer_latent"].shape == (M, 16, F)
        assert item["memory_buffer_scale"].shape == (M, 1)
        assert item["target_scale"].shape == (1,)
        for j in range(M):
            assert torch.equal(item["memory_buffer_latent"][j].float(), d["latent"][k - M + j].float())
            assert float(item["memory_buffer_scale"][j, 0]) == float(d["scale"][k - M + j])
        assert torch.equal(item["target_latent"].float(), d["latent"][k].float())
        assert float(item["target_scale"][0]) == float(d["scale"][k])


def test_latent_stats(toy_ar):
    """mu, std over the train-region frames [0, 8 s) of every file; population std (ddof 0), floored at 1e-5."""
    root, store = toy_ar
    ds = load_toy_dataset(root)
    train, _ = ds.get_splits()
    mu, std = ds.latent_stats(train)
    frames = np.concatenate([store.load_latents(root / "raw" / f)["latent"][:, :8000].double().numpy()
                             for f in sorted(ds.filenames)], axis=1)
    assert mu.shape == (16,) and std.shape == (16,)
    assert np.abs(mu.double().numpy() - frames.mean(axis=1)).max() <= 1e-6
    assert np.abs(std.double().numpy() - np.maximum(frames.std(axis=1), 1e-5)).max() <= 1e-6


def test_sample_enumeration(toy_ar):
    """all_indices = every t in S*Z with a full past, a target and a CLAP window that lie inside one region."""
    root, _ = toy_ar
    ds = load_toy_dataset(root)
    regions = [(0, 8000), (8000, 9000), (9000, 10000)]  # frames
    want = set()
    for fname in ds.filenames:
        for t in range(0, T_FILE, S):
            start, end = t - W, max(t + B, t + CTX)
            if start < 0 or end > T_FILE:
                continue
            if any(lo <= start and end <= hi for lo, hi in regions):
                want.add((fname, t))
    got = [(f, int(t)) for f, t in ds.all_indices]
    assert len(got) == len(set(got))
    assert set(got) == want
