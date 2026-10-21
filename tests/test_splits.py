"""Slice B3: time split with span-inclusion membership (design D11, "Training samples and split")."""
import json

import pytest

from conftest import load_toy_dataset, make_toy_dataset, stub_clap


def _span_s(ds, idx):
    """[past start, max(target end, CLAP window end)] in seconds, from the geometry."""
    g = ds.geom
    _, t = ds.all_indices[idx]
    start = t - g.past_frames  # AR: one window of W frames; atoms: M past atoms spaced by H
    assert g.past_frames == (g.memory_frames if g.mode == "autoregressive" else g.memory_units * g.unit_hop_frames)
    end = max((t + g.unit_frames) / g.frame_rate, t / g.frame_rate + g.context_seconds)
    return start / g.frame_rate, end


def test_region_bounds_and_assign():
    from SCAPES.data.splits import assign_region, region_bounds
    b = region_bounds(10.0, 0.1, 0.1)
    assert set(b) == {"train", "val", "reserve"}
    for name, (lo, hi) in {"train": (0.0, 8.0), "val": (8.0, 9.0), "reserve": (9.0, 10.0)}.items():
        assert b[name][0] == pytest.approx(lo, abs=1e-9) and b[name][1] == pytest.approx(hi, abs=1e-9)
    assert assign_region((7.5, 8.0), b) == "train"
    assert assign_region((8.0, 8.5), b) == "val"
    assert assign_region((9.5, 10.0), b) == "reserve"
    assert assign_region((7.9, 8.1), b) is None
    assert assign_region((8.9, 9.1), b) is None


def test_no_straddle(toy_ar):
    root, _ = toy_ar
    ds = load_toy_dataset(root)
    train, val = ds.get_splits()
    reserve = ds.get_reserve()
    bounds = {"train": (0.0, 8.0), "val": (8.0, 9.0), "reserve": (9.0, 10.0)}
    sets = {}
    for name, sub in (("train", train), ("val", val), ("reserve", reserve)):
        idxs = list(sub.indices)
        assert idxs, name
        sets[name] = set(idxs)
        lo, hi = bounds[name]
        for i in idxs:
            s, e = _span_s(ds, i)
            assert lo - 1e-9 <= s and e <= hi + 1e-9, (name, ds.all_indices[i], s, e)
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["reserve"]) and not (sets["val"] & sets["reserve"])


def test_same_regions_across_codecs(tmp_path):
    from SCAPES.pipeline import run_dataprep
    root = tmp_path / "toy"
    make_toy_dataset(root, "autoregressive", config_name="config_ar")
    make_toy_dataset(root, "atoms", config_name="config_atoms", write_wavs=False)
    stores = [run_dataprep(root, config_dir=root / c, device="cuda", clap_embedder=stub_clap)
              for c in ("config_ar", "config_atoms")]
    durations = [{k: v["duration_s"] for k, v in json.loads((s.root / "index.json").read_text()).items()} for s in stores]
    assert durations[0] == durations[1]
    bounds = {"train": (0.0, 8.0), "val": (8.0, 9.0), "reserve": (9.0, 10.0)}
    for c in ("config_ar", "config_atoms"):
        ds = load_toy_dataset(root, config_dir=root / c)
        train, val = ds.get_splits()
        for name, sub in (("train", train), ("val", val), ("reserve", ds.get_reserve())):
            assert len(sub) > 0, (c, name)
            lo, hi = bounds[name]
            for i in sub.indices:
                s, e = _span_s(ds, i)
                assert lo - 1e-9 <= s and e <= hi + 1e-9, (c, name, s, e)
