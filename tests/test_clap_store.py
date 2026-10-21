"""Slice B3: CLAP stored per representation on the start grid (design D9; CLAP alignment)."""
import csv

import torch

from conftest import load_toy_dataset

STRIDE_S = 0.016  # toy AR stride: S = 16 frames at 1000 Hz
CTX_S = 0.5


def _clap_key(store):
    keys = [p.name for p in (store.root / "clap").iterdir() if p.is_dir()]
    assert len(keys) == 1, keys
    return keys[0]


def test_alignment_known_answer(toy_ar):
    """b.wav is the ramp x(t) = t/20; the stub returns the window mean, so c_g[0] = (t_g + tau_c/2)/20."""
    root, store = toy_ar
    key = _clap_key(store)
    arr = store.load_grid_array("clap", key, root / "raw" / "b.wav")
    G = 594  # grid starts g*16 frames whose 0.5 s window fits in 10 s: g <= 593.75
    assert arr.shape == (G, 1024)
    t = torch.arange(G, dtype=torch.float64) * STRIDE_S
    want = (t + CTX_S / 2) / 20.0
    assert float((arr[:, 0].double() - want).abs().max()) <= 1e-5
    assert float(arr[:, 1:].abs().max()) == 0.0
    assert store.load_grid_array("clap", key, root / "raw" / "a.wav").shape == (G, 1024)


def test_semantics_folder_grid(toy_ar, tmp_path):
    from SCAPES.data.dataprep.semantic import build_semantics_folder
    root, store = toy_ar
    csv_path = tmp_path / "cherry_picking.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "start_sec", "end_sec", "flag", "icon", "description"])
        w.writerow(["b.wav", "1.0", "2.0", "ramp", "R", "ramp from 1 to 2 s"])
    out_dir = tmp_path / "semantics"
    ds = load_toy_dataset(root)
    assert build_semantics_folder(csv_path, out_dir, ds) == 1
    out = torch.load(out_dir / "ramp.pt", map_location="cpu", weights_only=True)
    # grid starts in [1, 2): g * 0.016 in [1, 2) -> g = 63 .. 124
    assert out.shape == (62, 1024)
    arr = store.load_grid_array("clap", _clap_key(store), root / "raw" / "b.wav")
    assert torch.equal(out[:, 0].cpu().float(), arr[63:125, 0].cpu().float())
