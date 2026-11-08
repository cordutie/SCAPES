"""The K grid and the representatives of the atlas (SCAPES/auxiliar/atlas.py, atlas_reps.py). No GPU, no audio. Run: python -m pytest tests -q"""
import numpy as np
import torch

from SCAPES.auxiliar.atlas import Atlas, K_RANGE, K_STEP
from SCAPES.auxiliar.atlas_reps import pick


def blobs(centres, n, spread=0.05, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = [torch.nn.functional.normalize(torch.as_tensor(c) + spread * torch.randn(n, d, generator=g), dim=1) for c in centres]
    return torch.cat(rows)


def axes(*ids, d=32):
    return [torch.eye(d)[i] for i in ids]


def test_only_the_multiples_of_three_of_the_range_are_tried(monkeypatch):
    import SCAPES.auxiliar.atlas as atlas_module
    tried = []
    real = atlas_module._kmeans
    monkeypatch.setattr(atlas_module, "_kmeans", lambda x, k, g, iters=30: (tried.append(k), real(x, k, g, iters))[1])
    atlas = Atlas.fit(blobs(axes(*range(12)), 200, d=32))
    assert tried == list(range(K_RANGE[0], K_RANGE[1] + 1, K_STEP)) and K_RANGE == (9, 60) and K_STEP == 3
    assert 1 <= atlas.K <= K_RANGE[1]          # a centre that ends up with no rows is dropped, so K is not always on the grid


def test_one_group_gives_one_representative_and_it_is_a_real_row():
    rows = blobs(axes(0), 300)
    reps = pick(rows, torch.zeros(300, dtype=torch.long), 1)
    assert len(reps) == 1 and reps[0][0] == 0 and 0 <= reps[0][1] < 300


def test_three_separate_groups_give_three_representatives():
    rows = blobs(axes(0, 1, 2), 100, spread=0.04)
    reps = pick(rows, torch.zeros(300, dtype=torch.long), 1)
    assert len(reps) == 3
    assert sorted(r[1] // 100 for r in reps) == [0, 1, 2]       # one from each blob
    assert len({r[1] for r in reps}) == 3


def test_a_small_cluster_has_one_representative():
    rows = blobs(axes(0, 1), 10)
    assert len(pick(rows, torch.zeros(20, dtype=torch.long), 1)) == 1


def test_a_tiny_group_is_not_a_representative():
    rows = torch.cat([blobs(axes(0), 200), blobs(axes(1), 6)])       # the second group is 3% of the cluster
    assert len(pick(rows, torch.zeros(206, dtype=torch.long), 1)) == 1


def test_every_cluster_has_at_least_one_and_at_most_three():
    rows = blobs(axes(*range(6)), 150, d=32)
    labels = torch.arange(6).repeat_interleave(150)
    reps = pick(rows, labels, 6)
    count = np.bincount([r[0] for r in reps], minlength=6)
    assert count.min() >= 1 and count.max() <= 3


def test_representatives_survive_save_and_load(tmp_path):
    rows = blobs(axes(*range(12)), 200, d=32)
    atlas = Atlas.fit(rows)
    files = [rows[:1200], rows[1200:]]
    stored = [atlas.autoencode(a) for a in files]
    starts = [np.arange(len(a)) * 0.2 for a in files]
    reps = atlas.add_representatives(files, stored, starts, ["a", "b"])
    atlas.save(tmp_path / "atlas.npz")
    loaded = Atlas.load(tmp_path / "atlas.npz")
    assert len(loaded.reps) == len(reps) > 0
    assert np.array_equal(loaded.reps.vec, reps.vec) and loaded.reps.files == ["a", "b"]
    assert all(0 <= s < 0.2 * max(len(a) for a in files) for s in loaded.reps.start)
    # a stored vector is the row as the model sees it, so it falls back in the cluster it represents
    assert (loaded.encode(torch.tensor(loaded.reps.vec))[0].numpy() == loaded.reps.cluster).all()


def test_an_atlas_without_representatives_loads(tmp_path):
    atlas = Atlas.fit(blobs(axes(*range(12)), 100, d=32))
    atlas.save(tmp_path / "atlas.npz")
    assert Atlas.load(tmp_path / "atlas.npz").reps is None
