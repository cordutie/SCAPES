"""Slice B9: S4 check of the dataprep of the S5 codecs.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/s4_check.py --dataset ../datasets/microtex --configs '../runs/s5/*/config' --out ../runs/s4
Checks (all must be true): exactly one rep folder per config and no other; each rep.json equals the resolved config;
16 files load, whole-file latents have T = q * ceil(N / (q * hop)); a second run_dataprep changes no mtime and calls
neither the encoder nor the CLAP model (spies); no file path or inode is shared between rep folders.
Writes <out>/s4_result.json, latent_hist.png and env.json. Per-codec peak GPU memory is read from <out>/peak_<codec>.json
(scripts/s4_peak.py) when present.
"""
import argparse
import glob
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SCAPES.codecs import get_codec_info
from SCAPES.data.config_loader import load_dataprep_config
from SCAPES.data.geometry import resolve_geometry
from SCAPES.data.representations import _IDENTITY, RepresentationStore

N_FILES = 16


# every file of a folder: relative path -> (mtime_ns, size)
def snapshot(folder: Path) -> dict:
    return {str(p.relative_to(folder)): (p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(folder.rglob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--configs", required=True, help="glob of config folders, e.g. '../runs/s5/*/config'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_dirs = sorted(Path(p) for p in glob.glob(args.configs))
    checks, per_codec = {}, {}

    # resolve every config to its representation
    reps = {}
    for cdir in config_dirs:
        cfg = load_dataprep_config(cdir / "dataprep.gin")
        info = get_codec_info(cfg.codec_name)
        geom = resolve_geometry(cfg, info)
        reps[cfg.codec_name] = {"config_dir": cdir, "cfg": cfg, "info": info, "geom": geom,
                                "store": RepresentationStore(dataset, geom, info)}
    keys = {c: r["store"].key for c, r in reps.items()}
    on_disk = sorted(p.name for p in (dataset / "representations").iterdir() if p.is_dir())

    # check 1: exactly the folders these configs ask for
    checks["four_rep_folders"] = len(config_dirs) == 4 and on_disk == sorted(keys.values())
    print("configs:", keys, "\non disk:", on_disk)

    # check 2: rep.json equals the resolved config
    rep_ok = True
    for c, r in reps.items():
        have = json.loads((r["store"].root / "rep.json").read_text())
        want = r["store"]._rep_dict()
        diff = {k: (have.get(k), want.get(k)) for k in _IDENTITY if have.get(k) != want.get(k)}
        per_codec.setdefault(c, {})["rep_json_diff"] = diff
        rep_ok &= not diff

    # check 3: 16 files, T = q * ceil(N / (q * hop)) for whole-file latents; stats of every file
    files_ok, t_ok = True, True
    for c, r in reps.items():
        store, geom = r["store"], r["geom"]
        index = store.read_index()
        raw = sorted((dataset / "raw").glob("*.wav"))
        files_ok &= len(raw) == N_FILES and sorted(index) == [p.stem for p in raw]
        q, hop, sr = geom.frame_quantum, geom.hop_length, geom.sample_rate
        stats = {}
        for p in raw:
            info = sf.info(str(p))
            n = round(info.frames * sr / info.samplerate)
            data = store.load_latents(p)
            z = data["latent"].float()
            entry = {"frames_or_atoms": int(z.shape[-1] if geom.encoding == "whole" else z.shape[0]),
                     "mean": float(z.mean()), "std": float(z.std()), "finite": bool(torch.isfinite(z).all())}
            if geom.encoding == "whole":
                expected = q * math.ceil(n / (q * hop))
                entry["expected_T"] = expected
                t_ok &= entry["frames_or_atoms"] == expected
            else:
                t_ok &= z.shape[-1] == geom.unit_frames and z.shape[0] >= 1
            t_ok &= entry["finite"] and entry["std"] > 0
            stats[p.stem] = entry
        per_codec.setdefault(c, {})["files"] = stats
    checks["rep_json_equals_config"] = bool(rep_ok)
    checks["sixteen_files_and_frame_counts"] = bool(files_ok and t_ok)

    # check 4: second run_dataprep; spies on the encoder and on the CLAP embedder
    import SCAPES.data.dataprep.atoms as atoms_module
    from SCAPES.pipeline import run_dataprep
    calls = {"encoder": 0, "clap": 0}
    real_get_codec = atoms_module.get_codec

    def spy_get_codec(*a, **k):
        calls["encoder"] += 1
        return real_get_codec(*a, **k)

    def spy_clap(*a, **k):
        calls["clap"] += 1
        raise RuntimeError("CLAP embedder called on a prepared dataset")

    atoms_module.get_codec = spy_get_codec
    second_ok = True
    for c, r in reps.items():
        before = snapshot(r["store"].root)
        calls["encoder"] = calls["clap"] = 0
        run_dataprep(dataset, config_dir=r["config_dir"], clap_embedder=spy_clap)
        after = snapshot(r["store"].root)
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        per_codec[c]["second_run"] = {"changed_files": changed, "encoder_calls": calls["encoder"], "clap_calls": calls["clap"]}
        second_ok &= not changed and calls["encoder"] == 0 and calls["clap"] == 0
    atoms_module.get_codec = real_get_codec
    checks["second_dataprep_noop"] = bool(second_ok)

    # check 5: no path or inode shared between rep folders
    seen_paths, seen_inodes, shared = set(), set(), []
    for c, r in reps.items():
        for p in r["store"].root.rglob("*"):
            if p.is_file():
                ino = (p.stat().st_dev, p.stat().st_ino)
                if str(p.resolve()) in seen_paths or ino in seen_inodes:
                    shared.append(str(p))
                seen_paths.add(str(p.resolve()))
                seen_inodes.add(ino)
    checks["no_shared_files"] = not shared

    # sizes on disk, peak GPU of the dataprep (scripts/s4_peak.py) and the latent histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    rng = np.random.default_rng(0)
    colors = {"encodec24k": "tab:blue", "spectrostream": "tab:orange", "codicodec": "tab:green", "encodec48k": "tab:red"}
    for c, r in reps.items():
        root = r["store"].root
        per_codec[c]["rep_dir_mb"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6
        peak = out / f"peak_{c}.json"
        per_codec[c]["dataprep_peak"] = json.loads(peak.read_text()) if peak.exists() else None
        vals = []
        for p in sorted((dataset / "raw").glob("*.wav")):
            v = r["store"].load_latents(p)["latent"].float().flatten()
            vals.append(v[torch.randint(0, v.numel(), (20000,))].numpy())
        vals = np.concatenate(vals)
        ax.hist(vals, bins=200, range=(np.percentile(vals, 0.5), np.percentile(vals, 99.5)), density=True, histtype="step",
                color=colors.get(c), label=f"{c} (std {vals.std():.2f})")
    ax.set_xlabel("raw latent value (before standardisation)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "latent_hist.png", dpi=120)

    result = {"checks": checks, "passed": all(checks.values()), "keys": keys, "on_disk": on_disk, "shared_files": shared,
              "per_codec": per_codec}
    (out / "s4_result.json").write_text(json.dumps(result, indent=1))
    (out / "env.json").write_text(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                                              "cuda": torch.version.cuda, "numpy": np.__version__}, indent=1))
    print(json.dumps(checks, indent=1))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
