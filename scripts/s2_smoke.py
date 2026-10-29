"""Slice B7: S2 smoke, codec swap by config on a small dataset (design S2, R1-C8).

For each codec: write the run config with pipeline.build_run_config, record the git state of both repos, run dataprep,
training and generation in three fresh subprocesses, record the git state again, and write <out>/<codec>/s2_result.json.
After the codecs, an optional 20-step check with training.spectral_representation=True on encodec48k (D13).

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/s2_smoke.py --dataset ../datasets/micro_microtex --out ../runs/s2 \
        --codecs encodec48k codicodec encodec24k spectrostream
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import audio_codec_wrappers
WRAPPERS = Path(audio_codec_wrappers.__file__).resolve().parents[1]
SNIPPETS = {
    "encodec48k": "encodec48k_atoms.gin",
    "codicodec": "codicodec_atoms.gin",
    "music2latent": "music2latent_atoms.gin",
    "codicodec_grains": "codicodec_grains.gin",
    "encodec24k": "encodec24k_ar.gin",
    "spectrostream": "spectrostream_ar.gin",
}
STAGES = ("dataprep", "training", "generation")
STEP0_BAND = (1.8, 2.2)
GEN_SECONDS = 10.0


# ─── helpers ───

def git_state():
    state = {}
    for name, path in (("SCAPES", REPO), ("audio_codec_wrappers", WRAPPERS)):
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=path, capture_output=True, text=True).stdout
        state[name] = {"head": head, "status": status.splitlines()}
    return state


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))


# ─── stage workers (run inside a fresh subprocess) ───

def peak_memory_bytes():
    import torch
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def worker_dataprep(args):
    from SCAPES.pipeline import run_dataprep
    store = run_dataprep(args.dataset, config_dir=args.config_dir)
    return {"rep_key": store.key, "rep_dir": str(store.root)}


def worker_training(args):
    from SCAPES.pipeline import run_training
    run_training(args.dataset, args.model_dir, config_dir=args.config_dir)
    return {}


# Prompt of each file: the CLAP array of the first sample of its val region (the first 1 s of that region); never reserve
def worker_generation(args):
    import torch
    from SCAPES.data.config_loader import load_dataprep_config
    from SCAPES.data.dataset import AtomSequenceDataset
    from SCAPES.pipeline import run_generation

    config = load_dataprep_config(Path(args.config_dir) / "dataprep.gin")
    dataset = AtomSequenceDataset(args.dataset, config, requested_keys=["target_semantic", "index"], device="cpu")
    outputs = []
    seen = set()
    for idx, (fname, start) in enumerate(dataset.all_indices):
        if dataset._regions[idx] != "val" or fname in seen:
            continue
        seen.add(fname)
        assert dataset._regions[idx] != "reserve"
        prompt = dataset[idx]["target_semantic"]
        stem = Path(fname).stem
        wav = run_generation(args.model_dir, Path(args.gen_dir) / stem, [prompt], seconds=GEN_SECONDS, seed=0)
        outputs.append({"file": fname, "prompt_start_frame": int(start), "wav": str(wav)})
    return {"prompts": outputs}


WORKERS = {"dataprep": worker_dataprep, "training": worker_training, "generation": worker_generation}


def run_worker(args):
    t0 = time.time()
    out = WORKERS[args.stage](args)
    out.update({"stage": args.stage, "seconds": time.time() - t0, "peak_gpu_bytes": peak_memory_bytes()})
    write_json(args.stage_json, out)


# ─── parent ───

def spawn_stage(stage, args, codec_dir, config_dir, log_path):
    stage_json = codec_dir / f"stage_{stage}.json"
    stage_json.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--stage", stage, "--dataset", str(args.dataset),
           "--config-dir", str(config_dir), "--model-dir", str(codec_dir / "model"), "--gen-dir", str(codec_dir / "gen"),
           "--stage-json", str(stage_json)]
    t0 = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n===== stage {stage} =====\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    result = {"exit_code": proc.returncode, "seconds": time.time() - t0, "ok": proc.returncode == 0}
    if stage_json.exists():
        result.update(json.loads(stage_json.read_text()))
    return result


def build_config(args, codec_dir, extra_overrides, max_steps):
    from SCAPES.pipeline import build_run_config
    overrides = [
        'model.size = "small"',
        f"training.max_steps = {max_steps}",
        "training.batch_size = 128",
        "training.log_every = 1",
        "training.seed = 0",
        "splits.val_split_ratio = 0.1",
        "splits.reserve_ratio = 0.1",
    ] + extra_overrides
    snippet = REPO / "SCAPES" / "config" / "codecs" / SNIPPETS[args.codec_now]
    return build_run_config(Path(args.dataset) / "config", [snippet], overrides, codec_dir / "config")


def check_artefacts(codec_dir, dataset):
    checks = {}
    reps = Path(dataset) / "representations"
    checks["rep_folder"] = reps.is_dir() and any(reps.iterdir())
    ckpt = codec_dir / "model" / "checkpoints"
    checks["best_flow_model"] = (ckpt / "best_flow_model.pt").exists()
    gin = ckpt / "inference.gin"
    text = gin.read_text() if gin.exists() else ""
    checks["inference_gin"] = "codec.name" in text and "generation.mode" in text
    # wav of 10 s +- one unit (unit duration from the checkpoint geometry is not read here: 0.5 s tolerance bound)
    wavs = sorted((codec_dir / "gen").glob("*/generation.wav"))
    durations = []
    if wavs:
        import soundfile as sf
        durations = [sf.info(str(w)).duration for w in wavs]
    checks["wav_durations_s"] = durations
    checks["wav_10s"] = bool(durations) and all(abs(d - GEN_SECONDS) <= 1.0 for d in durations)
    return checks


def run_codec(args, codec, max_steps, out_dir, extra_overrides=(), stages=STAGES):
    codec_dir = out_dir
    codec_dir.mkdir(parents=True, exist_ok=True)
    args.codec_now = codec
    config_dir = build_config(args, codec_dir, list(extra_overrides), max_steps)
    log_path = codec_dir / "s2.log"
    log_path.write_text("")

    before = git_state()
    result = {"codec": codec, "max_steps": max_steps, "git_before": before, "stages": {}}
    failed = False
    for stage in stages:
        if failed:
            result["stages"][stage] = {"ok": False, "skipped": True}
            continue
        result["stages"][stage] = spawn_stage(stage, args, codec_dir, config_dir, log_path)
        if not result["stages"][stage]["ok"]:
            failed = True
    result["git_after"] = git_state()
    result["git_unchanged_and_clean"] = (before == result["git_after"]
                                          and not any(before[n]["status"] for n in before))
    result["artefacts"] = check_artefacts(codec_dir, args.dataset) if not failed else {}
    result["stages_ok"] = sum(1 for s in result["stages"].values() if s.get("ok"))

    # step-0 loss and finiteness from the training history
    hist_path = codec_dir / "model" / "logs" / "history_steps.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text())
        losses = [h["loss"] for h in hist]
        import math
        result["train"] = {"n_logged": len(hist), "first_loss": losses[0] if losses else None,
                           "last_loss": losses[-1] if losses else None,
                           "all_finite": all(math.isfinite(x) for x in losses),
                           "first_step": hist[0]["step"] if hist else None}
    write_json(codec_dir / "s2_result.json", result)
    print(f"{codec}: {result['stages_ok']}/3 stages ok; artefacts {result['artefacts']}; "
          f"git unchanged+clean {result['git_unchanged_and_clean']}; train {result.get('train')}", flush=True)
    return result


# Loss against step, one colour per codec with the step-0 band; per-file val loss (2 files = 2 colours) per codec
def make_figures(out_dir, codecs):
    from matplotlib.figure import Figure
    colours = {"encodec48k": "tab:blue", "codicodec": "tab:orange", "encodec24k": "tab:green", "spectrostream": "tab:red", "music2latent": "tab:purple", "codicodec_grains": "tab:brown"}
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    ax.axhspan(*STEP0_BAND, color="gray", alpha=0.25, label="step-0 band [1.8, 2.2]")
    for codec in codecs:
        p = out_dir / codec / "model" / "logs" / "history_steps.json"
        if p.exists():
            h = json.loads(p.read_text())
            ax.plot([x["step"] for x in h], [x["loss"] for x in h], color=colours[codec], label=codec, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_step.png", dpi=120)

    fig = Figure(figsize=(12, 3.5))
    axes = fig.subplots(1, len(codecs), squeeze=False)[0]
    file_colours = ["tab:blue", "tab:orange"]
    for ax, codec in zip(axes, codecs):
        p = out_dir / codec / "model" / "logs" / "val_per_file.json"
        ax.set_title(codec, fontsize=9)
        if p.exists():
            v = json.loads(p.read_text())
            names = sorted({k for e in v for k in e["per_file"]})
            for i, name in enumerate(names):
                ax.plot([e["step"] for e in v if name in e["per_file"]], [e["per_file"][name] for e in v if name in e["per_file"]],
                        marker="o", color=file_colours[i % 2], label=name)
            ax.legend(fontsize=6)
        ax.set_xlabel("step")
        ax.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("val loss")
    fig.tight_layout()
    fig.savefig(out_dir / "val_per_file.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--out")
    ap.add_argument("--codecs", nargs="+", default=list(SNIPPETS))
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--no-spectral-check", action="store_true")
    # worker options (internal)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--stage")
    ap.add_argument("--config-dir")
    ap.add_argument("--model-dir")
    ap.add_argument("--gen-dir")
    ap.add_argument("--stage-json")
    args = ap.parse_args()
    if args.worker:
        run_worker(args)
        return

    args.dataset = Path(args.dataset).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for codec in args.codecs:
        results[codec] = run_codec(args, codec, args.max_steps, out / codec)

    # D13 check: spectral_representation on encodec48k, 20 steps, training only (the representation is the same)
    spectral = None
    if "encodec48k" in args.codecs and not args.no_spectral_check:
        spectral = run_codec(args, "encodec48k", 20, out / "spectral_check",
                             extra_overrides=["training.spectral_representation = True"], stages=("training",))
    make_figures(out, args.codecs)
    summary = {c: {"stages_ok": r["stages_ok"], "git_unchanged_and_clean": r["git_unchanged_and_clean"],
                   "artefacts": r["artefacts"], "train": r.get("train")} for c, r in results.items()}
    summary["spectral_check"] = None if spectral is None else {"stages_ok": spectral["stages_ok"],
                                                                "train": spectral.get("train")}
    write_json(out / "s2_summary.json", summary)


if __name__ == "__main__":
    main()
