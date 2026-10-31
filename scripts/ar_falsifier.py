"""Slice B8: AR continuity falsifier E3b (design E3b, R1-C2).

Trains the autoregressive model for a few thousand steps on a small dataset, then measures
  L_real, L_null   deterministic val loss with the real past and with the past replaced by the null embedding,
  rho_seam_gen     seam ratio of the generated latents (2 val prompts x seeds x seconds),
  rho_seam_real    seam ratio of the val-region latents of the same files (same block phase),
and writes falsifier.json. Pass iff L_real < 0.98 L_null and rho_seam_gen <= 2 rho_seam_real.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/ar_falsifier.py --dataset ../datasets/micro_microtex --codec encodec24k \
        --prefix-seconds 0 --steps 2000 --out ../runs/s3b_prefix0
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))


# Run config. Default: dataset gin minus its geometry lines + codec snippet + overrides (build_run_config).
# With --config-dir: that folder's files as they are, plus the three lines this tool owns.
def make_config(args, out_dir):
    from SCAPES.pipeline import build_run_config
    prefix = f"ar.prefix_seconds = {args.prefix_seconds}"
    steps = [f"training.max_steps = {args.steps}", "training.seed = 0"]
    cfg_dir = out_dir / "config"
    if args.config_dir is None:
        snippet = REPO / "SCAPES" / "config" / "codecs" / f"{args.codec}_ar.gin"
        overrides = ['model.size = "small"', "training.batch_size = 128", "training.log_every = 10",
                     "splits.val_split_ratio = 0.1", "splits.reserve_ratio = 0.1", prefix] + steps
        return build_run_config(Path(args.dataset) / "config", [snippet], overrides, cfg_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.config_dir)
    for name, extra in (("dataprep.gin", [prefix]), ("training.gin", steps)):
        (cfg_dir / name).write_text((src / name).read_text().rstrip("\n") + "\n" + "\n".join(extra) + "\n")
    if (src / "cherry_picking.csv").exists():
        (cfg_dir / "cherry_picking.csv").write_text((src / "cherry_picking.csv").read_text())
    return cfg_dir


# Latents (latent channels only, model units) of the generated clip for one prompt and seed: [d, T]
def generate_latents(engine, prompt, seed, n_blocks, nfe):
    import torch
    torch.manual_seed(seed)
    clap = [torch.as_tensor(prompt).reshape(-1).float() for _ in range(n_blocks)]
    timeline = engine.build_base_timeline([None] * n_blocks, clap, None, default_TF=False)
    timeline = engine.generate(timeline, NFE=nfe, cfg_scale=3.0)
    blocks = [step["atom_generated"][0, :engine.geom.latent_dim].cpu() for step in timeline]
    for b in blocks:
        assert b.shape[-1] == engine.geom.unit_frames, f"generated block has {b.shape[-1]} frames"
    return torch.cat(blocks, dim=-1)


def evaluate(args, trainer, model_dir, out_dir):
    import numpy as np
    import torch
    from SCAPES.inference.FlowInference import FlowInference
    from SCAPES.inference.diagnostics import seam_ratio

    result = {}
    # (i) deterministic val loss with the real past and with the null past
    usage = trainer.evaluate_past_usage(trainer.val_loader.dataset)
    result.update({k: float(v) for k, v in usage.items()})

    # (ii) generation from the last checkpoint (the weights that were just evaluated)
    engine = FlowInference(model_dir, checkpoint="last")
    g = engine.geom
    B = g.unit_frames
    if g.atom_type != "frames":
        raise NotImplementedError("ar_falsifier.py supports frame-based geometries only")
    n_blocks = max(2, math.ceil(args.gen_seconds * g.frame_rate / B))
    dataset = trainer.dataset
    regions = dataset._regions

    # per file: first val sample (prompt and start frame) and the val-region latents from that frame on
    gen_rhos, real_rhos, per_clip = [], [], []
    seen = set()
    for idx, (fname, start) in enumerate(dataset.all_indices):
        if regions[idx] != "val" or fname in seen:
            continue
        seen.add(fname)
        prompt = dataset[idx]["target_semantic"]
        val_starts = [t for i, (f, t) in enumerate(dataset.all_indices) if f == fname and regions[i] == "val"]
        last_t = max(val_starts)
        z_real = dataset._latent[fname][:g.latent_dim, start:last_t + B].float()
        if engine.mu is not None:
            z_real = (z_real - engine.mu.cpu().view(-1, 1)) / engine.std.cpu().view(-1, 1)
        rho_real = seam_ratio(z_real, B, phase=start % B)
        real_rhos.append(rho_real)
        for seed in args.gen_seeds:
            z_gen = generate_latents(engine, prompt, seed, n_blocks, nfe=16)
            rho_gen = seam_ratio(z_gen, B, phase=0)
            gen_rhos.append(rho_gen)
            per_clip.append({"file": fname, "seed": seed, "rho_seam_gen": rho_gen, "frames": int(z_gen.shape[-1]),
                             "latent_std": float(z_gen.std())})
        per_clip.append({"file": fname, "rho_seam_real": rho_real, "real_frames": int(z_real.shape[-1]),
                         "real_start_frame": int(start)})
    result["rho_seam_gen"] = float(np.mean(gen_rhos))
    result["rho_seam_real"] = float(np.mean(real_rhos))
    result["rho_seam_gen_all"] = gen_rhos
    result["rho_seam_real_all"] = real_rhos
    result["clips"] = per_clip
    result["block_frames"] = B
    result["prefix_frames"] = g.prefix_frames
    return result


def make_figure(out_dir, history, result):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 3.8))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot([h["step"] for h in history], [h["loss"] for h in history], color="tab:blue", linewidth=1.2, label="train loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    names = ["L_real", "L_null"]
    ax2.bar(names, [result["L_real"], result["L_null"]], color=["tab:green", "tab:red"])
    ax2.axhline(0.98 * result["L_null"], color="k", linestyle="--", label="0.98 L_null")
    ax2.set_ylabel("val loss")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "falsifier.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--prefix-seconds", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-dir", default=None)
    ap.add_argument("--gen-seconds", type=float, default=10.0)
    ap.add_argument("--gen-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    import torch
    from SCAPES.pipeline import build_trainer, run_dataprep

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model"
    cfg_dir = make_config(args, out_dir)
    t0 = time.time()

    # representation: reused when present (whole-file latents do not depend on the prefix)
    run_dataprep(args.dataset, config_dir=cfg_dir)
    t_prep = time.time() - t0

    # training, with early stopping off so that the run has exactly --steps steps
    t0 = time.time()
    trainer = build_trainer(args.dataset, model_dir, config_dir=cfg_dir)
    trainer.max_steps = args.steps
    trainer.train(epochs=args.steps, audio_val_freq=10 ** 9, val_nfe=8, patience=10 ** 9)
    t_train = time.time() - t0
    peak_train = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    # evaluation
    t0 = time.time()
    result = evaluate(args, trainer, model_dir, out_dir)
    t_eval = time.time() - t0
    result["ratio"] = result["L_real"] / result["L_null"]
    result["pass_loss"] = bool(result["L_real"] < 0.98 * result["L_null"])
    result["pass_seam"] = bool(result["rho_seam_gen"] <= 2 * result["rho_seam_real"])
    result["pass"] = bool(result["pass_loss"] and result["pass_seam"])

    hist_path = model_dir / "logs" / "history_steps.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    finite = all(math.isfinite(h["loss"]) for h in history)
    result.update({
        "codec": args.codec, "prefix_seconds": args.prefix_seconds, "steps": args.steps,
        "last_logged_step": history[-1]["step"] if history else None,
        "first_loss": history[0]["loss"] if history else None, "last_loss": history[-1]["loss"] if history else None,
        "all_train_losses_finite": finite,
        "seconds": {"dataprep": t_prep, "train": t_train, "eval": t_eval},
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "peak_gpu_bytes_train": peak_train,
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "python": sys.version.split()[0]},
    })
    write_json(out_dir / "falsifier.json", result)
    if history:
        make_figure(out_dir, history, result)
    print(json.dumps({k: result[k] for k in ("L_real", "L_null", "ratio", "rho_seam_gen", "rho_seam_real", "pass")}), flush=True)


if __name__ == "__main__":
    main()
