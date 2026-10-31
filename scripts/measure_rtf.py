"""Slice B6: S3, random-init real-time factor of atoms against autoregressive generation (design Metrics S3, R2-C3).

Protocol: one untimed session warm-up process; AB pairs in ABBA order (pair i runs atoms then AR if i is even, AR then
atoms if odd), each process a fresh subprocess; A/A control pairs (atoms-atoms and AR-AR) interleaved after every two AB
pairs. In each process: build the model with seed = seed_base + slot, output projection N(0, 0.02), an untimed 2 s
generation, then --reps timed generations. RTF = wall time / (batch * audio seconds); the clock runs from the first ODE
solve to the last audio sample on the host.

Usage (from SCAPES/):
    ../venv_SCAPES/bin/python scripts/measure_rtf.py --codec encodec24k --modes atoms autoregressive --size large \
        --seconds 10 --nfe 16 --cfg 3.0 --pairs 5 --aa-pairs 3 --seed-base 0 --reps 3 --session-warmup 60 \
        --random-init --out ../runs/s3_build
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODES = ("atoms", "autoregressive")


# ─── statistics (R2-C3) ───

def _median_rtf(p):
    return float(statistics.median(p["rtf"]))


# {pair: log(median_rtf_atoms / median_rtf_ar)} over the AB processes, paired by pair id (time slot, not seed)
def paired_log_ratios(processes):
    by_pair = {}
    for p in processes:
        if p["kind"] == "ab":
            by_pair.setdefault(p["pair"], {})[p["mode"]] = _median_rtf(p)
    return {pair: math.log(v["atoms"] / v["autoregressive"]) for pair, v in sorted(by_pair.items())
            if "atoms" in v and "autoregressive" in v}


# log(median_rtf_first / median_rtf_second) of each A/A pair (first = earlier slot)
def aa_log_ratios(processes):
    by_pair = {}
    for p in processes:
        if p["kind"] == "aa":
            by_pair.setdefault(p["pair"], []).append(p)
    out = []
    for pair, two in sorted(by_pair.items()):
        two = sorted(two, key=lambda q: q["slot"])
        if len(two) == 2:
            out.append(math.log(_median_rtf(two[0]) / _median_rtf(two[1])))
    return out


def aa_margin(aa_ratios):
    if len(aa_ratios) < 2:
        return 0.01
    return max(0.01, 2 * statistics.stdev(aa_ratios))


# 2.5 and 97.5 percentiles of bootstrap means over the units (pairs)
def bootstrap_mean_ci(values, n_resamples=10000, seed=0):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_resamples, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def classify(ci, margin):
    if ci[0] > margin:
        return "faster"
    if ci[1] < -margin:
        return "slower"
    return "indistinguishable"


# ─── environment readings ───

def gpu_state():
    q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        vals = [float(x) for x in out.split(",")]
        return dict(zip(["sm_clock_mhz", "mem_clock_mhz", "temp_c", "power_w", "util_pct"], vals))
    except Exception as e:  # nvidia-smi missing or failed: still a non-empty record
        return {"error": str(e)}


def cpu_governor():
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return "unknown"


def numeric_flags():
    import torch
    return {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


# ─── worker: one process, one mode ───

def build_engine(args, mode, seed):
    import torch
    from SCAPES.inference import FlowInference as fi
    engine = fi.FlowInference.from_config(args.codec, size=args.size, device=args.device, seed=seed)
    if args.random_init:
        # output projection N(0, 0.02): the V1 zero init would make all velocities identically zero
        torch.nn.init.normal_(engine.model.transformer.output_proj.weight, mean=0.0, std=0.02)
    if args.cfg_batch:
        patch_cfg_batch(engine.model)
    if engine.geom.atom_type != "frames":
        raise NotImplementedError("measure_rtf.py supports frame-based geometries only")
    return engine


# CFG as one batch-2 call (variant ii): same maths as FlowModel.vector_field, conditional and null in one transformer call
def patch_cfg_batch(model):
    import torch

    def vector_field(x, s, context_dict):
        memory, clap = context_dict["memory"], context_dict["clap"]
        structure, cfg = context_dict.get("structure"), context_dict.get("cfg_scale", 3.0)
        prefix = context_dict.get("prefix")
        o = 0
        if prefix is not None:
            x = torch.cat([prefix, x], dim=1)
            o = prefix.size(1)
        if cfg == 1.0:
            return model.transformer(x, memory, clap, s, structure)[:, o:]
        n = x.size(0)
        st2 = torch.cat([structure, structure], 0) if structure is not None else None
        v = model.transformer(torch.cat([x, x], 0), torch.cat([memory, memory], 0),
                              torch.cat([clap, torch.zeros_like(clap)], 0), torch.cat([s, s], 0), st2)[:, o:]
        v_cond, v_uncond = v[:n], v[n:]
        return v_uncond + cfg * (v_cond - v_uncond)

    model.vector_field = vector_field


def units_for(geom, seconds):
    frames = seconds * geom.frame_rate
    if geom.mode == "autoregressive":
        return max(1, math.ceil(frames / geom.unit_frames - 1e-6))
    return max(1, math.ceil((frames - geom.unit_frames) / geom.unit_hop_frames - 1e-6) + 1)


def make_contexts(n_streams, n_units, seed):
    import torch
    g = torch.Generator().manual_seed(1234 + seed)
    ctx = torch.randn(n_streams, 1024, generator=g)
    return ctx / ctx.norm(dim=-1, keepdim=True)


# Batched copy of the engine's generate loop (batch > 1 only; the engine itself is batch 1)
def generate_batch(engine, n, units, ctx, nfe, cfg):
    import torch
    import torch.nn.functional as F
    g, dev = engine.geom, engine.device
    D, U = g.frame_dim, g.unit_frames
    produced = []
    ctx = ctx.to(dev)
    for t in range(units):
        if g.mode == "autoregressive":
            W, o = g.memory_frames, g.prefix_frames
            past = torch.cat(produced, dim=-1)[..., -W:] if produced else torch.zeros(n, D, 0, device=dev)
            missing = W - past.shape[-1]
            if missing > 0:
                past = F.pad(past, (missing, 0))
            enc = engine.local_encoder(past.unsqueeze(1))
            if missing > 0:
                masked = (torch.arange(W, device=dev) < missing).view(1, 1, W, 1)
                enc = torch.where(masked, engine.model.null_past_embed[:, :, :W], enc)
            prefix = past[..., -o:].transpose(1, 2).contiguous() if o > 0 else None
        else:
            M = g.memory_units
            items = [torch.zeros(n, D, U, device=dev) if i < 0 else produced[i] for i in range(t - M, t)]
            enc = engine.local_encoder(torch.stack(items, dim=1))
            if M - t > 0:
                enc[:, :M - t] = engine.model.null_past_embed
            prefix = None
        x0 = torch.randn(n, U, D, device=dev)
        pred = engine.model.generate(x0=x0, encoded_past=enc, clap_context=ctx, structure_vector=None,
                                     max_nfe=nfe, cfg_scale=cfg, prefix=prefix)
        produced.append(pred.transpose(1, 2))
    return produced


def one_generation(engine, args, seed, seconds):
    """One generation. Returns (flow_s, decode_s, wall_s, audio_s_per_stream, n_units, finite)."""
    import torch
    n = args.batch
    units = units_for(engine.geom, seconds)
    ctx = make_contexts(n, units, seed)
    torch.manual_seed(seed)
    sr = engine.sr
    if n == 1:
        clap = [ctx[0]] * units
        timeline = engine.build_base_timeline([None] * units, clap, None, False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timeline = engine.generate(timeline, NFE=args.nfe, cfg_scale=args.cfg)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        audio = engine.decode_timeline(timeline)                 # ends with the host copy
        t2 = time.perf_counter()
        samples = audio.shape[-1]
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        produced = generate_batch(engine, n, units, ctx, args.nfe, args.cfg)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        audio = decode_batch_audio(engine, n, produced).cpu()
        t2 = time.perf_counter()
        samples = audio.shape[-1]
    finite = bool(torch.isfinite(audio).all())
    return t1 - t0, t2 - t1, t2 - t0, samples / sr, units, finite


# Batched decode, audio [n, C, T] on the device
def decode_batch_audio(engine, n, produced):
    import torch
    from SCAPES.inference.render import StreamRenderer, ola_render
    g, d = engine.geom, engine.geom.latent_dim
    with torch.no_grad():
        if g.mode == "autoregressive":
            r = StreamRenderer(engine.codec, batch_size=n)
            for a in produced:
                r.push(engine._unstandardise(a[:, :d, :]))
            return r.finish()
        auds, C = [], None
        for k, a in enumerate(produced):
            y = engine.codec.decode(engine._unstandardise(a[:, :d, :]).float(), None, k)   # [n, C, T]
            C = y.shape[1]
            auds.append(y.reshape(n * C, y.shape[-1]))
        out = ola_render(auds, g)                                # [n*C, T]
        return out.reshape(n, C, out.shape[-1])


def run_worker(args):
    import torch
    torch.cuda.reset_peak_memory_stats()
    engine = build_engine(args, args.mode, args.seed)
    gpu_before = gpu_state()
    # untimed warm-up: 2 s generation (loads the codec, compiles kernels)
    one_generation(engine, args, args.seed, 2.0)
    warmup_runs = 1
    torch.cuda.reset_peak_memory_stats()
    reps = []
    for r in range(args.reps):
        flow, dec, wall, audio_s, units, finite = one_generation(engine, args, args.seed + 1000 * (r + 1), args.seconds)
        if not finite:
            raise RuntimeError("non-finite audio (crash guard)")
        reps.append({"flow_s": flow, "decode_s": dec, "wall_s": wall, "audio_s": audio_s, "units": units})
    gpu_after = gpu_state()
    n = args.batch
    rtf = [r["wall_s"] / (n * r["audio_s"]) for r in reps]
    med = statistics.median
    solves = [r["units"] * n for r in reps]
    result = {
        "mode": args.mode, "seed": args.seed, "pid": os.getpid(), "warmup_runs": warmup_runs,
        "rtf": rtf, "median_rtf": med(rtf),
        "solves_per_s": med([s / r["wall_s"] for s, r in zip(solves, reps)]),
        "s_per_solve": med([r["flow_s"] / s for s, r in zip(solves, reps)]),
        "flow_s_per_audio_s": med([r["flow_s"] / (n * r["audio_s"]) for r in reps]),
        "decode_s_per_audio_s": med([r["decode_s"] / (n * r["audio_s"]) for r in reps]),
        "s_per_unit_wall": med([r["wall_s"] / (r["units"]) for r in reps]),
        "units": reps[0]["units"], "audio_s": reps[0]["audio_s"], "batch": n,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_before": gpu_before, "gpu_after": gpu_after, "flags": numeric_flags(),
    }
    Path(args.result).write_text(json.dumps(result))


# Untimed session warm-up: alternates the two modes for --session-warmup seconds so the GPU reaches thermal steady state
def run_session_warmup(args):
    import torch
    engines = {m: build_engine(args, m, args.seed_base) for m in MODES}
    t_end = time.perf_counter() + args.session_warmup
    k = 0
    while time.perf_counter() < t_end:
        one_generation(engines[MODES[k % 2]], args, args.seed_base + k, 2.0)
        k += 1
    print(f"session warm-up: {k} generations of 2 s in {args.session_warmup} s", flush=True)


# ─── parent: schedule, subprocesses, report ───

def schedule(n_pairs, n_aa):
    """Process list [(kind, pair, mode)] in time order: ABBA pairs; one A/A group (atoms-atoms, AR-AR) after every two AB pairs."""
    seq, aa_left, aa_id, g = [], n_aa, n_pairs, 0
    def aa_group():
        nonlocal aa_id, g
        order = MODES if g % 2 == 0 else MODES[::-1]
        for m in order:
            seq.extend([("aa", aa_id, m), ("aa", aa_id, m)])
            aa_id += 1
        g += 1
    for i in range(n_pairs):
        pair = ("atoms", "autoregressive") if i % 2 == 0 else ("autoregressive", "atoms")
        seq.extend(("ab", i, m) for m in pair)
        if i % 2 == 1 and aa_left > 0:
            aa_group()
            aa_left -= 1
    while aa_left > 0:
        aa_group()
        aa_left -= 1
    return seq


def common_args(args):
    a = ["--codec", args.codec, "--gin-dir", str(args.gin_dir), "--size", args.size, "--seconds", str(args.seconds),
         "--nfe", str(args.nfe), "--cfg", str(args.cfg), "--reps", str(args.reps), "--seed-base", str(args.seed_base),
         "--device", args.device, "--batch", str(args.batch), "--session-warmup", str(args.session_warmup)]
    if args.random_init:
        a.append("--random-init")
    if args.cfg_batch:
        a.append("--cfg-batch")
    return a


def spawn(cmd, log):
    t0 = time.perf_counter()
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    log.write(f"$ {' '.join(cmd)}\n{out.stdout}{out.stderr[-3000:]}\n")
    log.flush()
    if out.returncode != 0:
        raise RuntimeError(f"subprocess failed ({out.returncode}): {out.stderr[-3000:]}")
    return time.perf_counter() - t0


def run_parent(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "measure.log", "a")
    base = [sys.executable, str(Path(__file__).resolve())] + common_args(args)
    if args.session_warmup > 0:
        print(f"session warm-up {args.session_warmup} s", flush=True)
        spawn(base + ["--worker", "warmup"], log)

    processes = []
    for slot, (kind, pair, mode) in enumerate(schedule(args.pairs, args.aa_pairs)):
        res_path = out_dir / f"proc_{slot:02d}.json"
        dt = spawn(base + ["--worker", "run", "--mode", mode, "--seed", str(args.seed_base + slot),
                           "--result", str(res_path)], log)
        r = json.loads(res_path.read_text())
        r.update({"pair": pair, "slot": slot, "kind": kind})
        processes.append(r)
        print(f"slot {slot:2d} {kind} pair {pair} {mode:14s} seed {r['seed']} rtf {r['median_rtf']:.4f} "
              f"flow {r['flow_s_per_audio_s']:.4f} dec {r['decode_s_per_audio_s']:.4f} "
              f"sm {r['gpu_after'].get('sm_clock_mhz')} MHz {r['gpu_after'].get('temp_c')} C ({dt:.0f} s)", flush=True)
        res_path.unlink()

    lr = paired_log_ratios(processes)
    aa = aa_log_ratios(processes)
    margin = aa_margin(aa)
    vals = list(lr.values())
    ci = bootstrap_mean_ci(vals) if vals else (float("nan"), float("nan"))
    mean_lr = sum(vals) / len(vals) if vals else float("nan")
    result = {
        "processes": processes,
        "log_ratio_per_pair": {str(k): v for k, v in lr.items()},
        "ratio_per_pair": {str(k): math.exp(v) for k, v in lr.items()},
        "pairs_above_1": sum(v > 0 for v in vals),
        "aa_log_ratios": aa, "margin": margin, "ci95_log": list(ci),
        "speedup": math.exp(mean_lr), "speedup_ci95": [math.exp(ci[0]), math.exp(ci[1])],
        "outcome": classify(ci, margin) if vals else "indistinguishable",
        "cpu_governor": cpu_governor(), "session_warmup_s": args.session_warmup,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                   if k not in ("worker", "result", "mode", "seed")},
    }
    (out_dir / "rtf.json").write_text(json.dumps(result, indent=1))
    peak = {}
    for p in processes:
        peak[p["mode"]] = max(peak.get(p["mode"], 0), p["peak_memory_bytes"])
    (out_dir / "peak_memory.json").write_text(json.dumps(
        {"max_memory_allocated_bytes": peak, "max_gb": {k: v / 1e9 for k, v in peak.items()}}, indent=1))
    write_env(out_dir, args)
    try:
        plot(out_dir, result, args.variant_dirs)
    except Exception as e:
        log.write(f"plot failed: {e}\n")
        print(f"plot failed: {e}", flush=True)
    s = result
    print(f"ratio per pair {[round(v, 4) for v in s['ratio_per_pair'].values()]}, speedup {s['speedup']:.4f} "
          f"CI {s['speedup_ci95'][0]:.4f}-{s['speedup_ci95'][1]:.4f}, A/A margin {margin:.4f}, outcome {s['outcome']}", flush=True)


def write_env(out_dir, args):
    import torch
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha, dirty = "unknown", ""
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu_governor": cpu_governor(), "gpu_state_at_end": gpu_state(), "flags": numeric_flags(),
           "repo_sha": sha, "repo_dirty": bool(dirty), "requirements_lock": "SCAPES/requirements.lock",
           "command": " ".join(sys.argv)}
    (out_dir / "env.json").write_text(json.dumps(env, indent=1))


def plot(out_dir, result, variant_dirs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colour = {"atoms": "tab:blue", "autoregressive": "tab:red"}
    procs = result["processes"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # RTF against slot, coloured by mode, A/A marked
    a = ax[0, 0]
    for m in MODES:
        for kind, mk in (("ab", "o"), ("aa", "s")):
            pts = [(p["slot"], p["median_rtf"]) for p in procs if p["mode"] == m and p["kind"] == kind]
            if pts:
                a.scatter(*zip(*pts), c=colour[m], marker=mk, edgecolors="k" if kind == "aa" else "none",
                          label=f"{m} ({'A/A' if kind == 'aa' else 'AB'})", s=50)
    a.set_xlabel("time slot"); a.set_ylabel("median RTF"); a.set_title("RTF in time order (squares = A/A)")
    a.legend(fontsize=8)
    a2 = a.twinx()
    a2.plot([p["slot"] for p in procs], [p["gpu_after"].get("temp_c", float("nan")) for p in procs],
            c="gray", lw=0.8, ls=":")
    a2.set_ylabel("GPU temp after (C), dotted")

    # per-pair ratios, CI of the mean, margin band, outcome, variants
    a = ax[0, 1]
    ratios = list(result["ratio_per_pair"].values())
    a.scatter(range(len(ratios)), ratios, c="k", label="AB pairs")
    a.axhspan(math.exp(-result["margin"]), math.exp(result["margin"]), color="0.85", label="+- A/A margin")
    a.axhline(1.0, c="0.4", lw=0.8)
    x = len(ratios) + 0.5
    a.errorbar([x], [result["speedup"]], yerr=[[result["speedup"] - result["speedup_ci95"][0]],
                                               [result["speedup_ci95"][1] - result["speedup"]]],
               fmt="D", c="tab:green", capsize=4, label="mean and 95% CI")
    for j, vd in enumerate(variant_dirs or []):
        f = Path(vd) / "rtf.json"
        if f.exists():
            v = json.loads(f.read_text())
            a.errorbar([x + 1 + j], [v["speedup"]], yerr=[[v["speedup"] - v["speedup_ci95"][0]],
                                                          [v["speedup_ci95"][1] - v["speedup"]]],
                       fmt="^", capsize=3, label=f"variant {Path(vd).name}")
    a.set_xlabel("AB pair"); a.set_ylabel("RTF atoms / RTF AR (>1: AR faster)")
    a.set_title(f"outcome: {result['outcome']}")
    a.legend(fontsize=7)

    # stacked bars: flow and decode seconds per audio second
    a = ax[1, 0]
    for i, m in enumerate(MODES):
        ps = [p for p in procs if p["mode"] == m]
        fl = statistics.mean(p["flow_s_per_audio_s"] for p in ps)
        de = statistics.mean(p["decode_s_per_audio_s"] for p in ps)
        a.bar(i, fl, color=colour[m], label="flow" if i == 0 else None)
        a.bar(i, de, bottom=fl, color="orange", label="decode" if i == 0 else None)
    a.set_xticks([0, 1]); a.set_xticklabels(MODES); a.set_ylabel("s per audio s"); a.legend()
    a.set_title("flow and decode time per audio second")

    # per-unit wall time
    a = ax[1, 1]
    for m in MODES:
        pts = [(p["slot"], 1000 * p["s_per_unit_wall"]) for p in procs if p["mode"] == m]
        a.scatter(*zip(*pts), c=colour[m], label=m)
    a.set_xlabel("time slot"); a.set_ylabel("wall ms per block (AR) / atom (atoms)"); a.legend()
    a.set_title("per-unit wall time")
    fig.tight_layout()
    fig.savefig(out_dir / "rtf.png", dpi=120)
    plt.close(fig)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="encodec24k")
    ap.add_argument("--modes", nargs=2, default=list(MODES))
    ap.add_argument("--gin-dir", default=str(REPO / "SCAPES" / "config" / "codecs"))
    ap.add_argument("--size", default="large")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--aa-pairs", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--session-warmup", type=float, default=60)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cfg-batch", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant-dirs", nargs="*", default=[])
    ap.add_argument("--replot", action="store_true", help="redraw rtf.png of --out from its rtf.json (+ --variant-dirs)")
    # worker options (set by the parent)
    ap.add_argument("--worker", choices=["run", "warmup"], default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--result", default=None)
    a = ap.parse_args()
    if list(a.modes) != list(MODES):
        ap.error("--modes must be: atoms autoregressive")
    return a


if __name__ == "__main__":
    args = parse()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if args.worker == "run":
        run_worker(args)
    elif args.worker == "warmup":
        run_session_warmup(args)
    elif args.replot:
        plot(Path(args.out), json.loads((Path(args.out) / "rtf.json").read_text()), args.variant_dirs)
    else:
        if args.out is None:
            raise SystemExit("--out is required")
        run_parent(args)
