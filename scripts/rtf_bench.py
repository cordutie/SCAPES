"""Slice B15: generation-speed benchmark per block, for real-time porting.

For each model (large: runs/s5/<codec>/model, best; small: runs/s5_small/<codec>/model; encodec48k large is the symlink
runs/s5/encodec48k/model) the steady-state cost of ONE generation unit is timed over many calls, including the codec decode of the
new audio:
  AR codecs: one block of B frames; flow (past encoding + NFE ODE steps) then the codec's stateful streaming decoder push,
             the new audio is the block (B*h samples), copied to the host.
  atoms:     one atom; flow then codec decode of the whole atom, windowed overlap-add into a tail buffer, the new audio is the
             hop H (H*h samples), copied to the host.
The past is real (generated blocks of the model itself: the AR window / the M previous atoms), the CLAP context is a fixed random unit vector
(timing does not depend on its values). Batch 1, fp32 as trained; a bf16 autocast of the flow part is an extra column (quality untested).
Configs per model: nfe 32 / cfg 3, nfe 16 / cfg 3, nfe 16 / cfg off (cfg_scale 1.0: a single forward per step), and bf16 for the first two.
Per call: torch.cuda.synchronize() before and after each part. RTF_new = time / seconds of NEW audio (the one to judge); RTF_generated = time / seconds of the whole generated unit; the real-time requirement is
RTF < 1 per block with margin (headroom = 1 / RTF).

Usage (from SCAPES/, GPU idle, chain finished):
    ../venv_SCAPES/bin/python scripts/rtf_bench.py --out ../runs/rtf_bench
    ../venv_SCAPES/bin/python scripts/rtf_bench.py --dry           # lists the models and configs only
Refuses to run while another process uses the GPU or the chain (runs/s5_small/chain.log) is not done, unless --force.
Outputs: rtf_bench.json, rtf_bench.md, rtf_bench.png.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO.parent / "runs"
sys.path.insert(0, str(REPO))
CODECS = ["encodec24k", "spectrostream", "encodec48k"]
# (label, nfe, cfg scale, bf16)
CONFIGS = [("nfe32_cfg3", 32, 3.0, False), ("nfe16_cfg3", 16, 3.0, False), ("nfe16_cfgoff", 16, 1.0, False),
           ("nfe32_cfgoff", 32, 1.0, False), ("nfe32_cfg3_bf16", 32, 3.0, True), ("nfe16_cfg3_bf16", 16, 3.0, True)]


# Folders of the trained runs of each size (canonical names; the old runs keep their folders)
TRAINED = {"large": "s5", "small": "s5_small", "tiny": "s5_tiny", "petite": "s5_petite"}


def models(args):
    out = {}
    if args.trained:     # trained models of the given sizes: runs/<folder>/<codec>/model
        return {f"{c}_{size}": str(RUNS / TRAINED[size] / c / "model") for size in args.trained for c in CODECS}
    if args.random_init:      # untrained weights of any size, timing only
        return {f"{c}_{s}_random": f"random:{c}:{s}" for s in args.sizes for c in CODECS}
    for size, runs in (("large", args.large_runs), ("small", args.small_runs)):
        for c in CODECS:
            out[f"{c}_{size}"] = str(Path(runs) / c / "model")
    return out


def gpu_info():
    q = "name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout.strip().splitlines()[0].split(", ")
        return dict(zip(q.split(","), out))
    except Exception as e:
        return {"error": str(e)}


# CPU record for the CPU benchmark: model, logical cores, torch threads, load average
def cpu_info():
    import os
    import torch
    model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"name": model, "logical_cores": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(), "load_avg_1_5_15": list(os.getloadavg())}


def device_info(device):
    return cpu_info() if str(device) == "cpu" else gpu_info()


def other_gpu_processes():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return [x for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def summary(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "min": float(x.min()), "max": float(x.max()),
            "n": int(len(x))}


# One unit of generation from the given past units (list of [1, frame_dim, F], most recent last) -> [1, frame_dim, F].
# Same calls as FlowInference.generate for one step (non-spectral models): past encoding, then the ODE solve.
def generate_unit(engine, past_units, clap, nfe, cfg, bf16=False):
    import torch
    g = engine.geom
    steps = [{"TF": True, "atom_given": u, "atom_generated": None} for u in past_units]
    t = len(steps)
    ctx = clap.to(engine.device)
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)
    n = past_units[-1].shape[0] if past_units else 1       # batch (AR only for n > 1)
    if n > 1:
        ctx = ctx.expand(n, -1)
    with torch.no_grad(), torch.autocast("cuda" if str(engine.device).startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=bf16):
        encoded_past, prefix = engine._ar_past(steps, t) if g.mode == "autoregressive" else engine._atoms_past(steps, t)
        x0 = torch.randn(n, g.unit_frames, g.frame_dim, device=engine.device)
        pred = engine.model.generate(x0=x0, encoded_past=encoded_past, clap_context=ctx, structure_vector=None, max_nfe=nfe,
                                     cfg_scale=cfg, prefix=prefix)
    return pred.float().transpose(1, 2)


# Decoder state of the real-time path: AR streaming decoder, or the atoms overlap-add tail
class Decoder:
    def __init__(self, engine, batch=1):
        import torch
        from SCAPES.data.geometry import crossfade_window
        self.engine, self.g = engine, engine.geom
        if self.g.mode == "autoregressive":
            self.dec = engine.codec.streaming_decoder(batch)
        else:
            self.window = crossfade_window(self.g).to(engine.device)
            self.tail, self.k = None, 0
        self.new_samples = self.g.unit_frames * self.g.hop_length if self.g.mode == "autoregressive" else self.g.hop_samples

    @property
    def generated_seconds(self):          # the whole audio of one generated unit (atom, or the AR block)
        return self.g.unit_frames * self.g.hop_length / self.engine.sr

    @property
    def new_seconds(self):
        return self.new_samples / self.engine.sr

    # decodes one generated unit, returns the new audio on the host
    def push(self, unit):
        import torch
        e, g, d = self.engine, self.g, self.g.latent_dim
        with torch.no_grad():
            if g.mode == "autoregressive":
                return self.dec.push(e._unstandardise(unit[:, :d, :])).cpu()
            z = e._unstandardise(unit[:, :d, :]).float()
            scale = torch.abs(unit[:, d, :]).mean(dim=-1).reshape(-1).float() if g.has_scale else None
            audio = e.codec.decode(z, scale, self.k)[0] * self.window
            self.k += 1
            if self.tail is None:
                self.tail = torch.zeros_like(audio)
            self.tail = self.tail + audio
            out = self.tail[:, :self.new_samples]
            self.tail = torch.cat([self.tail[:, self.new_samples:], torch.zeros_like(audio[:, :self.new_samples])], dim=-1)
            return out.cpu()


def g_mode_is_atoms(model_dir, args):
    name = Path(model_dir).parts
    return any(c == "encodec48k" for c in name) or "encodec48k" in str(model_dir)


def bench_model(args, name, model_dir):
    import torch
    from SCAPES.inference.FlowInference import FlowInference
    batch = getattr(args, "batch", 1)
    if batch > 1 and g_mode_is_atoms(model_dir, args):
        raise SystemExit("--batch > 1 is implemented for the autoregressive codecs only")
    random_init = model_dir.startswith("random:")
    if random_init:
        import size_params
        _, rcodec, rsize = model_dir.split(":")
        engine = size_params.build(rcodec, rsize, device=args.device)
    else:
        engine = FlowInference(model_dir, device=args.device, checkpoint="best")
    g = engine.geom
    cuda = str(args.device).startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    gen = torch.Generator().manual_seed(1234)
    clap = torch.randn(1024, generator=gen)
    clap = clap / clap.norm()
    need = g.memory_units if g.mode == "atoms" else math.ceil(g.memory_frames / g.unit_frames)
    result = {"model": name, "model_dir": model_dir, "random_weights": random_init, "codec": g.codec, "mode": g.mode, "unit_frames": g.unit_frames,
              "unit_hop_frames": g.unit_hop_frames, "memory_units_needed": need, "sample_rate": engine.sr,
              "params": sum(p.numel() for p in list(engine.model.parameters()) + list(engine.local_encoder.parameters())),
              "precision": "fp32 weights and activations (bf16 columns: autocast of the flow part only)",
              "batch": batch, "device": str(args.device), "gpu_start": device_info(args.device), "configs": {}}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    wanted = getattr(args, "configs", None)
    for label, nfe, cfg, bf16 in CONFIGS:
        if bf16 and args.no_bf16:
            continue
        if wanted and label not in wanted:
            continue
        torch.manual_seed(0)
        dec = Decoder(engine, batch)
        past = [torch.randn(batch, g.frame_dim, g.unit_frames, device=engine.device)] if batch > 1 else []
        # fill the real past (untimed), then 3 discarded warm-up blocks
        for _ in range(need + 1 + args.warmup):
            unit = generate_unit(engine, past[-need:], clap, nfe, cfg, bf16)
            dec.push(unit)
            past.append(unit)
        flow, decode = [], []
        for _ in range(args.calls):
            sync()
            t0 = time.perf_counter()
            unit = generate_unit(engine, past[-need:], clap, nfe, cfg, bf16)
            sync()
            t1 = time.perf_counter()
            audio = dec.push(unit)
            sync()
            t2 = time.perf_counter()
            past.append(unit)
            flow.append(t1 - t0)
            decode.append(t2 - t1)
        finite = bool(torch.isfinite(audio).all()) and bool(torch.isfinite(unit).all())
        total = np.array(flow) + np.array(decode)
        dur = dec.new_seconds
        gdur = dec.generated_seconds
        result["configs"][label] = {
            "nfe": nfe, "cfg_scale": cfg, "cfg_on": cfg != 1.0, "bf16": bf16, "calls": args.calls, "new_audio_seconds": dur,
            "generated_audio_seconds": gdur, "overlap_seconds": gdur - dur, "rtf_new": summary(total / dur),
            "rtf_generated": summary(total / gdur), "rtf_generated_over_rtf_new": float(dur / gdur),
            "batch": batch, "rtf_per_stream": summary(total / (dur * batch)),
            "per_block_seconds": summary(total), "rtf": summary(total / dur), "flow_seconds": summary(flow),
            "decode_seconds": summary(decode), "flow_rtf": summary(np.array(flow) / dur), "decode_rtf": summary(np.array(decode) / dur),
            "headroom_x": float(dur / total.mean()), "latency_below_block_duration": bool(total.max() < dur), "finite": finite}
        print(f"[{name}] {label}: block {dur:.3f} s, time {total.mean():.4f} s, RTF {total.mean() / dur:.3f} (flow {np.mean(flow):.4f}, decode {np.mean(decode):.4f})", flush=True)
    result["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated()) if cuda else 0
    result["gpu_end"] = device_info(args.device)
    result["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "threads": torch.get_num_threads(),
                     "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                     "cudnn_benchmark": torch.backends.cudnn.benchmark, "matmul_precision": torch.get_float32_matmul_precision()}
    return result


def write_outputs(out: Path, res: dict):
    (out / "rtf_bench.json").write_text(json.dumps(res, indent=1))
    L = ["# Generation speed per block (steady state, batch 1, flow + codec decode of the new audio)", ""]
    if any(m.get("random_weights") for m in res["models"].values()):
        L += ["**Some or all models are RANDOM-INIT (untrained) weights: timing only, the audio has no quality meaning.**", ""]
    L += [
         "Two factors: RTF_new = time per call / duration of only the NEW audio (AR: the block; atoms: the hop = atom minus overlap) is the one "
         "to judge real time (rtf column; headroom = 1 / RTF_new; RTF_new <= 0.5 gives 2x); RTF_generated = time per call / duration of the whole audio "
         "produced (AR: the same; atoms: the whole atom, so smaller by hop / atom). Block = one AR block, or the hop of one atom. fp32; bf16 rows autocast the flow part only (quality untested). "
         "cfg off = cfg_scale 1.0 (one forward per ODE step instead of two). Quality at nfe 16 against nfe 32 was NOT measured here "
         "(follow-up: the teacher-forced MSS/KAD at both).", "",
         "| model | config | new audio (s) | generated audio (s) | overlap (s) | time mean +- std (ms) | min / max (ms) | RTF_new (judge this) | RTF_generated | RTF_gen / RTF_new | flow RTF_new | decode RTF_new | headroom | max < new audio |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in res["models"].items():
        for label, c in m["configs"].items():
            L.append(f"| {name} | {label} | {c['new_audio_seconds']:.3f} | {c['generated_audio_seconds']:.3f} | {c['overlap_seconds']:.3f} | {c['per_block_seconds']['mean'] * 1e3:.1f} +- {c['per_block_seconds']['std'] * 1e3:.1f} | "
                     f"{c['per_block_seconds']['min'] * 1e3:.1f} / {c['per_block_seconds']['max'] * 1e3:.1f} | {c['rtf']['mean']:.3f} | "
                     f"{c['rtf_generated']['mean']:.3f} | {c['rtf_generated_over_rtf_new']:.3f} | {c['flow_rtf']['mean']:.3f} | {c['decode_rtf']['mean']:.3f} | {c['headroom_x']:.2f}x | {c['latency_below_block_duration']} |")
    first = next(iter(res["models"].values()))
    if first.get("device") == "cpu":
        c0, c1 = res["gpu_start"], res["gpu_end"]
        L += ["", f"CPU: {c0.get('name')}, {c0.get('logical_cores')} logical cores, torch threads {first['env']['threads']}, load average (1/5/15 min) "
              f"before {c0.get('load_avg_1_5_15')} and after {c1.get('load_avg_1_5_15')}; {res.get('calls')} timed blocks after {res.get('warmup')} warm-up blocks; "
              f"torch {first['env']['torch']}. Quality with cfg off is NOT measured.", ""]
        (out / "rtf_bench.md").write_text("\n".join(L))
    else:
      L += ["", f"GPU: {res['gpu_start'].get('name')} driver {res['gpu_start'].get('driver_version')}; at start SM clock {res['gpu_start'].get('clocks.sm')} MHz, "
          f"temperature {res['gpu_start'].get('temperature.gpu')} C; at end {res['gpu_end'].get('clocks.sm')} MHz, {res['gpu_end'].get('temperature.gpu')} C. "
          f"torch {first['env']['torch']}, CUDA {first['env']['cuda']}, TF32 matmul {first['env']['tf32_matmul']}, cudnn TF32 {first['env']['tf32_cudnn']}.", ""]
      (out / "rtf_bench.md").write_text("\n".join(L))
    from matplotlib.figure import Figure
    names = list(res["models"])
    have = set.intersection(*[set(m["configs"]) for m in res["models"].values()])
    labels = [l for l in ("nfe32_cfg3", "nfe16_cfg3") if l in have] or [l for l in ("nfe32_cfgoff", "nfe16_cfgoff") if l in have]
    fig = Figure(figsize=(13, 5))
    ax = fig.subplots()
    w = 0.38
    for j, label in enumerate(labels):
        fl = [res["models"][n]["configs"][label]["flow_rtf"]["mean"] for n in names]
        de = [res["models"][n]["configs"][label]["decode_rtf"]["mean"] for n in names]
        x = np.arange(len(names)) + (j - 0.5) * w
        ax.bar(x, fl, w, color="tab:blue" if j == 0 else "tab:cyan", label=f"flow, {label}")
        ax.bar(x, de, w, bottom=fl, color="tab:red" if j == 0 else "tab:orange", label=f"decode, {label}")
    for j, label in enumerate(labels):      # RTF_generated (whole audio of the unit) as a marker on each bar
        rg = [res["models"][n]["configs"][label]["rtf_generated"]["mean"] for n in names]
        ax.scatter(np.arange(len(names)) + (j - 0.5) * w, rg, color="k", marker="D" if j == 0 else "o", zorder=5, s=28,
                   label=f"RTF_generated, {label}")
    ax.axhline(1.0, color="k", linestyle="--", label="RTF 1.0")
    ax.axhline(0.5, color="grey", linestyle=":", label="RTF 0.5")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("RTF_new (bars; judge real time) and RTF_generated (markers)")
    ax.set_title(f"RTF_new per model: {labels[0]} (left bar) and {labels[1]} (right bar), fp32")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "rtf_bench.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large-runs", default=str(RUNS / "s5"))
    ap.add_argument("--small-runs", default=str(RUNS / "s5_small"))
    ap.add_argument("--out", default=str(RUNS / "rtf_bench"))
    ap.add_argument("--calls", type=int, default=None, help="timed blocks (default 20 on GPU, 5 on CPU)")
    ap.add_argument("--warmup", type=int, default=None, help="discarded warm-up blocks (default 3 on GPU, 2 on CPU)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (cpu: batch 1, fp32, default configs nfe16_cfgoff and nfe32_cfgoff, default out runs/rtf_bench_cpu)")
    ap.add_argument("--configs", nargs="+", default=None, choices=[c[0] for c in CONFIGS], help="subset of configs (default all, or the two cfg-off configs on CPU)")
    ap.add_argument("--threads", type=int, default=None, help="torch threads (CPU; default: half the logical cores, fixed and reported)")
    ap.add_argument("--batch", type=int, default=1, help="parallel streams (autoregressive codecs only); labelled as the launch-bound check, not real-time latency")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="model names such as encodec24k_small (default all 8)")
    ap.add_argument("--worker", default=None, help="internal: bench this model name in this process")
    ap.add_argument("--trained", nargs="+", choices=list(TRAINED), default=None,
                    help="bench the trained models of these sizes (large small tiny petite) instead of the default large+small pair; "
                         "bf16 is off unless --bf16 is given; default --out is runs/rtf_bench_all")
    ap.add_argument("--bf16", action="store_true", help="with --trained: add the bf16 columns")
    ap.add_argument("--random-init", action="store_true", help="time untrained (random-weight) models of the sizes in --sizes: timing only, no quality meaning")
    ap.add_argument("--sizes", nargs="+", default=["petite", "tiny", "small", "medium", "large", "extra_large", "audiobox"],
                    help="with --random-init: size names of scripts/size_params.py")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cpu = str(a.device) == "cpu"
    if a.calls is None:
        a.calls = 5 if cpu else 20
    if a.warmup is None:
        a.warmup = 2 if cpu else 3
    if cpu:
        import os
        import torch
        a.threads = a.threads or max(1, (os.cpu_count() or 2) // 2)
        torch.set_num_threads(a.threads)
        a.no_bf16 = True
        a.configs = a.configs or ["nfe16_cfgoff", "nfe32_cfgoff"]
        if a.out == str(RUNS / "rtf_bench"):
            a.out = str(RUNS / ("rtf_bench_cpu"))
    if a.trained:
        a.no_bf16 = a.no_bf16 or not a.bf16
        if a.out == str(RUNS / "rtf_bench"):
            a.out = str(RUNS / "rtf_bench_all")
    allm = models(a)
    names = a.only or list(allm)
    if a.worker:
        out = Path(a.out)
        res = bench_model(a, a.worker, allm[a.worker])
        (out / f"part_{a.worker}.json").write_text(json.dumps(res, indent=1))
        return
    if a.dry:
        for n in names:
            print(n, allm[n], "random weights" if a.random_init else
                  ("exists" if (Path(allm[n]) / "checkpoints" / "inference.gin").exists() else "MISSING"))
        print("configs:", [c[0] for c in CONFIGS], "calls", a.calls, "warmup", a.warmup)
        return
    if not a.force:
        if cpu:
            import os
            load = os.getloadavg()[0]
            if load > 1.5:
                sys.exit(f"the machine is not idle (1-min load average {load:.1f}); use --force to override")
        else:
            busy = other_gpu_processes()
            if busy:
                sys.exit("another process uses the GPU: " + "; ".join(busy) + " (use --force to override)")
        log = Path(a.small_runs) / "chain.log"
        if a.trained:
            missing = [z for z in a.trained if z in ("tiny", "petite") and not (RUNS / TRAINED[z] / "chain_done_marker").exists()]
            if missing:
                sys.exit(f"the training chain of {missing} has not finished (no chain_done_marker); use --force to override")
        elif not (log.exists() and "chain done" in log.read_text()):
            sys.exit("the training chain has not finished (no 'chain done' in chain.log); use --force to override")
    out = Path(a.out)
    if out.exists() and (out / "rtf_bench.json").exists():
        sys.exit(f"{out} already has results")
    out.mkdir(parents=True, exist_ok=True)
    gpu0 = device_info(a.device)
    res = {"models": {}, "gpu_start": gpu0, "calls": a.calls, "warmup": a.warmup}
    for n in names:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", n, "--out", str(out), "--calls", str(a.calls),
               "--warmup", str(a.warmup), "--device", a.device, "--batch", str(a.batch)] + (["--threads", str(a.threads)] if cpu else []) + \
              (["--configs", *a.configs] if a.configs else []) + ["--large-runs", a.large_runs, "--small-runs", a.small_runs] + \
              (["--no-bf16"] if a.no_bf16 else []) + (["--trained", *a.trained] if a.trained else []) + (["--random-init", "--sizes", *a.sizes] if a.random_init else [])
        with open(out / f"{n}.log", "w") as fh:
            code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        if code != 0:
            print(f"{n}: worker failed (exit {code}), see {out / (n + '.log')}", flush=True)
            continue
        res["models"][n] = json.loads((out / f"part_{n}.json").read_text())
        print(n, "done", flush=True)
    res["gpu_end"] = device_info(a.device)
    write_outputs(out, res)


if __name__ == "__main__":
    main()
