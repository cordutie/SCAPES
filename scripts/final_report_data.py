"""Final report data for 16 models (4 codecs x petite, tiny, small, large): training times, teacher-forced MSS/KAD at 16 and 32 ODE
steps without CFG, sound examples, RTF on GPU and CPU. One driver, stage by stage; all outputs go to ../final/ (data/ and audio/);
scratch goes to ../runs/final_work/.

    cd SCAPES
    ../venv_SCAPES/bin/python scripts/final_report_data.py training          # 0: training times (cheap, no GPU)
    ../venv_SCAPES/bin/python scripts/final_report_data.py metrics-audio     # 1 + 3: teacher-forced eval (32 evals) + semantic resynthesis + wavs
    ../venv_SCAPES/bin/python scripts/final_report_data.py rtf-gpu           # 2: speed benchmark on the GPU (idle GPU)
    ../venv_SCAPES/bin/python scripts/final_report_data.py rtf-cpu           # 2: the same on the CPU (idle machine)

Protocol: best checkpoint of every model (path and sha256 recorded in data/provenance.json), NFE 16 and 32, cfg_scale 1.0 (no CFG, verified by
counting the forward passes of the vector field), seed 0. Teacher forcing and semantic resynthesis use the first 10 s of the files of
the training dataset (train region of the files). Sound examples: forest, park, rain, tokio.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROOT = REPO.parent
RUNS = ROOT / "runs"
FINAL = ROOT / "final"
WORK = RUNS / "final_work"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

SIZES = {"petite": "s5_petite", "tiny": "s5_tiny", "small": "s5_small", "large": "s5"}     # size order
CODECS = ["encodec24k", "spectrostream", "codicodec", "encodec48k"]
MODE = {"encodec24k": "ar", "spectrostream": "ar", "codicodec": "atoms", "encodec48k": "atoms"}
AUDIO_FILES = ["forest", "park", "rain", "tokio"]
NFES = [16, 32]
CFG = 1.0
SEED = 0
DATASET = ROOT / "datasets" / "microtex"
ALIAS = {"petite": "pp", "tiny": "p", "small": "mp", "large": "f"}


def models():
    return [(c, s) for s in SIZES for c in CODECS]


def model_dir(codec, size):
    return RUNS / SIZES[size] / codec / "model"


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()[:12]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def boot(values, n=10000, seed=0):
    v = np.asarray(values, float)
    m = v[np.random.RandomState(seed).randint(0, len(v), (n, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": len(v)}


# ─── stage 0: training times ───

def chain_wall_minutes(codec, size):
    import datetime as dt
    log = RUNS / ("s5_small/chain.log" if size == "small" else "chain_petite_tiny.log" if size in ("petite", "tiny") else "none")
    if not log.exists():
        return None
    start = end = None
    for line in log.read_text().splitlines():
        m = re.match(r"(\S+) (start|end) (?:(\w+) )?(\w+)", line)
        if not m:
            continue
        when, kind, sz, c = m.groups()
        if c != codec and sz != codec:
            continue
        if size in ("petite", "tiny") and sz != size and c != codec:
            continue
        if size in ("petite", "tiny") and (sz if sz in SIZES else None) != size:
            continue
        t = dt.datetime.strptime(when, "%Y-%m-%d_%H:%M:%S")
        if kind == "start":
            start = t
        elif kind == "end":
            end = t
    return (end - start).total_seconds() / 60 if start and end else None


def stage_training():
    import val_split_report as vsr
    rows = []
    for codec, size in models():
        md = model_dir(codec, size)
        real = md.resolve()
        run = RUNS / SIZES[size] / codec
        hist = vsr.history(md)
        rep = vsr.report(run)
        parts = len([d for d in vsr.log_dirs(md) if (d / "history_steps.json").exists()])
        lh = load(md / "loss" / "loss_history.json") or {}
        n_epochs = len((lh.get("train") or {}).get("total") or [])
        log_text = ""
        for name in ("train.log", "train_part2.log"):
            for base in (run, real.parent):
                if (base / name).exists():
                    log_text += (base / name).read_text(errors="ignore")
                    break
        m = re.search(r"Early stopping at (epoch|step) (\d+) \(no improvement for (\d+) (epochs|validations)\)", log_text)
        env = load(md / "logs" / "env.json") or load(md / "logs_part1" / "env.json") or {}
        exit_code = (run / "exit_code").read_text().strip() if (run / "exit_code").exists() else None
        cfg_epochs = None
        for line in (run / "config" / "training.gin").read_text().splitlines():
            if line.startswith("training.epochs"):
                cfg_epochs = int(line.split("=")[1].split("#")[0])
        best = rep.get("best_by_total", {})
        rows.append({
            "codec": codec, "size": size, "size_alias": ALIAS[size], "run_dir": str(real.parent),
            "wall_minutes_history_clock": hist[-1]["time"] / 60 if hist else None,
            "wall_minutes_chain_log": chain_wall_minutes(codec, size),
            "parts": parts, "total_steps_logged": hist[-1]["step"] if hist else None, "epochs": n_epochs,
            "validations": rep.get("n_validations"), "best_validation_index": best.get("index"), "best_step": best.get("step"),
            "val_loss_at_best": best.get("val_total"),
            "stop_rule": f"patience {m.group(3)} {m.group(4)}" if m else None,
            "stopped_by_patience": bool(m), "stop_at": f"{m.group(1)} {m.group(2)}" if m else None,
            "epoch_cap_in_config": cfg_epochs, "cap_limited": bool(cfg_epochs and m is None),
            "exit_code": exit_code, "gpu": env.get("gpu"), "torch": env.get("torch"),
            "source": "logs/history_steps.json (merged over parts), loss/loss_history.json, val_per_file.json, train.log, chain logs"})
    (FINAL / "data").mkdir(parents=True, exist_ok=True)
    (FINAL / "data" / "training_times.json").write_text(json.dumps(rows, indent=1))
    L = ["# Training times", "",
         "Wall clock = the run's own clock at its last logged step (includes validation and checkpoint writes), from logs/history_steps.json; the second column "
         "is the chain log's start-to-end (includes process start and the end of the run) where a chain log exists (small, tiny, petite). GPU: NVIDIA GeForce "
         "RTX 4090 for every run. Every run trained until patience fired (10 epochs, or 10 validations of 100 steps for codicodec), best + last checkpoints kept. "
         "Large encodec48k is the V1-style run trained in two parts (part 1 stopped at the 120-epoch cap, part 2 resumed and stopped by patience); its times add up "
         "(5869 s + 3943 s). For codicodec an epoch is 15 steps and validation happens every 100 steps; the best index is a validation index.", "",
         "| codec | size | wall (min, run clock) | wall (min, chain log) | steps | epochs | best validation index | best step | val loss at best | stop | cap in config | cap-limited | exit |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, d=1: "n/a" if x is None else f"{x:.{d}f}"
    for r in rows:
        L.append(f"| {r['codec']} | {r['size']} ({r['size_alias']}) | {f(r['wall_minutes_history_clock'])} | {f(r['wall_minutes_chain_log'])} | {r['total_steps_logged']} | "
                 f"{r['epochs']} | {r['best_validation_index']} | {r['best_step']} | {f(r['val_loss_at_best'], 4)} | {r['stop_rule']} at {r['stop_at']} | "
                 f"{r['epoch_cap_in_config']} | {r['cap_limited']} | {r['exit_code']} |")
    (FINAL / "data" / "training_times.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ─── stage 1 + 3: teacher-forced metrics and sound examples ───

def run(cmd, log):
    with open(log, "w") as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def count_vector_field_calls(engine, nfe, cfg):
    """Forward passes of the vector field for one generated unit (cfg off: nfe; cfg on: 2 nfe) and the checkpoint really loaded."""
    import torch
    n = {"calls": 0}
    tr = engine.model.transformer
    orig = tr.forward

    def counting(*a, **k):
        n["calls"] += 1
        return orig(*a, **k)
    tr.forward = counting
    g = engine.geom
    zero_past = [torch.zeros(1, g.frame_dim, g.unit_frames, device=engine.device)] * max(1, g.memory_units)
    steps = [{"TF": True, "atom_given": u, "atom_generated": None} for u in zero_past]
    ctx = torch.randn(1, 1024, device=engine.device)
    with torch.no_grad():
        enc, prefix = engine._ar_past(steps, len(steps)) if g.mode == "autoregressive" else engine._atoms_past(steps, len(steps))
        engine.model.generate(x0=torch.randn(1, g.unit_frames, g.frame_dim, device=engine.device), encoded_past=enc, clap_context=ctx,
                              structure_vector=None, max_nfe=nfe, cfg_scale=cfg, prefix=prefix)
    tr.forward = orig
    return n["calls"]


def semantic_worker(codec, size):
    """One process per model: provenance, cfg check, semantic resynthesis of the 4 files at both NFE (real per-unit CLAP, TF False)."""
    import torch
    import eval_metrics as em
    import teacher_forced_eval as tf
    import semantic_resynthesis as sr
    from SCAPES.data.dataprep.semantic import load_audio_48k
    from SCAPES.data.representations import RepresentationStore
    from SCAPES.inference.FlowInference import FlowInference

    md = model_dir(codec, size)
    best = md / "checkpoints" / "best_flow_model.pt"
    assert best.exists(), f"no best checkpoint: {best}"
    engine = FlowInference(md, device="cuda", checkpoint="best")
    names = engine._resolve_checkpoint_names("best", md)
    assert names[0] == "best_flow_model.pt", names
    prov = {"codec": codec, "size": size, "checkpoint": str(best.resolve()), "sha256_12": sha12(best), "bytes": best.stat().st_size,
            "resolved_names": list(names), "geometry": {"mode": engine.geom.mode, "unit_frames": engine.geom.unit_frames,
                                                         "hop_frames": engine.geom.unit_hop_frames, "sample_rate": engine.sr},
            "vector_field_calls_per_unit": {f"nfe{n}_cfg{CFG}": count_vector_field_calls(engine, n, CFG) for n in NFES}}
    prov["vector_field_calls_per_unit_cfg3_reference_nfe16"] = count_vector_field_calls(engine, 16, 3.0)
    assert prov["vector_field_calls_per_unit"]["nfe16_cfg1.0"] == 16 and prov["vector_field_calls_per_unit"]["nfe32_cfg1.0"] == 32, prov
    assert prov["vector_field_calls_per_unit_cfg3_reference_nfe16"] == 32, prov
    geom, info = engine.geom, engine.info
    store = RepresentationStore(DATASET, geom, info)
    store.ensure()
    clap_key = store.clap_key("2023", geom.context_seconds, engine.semantic_random_extension)
    starts = tf.unit_starts(geom, tf.SECONDS)
    out = FINAL / "audio" / "semantic_resynthesis"
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for stem in AUDIO_FILES:
        path = DATASET / "raw" / f"{stem}.wav"
        latents = store.load_latents(path)
        clap = store.load_grid_array("clap", clap_key, path)
        _, claps = tf.real_units(geom, latents, clap, starts, engine._standardise)
        for nfe in NFES:
            audio = sr.semantic_resynthesis(engine, claps, nfe, CFG, SEED)
            a = audio.numpy()
            name = f"{codec}_{size}_nfe{nfe}_{stem}.wav"
            tf.write_wav(out / name, a, info.sample_rate)
            clips.append({"file": name, "seconds": a.shape[-1] / info.sample_rate, "finite": bool(np.isfinite(a).all()),
                          "peak": float(np.abs(a).max()), "sample_rate": info.sample_rate})
    (WORK / "semantic").mkdir(parents=True, exist_ok=True)
    (WORK / "semantic" / f"{codec}_{size}.json").write_text(json.dumps({"provenance": prov, "clips": clips}, indent=1))


def stage_metrics_audio(only=None):
    import eval_metrics as em
    tfdir = WORK / "tf"
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    for codec, size in models():
        if only and f"{codec}_{size}" not in only:
            continue
        for nfe in NFES:
            name = f"{codec}_{size}_nfe{nfe}"
            if (tfdir / name / "metrics.json").exists():
                continue
            cmd = [sys.executable, "scripts/teacher_forced_eval.py", "--model-dir", str(model_dir(codec, size)), "--dataset", str(DATASET),
                   "--name", name, "--out", str(tfdir), "--codec-roundtrip", "on", "--checkpoint", "best", "--nfe", str(nfe),
                   "--cfg", str(CFG), "--seed", str(SEED), "--note", f"{codec} {size} nfe {nfe} cfg off, first 10 s (train region), best checkpoint"]
            code = run(cmd, WORK / "logs" / f"tf_{name}.log")
            print(name, "teacher-forced exit", code, flush=True)
        if not (WORK / "semantic" / f"{codec}_{size}.json").exists():
            code = run([sys.executable, str(Path(__file__).resolve()), "semantic-worker", codec, size], WORK / "logs" / f"sem_{codec}_{size}.log")
            print(codec, size, "semantic exit", code, flush=True)
    if only:
        return
    # band-limited MSS from the wavs, then the tables and the audio folder
    run([sys.executable, "scripts/eval_extra.py", "--out", str(tfdir), "--no-clap"], WORK / "logs" / "eval_extra.log")
    collect_metrics_audio()


def read_wav_finite(path):
    import soundfile as sf
    x, sr = sf.read(str(path), always_2d=True)
    return {"seconds": len(x) / sr, "finite": bool(np.isfinite(x).all()), "sample_rate": sr}


def collect_metrics_audio():
    import shutil
    tfdir = WORK / "tf"
    audio = FINAL / "audio"
    (audio / "teacher_forcing").mkdir(parents=True, exist_ok=True)
    (audio / "original").mkdir(parents=True, exist_ok=True)
    for stem in AUDIO_FILES:
        shutil.copy2(tfdir / "original_10s" / f"{stem}.wav", audio / "original" / f"{stem}.wav")
    rows, prov, audio_checks = [], {}, []
    for codec, size in models():
        sem = load(WORK / "semantic" / f"{codec}_{size}.json")
        if sem:
            prov[f"{codec}_{size}"] = sem["provenance"]
            audio_checks += [dict(c, mode="semantic_resynthesis") for c in sem["clips"]]
        for nfe in NFES:
            name = f"{codec}_{size}_nfe{nfe}"
            m = load(tfdir / name / "metrics.json")
            if not m:
                continue
            band = load(tfdir / name / "mss_band.json") or {}
            pf = m["mss"]["per_file"]["generated"]
            rt = m["mss"]["rows"].get("codec_roundtrip", {})
            k = m["kad"]
            row = {"codec": codec, "size": size, "size_alias": ALIAS[size], "nfe": nfe, "cfg_scale": m["generation"]["cfg"], "seed": m["generation"]["seed"],
                   "checkpoint": m["checkpoint"]["flow"], "checkpoint_sha256_12": m["checkpoint"]["flow_sha256_12"],
                   "region_s": [m["region"]["start_s"], m["region"]["end_s"]], "n_files": len(m["generation"]["files"]),
                   "mss_full_log_l1": boot([v["mean_log_l1"] for v in pf.values()]), "mss_full_spec_conv": boot([v["mean_spec_conv"] for v in pf.values()]),
                   "mss_band_log_l1": (band.get("rows", {}).get("generated", {}) or {}).get("mean_log_l1"),
                   "mss_band_spec_conv": (band.get("rows", {}).get("generated", {}) or {}).get("mean_spec_conv"),
                   "roundtrip_mss_full_log_l1": rt.get("mean_log_l1"), "roundtrip_mss_full_spec_conv": rt.get("mean_spec_conv"),
                   "roundtrip_mss_band_log_l1": (band.get("rows", {}).get("codec_roundtrip", {}) or {}).get("mean_log_l1"),
                   "kad_clip": k["clip_level"]["sets"]["generated"], "kad_clip_floor": k["clip_level"]["sets"]["floor"]["kad"],
                   "kad_clip_roundtrip": k["clip_level"]["sets"].get("codec_roundtrip", {}).get("kad"),
                   "kad_segment": k["kad"]["segment_level"]["sets"]["generated"] if "kad" in k else k["segment_level"]["sets"]["generated"]}
            rows.append(row)
            for stem in AUDIO_FILES:
                src = tfdir / name / "generated" / f"{stem}.wav"
                dst = audio / "teacher_forcing" / f"{codec}_{size}_nfe{nfe}_{stem}.wav"
                shutil.copy2(src, dst)
                audio_checks.append(dict(read_wav_finite(dst), file=dst.name, mode="teacher_forcing"))
    data = FINAL / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "metrics.json").write_text(json.dumps(rows, indent=1))
    (data / "provenance.json").write_text(json.dumps(prov, indent=1))
    (data / "audio_checks.json").write_text(json.dumps(audio_checks, indent=1))
    f = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"
    ci = lambda s, d=3: f"{s['mean']:.{d}f} [{s['ci95'][0]:.{d}f}, {s['ci95'][1]:.{d}f}]"
    L = ["# Teacher-forced MSS and KAD, 16 models x 16 and 32 ODE steps, no CFG (cfg_scale 1.0)", "",
         "Protocol of runs/eval_tf (scripts/teacher_forced_eval.py): the first 10 s of each of the 16 microtex files, every unit generated from the real past and the "
         "real per-unit CLAP; this is the training part of the files (a measure of fit, not of generalisation). Mono 48 kHz, region [0.4 s, end). MSS = natural-log "
         "magnitude L1 and spectral convergence (SC), mean of three big windows, mean over files [95% bootstrap over files]; band = bins below 12 kHz. "
         "KAD = 100 x unbiased MMD^2 of CLAP embeddings, clip level (16 vs 16), point [95% bootstrap over clips, 200 resamples]; floor = the original's seconds 10-20 against 0-10. "
         "Round trip = the codec alone (the ceiling). Seed 0, one run per cell, best checkpoint.", "",
         "| codec | size | nfe | MSS L1 | MSS SC | MSS band L1 | MSS band SC | KAD clip | KAD floor | round trip: L1 / band L1 / KAD |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        kad = r["kad_clip"]
        L.append(f"| {r['codec']} | {r['size']} ({r['size_alias']}) | {r['nfe']} | {ci(r['mss_full_log_l1'])} | {ci(r['mss_full_spec_conv'])} | {f(r['mss_band_log_l1'])} | "
                 f"{f(r['mss_band_spec_conv'])} | {kad['kad']:.2f} [{kad['ci95'][0]:.2f}, {kad['ci95'][1]:.2f}] | {r['kad_clip_floor']:.2f} | "
                 f"{f(r['roundtrip_mss_full_log_l1'])} / {f(r['roundtrip_mss_band_log_l1'])} / {f(r['kad_clip_roundtrip'], 2)} |")
    (data / "metrics.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ─── stage 2: RTF ───

def rtf_tables(src: Path, label: str):
    r = load(src / "rtf_bench.json")
    dev = "gpu" if label == "gpu" else "cpu"
    rows = []
    for name, m in r["models"].items():
        codec, size = name.rsplit("_", 1)
        for cfgname, c in m["configs"].items():
            rows.append({"codec": codec, "size": size, "size_alias": ALIAS[size], "config": cfgname, "nfe": c["nfe"], "cfg_scale": c["cfg_scale"],
                         "new_audio_seconds": c["new_audio_seconds"], "generated_audio_seconds": c["generated_audio_seconds"],
                         "calls": c["calls"], "checkpoint_loaded": m["model_dir"], "params": m["params"],
                         "time_per_block_s": c["per_block_seconds"], "rtf_new": c["rtf_new"], "flow_rtf_new": c["flow_rtf"], "decode_rtf_new": c["decode_rtf"],
                         "rtf_generated": c["rtf_generated"], "finite": c["finite"]})
    meta = {"device_start": r["gpu_start"], "device_end": r["gpu_end"], "calls": r["calls"], "warmup": r["warmup"]}
    (FINAL / "data").mkdir(parents=True, exist_ok=True)
    (FINAL / "data" / f"rtf_{label}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    L = [f"# RTF_new on the {'GPU (RTX 4090)' if label == 'gpu' else 'CPU'}, no CFG", "",
         "RTF_new = time to produce one unit (flow with NFE steps + codec decode of the new audio, copied to host) / duration of the NEW audio only (AR: the block; atoms: the hop, "
         "overlap not counted). Batch 1, fp32, best checkpoint of each model, cfg_scale 1.0 (a single forward per step). Steady state: the past is filled first, then "
         f"{r['warmup']} warm-up blocks are discarded, then {r['calls']} timed blocks per configuration. mean +- std over timed blocks. Quality at these settings is measured in "
         "metrics.md (teacher-forced).", ""]
    d0, d1 = meta["device_start"], meta["device_end"]
    if label == "gpu":
        L += [f"GPU before: {d0.get('name')}, SM clock {d0.get('clocks.sm')} MHz, {d0.get('temperature.gpu')} C, utilisation {d0.get('utilization.gpu')} %; after: SM clock {d1.get('clocks.sm')} MHz, {d1.get('temperature.gpu')} C.", ""]
    else:
        L += [f"CPU: {d0.get('name')}, {d0.get('logical_cores')} logical cores, torch threads {d0.get('torch_threads')}; load average (1/5/15 min) before {d0.get('load_avg_1_5_15')}, after {d1.get('load_avg_1_5_15')}.", ""]
    L += ["| codec | size | nfe | new audio (s) | time per block (ms) | RTF_new | flow | decode | finite |", "|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda x: (CODECS.index(x["codec"]), list(SIZES).index(x["size"]), x["nfe"])):
        L.append(f"| {x['codec']} | {x['size']} ({x['size_alias']}) | {x['nfe']} | {x['new_audio_seconds']:.3f} | {x['time_per_block_s']['mean'] * 1e3:.1f} +- {x['time_per_block_s']['std'] * 1e3:.1f} | "
                 f"{x['rtf_new']['mean']:.3f} +- {x['rtf_new']['std']:.3f} | {x['flow_rtf_new']['mean']:.3f} | {x['decode_rtf_new']['mean']:.3f} | {x['finite']} |")
    (FINAL / "data" / f"rtf_{label}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def stage_rtf(device):
    out = WORK / f"rtf_{device}"
    cmd = [sys.executable, "scripts/rtf_bench.py", "--trained", *SIZES, "--configs", "nfe16_cfgoff", "nfe32_cfgoff", "--out", str(out), "--device",
           "cuda" if device == "gpu" else "cpu"]
    code = run(cmd, WORK / "logs" / f"rtf_{device}.log")
    print("rtf", device, "exit", code, flush=True)
    rtf_tables(out, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["training", "metrics-audio", "collect", "rtf-gpu", "rtf-cpu", "semantic-worker"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--only", nargs="+", default=None, help="metrics-audio: models such as encodec24k_tiny (debug)")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.stage == "training":
        stage_training()
    elif a.stage == "metrics-audio":
        stage_metrics_audio(a.only)
    elif a.stage == "collect":
        collect_metrics_audio()
    elif a.stage == "semantic-worker":
        semantic_worker(*a.args)
    elif a.stage == "rtf-gpu":
        stage_rtf("gpu")
    else:
        stage_rtf("cpu")


if __name__ == "__main__":
    main()
