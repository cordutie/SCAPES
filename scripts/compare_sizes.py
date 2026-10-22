"""Side-by-side of the trained model sizes petite (pp), tiny (p), small (mp) and large (f) for every metric, with deltas to large.
Read-only; writes <out>.md and <out>.json. Missing inputs give n/a.

    ../venv_SCAPES/bin/python scripts/compare_sizes.py --out ../runs/size_comparison
Per size the outputs of eval_size.py are read (runs/eval_tf<suffix>, s5_val<suffix>, semantic_resynth<suffix>, runs/<folder>/val_split via the
logs, past_usage_all/<size>/<codec>.json). The large and small results live in the older folders (eval_tf, s5_val, semantic_resynth,
eval_tf_small, ...). The speed table comes from runs/rtf_bench_all/rtf_bench.json when it exists.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_small_large as csl

R = HERE.parent.parent / "runs"
CODECS = csl.CODECS
# size -> (eval_tf, s5_val, semantic_resynth, runs folder)
SIZES = {
    "petite": (R / "eval_tf_petite", R / "s5_val_petite", R / "semantic_resynth_petite", R / "s5_petite"),
    "tiny": (R / "eval_tf_tiny", R / "s5_val_tiny", R / "semantic_resynth_tiny", R / "s5_tiny"),
    "small": (R / "eval_tf_small", R / "s5_val_small", R / "semantic_resynth_small", R / "s5_small"),
    "large": (R / "eval_tf", R / "s5_val", R / "semantic_resynth", R / "s5"),
}
ALIAS = {"petite": "pp", "tiny": "p", "small": "mp", "large": "f"}

CAVEATS = csl.CAVEATS + [
    "Sizes are compared on one training run each (seed 0); petite and tiny encodec48k runs use a training.epochs cap of 800 (small and large: 400), "
    "and a run that stops at the cap is cap-limited, not converged.",
    "Past-usage numbers use the per-file split (clean forest, park, rain, tokio vs 12 repeat files) for every size, from runs/past_usage_all.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "size_comparison"))
    ap.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    a = ap.parse_args()
    data = {}
    for size in a.sizes:
        tf, val, sem, runs = SIZES[size]
        d = csl.collect(tf, val, sem, R / "past_usage_all" / size, runs, "{codec}.json")
        data[size] = d
    params = (csl.load(R / "size_params.json") or {})
    speed = (csl.load(R / "rtf_bench_all" / "rtf_bench.json") or {}).get("models", {})
    for size in data:
        for c in CODECS:
            data[size][c]["params_exact_M_flow_plus_local"] = (csl.get(params, size, "codecs", c, "total") or 0) / 1e6 or None
            for label in ("nfe32_cfg3", "nfe16_cfg3", "nfe16_cfgoff"):
                data[size][c][f"bench_rtf_new_{label}"] = csl.get(speed, f"{c}_{size}", "configs", label, "rtf", "mean")
    Path(a.out + ".json").write_text(json.dumps({"sizes": data, "caveats": CAVEATS}, indent=1))
    names = [f"{s} ({ALIAS[s]})" for s in a.sizes]
    L = ["# Model sizes side by side: " + ", ".join(names), "", "Caveats:", ""] + [f"{i + 1}. {c}" for i, c in enumerate(CAVEATS)]
    L += ["", "Lower is better for MSS, KAD, D-a, D-d, RTF, times; higher for D-b generated, past-usage gain, production pass. Teacher-forced and "
          "semantic numbers are on train-region excerpts; production and D-a..D-e on the val region. Delta columns are size minus large.", ""]
    keys = list(data[a.sizes[0]][CODECS[0]])
    for c in CODECS:
        L += [f"## {c}", "", "| metric | " + " | ".join(names) + " | " + " | ".join(f"{s} - large" for s in a.sizes if s != "large") + " |",
              "|---|" + "---|" * (len(a.sizes) + len(a.sizes) - (1 if "large" in a.sizes else 0))]
        for k in keys:
            vals = [data[s][c][k] for s in a.sizes]
            base = data["large"][c][k] if "large" in data else None
            deltas = []
            for s in a.sizes:
                if s == "large":
                    continue
                v = data[s][c][k]
                deltas.append(csl.fmt(v - base) if isinstance(v, (int, float)) and isinstance(base, (int, float)) else "n/a")
            L.append(f"| {k} | " + " | ".join(csl.fmt(v) for v in vals) + " | " + " | ".join(deltas) + " |")
        L.append("")
    Path(a.out + ".md").write_text("\n".join(L))
    print("wrote", a.out + ".md")


if __name__ == "__main__":
    main()
