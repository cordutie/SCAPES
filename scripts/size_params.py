"""Exact parameter counts (flow model + local encoder) of every model size for the four codecs, random init on CPU.

    ../venv_SCAPES/bin/python scripts/size_params.py --out ../runs/size_params
Sizes: the loader's table (small, medium, large, extra_large, audiobox) plus petite (256/4/4/1024) and mid (384/6/5/1536).
Geometry: the S5 geometries (AR 8/120 frames for encodec24k and spectrostream, atoms for the others; encodec48k with M = 1 as the trained run).
Also fits per-layer parameters against 14 d^2 + 2 d ff and breaks the model into blocks. No GPU, no training.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SIZES = {"petite": (256, 4, 4, 1024), "tiny": (384, 6, 5, 1536), "small": (512, 8, 6, 2048), "medium": (768, 12, 8, 2048),
         "large": (1024, 16, 12, 2048), "extra_large": (1280, 20, 16, 2048), "audiobox": (1024, 16, 24, 4096)}
CODECS = {"encodec24k": "autoregressive", "spectrostream": "autoregressive", "encodec48k": "atoms", "codicodec": "atoms"}
GEOMETRY = {"encodec48k": {"dataset.memory_buffer_atoms": 1}}


def build(codec, size, device="cpu"):
    from SCAPES.inference.FlowInference import FlowInference
    d, h, n, ff = SIZES[size]
    geometry = {"model.d_model": d, "model.nhead": h, "model.num_layers": n, "model.dim_feedforward": ff, **GEOMETRY.get(codec, {})}
    return FlowInference.from_config(codec, CODECS[codec], size="custom", geometry=geometry, device=device)


def count(m):
    return sum(p.numel() for p in m.parameters())


# parameters of the flow model by block
def blocks(engine):
    model = engine.model
    out = {}
    for name, p in model.named_parameters():
        key = name.split(".")
        k = "layers (per layer x n)" if "layers" in key else ("memory_pos_enc / null_past" if key[0] in ("memory_pos_enc", "null_past_embed") else
             "/".join(key[:2]) if key[0] == "transformer" else key[0])
        out[k] = out.get(k, 0) + p.numel()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="+", default=list(SIZES))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    res = {}
    for size in a.sizes:
        d, h, n, ff = SIZES[size]
        assert d % h == 0 and (d // h) == 64, (size, d, h)
        for codec in CODECS:
            e = build(codec, size)
            flow, local = count(e.model), count(e.local_encoder)
            layer = sum(p.numel() for p in e.model.transformer.layers[0].parameters())
            res.setdefault(size, {"d_model": d, "nhead": h, "num_layers": n, "dim_feedforward": ff, "head_dim": d // h, "codecs": {}})
            res[size]["codecs"][codec] = {"flow": flow, "local_encoder": local, "total": flow + local, "per_layer": layer,
                                          "layers_total": layer * n, "outside_layers_flow": flow - layer * n,
                                          "blocks": blocks(e) if codec == "encodec24k" else None}
            print(size, codec, flow + local, flush=True)
            del e
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    L = ["# Parameter counts (flow model + local encoder), random init, exact", "",
         "| size | d_model / heads / layers / ff | encodec24k | spectrostream | encodec48k | codicodec | per layer | formula 14 d^2 + 2 d ff (per layer) | layers share (encodec24k) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for size, r in res.items():
        c = r["codecs"]
        d, ff, n = r["d_model"], r["dim_feedforward"], r["num_layers"]
        pl = c["encodec24k"]["per_layer"]
        L.append(f"| {size} | {d} / {r['nhead']} / {n} / {ff} | " + " | ".join(f"{c[k]['total'] / 1e6:.2f} M" if k in c else "n/a" for k in CODECS) +
                 f" | {pl / 1e6:.3f} M | {(14 * d * d + 2 * d * ff) / 1e6:.3f} M | {100 * c['encodec24k']['layers_total'] / c['encodec24k']['flow']:.0f}% |")
    L += ["", "Per-layer structure (TransformerLayer): 3 AdaLN (each Linear d -> 2d: 2 d^2 + 2 d), self-attention (4 d^2 + 4 d), cross-attention (4 d^2 + 4 d), "
          "FFN (2 d ff + ff + d): 14 d^2 + 2 d ff + biases. Outside the layers: time MLP (2 d^2), context MLP (1024 d + d^2), final AdaLN (2 d^2), target / output "
          "projections, the null-past embedding (window x d) and the atom position embedding; the local encoder (hidden 256) adds 0.8-1.5 M and grows with d."]
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
