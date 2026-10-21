"""Make (or check) the V1 reference fixtures with the unchanged V1 code and venv.

Run before any environment change:
    ../venv_SCAPES/bin/python scripts/make_v1_fixtures.py          # write fixtures/v1/
    ../venv_SCAPES/bin/python scripts/make_v1_fixtures.py --check  # verify shapes, finiteness, hashes
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import soundfile as sf
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "fixtures" / "v1"
PROJECT = REPO.parent
WATER = PROJECT / "datasets" / "micro_water" / "raw" / "water_fast.wav"
MODEL_DIR = PROJECT / "models" / "micro_water"
NAMES = ["clap_ref.pt", "encodec48_ref.pt", "v1_gen_ref.pt"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# First n samples of water_fast.wav as [1, 2, n] float32.
def load_water(n):
    x, sr = sf.read(str(WATER), dtype="float32", always_2d=True)
    assert sr == 48000, sr
    return torch.from_numpy(x[:n].T.copy()).unsqueeze(0)


def make_clap():
    from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
    audio = load_water(48000)
    clap = CLAPWrapper(version="2023", use_cuda=True)
    emb = clap.compute_embedding(audio.cuda(), og_sr=48000, random_extension=False)[0]
    torch.save({"audio": audio, "embedding": emb.detach().float().cpu()}, OUT / "clap_ref.pt")


def make_encodec():
    from SCAPES.auxiliar.encodec_wrapper import EncodecProcessor
    audio = load_water(15360)
    proc = EncodecProcessor(sr=48000, streamable=True, device="cuda")
    latents, meta = proc.audio_to_latents(audio, 48000)
    latent = torch.cat(latents, dim=-1).detach().float().cpu()
    scale = meta["audio_scales"][0].detach().float().cpu()
    torch.save({"audio": audio, "latent": latent, "scale": scale}, OUT / "encodec48_ref.pt")


def make_generation():
    from SCAPES.inference.FlowInference import FlowInference
    engine = FlowInference(model_dir=str(MODEL_DIR), device="cuda", checkpoint="best")
    g = torch.Generator().manual_seed(1)
    clap = [F.normalize(torch.randn(1024, generator=g), dim=0) for _ in range(6)]
    torch.manual_seed(0)
    audio = engine.generate_soundscape(clap, nfe=16, cfg_scale=3.0)
    atoms = torch.stack([s["atom_generated"].detach().float().cpu() for s in engine.timeline])
    torch.save({"clap": torch.stack(clap), "atoms": atoms, "audio": audio.detach().float().cpu()},
               OUT / "v1_gen_ref.pt")


def git_out(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def write_meta():
    import numpy
    props = torch.cuda.get_device_properties(0)
    meta = {
        "git_head": git_out("rev-parse", "HEAD"),
        "git_main": git_out("rev-parse", "main"),
        "pip_freeze": "requirements.pre_v2.lock",
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": props.name,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    sums = "".join(f"{sha256(OUT / n)}  {n}\n" for n in NAMES)
    (OUT / "SHA256SUMS").write_text(sums)


def check():
    ok = True

    def expect(cond, msg):
        nonlocal ok
        print(("ok    " if cond else "FAIL  ") + msg)
        ok = ok and cond

    # Hashes.
    for line in (OUT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        expect(sha256(OUT / name) == digest, f"sha256 {name}")
    load = lambda n: torch.load(OUT / n, map_location="cpu", weights_only=True)
    shapes = {
        "clap_ref.pt": {"audio": (1, 2, 48000), "embedding": None},
        "encodec48_ref.pt": {"audio": (1, 2, 15360), "latent": (1, 128, 48), "scale": (1, 1)},
        "v1_gen_ref.pt": {"clap": (6, 1024), "atoms": (6, 1, 129, 48), "audio": (2, 39360)},
    }
    for name, spec in shapes.items():
        d = load(name)
        for key, shape in spec.items():
            t = d[key]
            if shape is not None:
                expect(tuple(t.shape) == shape, f"{name}:{key} shape {tuple(t.shape)}")
            expect(bool(torch.isfinite(t).all()), f"{name}:{key} finite")
    emb = load("clap_ref.pt")["embedding"]
    expect(emb.numel() == 1024, f"clap embedding numel {emb.numel()}")
    print("CHECK PASSED" if ok else "CHECK FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    OUT.mkdir(parents=True, exist_ok=True)
    make_clap()
    make_encodec()
    make_generation()
    write_meta()
    print("written", OUT)
