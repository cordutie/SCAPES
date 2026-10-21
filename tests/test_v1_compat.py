"""Slice B5: a V1 checkpoint plus the two migration lines reproduces V1 generation (design D14, brief Non-goals)."""
import torch

from conftest import FIXTURES, V1_MODEL, max_abs


def test_v1_checkpoint_reproduces(tmp_path):
    from SCAPES.inference.FlowInference import FlowInference
    ref = torch.load(FIXTURES / "v1_gen_ref.pt", map_location="cpu", weights_only=True)
    ckpt = tmp_path / "micro_water" / "checkpoints"
    ckpt.mkdir(parents=True)
    for p in (V1_MODEL / "checkpoints").glob("*.pt"):
        (ckpt / p.name).symlink_to(p)
    gin = (V1_MODEL / "checkpoints" / "inference.gin").read_text()
    (ckpt / "inference.gin").write_text(gin + '\ncodec.name = "encodec48k"\ngeneration.mode = "atoms"\n')
    engine = FlowInference(model_dir=tmp_path / "micro_water", device="cuda", checkpoint="best")
    clap = [c for c in ref["clap"]]
    torch.manual_seed(0)
    audio = engine.generate_soundscape(clap, nfe=16, cfg_scale=3.0)
    atoms = torch.stack([s["atom_generated"].cpu() for s in engine.timeline])
    assert atoms.shape == ref["atoms"].shape == (6, 1, 129, 48)
    assert max_abs(atoms, ref["atoms"]) <= 1e-4
    assert audio.shape == ref["audio"].shape == (2, 39360)
    assert max_abs(audio, ref["audio"]) <= 1e-4
