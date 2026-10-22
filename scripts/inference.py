"""
SCAPES Inference Pipeline

Loads a trained model from a model directory and runs inference.
All model/architecture/inference settings come from
<model_dir>/checkpoints/inference.gin (generated during training).

Examples:
    python scripts/inference.py --model_dir ../models/my_model reconstruct --audio file.wav
    python scripts/inference.py --model_dir ../models/my_model generate --prompt_audio file.wav --seconds 10 --out out.wav
    python scripts/inference.py --model_dir ../models/my_model interpolate --audio_1 a.wav --audio_2 b.wav
    python scripts/inference.py --model_dir ../models/microtex semantic_dir --input_dir ../datasets/microtex/raw --nfe 32 --duration 30 --cfg 1.0

Can be run from any directory.
"""

import sys, pathlib
import argparse
import torch

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from SCAPES.inference.FlowInference import FlowInference, run_resynthesis_pipeline, run_batch_resynthesis_pipeline, run_interpolation_pipeline
from SCAPES.pipeline import run_generation


def _build_engine(args):
    return FlowInference(
        args.model_dir,
        device=args.device,
        verbose=args.verbose,
        checkpoint=args.checkpoint,
    )


def cmd_reconstruct(args):
    engine = _build_engine(args)
    run_resynthesis_pipeline(
        engine=engine,
        audio_path=args.audio,
        duration=args.duration,
        play=False,
        save_path=args.save or None,
        TF=not args.no_tf,
        NFE=args.nfe,
        cfg_scale=args.cfg_scale,
        decode_method="ola_smooth"
    )


def cmd_semantic(args):
    engine = _build_engine(args)
    run_resynthesis_pipeline(
        engine=engine,
        audio_path=args.audio,
        duration=args.duration,
        play=False,
        save_path=args.save or None,
        TF="partial",
        NFE=args.nfe,
        cfg_scale=args.cfg_scale,
        decode_method="ola_smooth"
    )


def cmd_reconstruct_dir(args):
    engine = _build_engine(args)
    run_batch_resynthesis_pipeline(
        engine=engine,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        duration=args.duration,
        NFE=args.nfe,
        cfg_scale=args.cfg_scale,
        TF=not args.no_tf,
        decode_method="ola_smooth",
    )


def cmd_semantic_dir(args):
    engine = _build_engine(args)
    run_batch_resynthesis_pipeline(
        engine=engine,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        duration=args.duration,
        NFE=args.nfe,
        cfg_scale=args.cfg_scale,
        TF="partial",
        decode_method="ola_smooth",
    )


def cmd_generate(args):
    # prompt: a CLAP embedding file (.pt, [1024] or [N, 1024]) or the first second of an audio file
    if args.prompt_pt:
        prompts = list(torch.load(args.prompt_pt, map_location="cpu", weights_only=True).reshape(-1, 1024))
    else:
        from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
        from SCAPES.data.dataprep.semantic import load_audio_48k
        clap = CLAPWrapper(version="2023", use_cuda=torch.cuda.is_available())
        audio = load_audio_48k(args.prompt_audio)[:, :48000].unsqueeze(0)
        prompts = [clap.compute_embedding(audio, og_sr=48000, random_extension=True)[0].detach().cpu()]
    run_generation(args.model_dir, args.out, prompts, seconds=args.seconds, seed=args.seed, nfe=args.nfe,
                   cfg_scale=args.cfg_scale, checkpoint=args.checkpoint)


def cmd_interpolate(args):
    engine = _build_engine(args)
    run_interpolation_pipeline(
        engine=engine,
        audio_path_1=args.audio_1,
        audio_path_2=args.audio_2,
        timeline_size=args.timeline,
        stay_time=args.stay,
        stickyness=args.stickiness,
        play=False,
        save_path=args.save or None,
        NFE=args.nfe,
        cfg_scale=args.cfg_scale,
        context_static=not args.dynamic_context,
        decode_method="ola_smooth"
    )


def main():
    parser = argparse.ArgumentParser(description="SCAPES Inference")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory (contains checkpoints/)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str, default="best",
                        help="Which checkpoint to load: 'best', 'last', or an epoch number")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- reconstruct ---
    p_rec = subparsers.add_parser("reconstruct", help="Reconstruct audio from its latent representation")
    p_rec.add_argument("--audio", required=True)
    p_rec.add_argument("--duration", type=int, default=None)
    p_rec.add_argument("--save", default=None)
    p_rec.add_argument("--no-tf", action="store_true", default=False)
    p_rec.add_argument("--nfe", type=int, default=32)
    p_rec.add_argument("--cfg_scale", type=float, default=3.0)
    p_rec.set_defaults(func=cmd_reconstruct)

    # --- reconstruct_dir ---
    p_recdir = subparsers.add_parser("reconstruct_dir", help="Reconstruct all audio files in a directory")
    p_recdir.add_argument("--input_dir", required=True)
    p_recdir.add_argument("--output_dir", default="reconstructed")
    p_recdir.add_argument("--duration", type=int, default=None)
    p_recdir.add_argument("--no-tf", action="store_true", default=False)
    p_recdir.add_argument("--nfe", type=int, default=32)
    p_recdir.add_argument("--cfg_scale", type=float, default=3.0)
    p_recdir.set_defaults(func=cmd_reconstruct_dir)

    # --- semantic_dir ---
    p_semdir = subparsers.add_parser("semantic_dir", help="Reconstruct all audio files from semantics only")
    p_semdir.add_argument("--input_dir", required=True)
    p_semdir.add_argument("--output_dir", default="semantic_reconstructed")
    p_semdir.add_argument("--duration", type=int, default=None)
    p_semdir.add_argument("--nfe", type=int, default=32)
    p_semdir.add_argument("--cfg_scale", type=float, default=3.0)
    p_semdir.set_defaults(func=cmd_semantic_dir)

    # --- semantic ---
    p_sem = subparsers.add_parser("semantic", help="Reconstruct audio from semantics only")
    p_sem.add_argument("--audio", required=True)
    p_sem.add_argument("--duration", type=int, default=None)
    p_sem.add_argument("--save", default=None)
    p_sem.add_argument("--nfe", type=int, default=32)
    p_sem.add_argument("--cfg_scale", type=float, default=3.0)
    p_sem.set_defaults(func=cmd_semantic)

    # --- generate ---
    p_gen = subparsers.add_parser("generate", help="Generate audio from a CLAP prompt (audio file or .pt embeddings)")
    group = p_gen.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt_audio", default=None)
    group.add_argument("--prompt_pt", default=None)
    p_gen.add_argument("--seconds", type=float, default=10.0)
    p_gen.add_argument("--out", required=True, help="Output wav (or a folder: generation.wav inside)")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--nfe", type=int, default=16)
    p_gen.add_argument("--cfg_scale", type=float, default=3.0)
    p_gen.set_defaults(func=cmd_generate)

    # --- interpolate ---
    p_int = subparsers.add_parser("interpolate", help="Interpolate between two audio files")
    p_int.add_argument("--audio_1", required=True)
    p_int.add_argument("--audio_2", required=True)
    p_int.add_argument("--timeline", type=int, default=200)
    p_int.add_argument("--stay", type=int, default=20)
    p_int.add_argument("--stickiness", type=float, default=3.0)
    p_int.add_argument("--save", default=None)
    p_int.add_argument("--nfe", type=int, default=32)
    p_int.add_argument("--cfg_scale", type=float, default=3.0)
    p_int.add_argument("--dynamic-context", action="store_true", default=False)
    p_int.set_defaults(func=cmd_interpolate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
