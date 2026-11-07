"""Two calls for the quickstart notebook: data_preparation and training_loop.

Each takes the wav folder, the build folder and the config file (SCAPES/config/config_simplified.gin or config.gin).
The build folder holds dataset/ (the dataprep output) and models/<size>_mem<ms>/ (the trained models).
"""

# 1. SETUP -------------------------------------------------------------------------------

import os
import subprocess
import sys
from pathlib import Path

from SCAPES.data.config_loader import load_training_config, load_dataprep_config
from SCAPES.pipeline import model_folder
from SCAPES.generation import resolve_generation

def parameter_wrapper(corpus_path, build_path, config_path):
    # Paths. Edit the placeholders, or set the environment variable SCAPES_<NAME> instead.
    CORPUS_PATH = Path(corpus_path)
    BUILD_PATH  = Path(build_path)
    CONFIG_PATH = Path(config_path)

    TRAINING   = load_training_config(CONFIG_PATH)
    MODEL_PATH = model_folder(BUILD_PATH, CONFIG_PATH)
    dataprep   = load_dataprep_config(CONFIG_PATH)
    resolution = resolve_generation(dataprep.codec_name, 
                                    dataprep.conditioning_rate,
                                    dataprep.conditioning_memory_seconds)

    print("Your corpus will be read from:    ",CORPUS_PATH)
    print("Your data will be encoded at:     ",BUILD_PATH / "dataset")
    print("Your model will be saved at:      ",MODEL_PATH)
    print("Your configuration were read from:", CONFIG_PATH)

    # Fields of Resolution
    codec = resolution.codec
    mode = resolution.mode
    latent_rate = resolution.latent_rate
    requested_rate = resolution.requested_rate
    requested_memory_seconds = resolution.requested_memory_seconds
    n = resolution.n
    rate = resolution.rate
    memory_latents = resolution.memory_latents
    crossfade = resolution.crossfade
    overlapped = resolution.overlapped
    atom = resolution.atom
    memory_atoms = resolution.memory_atoms
    k_max = resolution.k_max
    memory_window = resolution.memory_window
    attended = resolution.attended
    notes = resolution.notes

    # Properties of Resolution
    hop_seconds = resolution.hop_seconds
    atom_seconds = resolution.atom_seconds
    crossfade_seconds = resolution.crossfade_seconds
    memory_window_seconds = resolution.memory_window_seconds
    covered_latents = resolution.covered_latents
    buffer_span_latents = resolution.buffer_span_latents

    print("Model configurations:")
    print("    codec:", codec)
    print("    model size:", TRAINING.size)
    print("    conditioning rate (Hz):", round(1/hop_seconds,2))
    memory_seconds = covered_latents / latent_rate if mode == "atoms" else memory_window_seconds
    print("    memory buffer (s):     ", round(memory_seconds, 2))

    return CORPUS_PATH, BUILD_PATH, CONFIG_PATH, TRAINING, MODEL_PATH

import time
import torch

from SCAPES.data.config_loader import load_dataprep_config
from SCAPES.inference.FlowInference import FlowInference
from SCAPES.pipeline import units_for_seconds

def test(device, CONFIG_PATH, TRAINING):
    initialization_timer_1 = time.perf_counter()
    dataprep = load_dataprep_config(CONFIG_PATH)     
    geometry = {"conditioning_rate": dataprep.conditioning_rate,
                "conditioning_memory_seconds": dataprep.conditioning_memory_seconds}
    engine = FlowInference.from_config(dataprep.codec_name, size=TRAINING.size, geometry=geometry, device=device)
    units = units_for_seconds(engine.geom, 10)
    random_sound = lambda: [torch.randn(1024) for _ in range(units)]  # one random CLAP-like vector per unit
    engine.generate_soundscape(random_sound(), nfe=TRAINING.nfe, cfg_scale=TRAINING.cfg_scale)  # warm-up: not timed
    initialization_time = time.perf_counter() - initialization_timer_1
    wall = seconds = 0.0
    for _ in range(3):
        start = time.perf_counter()
        audio = engine.generate_soundscape(random_sound(), nfe=TRAINING.nfe, cfg_scale=TRAINING.cfg_scale)
        if device == "cuda":
            torch.cuda.synchronize()
        wall += time.perf_counter() - start
        seconds += audio.shape[-1] / engine.sr
    per_second = wall / seconds
    # gpu = torch.cuda.get_device_name(0) if device == "cuda" else "no GPU (CPU)"
    print("Results:")
    print(f"    Initialization time (s):   {initialization_time:.2f}")
    print(f"    Time to make 1 s of sound: {per_second:.2f}")
    print(f"    Real-Time:                 {'yes' if per_second < 1.0 else 'no'}")
    print("")

def test_gpu(CONFIG_PATH, TRAINING):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device=="cpu":
        raise Exception("Your machine is not connected to a GPU")   
    print("GPU test starts!")
    test(device, CONFIG_PATH, TRAINING)

def test_cpu(CONFIG_PATH, TRAINING):
    print("CPU test starts!")
    test("cpu", CONFIG_PATH, TRAINING)



from SCAPES.pipeline import prepare_build, train_in_build

# Encodes the wav folder into <build>/dataset, or reuses it if the settings and the wavs are the same
def data_preparation(wav_path, build_path, config_path):
    return prepare_build(wav_path, build_path, config_path)


# Trains one model into <build>/models/<size>_mem<ms>, and returns that folder
def training_loop(wav_path, build_path, config_path, resume=None):
    return train_in_build(wav_path, build_path, config_path, resume=resume)