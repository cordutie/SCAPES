"""The whole SCAPES flow in one plain script: prepare data, train, resynthesise a file.

Run from the repository root:  python quickstart/example.py
Edit the paths and the three generation settings below. The same settings are documented in
docs/generation_modes.html, with every conditioning rate each codec can use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quickstart.wrappers import data_preparation, training_loop

DATASET_PATH = "path/to/your/dataset"      # folder with raw/ and config/
MODEL_PATH = "path/to/save/my_model"
AUDIO_PATH = "path/to/some/audio.wav"      # file to resynthesise after training
OUTPUT_PATH = "resynthesis.wav"

CODEC = "music2latent"                     # encodec24 | spectrostream | encodec48 | music2latent
CONDITIONING_RATE = 10                     # Hz, snapped to the nearest rate the codec offers
CONDITIONING_MEMORY_SECONDS = 0.5          # seconds of past audio the model sees

settings = dict(codec=CODEC, conditioning_rate=CONDITIONING_RATE,
                conditioning_memory_seconds=CONDITIONING_MEMORY_SECONDS)

# 1. Encode the dataset (prints the resolved geometry once)
data_preparation(DATASET_PATH, **settings)

# 2. Train until the validation loss stops improving
training_loop(DATASET_PATH, MODEL_PATH, **settings)

# 3. Resynthesise a file with the best checkpoint (teacher forced: each step sees the real past)
from SCAPES.inference.FlowInference import FlowInference, run_resynthesis_pipeline

engine = FlowInference(model_dir=MODEL_PATH, device="cuda", checkpoint="best")
run_resynthesis_pipeline(engine, AUDIO_PATH, duration=10, play=False, save_path=OUTPUT_PATH, TF=True, NFE=16)
