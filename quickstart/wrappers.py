"""Two calls for the quickstart notebook: data_preparation and training_loop.

Each takes the wav folder, the build folder and the config file (SCAPES/config/config_simplified.gin or config.gin).
The build folder holds dataset/ (the dataprep output) and models/<size>_mem<ms>/ (the trained models).
"""
from SCAPES.pipeline import prepare_build, train_in_build


# Encodes the wav folder into <build>/dataset, or reuses it if the settings and the wavs are the same
def data_preparation(wav_path, build_path, config_path):
    return prepare_build(wav_path, build_path, config_path)


# Trains one model into <build>/models/<size>_mem<ms>, and returns that folder
def training_loop(wav_path, build_path, config_path, resume=None):
    return train_in_build(wav_path, build_path, config_path, resume=resume)
