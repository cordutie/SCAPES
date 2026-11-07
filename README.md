# SCAPES

**SCAPES** (Semantically Conditioned Auto-Regressive Prior for Environmental Sounds) is a generative audio framework for high-fidelity environmental texture synthesis with semantic control.

Try the online demo: [![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/cordutie/SCAPES-demo)

## Installation

Requirements: Linux, Python 3.12, an NVIDIA GPU with a recent driver (CUDA 12), git.

Clone the repository with its codec wrappers (the git submodule `external/audio_codec_wrappers/`):

<!-- install:get -->
```bash
git clone --recurse-submodules https://github.com/cordutie/SCAPES.git
cd SCAPES
git submodule update --init    # only needed if you cloned without --recurse-submodules
```

Create the virtual environment and install SCAPES in it:

<!-- install:env -->
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install --no-deps msclap==1.3.4 "magenta-rt @ git+https://github.com/magenta/magenta-realtime@694a545e4ba0b88bf1150137b129582166d3e07f"
pip install music2latent==0.1.8
pip install -e ./external/audio_codec_wrappers
python external/audio_codec_wrappers/scripts/fetch_weights.py --codec spectrostream
```

The `--extra-index-url` line gives the PyTorch CUDA 12.8 build that `requirements.txt` pins (torch 2.10.0+cu128, as in the lock file). The last line downloads the SpectroStream weights once. EnCodec and Music2Latent download their weights on first use.

The `--no-deps` flags: msclap declares `numpy<2` and `transformers<5`, which SCAPES does not follow; magenta-rt declares git and cloud dependencies that SCAPES does not use. music2latent and the codec wrappers take their dependencies normally: a `pip install --dry-run` in a fresh environment showed nothing to change for them.

### As a library (pip)

SCAPES can also be installed as a package, for example by the demos. Install PyTorch first (the build for your CUDA, see `requirements.txt`), then the codec wrappers and SCAPES:

```bash
pip install "audio_codec_wrappers @ git+https://github.com/cordutie/audio_codec_wrappers"
pip install "scapes[atlas] @ git+https://github.com/cordutie/SCAPES@v0.1.0"
```

The package lists only the libraries without a special build. `msclap` (installed with `--no-deps`) and the jax set of SpectroStream are still installed as in the steps above.

Check that the installation works:

```bash
python -c "import SCAPES.pipeline, audio_codec_wrappers"
```

## Tutorials

Both tutorials run on Google Colab and on your computer. Each has a setup cell (it installs SCAPES on Colab, or finds the repository locally) and a parameters cell (the wav folder, the build folder and the config file; the codec and the generation settings are in the config file). Edit the placeholders in the parameters cell, or set the environment variables `SCAPES_<NAME>` (for example `SCAPES_WAV_PATH`) before you start Jupyter.

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cordutie/SCAPES/blob/main/quickstart/tutorial_simplified.ipynb) **Simplified tutorial** ([`quickstart/tutorial_simplified.ipynb`](quickstart/tutorial_simplified.ipynb)): prepare a dataset, train with the default settings and generate sounds, in a few cells.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cordutie/SCAPES/blob/main/quickstart/tutorial.ipynb) **Full tutorial** ([`quickstart/tutorial.ipynb`](quickstart/tutorial.ipynb)): the whole pipeline, one cell per stage: dataset preparation, training, and inference (full reconstruction, semantic reconstruction, interpolation and generation).

To run a notebook on your computer, activate the environment, install Jupyter and open the notebook:

```bash
pip install jupyter
jupyter notebook quickstart/tutorial_simplified.ipynb
```

The simplified tutorial calls two functions from [`quickstart/wrappers.py`](quickstart/wrappers.py): `data_preparation` and `training_loop`. Each reads one config file and a build folder (`dataset/` and `models/<size>_mem<ms>/`) and runs the step.

## Documentation

The generation modes (`encodec24`, `spectrostream`, `encodec48`, `music2latent`), their settings (`codec`, `conditioning_rate`, `conditioning_memory_seconds`) and their geometry are described in [docs/generation_modes.html](docs/generation_modes.html).

**GUI = True.** With `GUI = True` in the config, dataprep ends by fitting an atlas (`SCAPES/auxiliar/atlas.py`) on the CLAP embeddings and replaces them by their atlas version, so the model trains on those. The atlas is saved as `atlas.npz` in the dataset folder and copied into each model folder; `Atlas.load(path)` reads it (`encode`, `decode`, `autoencode`). Inference then rewrites the CLAP embeddings that come from audio with it. A dataset needs a new build folder to change this key.

## Citation

If you use SCAPES in your research, please cite:

```bibtex
@inproceedings{scapes,
  title     = {Semantically Conditioned Autoregresive Prior for Environmental Sounds},
  author    = {Esteban Gutiérrez and Lonce Wyse and Frederic Font and Xavier Serra},
  booktitle = {Proceedings of the International Conference on Digital Audio Effects (DAFx)},
  year      = {2026},
  address   = {Cambridge, USA}
}
```

## Acknowledgments

This work has been supported by the project "IA y Música: Cá-tedra en Inteligencia Artificial y Música (TSI-100929-2023-1)", funded by the "Secretaría de Estado de Digitalización e Inteligencia Artificial and the Unión Europea-Next Generation EU". We also acknowledge support from NVIDIA Corporation and Meta through academic grant programs. Additionally, we would like to express our sincere gratitude to Daniela Quimis, Clara Charbonnier, Jan Pol Obrador, Marcel Manzano, Omar Hamze, Jordi Fabregat, Eduard Herrera, and Eric Mas, for their valuable insights into extending the model to diverse audio sources, exploring alternative training techniques, and helping shape the design of an intuitive demonstration of SCAPES.
