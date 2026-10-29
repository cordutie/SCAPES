# SCAPES

**SCAPES** (Semantically Conditioned Auto-Regressive Prior for Environmental Sounds) is a generative audio framework for high-fidelity environmental texture synthesis with semantic control.

## Features

- **Semantic control** with CLAP-based conditioning (no manual labels)
- **Continuous latent generation** (no token quantization bottleneck)
- **Efficient training** on consumer GPUs
- **Long-term texture stability** in autoregressive generation
- **Smooth semantic interpolation** between environmental classes

## Demo

[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/cordutie/SCAPES-demo)

Try the online demo to generate environmental sounds with semantic control.

## Quick Start

### Simplified Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cordutie/SCAPES/blob/main/quickstart/tutorial_simplified.ipynb) 

Train with default parameters and generate sounds quickly.


### Full Tutorial
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cordutie/SCAPES/blob/main/quickstart/tutorial_full.ipynb) 

Dive deep into the code and customize the training loop.

## Installation

SCAPES installs together with its codec wrappers ([audio_codec_wrappers](https://github.com/cordutie/audio_codec_wrappers), included as the git submodule `external/audio_codec_wrappers/`) in a single virtual environment.

Requirements: Linux, Python 3.12, an NVIDIA GPU with a recent driver (CUDA 12), git.

```bash
git clone --recurse-submodules https://github.com/cordutie/SCAPES.git
cd SCAPES
git checkout v2        # until v2 is merged into main
git submodule update --init   # only needed if you cloned without --recurse-submodules

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 1. Pinned packages (torch, jax, ...). If you need a specific CUDA build of PyTorch, install it first.
pip install -r requirements.txt

# 2. Packages that must be installed without their own dependency pins:
#    msclap, magenta-rt, codicodec, and the wrappers (editable, from ./external/audio_codec_wrappers)
pip install --no-deps -r requirements-nodeps.txt

# 3. One-time download of the SpectroStream and CoDiCodec weights (EnCodec downloads itself on first use)
for c in spectrostream codicodec; do python external/audio_codec_wrappers/scripts/fetch_weights.py --codec $c; done
```

Check the installation (needs the GPU; the last test trains 20 steps for each of the four codecs and takes a few minutes):

```bash
python -m pytest -q tests_new/test_wrappers_integration.py
```

Notes:
- The wrappers are pinned by the submodule commit. To move to a newer version: `git -C external/audio_codec_wrappers pull origin main`, then `git add external/audio_codec_wrappers` and commit. To change the wrappers, edit them in `external/audio_codec_wrappers/` (it is its own git repository; commit and push there, then commit the new submodule pointer here).
- The weights are stored in `~/.cache/audio_codec_wrappers` (set `ACW_CACHE` to change it).
- `msclap` asks for numpy < 2; this is why it is installed with `--no-deps`. Its embeddings are unchanged with numpy 2 (checked by `tests/test_env_scapes.py`).

## Pipeline

1. **Audio → Atoms**: Segment audio into overlapping windows and encode with EnCodec
2. **Semantic Context**: Precompute CLAP embeddings for conditioning

<p align="center">
  <img src="SCAPES/auxiliar/figures/dataprep.svg" width="100%" alt="Data Preparation Pipeline">
</p>

3. **Flow Training**: Train conditional CNF (Flow Matching) to generate atoms from context
4. **Autoregressive Synthesis**: Generate atoms iteratively with overlap-and-add crossfade

<p align="center">
  <img src="SCAPES/auxiliar/figures/architecture.svg" width="100%" alt="SCAPES Architecture">
</p>

## Future Work

Potential areas for collaboration:

- **New Audio Codecs**: Explore alternatives to EnCodec
- **Perceptual Losses**: Train surrogate losses beyond MSE for latent comparison
- **Alternative Conditioning**: Experiment with features beyond CLAP

## Acknowledgments

This work has been supported by the project "IA y Música: Cá-tedra en Inteligencia Artificial y Música (TSI-100929-2023-1)", funded by the "Secretaría de Estado de Digitalización e Inteligencia Artificial and the Unión Europea-Next Generation EU". We also acknowledge support from NVIDIA Corporation and Meta through academic grant programs. Additionally, we would like to express our sincere gratitude to Daniela Quimis, Clara Charbonnier, Jan Pol Obrador, Marcel Manzano, Omar Hamze, Jordi Fabregat, Eduard Herrera, and Eric Mas, for their valuable insights into extending the model to diverse audio sources, exploring alternative training techniques, and helping shape the design of an intuitive demonstration of SCAPES.

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
