# SCAPES

**SCAPES** (Semantically Conditioned Auto-Regressive Prior for Environmental Sounds) is a lightweight generative audio framework for high-fidelity environmental texture synthesis with **semantic control**.

It combines:
- continuous EnCodec latent representations,
- CLAP-based semantic conditioning,
- a Transformer-parameterized Continuous Normalizing Flow (CNF),
- and segment-level autoregressive generation with overlap-aware decoding.

This repository includes the core library, quickstart scripts, experiment notebooks, and model/data organization used in the paper workflow.

---

## Why SCAPES?

SCAPES is built for open, small-scale, reproducible generative audio research:

- **Semantic control with no manual labels** (CLAP-based conditioning).
- **Continuous latent generation** (no token quantization bottleneck).
- **Efficient training** on consumer GPUs.
- **Strong long-term texture stability** in autoregressive generation.
- **Smooth semantic interpolation** between environmental classes.

---

## TLDR Method

1. **Audio → atoms**: audio is segmented into overlapping windows and encoded with EnCodec into latent representations (+ scale). We call this an atom.
2. **Semantic context**: CLAP embeddings are precomputed to be used as conditioning semantic context.
3. **Flow training**: a conditional CNF (Flow Matching objective) learns to generate the next atom from past atoms + semantic context.
4. **Autoregressive synthesis**: atoms are generated iteratively and stitched using overlap-and-add with crossfade masking.

---

## Installation

### 1) Create/activate environment

Use your preferred environment manager. Example with `venv`:

```bash
cd /path/to/SCAPES
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Current `requirements.txt` includes:
- `torch`, `torchaudio`, `torchdiffeq`
- `librosa`, `soundfile`, `numpy`, `matplotlib`, `tqdm`
- `transformers`

> Note: depending on your CUDA/driver setup, you may want to install a CUDA-specific PyTorch build first.

---

## Quickstart pipeline notebooks

You can quickly prepare your data and train a model using the quickstart notebook. Use the quickstart notebooks in this exact order:

1. `quickstart/1. dataprep.ipynb`  
	Build atoms, splits, and annotations.
2. `quickstart/2. training.ipynb`  
	Train the flow model.
3. `quickstart/3. inference.ipynb`  
	Run generation/resynthesis and inspect outputs.

---

## Acknowledgments

SCAPES builds on a rich ecosystem including EnCodec, CLAP, Neural ODE/CNF methods, flow matching and Freesound.

## Future work

There is a lot of future work to be done in SCAPES in case anyone wants to collaborate in extending the capacities of the model. The following is a list of options:
1. New Neural Audio Codecs appear every year. Maybe EnCodec is not the best one suited for the job anymore?
2. Currently SCAPES uses a MSE loss to compare EnCodec latents, however this loss has proven to be not enough in some scenarios. Maybe one could train a surrogate loss that learns how to replicate other perceptual audio losses between audio latents.
3. Currently SCAPES is conditioned on CLAP. A lot of exploration could be made by trying other conditioning features.
