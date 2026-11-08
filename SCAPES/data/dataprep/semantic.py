import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from SCAPES.data.dataprep.snippets import write_snippets
from SCAPES.data.representations import grid_starts_48k
from audio_codec_wrappers.audio import adapt_audio, load_audio

CLAP_VERSION = "2023"


# Raw audio of a file as stereo float32 [2, N] at 48 kHz
def load_audio_48k(path) -> torch.Tensor:
    x, sr = load_audio(path)
    return adapt_audio(x, sr, 48000, 2)[0]


# Default embedder: the CLAP wrapper, [B, 2, n] at 48 kHz -> [B, 1024]
def make_clap_embedder(device="cuda", random_extension=True):
    from SCAPES.auxiliar.clap_wrapper import CLAPWrapper
    model = CLAPWrapper(version=CLAP_VERSION, use_cuda=str(device).startswith("cuda"))

    def embed(audio, sr):
        return model.compute_embedding(audio, og_sr=sr, random_extension=random_extension)

    return embed


def precompute_semantic_annotations(dataset, embedder=None, batch_size: int = 128, device: str = "auto"):
    """
    CLAP embedding of the context window that starts at every grid start g * stride_frames of every file.
    One [G, 1024] float32 array per file, stored under clap/<clap_key>/ of the representation.
    embedder: Callable[[Tensor[B, 2, n] at 48 kHz, int], Tensor[B, 1024]].
    """
    if device in (None, "auto"):
        device = dataset.device
    if embedder is None:
        embedder = make_clap_embedder(device, dataset.semantic_random_extension)
    store, geom, key = dataset.store, dataset.geom, dataset.clap_key
    window = dataset.context_samples
    print(f"\n--- CLAP pre-computation ({key}) ---")
    with torch.no_grad():
        for fname in tqdm(dataset.filenames, desc="CLAP"):
            path = dataset.audio_path(fname)
            if store.has_grid_array("clap", key, path):
                continue
            audio = load_audio_48k(path)
            starts = grid_starts_48k(audio.shape[-1], geom, window)
            out = []
            for i in range(0, len(starts), batch_size):
                batch = torch.stack([audio[:, int(s): int(s) + window] for s in starts[i: i + batch_size]]).to(device)
                out.append(embedder(batch, 48000).detach().float().cpu())
            arr = torch.cat(out) if out else torch.zeros(0, 1024)
            store.save_grid_array("clap", key, path, arr)
    print("CLAP embeddings saved.")


ATLAS_FILE = "atlas.npz"
SNIPPETS_DIR = "snippets"        # next to the atlas: the audio of its representatives


def apply_atlas(dataset):
    """
    GUI = True: fit an Atlas (SCAPES/auxiliar/atlas.py) on all the CLAP arrays, save it as <dataset>/atlas.npz, and
    replace every array by its autoencode. The original embeddings are not kept: training reads the new ones.
    Not run again when atlas.npz exists (the arrays are then already the autoencoded ones).
    """
    from SCAPES.auxiliar.atlas import Atlas
    store, key = dataset.store, dataset.clap_key
    path = Path(dataset.dataset_path) / ATLAS_FILE
    if path.exists():
        print(f"Atlas already fitted: {path}")
        return path
    print("\n--- Atlas (GUI = True) ---")
    files = [dataset.audio_path(f) for f in dataset.filenames]
    arrays = [store.load_grid_array("clap", key, f) for f in files]
    atlas = Atlas.fit(torch.cat(arrays))
    stems = [Path(f).stem for f in files]
    atlas.add_names_and_prior([a for a in arrays], stems)
    stored = [atlas.autoencode(a.float()) if len(a) else a for a in arrays]
    # the representatives: found in the CLAP embeddings as they are, kept as the model will see them; then their audio
    geom = dataset.geom
    starts = [np.arange(len(a)) * geom.stride_samples / geom.sample_rate for a in arrays]
    reps = atlas.add_representatives(arrays, stored, starts, stems)
    paths = dict(zip(stems, files))
    n = write_snippets(reps, lambda stem: paths[stem], Path(dataset.dataset_path) / SNIPPETS_DIR)
    for f, arr in zip(files, stored):
        if len(arr):
            store.save_grid_array("clap", key, f, arr)
    atlas.save(path)
    print(f"Atlas with {atlas.K} clusters and {len(reps)} representatives ({n} audio snippets) saved to {path}; "
          f"CLAP embeddings replaced by their atlas version.")
    return path


def build_semantics_folder(csv_path, output_dir, dataset, overwrite: bool = False) -> int:
    """
    Reads the cherry-picking CSV (columns filename, start_sec, end_sec, flag, icon, description) and writes, for each
    row, the stored CLAP vectors of the grid starts g * stride_frames / frame_rate in [start_sec, end_sec) as a
    [N, 1024] tensor to output_dir/{flag}.pt, plus semantics.csv. Returns the number of flags created.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(csv_path).exists():
        print(f"  Cherry-picking CSV not found: {csv_path}; skipping.")
        return 0

    # filename lookup (handles both "file.wav" and "file")
    lookup = {}
    for fname in dataset.filenames:
        lookup[fname] = fname
        lookup[Path(fname).stem] = fname

    store, geom, key = dataset.store, dataset.geom, dataset.clap_key
    print(f"\n--- Building semantics folder ---")
    print(f"  CSV:     {csv_path}")
    print(f"  Output:  {output_dir}")

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("  CSV is empty; nothing to do.")
        return 0

    csv_rows_out = []
    created = 0
    skipped = 0
    for row in rows:
        raw_name = row.get("filename", "").strip()
        if not raw_name:
            skipped += 1
            continue
        name = lookup.get(raw_name) or lookup.get(raw_name + ".wav")
        if name is None:
            print(f"  File '{raw_name}' not found in the index; skipping.")
            skipped += 1
            continue
        try:
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
        except (KeyError, ValueError):
            print(f"  Invalid start_sec/end_sec for '{raw_name}'; skipping.")
            skipped += 1
            continue

        flag = row.get("flag", "").strip()
        if not flag:
            flag = f"{Path(raw_name).stem}_{start_sec}_{end_sec}"
            print(f"  No flag for '{raw_name}' [{start_sec}, {end_sec}]; using '{flag}'")
        icon = row.get("icon", "").strip()
        description = row.get("description", "").strip()

        out_path = output_dir / f"{flag}.pt"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        if end_sec <= start_sec:
            print(f"  Empty range [{start_sec}, {end_sec}] for '{raw_name}'; skipping.")
            skipped += 1
            continue

        # grid vectors whose start time lies in [start_sec, end_sec)
        arr = store.load_grid_array("clap", key, dataset.audio_path(name))
        times = torch.arange(arr.shape[0], dtype=torch.float64) * geom.stride_samples / geom.sample_rate
        chosen = (times >= start_sec) & (times < end_sec)
        if not bool(chosen.any()):
            print(f"  No semantic embeddings for '{raw_name}' [{start_sec}, {end_sec}]s; skipping.")
            skipped += 1
            continue

        torch.save(arr[chosen].contiguous().clone(), out_path)
        csv_rows_out.append({"name": flag, "icon": icon, "description": description})
        created += 1
        print(f"  {flag}.pt: {int(chosen.sum())} embeddings")

    # semantics.csv is a derived artifact, always rewritten
    with open(output_dir / "semantics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "icon", "description"])
        writer.writeheader()
        writer.writerows(csv_rows_out)

    print(f"Semantics folder done: {created} created, {skipped} skipped.")
    return created
