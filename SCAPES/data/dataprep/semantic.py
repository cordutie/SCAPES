import csv
from pathlib import Path

import torch
from tqdm import tqdm

from SCAPES.auxiliar.clap_wrapper import CLAPWrapper


def precompute_semantic_annotations(
    dataset,
    model=None,
    batch_size: int = 32,
    device: str = "auto",
):
    """
    Computes CLAP embeddings from target_context_audio and saves them to:
    annotations/semantic/semantic_<idx>.pt
    """
    if dataset.annotations_dir is None:
        raise ValueError("Dataset annotations_dir is not set.")

    save_dir = dataset.annotations_dir / "semantic"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting Semantic Pre-computation ---")
    print(f"Save Path: {save_dir}")

    if device in [None, "auto"]:
        device = getattr(dataset, "device", "cpu")

    if model is None:
        use_cuda = device == "cuda"
        model = CLAPWrapper(version="2023", use_cuda=use_cuda)

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing semantic"):
            batch_inputs = []
            indices_to_compute = []

            for j in range(batch_size):
                idx = i + j
                if idx >= len(dataset):
                    break

                save_path = save_dir / f"semantic_{idx}.pt"
                if save_path.exists():
                    continue

                indices_to_compute.append(idx)
                raw_audio = dataset.get_raw_audio(idx, part="context")
                batch_inputs.append(raw_audio)

            if not indices_to_compute:
                continue

            batched_audio = torch.stack(batch_inputs).to(device)
            random_extension = getattr(dataset, "semantic_random_extension", True)
            embedding = model.compute_embedding(
                batched_audio,
                og_sr=dataset.sr,
                random_extension=random_extension,
            )

            embedding = embedding.detach().cpu()

            for k, idx in enumerate(indices_to_compute):
                save_path = save_dir / f"semantic_{idx}.pt"
                torch.save(embedding[k], save_path)

    print("✅ Semantic annotations saved.")


def build_semantics_folder(
    csv_path,
    semantic_dir,
    output_dir,
    dataset,
    overwrite: bool = False,
) -> int:
    """
    Reads config/cherry_picking.csv and extracts precomputed CLAP embeddings
    for the specified (file, time_range) segments, saving each as a stacked
    tensor [N, 1024] to output_dir/{flag}.pt, plus semantics.csv.

    CSV columns: filename, start_sec, end_sec, flag, icon, description

    Returns number of flags created.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(csv_path).exists():
        print(f"  Cherry-picking CSV not found: {csv_path}; skipping.")
        return 0

    semantic_dir = Path(semantic_dir)

    # Build filename lookup (handle both "file.wav" and "file")
    fname_to_manifest = {}
    for fname in dataset.filenames:
        fname_to_manifest[fname] = fname
        fname_to_manifest[Path(fname).stem] = fname

    # Build per-filename reverse lookup: filename → [(index, start_atom), ...]
    idx_to_entry = {idx: entry for idx, entry in enumerate(dataset.all_indices)}
    file_to_indices: dict[str, list[tuple[int, int]]] = {}
    for idx, (fname, start) in idx_to_entry.items():
        file_to_indices.setdefault(fname, []).append((idx, start))

    hop_sec = dataset.atoms_hop_frames / dataset.frame_rate

    print(f"\n--- Building semantics folder ---")
    print(f"  CSV:     {csv_path}")
    print(f"  Output:  {output_dir}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

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

        manifest_name = fname_to_manifest.get(raw_name) or fname_to_manifest.get(
            raw_name + ".wav"
        )
        if manifest_name is None:
            print(f"  ⚠️  File '{raw_name}' not found in manifest; skipping.")
            skipped += 1
            continue

        try:
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
        except (KeyError, ValueError):
            print(f"  ⚠️  Invalid start_sec/end_sec for '{raw_name}'; skipping.")
            skipped += 1
            continue

        flag = row.get("flag", "").strip()
        if not flag:
            flag = f"{Path(raw_name).stem}_{start_sec}_{end_sec}"
            print(f"  ⚠️  No flag for '{raw_name}' [{start_sec}, {end_sec}]; using '{flag}'")

        icon = row.get("icon", "").strip()
        description = row.get("description", "").strip()

        out_path = output_dir / f"{flag}.pt"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        if end_sec <= start_sec:
            print(f"  ⚠️  Empty range [{start_sec}, {end_sec}] for '{raw_name}'; skipping.")
            skipped += 1
            continue

        # Compute atom range
        start_atom = int(start_sec / hop_sec)
        end_atom = int(end_sec / hop_sec)

        # Collect matching indices
        matching_indices: list[int] = []
        entries = file_to_indices.get(manifest_name, [])
        for idx, atom_pos in entries:
            if start_atom <= atom_pos < end_atom:
                matching_indices.append(idx)

        if not matching_indices:
            print(
                f"  ⚠️  No semantic embeddings for '{raw_name}' "
                f"[{start_sec}, {end_sec}]s; skipping."
            )
            skipped += 1
            continue

        # Load and stack embeddings
        embeddings = []
        for idx in matching_indices:
            emb_path = semantic_dir / f"semantic_{idx}.pt"
            if not emb_path.exists():
                continue
            embeddings.append(torch.load(emb_path, map_location="cpu", weights_only=True))

        if not embeddings:
            print(f"  ⚠️  No valid embeddings for '{raw_name}' [{start_sec}, {end_sec}]; skipping.")
            skipped += 1
            continue

        stacked = torch.stack(embeddings)
        torch.save(stacked, out_path)
        csv_rows_out.append({"name": flag, "icon": icon, "description": description})
        created += 1
        print(f"  ✓ {flag}.pt — {len(embeddings)} embeddings")

    # Write semantics.csv (overwrite always — it's a derived artifact)
    csv_out_path = output_dir / "semantics.csv"
    with open(csv_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "icon", "description"])
        writer.writeheader()
        writer.writerows(csv_rows_out)

    print(f"✅ Semantics folder done: {created} created, {skipped} skipped.")
    return created


def precompute_gui_annotations(dataset):
    """Deprecated: use build_semantics_folder instead."""
    csv_path = Path(dataset.dataset_path) / "config" / "cherry_picking.csv"
    if not csv_path.exists():
        return
    save_dir = dataset.annotations_dir / "GUI"
    build_semantics_folder(
        csv_path=csv_path,
        semantic_dir=dataset.annotations_dir / "semantic",
        output_dir=save_dir,
        dataset=dataset,
    )
