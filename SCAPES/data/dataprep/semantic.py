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
