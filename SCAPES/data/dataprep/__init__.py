from .atoms import latents_maker, list_audio_files
from .structure import precompute_structure_annotations, structure_key
from .semantic import build_semantics_folder, make_clap_embedder, precompute_semantic_annotations


def precompute_annotations(*_args, **_kwargs):
    raise RuntimeError(
        "precompute_annotations was split into precompute_semantic_annotations "
        "and precompute_structure_annotations."
    )


__all__ = [
    "latents_maker",
    "list_audio_files",
    "build_semantics_folder",
    "make_clap_embedder",
    "precompute_semantic_annotations",
    "precompute_structure_annotations",
    "structure_key",
    "precompute_annotations",
]
