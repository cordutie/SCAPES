from .atoms import atoms_maker, extractor_atoms, make_atom_path, torch_save_atoms, to_cpu
from .structure import precompute_structure_annotations
from .semantic import precompute_semantic_annotations


def precompute_annotations(*_args, **_kwargs):
    raise RuntimeError(
        "precompute_annotations was split into precompute_semantic_annotations "
        "and precompute_structure_annotations."
    )


__all__ = [
    "atoms_maker",
    "extractor_atoms",
    "make_atom_path",
    "torch_save_atoms",
    "to_cpu",
    "precompute_semantic_annotations",
    "precompute_structure_annotations",
    "precompute_annotations",
]
