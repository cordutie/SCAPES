"""Entry points of the V2 pipeline. B2 holds only build_run_config; run_dataprep, run_training and run_generation come later."""
from pathlib import Path

# gin keys owned by the codec snippet: a dataset's own lines in these namespaces are dropped
_GEOMETRY_PREFIXES = ("atoms.", "ar.", "codec.", "generation.", "dataset.memory_buffer_atoms")


def _is_geometry_line(line: str) -> bool:
    key = line.split("=", 1)[0].strip() if "=" in line else ""
    return key.startswith(_GEOMETRY_PREFIXES)


# Writes out_dir/{dataprep,training}.gin: the dataset's files without their geometry lines, then each snippet, then the
# overrides (the last occurrence of a key wins). The dataset's own files are never edited.
def build_run_config(dataset_config_dir, snippets: list[Path], overrides: list[str], out_dir) -> Path:
    src, out = Path(dataset_config_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tail = []
    for snippet in snippets:
        tail += Path(snippet).read_text().splitlines()
    tail += list(overrides)
    for name in ("dataprep.gin", "training.gin"):
        kept = [ln for ln in (src / name).read_text().splitlines() if not _is_geometry_line(ln)]
        (out / name).write_text("\n".join(kept + [""] + tail) + "\n")
    return out
