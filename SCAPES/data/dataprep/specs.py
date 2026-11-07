"""Dataprep specs of a build folder: what the stored dataset was made from (dataprep_specs.json), and the checks that
refuse a dataset or a wav folder that no longer matches the config. Pure Python: no torch, so it can be read and tested
without a GPU."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from SCAPES.errors import ConfigError
from SCAPES.generation import resolve_generation

SPEC_FILE = "dataprep_specs.json"
SPEC_FORMAT = 1
# the same tuple as AUDIO_EXTENSIONS of SCAPES/data/dataprep/atoms.py (not imported: that module needs torch)
AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.flac", "*.ogg")
# the spec keys that the config must match (the file also holds wavs, format and created, which are not compared here)
CONFIG_SPEC_KEYS = ("codec", "conditioning_rate", "semantic.context_seconds", "semantic.random_extension",
                    "splits.val_split_ratio", "splits.test_ratio", "dataset.hop_atoms", "seed", "cherry_picking", "GUI")
MAX_NAMES = 10


# Every audio file under the folder (as dataprep lists them), sorted by path: relative path, size and modification time.
def audio_files(folder) -> list[dict]:
    folder = Path(folder)
    found = sorted(p for ext in AUDIO_EXTENSIONS for p in folder.rglob(ext))
    out = []
    for p in found:
        st = p.stat()
        out.append({"path": p.relative_to(folder).as_posix(), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return out


# The spec dict of a config: the resolved values (codec name, rate of the nearest latent spacing, number of latents per
# step, atom spacing for atoms codecs) and the absolute cherry-picking path. config_file is the universal config file;
# a relative cherry_picking path starts at its folder. A cherry path that does not exist is refused.
def dataprep_specs(cfg, wav_folder, config_file) -> dict:
    res = resolve_generation(cfg.codec_name, cfg.conditioning_rate)
    cherry = ""
    if cfg.cherry_picking:
        path = Path(config_file).parent / cfg.cherry_picking
        if not path.is_file():
            raise ConfigError(f"{config_file}: cherry_picking {cfg.cherry_picking!r} is not a file ({path}). Fix the path, "
                              f"or set cherry_picking = \"\" for none.")
        cherry = str(path.resolve())
    return {
        "format": SPEC_FORMAT,
        "codec": res.codec,
        "conditioning_rate": round(res.rate, 9),
        "resolved.new_latents": res.n,
        "semantic.context_seconds": cfg.semantic_context_seconds,
        "semantic.random_extension": cfg.semantic_random_extension,
        "splits.val_split_ratio": cfg.val_split_ratio,
        "splits.test_ratio": cfg.test_split_ratio,
        # resolved: the autoregressive codecs have no atom spacing (1)
        "dataset.hop_atoms": cfg.hop_atoms if res.mode == "atoms" else 1,
        "seed": cfg.seed,
        "cherry_picking": cherry,
        "GUI": cfg.GUI,
        "wavs": {"folder": str(Path(wav_folder).resolve()), "files": audio_files(wav_folder)},
    }


# Writes the specs into the dataset folder, with the creation time. The file is written under a temporary name and
# renamed, so a half-written file never exists.
def write_specs(dataset_dir, specs) -> Path:
    dataset_dir = Path(dataset_dir)
    data = dict(specs, created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    target = dataset_dir / SPEC_FILE
    tmp = dataset_dir / (SPEC_FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, target)
    return target


# The stored specs of a dataset folder, or None when it has no spec file.
def read_specs(dataset_dir):
    path = Path(dataset_dir) / SPEC_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text())


# Differences between the stored specs and the current ones, as two lists of lines: the config keys, and the wav files
# (added, removed, changed). Empty lists: the same dataset can be reused. The rate is compared through its value: two
# rates with the same latent count have the same value, so the count is only shown.
def compare_specs(stored, current):
    config_lines = []
    for key in CONFIG_SPEC_KEYS:
        a, b = stored.get(key, False if key == "GUI" else None), current[key]
        if a != b:
            config_lines.append(f"  {key}: dataset has {_show(key, stored)}, config has {_show(key, current)}")

    wav_lines = []
    old = {f["path"]: f for f in stored.get("wavs", {}).get("files", [])}
    new = {f["path"]: f for f in current["wavs"]["files"]}
    added = sorted(p for p in new if p not in old)
    removed = sorted(p for p in old if p not in new)
    changed = []
    for p in sorted(set(old) & set(new)):
        if old[p]["size"] != new[p]["size"]:
            changed.append(f"{p} (size {old[p]['size']} -> {new[p]['size']})")
        elif old[p]["mtime_ns"] != new[p]["mtime_ns"]:
            changed.append(f"{p} (modification time changed)")
    wav_lines += _name_lines("added", added) + _name_lines("removed", removed) + _name_lines("changed", changed)
    return config_lines, wav_lines


# How one config value is written in a message: the rate also shows its number of latents per step.
def _show(key, specs):
    value = specs.get(key)
    if key == "conditioning_rate":
        return f"{value} Hz ({specs.get('resolved.new_latents')} latents per step)"
    return repr(value)


# One line per category with the names (at most MAX_NAMES), or nothing when the category is empty.
def _name_lines(label, names):
    if not names:
        return []
    shown = ", ".join(names[:MAX_NAMES])
    more = f" and {len(names) - MAX_NAMES} more" if len(names) > MAX_NAMES else ""
    return [f"  {label} ({len(names)}): {shown}{more}"]


# Decides whether the build folder's dataset can be used: "new" when there is no dataset folder, "reuse" when its specs
# match the config and the wav folder. Anything else is refused with a ConfigError that names the differences. Writes
# nothing.
def check_dataset(build_path, wav_folder, specs) -> str:
    build_path = Path(build_path)
    dataset = build_path / "dataset"
    if not dataset.exists():
        return "new"
    stored = read_specs(dataset)
    if stored is None:
        raise ConfigError(f"Build folder {build_path} has a dataset without {SPEC_FILE} (built before this version, or "
                          f"dataprep did not finish). Use a different build folder, or delete {dataset} to prepare it "
                          f"again. Nothing was written.")
    config_lines, wav_lines = compare_specs(stored, specs)
    if config_lines or wav_lines:
        raise ConfigError(_refusal_message(build_path, wav_folder, config_lines, wav_lines))
    _check_link(dataset, wav_folder)
    return "reuse"


# The refusal text: the build folder, the differences (config keys first, then wav files), the advice, and "Nothing was
# written." The wav block names the wav folder the dataset was prepared from when it differs from the one given.
def _refusal_message(build_path, wav_folder, config_lines, wav_lines):
    parts = []
    if config_lines:
        parts.append(f"Build folder {build_path} cannot be used with this config: its dataset was prepared with other "
                     f"dataprep specs.")
        parts += config_lines
    if wav_lines:
        parts.append(f"Build folder {build_path} cannot be used: the wav folder {Path(wav_folder).resolve()} is not the "
                     f"one the dataset was prepared from.")
        parts += wav_lines
    if config_lines and wav_lines:
        advice = "Use a different build folder, or change the config back and use the same wavs."
    elif config_lines:
        advice = "Use a different build folder, or change the config back."
    else:
        advice = "Use the same wavs, or a different build folder."
    parts.append(f"{advice} Nothing was written.")
    return "\n".join(parts)


# dataset/raw must be a symbolic link to the wav folder (no copy of the audio). A link to another folder is refused.
def _check_link(dataset, wav_folder):
    raw = dataset / "raw"
    if not raw.is_symlink() or os.path.realpath(raw) != str(Path(wav_folder).resolve()):
        raise ConfigError(f"{dataset / 'raw'} is not a link to the wav folder {Path(wav_folder).resolve()}. Use the same "
                          f"wav folder, or a different build folder. Nothing was written.")
