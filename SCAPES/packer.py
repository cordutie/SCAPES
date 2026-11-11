"""GUI packer: one zip (or, with zip=False, one folder) per trained model of a build folder, with only what the SCAPES GUI needs to run it.

A zip holds a single top folder <name>/ (a folder pack is that same <name>/ folder) with
    atlas.npz                          the atlas of the build (the model needs GUI = True)
    snippets/                          the audio of the atlas representatives (Score mode)
    checkpoints/inference.gin          the geometry of the model
    checkpoints/best_flow_model.pt     the best checkpoints (the last ones only if the run has no best)
    checkpoints/best_local_encoder.pt
    checkpoints/latent_stats.pt
Everything else (last checkpoints, logs, loss curves, validation audio, semantics) is left out. Nothing in the build folder
is changed or deleted, and an existing zip or folder is never overwritten.
"""
import shutil
import zipfile
from pathlib import Path

from SCAPES.errors import ConfigError

CHECKPOINT_BEST = ["best_flow_model.pt", "best_local_encoder.pt"]
CHECKPOINT_LAST = ["last_flow_model.pt", "last_local_encoder.pt"]
CHECKPOINT_COMMON = ["inference.gin", "latent_stats.pt"]


# The files of one model folder that go into its zip, as paths relative to the model folder. Raises ConfigError (and says
# what is missing) when the folder cannot run in the GUI.
def gui_files(model_dir) -> list[Path]:
    model_dir = Path(model_dir)
    ckpt = model_dir / "checkpoints"
    if not (model_dir / "atlas.npz").exists():
        raise ConfigError(f"{model_dir}: no atlas.npz (the model was not trained with GUI = True)")
    # same rule as FlowInference: the best checkpoints, the last ones only when there is no best flow model
    models = CHECKPOINT_BEST if (ckpt / "best_flow_model.pt").exists() else CHECKPOINT_LAST
    wanted = [Path("checkpoints") / f for f in CHECKPOINT_COMMON + models] + [Path("atlas.npz")]
    missing = [str(p) for p in wanted if not (model_dir / p).exists()]
    if missing:
        raise ConfigError(f"{model_dir}: missing {missing}")
    snippets = sorted(p.relative_to(model_dir) for p in (model_dir / "snippets").rglob("*") if p.is_file()) \
        if (model_dir / "snippets").is_dir() else []
    return wanted + snippets


# One pack for one model folder. zip=True: the zip out_path, with the top folder `name` inside; zip=False: the folder out_path
# (named `name` by the caller), without a zip. Built under a temporary name and renamed at the end.
def pack_model(model_dir, out_path, name=None, zip=True) -> Path:
    model_dir, out_path = Path(model_dir), Path(out_path)
    if out_path.exists():
        raise ConfigError(f"{out_path} already exists; it is not overwritten")
    files = gui_files(model_dir)
    top = name or out_path.stem
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    if tmp.exists():
        raise ConfigError(f"{tmp} exists (an interrupted pack?): look at it and remove it by hand")
    if zip:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in files:
                z.write(model_dir / rel, f"{top}/{rel.as_posix()}")
    else:
        for rel in files:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_dir / rel, tmp / rel)
    tmp.rename(out_path)
    return out_path


# One pack per model of the build folder (<build>/models/*), into out_dir (default <build>/gui_zips): a zip each, or with
# zip=False a folder each. The pack and its top folder are named <build folder>_<model folder> unless `name` is given: a function (build_path, model_dir) -> str.
# A model that cannot run in the GUI (no atlas, unfinished) is skipped and reported, not packed. Returns the zip paths.
def gui_packer(build_path, out_dir=None, name=None, zip=True) -> list[Path]:
    build_path = Path(build_path)
    models_dir = build_path / "models"
    if not models_dir.is_dir():
        raise ConfigError(f"{build_path} has no models/ folder")
    out_dir = Path(out_dir) if out_dir is not None else build_path / "gui_zips"
    made, skipped = [], []
    for model_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        label = name(build_path, model_dir) if name else f"{build_path.resolve().name}_{model_dir.name}"
        try:
            zip_path = pack_model(model_dir, out_dir / (f"{label}.zip" if zip else label), name=label, zip=zip)
        except ConfigError as err:
            skipped.append(str(err)); continue
        size = zip_path.stat().st_size if zip else sum(f.stat().st_size for f in zip_path.rglob("*") if f.is_file())
        print(f"{zip_path.name}: {size / 1e6:.1f} MB")
        made.append(zip_path)
    for s in skipped:
        print("SKIPPED:", s)
    print(f"{len(made)} {'zip(s)' if zip else 'folder(s)'} in {out_dir}" + (f", {len(skipped)} skipped" if skipped else ""))
    return made
