"""Run logging for training: step history, per-file validation loss, peak GPU memory, environment record, figures."""
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import torch
from matplotlib.figure import Figure


# Output of a command, or None if it fails
def _run(cmd, cwd=None):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


# HEAD and dirty flag (tracked files changed) of the git repo that holds `path`
def _git_record(path):
    if path is None:
        return {"head": None, "dirty": None}
    head = _run(["git", "rev-parse", "HEAD"], cwd=path)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path)
    return {"head": head, "dirty": None if status is None else bool(status)}


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# Versions, GPU and git state. jax is read from the package metadata, not imported (importing it would take GPU memory).
def environment_record() -> dict:
    import numpy
    scapes_root = Path(__file__).resolve().parents[2]
    try:
        import audio_codec_wrappers
        wrappers_root = Path(audio_codec_wrappers.__file__).resolve().parents[1]
    except ImportError:
        wrappers_root = None

    lock = scapes_root / "requirements.lock"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "jax": _package_version("jax"),
        "numpy": numpy.__version__,
        "gpu": gpu,
        "driver": driver.splitlines()[0] if driver else None,
        "requirements_lock": str(lock) if lock.exists() else None,
        "requirements_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "git": {"SCAPES": _git_record(scapes_root), "audio_codec_wrappers": _git_record(wrappers_root)},
    }


class RunLogger:
    """Writes <out_dir>/{history_steps,val_per_file,peak_memory,env}.json and two figures."""

    def __init__(self, out_dir, log_every: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = max(1, int(log_every))
        self.history = []
        self.val_history = []
        self.t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _write(self, name, obj):
        (self.out_dir / name).write_text(json.dumps(obj, indent=1))

    # One entry per logged step; the file is rewritten at each entry so a crash keeps the history so far.
    def log_step(self, step, loss, loss_latent, loss_scale, grad_norm, lr) -> None:
        self.history.append({
            "step": int(step), "loss": float(loss), "loss_latent": float(loss_latent), "loss_scale": float(loss_scale),
            "grad_norm": float(grad_norm), "lr": float(lr), "time": time.time() - self.t0,
        })
        self._write("history_steps.json", self.history)

    # Validation loss of every file (mean over its samples and the stratified s values) at a given step
    def log_val(self, step, per_file: dict) -> None:
        self.val_history.append({"step": int(step), "per_file": {k: float(v) for k, v in per_file.items()}})
        self._write("val_per_file.json", self.val_history)

    def finish(self) -> None:
        peak = {"max_memory_allocated_bytes": 0, "max_memory_reserved_bytes": 0}
        if torch.cuda.is_available():
            peak = {"max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        self._write("peak_memory.json", peak)
        self._write("env.json", environment_record())
        self._write("history_steps.json", self.history)
        self._figures()

    # loss_vs_step.png: train loss terms and grad norm; val_per_file.png: one coloured line per validation file
    def _figures(self):
        if self.history:
            steps = [h["step"] for h in self.history]
            fig = Figure(figsize=(12, 4))
            ax = fig.subplots(1, 2)
            for key, colour in (("loss", "tab:purple"), ("loss_latent", "tab:blue"), ("loss_scale", "tab:green")):
                ax[0].plot(steps, [h[key] for h in self.history], label=key, color=colour, linewidth=1.5)
            ax[0].set_xlabel("step")
            ax[0].set_ylabel("train loss (mean over each logging window)")
            ax[0].grid(True, linestyle="--", alpha=0.6)
            ax[0].legend()
            ax[1].plot(steps, [h["grad_norm"] for h in self.history], color="tab:red", label="grad_norm")
            ax[1].set_xlabel("step")
            ax[1].set_ylabel("gradient norm before clipping")
            ax[1].grid(True, linestyle="--", alpha=0.6)
            ax[1].legend()
            fig.tight_layout()
            fig.savefig(self.out_dir / "loss_vs_step.png", dpi=120)
        if self.val_history:
            names = sorted({k for v in self.val_history for k in v["per_file"]})
            colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink",
                       "tab:gray", "tab:olive", "tab:cyan"]
            fig = Figure(figsize=(7, 4))
            ax = fig.subplots()
            for i, name in enumerate(names):
                xs = [v["step"] for v in self.val_history if name in v["per_file"]]
                ys = [v["per_file"][name] for v in self.val_history if name in v["per_file"]]
                ax.plot(xs, ys, marker="o", color=colours[i % len(colours)], label=name,
                        linestyle="-" if i < len(colours) else "--")
            ax.set_xlabel("step")
            ax.set_ylabel("validation loss")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(self.out_dir / "val_per_file.png", dpi=120)
