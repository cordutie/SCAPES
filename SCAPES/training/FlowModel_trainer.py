import os
import json
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from SCAPES.auxiliar.losses_flow import (
    flow_matching_loss, spectral_flow_matching_loss,
    time_phase_regularizer, fft_phase_regularizer,
    time_to_spectral, spectral_to_time,
)
from SCAPES.data.config_loader import TrainingConfig
from SCAPES.models.factorization.AtomDiscriminator import AtomDiscriminator
from SCAPES.training.run_logging import RunLogger

# Deterministic validation (design D17): four stratified flow times per validation sample
VAL_S_VALUES = (0.125, 0.375, 0.625, 0.875)


# ==========================================
# PAST DROPOUT AND PAST NOISE
# ==========================================

# Atoms: with probability p the oldest j ~ U{1..M} past units are masked (V1 rule). bool [batch, M], a prefix from index 0.
def past_dropout_mask(batch_size: int, M: int, p: float, device="cpu") -> torch.Tensor:
    rows = torch.rand(batch_size, device=device) < p
    j = torch.randint(1, M + 1, (batch_size,), device=device)
    return rows[:, None] & (torch.arange(M, device=device)[None, :] < j[:, None])


# AR (one window of W frames, blocks of B): with probability p the oldest W - k*B frames are masked, k ~ U{0..ceil(W/B)-1}.
# These are the histories that exist at generation start. bool [batch, W], a prefix from the oldest frame.
def past_dropout_frame_mask(batch_size: int, W: int, B: int, p: float, device="cpu") -> torch.Tensor:
    rows = torch.rand(batch_size, device=device) < p
    k = torch.randint(0, math.ceil(W / B), (batch_size,), device=device)
    length = W - k * B
    return rows[:, None] & (torch.arange(W, device=device)[None, :] < length[:, None])


# Exposure-bias noise on the past: past + alpha * xi, alpha ~ U[0, gamma] per sample, xi ~ N(0, I). Never in place.
def apply_past_noise(past: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma == 0:
        return past
    alpha = torch.rand(past.size(0), device=past.device, dtype=past.dtype) * gamma
    alpha = alpha.view(-1, *([1] * (past.dim() - 1)))
    return past + alpha * torch.randn_like(past)


class FlowTrainer:
    def __init__(
            self,
            model,
            local_encoder,
            train_loader,
            dataset,
            processor,
            optimizer,
            config: TrainingConfig,
            val_loader=None,
            model_path="checkpoints/flow_model",
            resume_from=None
        ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(device)
        self.local_encoder = local_encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset = dataset
        self.processor = processor
        self.optimizer = optimizer
        self.device = device
        self.context_source = config.context_source
        self.past_dropout = config.past_dropout
        self.conditioning_dropout = config.conditioning_dropout
        self.past_noise = config.past_noise
        self.max_steps = config.max_steps
        self.log_every = config.log_every
        self.batch_size = config.batch_size
        self.global_step = 0
        self._last_val_step = None
        self._last_val_metric = None
        self._last_val_per_file = {}

        # Geometry and dimensions come from the dataset, not from gin
        self.geom = geom = dataset.geom
        self.mode = geom.mode
        self.n_latent = geom.latent_dim
        self.has_scale = geom.has_scale
        self.unit_frames = geom.unit_frames          # F (atoms) | B (AR)
        self.prefix_frames = geom.prefix_frames      # o (AR clean-prefix fallback)
        self.num_past_atoms = geom.memory_units      # M (atoms) | 1 (AR)
        self.memory_frames = geom.memory_frames      # F (atoms) | W (AR)
        self.atom_frames = geom.unit_frames
        self.context_vector_dim = 1024
        self.structure_dim = len(dataset.structure_feature_names)

        self.spectral_mode = config.spectral_representation
        if self.spectral_mode:
            if geom.codec != "encodec48k":
                raise NotImplementedError("spectral_representation is only available for encodec48k (D13)")
            self.spectral_frames = self.unit_frames // 2 + 1
            self.frame_dim = 385  # 128*3 + 1
            model_frames = self.spectral_frames
        else:
            self.spectral_frames = self.unit_frames
            self.frame_dim = geom.frame_dim
            model_frames = self.unit_frames + self.prefix_frames
        if config.use_discriminator and geom.codec != "encodec48k":
            raise NotImplementedError("use_discriminator is only available for encodec48k (D13)")

        # Build config dicts for JSON serialization
        self.model_config = {
            "frame_dim": self.frame_dim,
            "context_vector_dim": self.context_vector_dim,
            "num_past_atoms": self.num_past_atoms,
            "frames_per_atom": model_frames,
            "memory_frames": self.memory_frames,
            "semantic_random_extension": dataset.semantic_random_extension,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_layers": config.num_layers,
            "dim_feedforward": config.dim_feedforward,
            "structure_dim": self.structure_dim,
            "cfg_scale": config.cfg_scale,
        }
        self.encoder_config = {
            "in_channels": self.frame_dim,
            "hidden_dim": config.local_encoder_hidden_dim,
            "out_channels": config.d_model,
            "time_entanglement": config.local_encoder_time_entanglement,
            "temporal_compression": config.local_encoder_temporal_compression,
        }
        if self.mode == "autoregressive" and config.local_encoder_temporal_compression != 1:
            raise NotImplementedError("autoregressive mode needs local_encoder.temporal_compression = 1 "
                                      "(one encoded frame per latent frame)")

        self.use_structure = self.structure_dim > 0

        if self.use_structure and "target_structure" not in self.dataset.requested_keys:
            raise ValueError("Model expects structure but dataset does not request it")

        if self.use_structure:
            self.model_config["structure_feature_names"] = getattr(self.dataset, "structure_feature_names", None)

        # ─── Regularizers — incompatible with spectral representation ───
        self.regularizers_and_weights = config.regularizers_and_weights
        if self.spectral_mode and self.regularizers_and_weights:
            raise ValueError(
                "Regularizers (time_phase, fft_phase) are incompatible with spectral_representation=True. "
                "Phase is learned explicitly in the spectral domain. "
                "Set training.regularizers_and_weights = [] or remove them from training.gin."
            )

        # ─── Adversarial discriminator (optional, 2-stage) ───
        self.use_discriminator = config.use_discriminator
        self.disc_epochs = config.disc_epochs
        self.stage2_epochs = config.stage2_epochs
        self.x_hat_buffer = []          # buffer of (X_hat_latent, s) pairs for discriminator
        self.adv_buffer_max = 5000      # max samples in buffer
        self.adv_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.checkpoint_freq = config.checkpoint_freq
        self.save_resume_states = config.save_resume_states
        self.start_epoch = 1
        self.best_metric = float('inf')

        self.val_duration = config.val_duration
        val_audio_files = config.val_files
        if val_audio_files is None:
            self.val_audio_files = []
        elif isinstance(val_audio_files, str):
            self.val_audio_files = [val_audio_files]
        elif val_audio_files:
            resolved = []
            for item in val_audio_files:
                if item == "all":
                    resolved = list(dataset.filenames)
                    break
                elif item.startswith("random="):
                    n = int(item.split("=", 1)[1])
                    import random as _random
                    chosen = _random.sample(dataset.filenames, min(n, len(dataset.filenames)))
                    resolved.extend(chosen)
                else:
                    resolved.append(item)
            self.val_audio_files = resolved
        else:
            self.val_audio_files = dataset.filenames[:1]

        self.model_path = Path(model_path)
        self.ckpt_dir = self.model_path / "checkpoints"
        self.loss_dir = self.model_path / "loss"
        self.val_dir = self.model_path / "validation"

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        existing_ckpts = list(self.ckpt_dir.glob("*.pt"))
        if existing_ckpts and not resume_from:
            raise FileExistsError(
                f"⚠️ Model directory '{self.ckpt_dir}' already contains checkpoints! "
                "To prevent accidentally overwriting your trained models, please either specify a different `model_path`, "
                "or set `resume_from='latest'` (or a specific epoch number) to continue training."
            )

        # Per-channel standardisation of the latents: statistics of the train region, saved with the checkpoints
        self.standardize = config.standardize_latents
        self.mu = self.std = None
        if self.standardize:
            stats_path = self.ckpt_dir / "latent_stats.pt"
            if resume_from and stats_path.exists():
                stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            else:
                mu, std = dataset.latent_stats(dataset.get_splits()[0])
                stats = {"mu": mu, "std": std}
                torch.save(stats, stats_path)
            self.mu, self.std = stats["mu"].to(device), stats["std"].to(device)

        self._save_inference_gin()
        self.logger = RunLogger(self.model_path / "logs", self.log_every)

        if self.spectral_mode:
            self.train_losses = {"total": [], "mag": [], "cos": [], "sin": [], "scale": []}
            self.val_losses   = {"total": [], "mag": [], "cos": [], "sin": [], "scale": []}
        else:
            self.train_losses = {"total": [], "latent": [], "scale": []}
            self.val_losses   = {"total": [], "latent": [], "scale": []}

        # Epochs without improvement at the end of the saved history (restored on resume, so patience counts exactly)
        self.resume_no_improve = 0
        if resume_from is not None and resume_from is not False:
            resolved_path = self._resolve_resume_path(resume_from)
            if resolved_path:
                self._resume_from_state(resolved_path)
            else:
                print("⚠️ Could not resolve a valid trainer state to resume from.")

    def _resolve_resume_path(self, resume_from):
        if resume_from is True or resume_from in ["latest", "last"]:
            target_path = self.ckpt_dir / "last_trainer_state.pt"
            if target_path.exists():
                return target_path
            return None

        elif isinstance(resume_from, int):
            target_path = self.ckpt_dir / f"epoch_{resume_from}_trainer_state.pt"
            if target_path.exists():
                return target_path
            raise FileNotFoundError(f"Requested to resume from epoch {resume_from}, but {target_path} does not exist.")

        elif resume_from == "best":
            target_path = self.ckpt_dir / "best_trainer_state.pt"
            if target_path.exists():
                return target_path
            raise FileNotFoundError(f"Requested 'best' resume, but {target_path} does not exist.")

        elif isinstance(resume_from, (str, Path)):
            target_path = Path(resume_from)
            if target_path.exists():
                return target_path
            raise FileNotFoundError(f"Manual resume path {target_path} does not exist.")

        return None

    def _resume_from_state(self, state_path):
        state_path = Path(state_path)
        if not state_path.exists():
            raise FileNotFoundError(f"Resume state not found at {state_path}")

        print(f"🔄 Resuming training from {state_path.name}...")

        state_name = state_path.stem
        flow_name = state_name.replace("trainer_state", "flow_model")
        enc_name = state_name.replace("trainer_state", "local_encoder")

        flow_path = state_path.parent / f"{flow_name}.pt"
        enc_path = state_path.parent / f"{enc_name}.pt"

        if not flow_path.exists() or not enc_path.exists():
            raise FileNotFoundError(f"Could not find accompanying model files for {state_name}. Looked for {flow_name}.pt")

        self.model.load_state_dict(torch.load(flow_path, map_location=self.device)['model_state_dict'])
        self.local_encoder.load_state_dict(torch.load(enc_path, map_location=self.device)['model_state_dict'])

        state = torch.load(state_path, map_location=self.device)
        try:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        except Exception as e:
            print(f"⚠️ Could not load optimizer state: {e}")

        self.start_epoch = state['epoch'] + 1
        self.global_step = state.get('global_step', 0)
        self.best_metric = state['best_metric']

        self.train_losses = state.get('train_losses', self.train_losses)
        self.val_losses = state.get('val_losses', self.val_losses)

        # Patience counter: validations since the first occurrence of the best val loss in the saved history
        val_total = self.val_losses.get("total", [])
        if val_total and min(val_total) <= self.best_metric:
            self.resume_no_improve = len(val_total) - 1 - val_total.index(min(val_total))
        print(f"   Patience counter restored: {self.resume_no_improve}")

        print(f"✅ Successfully resumed! Starting at Epoch {self.start_epoch} (Best Metric so far: {self.best_metric:.4f})")

    def _save_inference_gin(self):
        g = self.geom
        lines = [
            "# inference.gin — generated by SCAPES training",
            "# Load this file with FlowInference(model_dir=...) to reconstruct models and run inference.",
            "",
            "# ─── Model architecture ───",
            f"model.d_model = {self.model_config['d_model']}",
            f"model.nhead = {self.model_config['nhead']}",
            f"model.num_layers = {self.model_config['num_layers']}",
            f"model.dim_feedforward = {self.model_config['dim_feedforward']}",
            "",
            "# ─── LocalEncoder architecture ───",
            f"local_encoder.hidden_dim = {self.encoder_config['hidden_dim']}",
            f"local_encoder.time_entanglement = {str(self.encoder_config['time_entanglement'])}",
            f"local_encoder.temporal_compression = {self.encoder_config['temporal_compression']}",
            "",
            "# ─── Codec and generation geometry (frames, machine-written; effective seconds in the comments) ───",
        ]
        lines += g.to_gin_lines()
        f = g.frame_rate
        if g.mode == "atoms":
            lines += [f"# atoms.seconds = {g.unit_frames / f:.6g}", f"# atoms.hop_seconds = {g.unit_hop_frames / f:.6g}",
                      f"# atoms.crossfade_seconds = {g.crossfade_samples / g.sample_rate:.6g}"]
        else:
            lines += [f"# ar.block_seconds = {g.unit_frames / f:.6g}", f"# ar.memory_seconds = {g.memory_frames / f:.6g}",
                      f"# ar.stride_seconds = {g.stride_frames / f:.6g}", f"# ar.prefix_seconds = {g.prefix_frames / f:.6g}"]
        lines += [
            f"dataset.semantic_random_extension = {self.model_config['semantic_random_extension']}",
            f"training.standardize_latents = {self.standardize}",
        ]

        struct_names = self.model_config.get("structure_feature_names")
        if struct_names:
            lines.append("")
            lines.append("# ─── Structure features ───")
            lines.append(f"structure.features = {struct_names}")

        if self.spectral_mode:
            lines.extend([
                "",
                "# ─── Spectral representation ───",
                "training.spectral_representation = True",
                f"atoms.spectral_frames = {self.spectral_frames}",
            ])

        lines.extend([
            "",
            "# ─── Inference defaults ───",
            f"inference.cfg_scale = {self.model_config['cfg_scale']}",
        ])

        gin_path = self.ckpt_dir / "inference.gin"
        with open(gin_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # Batch -> (past [B, M, C, T], target [B, U, C], context, structure) on the device, latents standardised if asked.
    # C = frame_dim (latents + scale channel). Deterministic: noise and dropout are applied in _encode_past.
    def _prepare_batch(self, batch):
        past_latent = batch["memory_buffer_latent"].to(self.device)
        present_latent = batch["target_latent"].to(self.device)
        if self.standardize:
            d = self.n_latent
            past_latent = (past_latent - self.mu.view(1, 1, d, 1)) / self.std.view(1, 1, d, 1)
            present_latent = (present_latent - self.mu.view(1, d, 1)) / self.std.view(1, d, 1)

        if self.has_scale:
            past_scale = batch["memory_buffer_scale"].to(self.device)
            present_scale = batch["target_scale"].to(self.device)
        else:
            past_scale = present_scale = None

        if self.spectral_mode:
            # Transform past latents to spectral domain
            B, N = past_latent.shape[:2]
            past_latent_flat = past_latent.reshape(B * N, 128, self.atom_frames).transpose(1, 2)
            past_spec_flat = time_to_spectral(past_latent_flat)
            past_spec = past_spec_flat.reshape(B, N, self.spectral_frames, 384).transpose(2, 3)

            past_scale_exp = past_scale.unsqueeze(-1).expand(
                -1, -1, -1, self.spectral_frames
            )

            past_memory = torch.cat([past_spec, past_scale_exp], dim=2)
        elif self.has_scale:
            past_scale_exp = past_scale.unsqueeze(-1).expand(
                -1, -1, -1, past_latent.size(-1)
            )

            past_memory = torch.cat(
                [past_latent, past_scale_exp],
                dim=2
            )
        else:
            past_memory = past_latent

        if self.has_scale:
            present_scale_exp = present_scale.unsqueeze(-1).expand(
                -1, -1, self.unit_frames
            )
            present_target = torch.cat([present_latent, present_scale_exp], dim=1).transpose(1, 2)
        else:
            present_target = present_latent.transpose(1, 2)

        if self.context_source not in ["clap", None]:
            raise ValueError(
                "Only 'clap' target_semantic is supported."
            )

        context = batch["target_semantic"].to(self.device)

        structure = batch.get("target_structure", None)
        if structure is not None:
            structure = structure.to(self.device)

        if self.use_structure and structure is None:
            raise ValueError("Model expects structure_vector but dataset did not provide it")

        return past_memory, present_target, context, structure

    # Past -> (encoded past [B, M, T', d_model], prefix [B, o, C] or None).
    # train=True: past noise (latent channels), then past dropout (AR: frames zeroed before LocalEncoder; atoms: V1 rule),
    # and the masked units/frames of the encoded past are replaced by the null embedding. The AR prefix is the last o frames
    # of the past after noise and masking (zeros where masked), never the null embedding (R2-C5).
    # null_past=True: every unit/frame is the null embedding and the prefix is zero (B8 falsifier).
    def _encode_past(self, past_memory, train=False, null_past=False):
        pm = past_memory
        B, N, C, T = pm.shape
        ar = self.mode == "autoregressive"

        if train and self.past_noise > 0:
            n_noisy = C - int(self.has_scale)
            pm = torch.cat([apply_past_noise(pm[:, :, :n_noisy], self.past_noise), pm[:, :, n_noisy:]], dim=2)

        mask = None
        if null_past:
            mask = torch.ones(B, T if ar else N, dtype=torch.bool, device=pm.device)
        elif train and self.past_dropout > 0:
            if ar:
                mask = past_dropout_frame_mask(B, T, self.unit_frames, self.past_dropout, device=pm.device)
            else:
                mask = past_dropout_mask(B, N, self.past_dropout, device=pm.device)
        if ar and mask is not None:
            pm = pm * (~mask)[:, None, None, :].to(pm.dtype)

        prefix = None
        if self.prefix_frames > 0:
            prefix = pm[:, -1, :, -self.prefix_frames:].transpose(1, 2).contiguous()

        encoded_past = self.local_encoder(pm)
        if mask is not None:
            m = mask[:, None, :, None] if ar else mask[:, :, None, None]
            encoded_past = torch.where(m, self.model.null_past_embed[:, :, :encoded_past.size(2)], encoded_past)
        return encoded_past, prefix

    def _plot_and_save_losses(self, current_epoch):
        has_val = self.val_loader is not None

        if self.spectral_mode:
            loss_types = [
                ("total", "Total Loss", "tab:purple"),
                ("mag", "Magnitude Loss", "tab:blue"),
                ("cos", "Cos-Phase Loss", "tab:orange"),
                ("sin", "Sin-Phase Loss", "tab:red"),
                ("scale", "Scale Loss", "tab:green"),
            ]
        else:
            loss_types = [
                ("total", "Total Loss", "tab:purple"),
                ("latent", "Latent Loss", "tab:blue"),
                ("scale", "Scale Loss", "tab:green"),
            ]

        cols = len(loss_types)
        rows = 2 if has_val else 1

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

        if rows == 1:
            axes = axes.reshape(1, -1)

        for col_idx, (key, title, color) in enumerate(loss_types):
            axes[0, col_idx].plot(range(1, len(self.train_losses[key]) + 1), self.train_losses[key], label=f"Train {title}", color=color, linewidth=2)
            axes[0, col_idx].set_title(f"Train {title}")
            axes[0, col_idx].set_xlabel("Epoch")
            axes[0, col_idx].set_ylabel("Loss")
            axes[0, col_idx].grid(True, linestyle='--', alpha=0.6)
            axes[0, col_idx].legend()

            if has_val:
                axes[1, col_idx].plot(range(1, len(self.val_losses[key]) + 1), self.val_losses[key], label=f"Val {title}", color=color, linewidth=2, linestyle="--")
                axes[1, col_idx].set_title(f"Val {title}")
                axes[1, col_idx].set_xlabel("Validation")
                axes[1, col_idx].set_ylabel("Loss")
                axes[1, col_idx].grid(True, linestyle='--', alpha=0.6)
                axes[1, col_idx].legend()

        plt.tight_layout()

        save_path = self.loss_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


    def train_epoch(self, discriminator=None, collect_adv=False):
        self.model.train()
        self.local_encoder.train()
        zero = torch.zeros((), device=self.device)
        # epoch sums (tensors on the device, no sync per step) and sums since the last log line
        keys = ("total", "lat", "cos", "sin", "scale", "adv", "reg", "grad")
        ep = {k: zero.clone() for k in keys}
        win = {k: zero.clone() for k in keys}
        n_steps = n_win = 0

        # Validation inside long epochs (only with max_steps): every 500 steps
        val_every = 500 if (self.max_steps is not None and self.val_loader is not None
                            and len(self.train_loader) > 500) else None

        # Build regularizer list once
        active_regularizers = []
        if self.regularizers_and_weights:
            REG_MAP = {
                "time_phase": time_phase_regularizer,
                "fft_phase": fft_phase_regularizer,
            }
            for name, weight in self.regularizers_and_weights:
                fn = REG_MAP.get(name)
                if fn is None:
                    print(f"⚠️ Unknown regularizer '{name}', skipping.")
                else:
                    active_regularizers.append((fn, weight))

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            past_memory, present_target, context, structure = self._prepare_batch(batch)

            if self.conditioning_dropout > 0.0:
                mask = torch.rand(context.shape[0], 1, device=self.device) < self.conditioning_dropout
                context = torch.where(mask, torch.zeros_like(context), context)

            self.optimizer.zero_grad()

            encoded_past, prefix = self._encode_past(past_memory, train=True)

            if self.spectral_mode:
                noise = torch.randn(present_target.size(0), self.spectral_frames, self.frame_dim,
                                    device=self.device)
                loss, l_mag, l_cos, l_sin, l_scale, X_hat, s = spectral_flow_matching_loss(
                    self.model, noise, present_target, context, encoded_past,
                    structure_vector=structure
                )
                parts = {"lat": l_mag, "cos": l_cos, "sin": l_sin, "scale": l_scale}
            else:
                noise = torch.randn_like(present_target)
                loss, l_lat, l_scale, X_hat, s = flow_matching_loss(
                    self.model, noise, present_target, context, encoded_past,
                    structure_vector=structure, n_latent=self.n_latent, has_scale=self.has_scale, prefix=prefix
                )
                parts = {"lat": l_lat, "scale": l_scale}

            # Regularizers (only used in non-spectral mode; blocked by validation in __init__)
            reg_loss = torch.tensor(0.0, device=self.device)
            for fn, weight in active_regularizers:
                reg_loss += weight * fn(X_hat, present_target, s, self.n_latent)

            combined_loss = loss + reg_loss

            if discriminator is not None:
                X_hat_latent = X_hat[:, :, :self.n_latent].transpose(1, 2).contiguous()
                logits = discriminator(X_hat_latent).squeeze(-1)
                adv_loss = self.adv_loss_fn(logits, torch.zeros_like(logits)).mean()
                combined_loss = combined_loss + adv_loss
                parts["adv"] = adv_loss

            combined_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.local_encoder.parameters()), 1.0
            )
            self.optimizer.step()

            # Accumulate detached values; they are read only at the log points
            parts.update({"total": combined_loss, "reg": reg_loss, "grad": grad_norm})
            for k, v in parts.items():
                ep[k] += v.detach()
                win[k] += v.detach()
            n_steps += 1
            n_win += 1
            self.global_step += 1

            # Collect adversarial data for next discriminator round
            if collect_adv:
                remaining = self.adv_buffer_max - len(self.x_hat_buffer)
                if remaining > 0:
                    X_hat_latent = X_hat[:, :, :self.n_latent].transpose(1, 2).contiguous()
                    B_curr = X_hat_latent.shape[0]
                    take = min(remaining, B_curr)
                    self.x_hat_buffer.append((
                        X_hat_latent[:take].detach().cpu(),
                        s[:take].detach().cpu().reshape(take, 1)
                    ))

            if self.global_step % self.log_every == 0:
                m = {k: v.item() / n_win for k, v in win.items()}
                self.logger.log_step(self.global_step, m["total"], m["lat"], m["scale"], m["grad"],
                                     self.optimizer.param_groups[0]["lr"])
                pbar.set_postfix({"step": self.global_step, "L": f"{m['total']:.4f}", "Lat": f"{m['lat']:.4f}",
                                  "Sca": f"{m['scale']:.4f}"})
                win = {k: zero.clone() for k in keys}
                n_win = 0

            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

            if val_every is not None and self.global_step % val_every == 0:
                self._validate()
                self.model.train()
                self.local_encoder.train()

        n = max(n_steps, 1)
        e = {k: v.item() / n for k, v in ep.items()}
        if self.spectral_mode:
            return e["total"], e["lat"], e["cos"], e["sin"], e["scale"], e["adv"], e["reg"]
        else:
            return e["total"], e["lat"], e["scale"], e["adv"], e["reg"]

    # Deterministic validation loss (design D17). Sample i gets x0 from a generator seeded with its dataset index and is
    # evaluated at 4 stratified s values; the loss is the mean over samples and s. Returns
    # (total, latent, scale, per_file), per_file = {filename: total loss over the samples of that file}.
    # null_past=True replaces the whole past by the null embedding (past-usage test). CPU and CUDA RNG are left unchanged.
    @torch.no_grad()
    def _eval_loss(self, loader, null_past=False):
        self.model.eval()
        self.local_encoder.eval()
        zero = torch.zeros((), device=self.device)
        tot, lat, sca, n = zero.clone(), zero.clone(), zero.clone(), 0
        file_sum, file_n = {}, {}

        with torch.random.fork_rng():
            for batch in loader:
                past_memory, present_target, context, structure = self._prepare_batch(batch)
                encoded_past, prefix = self._encode_past(past_memory, train=False, null_past=null_past)

                # x0 per sample, from its own seed
                U, P = present_target.shape[1:]
                x0 = torch.stack([
                    torch.randn(U, P, generator=torch.Generator().manual_seed(int(i))) for i in batch["index"].tolist()
                ]).to(self.device)

                # Samples grouped by file so that the per-file loss is exact
                groups = {}
                for pos, label in enumerate(batch["label"]):
                    groups.setdefault(label, []).append(pos)
                for label, pos in groups.items():
                    sel = torch.tensor(pos, device=self.device)
                    b = len(pos)
                    for s_val in VAL_S_VALUES:
                        s = torch.full((b, 1, 1), s_val, device=self.device)
                        loss, l_lat, l_sca, _, _ = flow_matching_loss(
                            self.model, x0[sel], present_target[sel], context[sel], encoded_past[sel],
                            structure_vector=None if structure is None else structure[sel],
                            n_latent=self.n_latent, has_scale=self.has_scale,
                            prefix=None if prefix is None else prefix[sel], s=s
                        )
                        tot += loss * b
                        lat += l_lat * b
                        sca += l_sca * b
                        file_sum[label] = file_sum.get(label, zero) + loss * b
                    file_n[label] = file_n.get(label, 0) + b
                    n += b * len(VAL_S_VALUES)

        n = max(n, 1)
        per_file = {k: file_sum[k].item() / (file_n[k] * len(VAL_S_VALUES)) for k in file_sum}
        return tot.item() / n, lat.item() / n, sca.item() / n, per_file

    # Spectral mode (encodec48k only): V1's validation, random x0 and s, with the RNG states restored afterwards
    @torch.no_grad()
    def _eval_spectral(self, loader):
        self.model.eval()
        self.local_encoder.eval()
        sums = [0.0] * 5
        with torch.random.fork_rng():
            for batch in loader:
                past_memory, present_target, context, structure = self._prepare_batch(batch)
                encoded_past, _ = self._encode_past(past_memory, train=False)
                noise = torch.randn(present_target.size(0), self.spectral_frames, self.frame_dim, device=self.device)
                _, l_mag, l_cos, l_sin, l_scale, _, _ = spectral_flow_matching_loss(
                    self.model, noise, present_target, context, encoded_past, structure_vector=structure
                )
                for i, v in enumerate((l_mag + l_cos + l_sin + l_scale, l_mag, l_cos, l_sin, l_scale)):
                    sums[i] += v.item()
        n = len(loader)
        return tuple(v / n for v in sums)

    def val_epoch(self):
        if self.val_loader is None:
            return 0.0, 0.0, 0.0
        if self.spectral_mode:
            return self._eval_spectral(self.val_loader)
        total, lat, sca, per_file = self._eval_loss(self.val_loader)
        self._last_val_per_file = per_file
        return total, lat, sca

    # Deterministic val loss of a subset with the real past and with the whole past replaced by the null embedding.
    def evaluate_past_usage(self, subset) -> dict:
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)
        l_real = self._eval_loss(loader)[0]
        l_null = self._eval_loss(loader, null_past=True)[0]
        return {"L_real": l_real, "L_null": l_null, "ratio": l_real / l_null}

    # Teacher-forced validation audio: for each chosen file, consecutive units of its val region (train if it has none) are
    # predicted from the real past and rendered by inference/render.py (overlap-add for atoms, the streaming decoder for AR).
    @torch.no_grad()
    def generate_validation_audio(self, epoch, NFE=32):
        if not self.val_audio_files:
            return
        import soundfile as sf
        from SCAPES.codecs import get_codec
        from SCAPES.inference.render import StreamRenderer, render_atoms

        self.model.eval()
        self.local_encoder.eval()
        g = self.geom
        codec = get_codec(g.codec, self.device)
        step = g.unit_frames if self.mode == "autoregressive" else g.unit_hop_frames
        num_units = max(1, int(self.val_duration * g.frame_rate // step))
        regions = self.dataset._regions

        for target_file in self.val_audio_files:
            by_start = {}
            for region in ("train", "val"):
                for i, (fname, start) in enumerate(self.dataset.all_indices):
                    if fname == target_file and regions[i] == region:
                        by_start.setdefault(region, {})[start] = i
            starts = by_start.get("val") or by_start.get("train")
            if not starts:
                continue
            # consecutive units: each target starts one step after the previous one
            first = min(starts)
            seq = []
            while first + len(seq) * step in starts and len(seq) < num_units:
                seq.append(starts[first + len(seq) * step])

            atoms = []
            for idx in tqdm(seq, desc=f"Generating {target_file} (TF)"):
                raw_batch = self.dataset[idx]
                for k in raw_batch:
                    if isinstance(raw_batch[k], torch.Tensor):
                        raw_batch[k] = raw_batch[k].unsqueeze(0)
                gt_past, _, context, structure = self._prepare_batch(raw_batch)[:4]
                encoded_past, prefix = self._encode_past(gt_past)
                if self.spectral_mode:
                    x0 = torch.randn(1, self.spectral_frames, self.frame_dim, device=self.device)
                else:
                    x0 = torch.randn(1, self.unit_frames, self.frame_dim, device=self.device)
                pred_raw = self.model.generate(x0, encoded_past, context, structure_vector=structure, max_nfe=NFE,
                                               cfg_scale=self.model_config['cfg_scale'], prefix=prefix)
                if self.spectral_mode:
                    latent_time = spectral_to_time(pred_raw[:, :, :384], self.atom_frames)
                    scale_time = pred_raw[:, :, 384:].mean(dim=1, keepdim=True).expand(-1, self.atom_frames, -1)
                    pred = torch.cat([latent_time, scale_time], dim=-1).transpose(1, 2)
                else:
                    pred = pred_raw.transpose(1, 2)
                atoms.append(pred)

            if self.mode == "autoregressive":
                renderer = StreamRenderer(codec)
                for a in atoms:
                    z = a[:, :self.n_latent, :]
                    if self.standardize:
                        z = z * self.std.view(1, -1, 1) + self.mu.view(1, -1, 1)
                    renderer.push(z)
                audio = renderer.finish()
            else:
                audio = render_atoms(codec, atoms, g, self.mu, self.std)

            file_stem = Path(target_file).stem
            sf.write(self.val_dir / f"epoch_{epoch}_{file_stem}_TF.wav", audio.cpu().T.numpy(), g.sample_rate)

        print(f"✅ Validation audio for epoch {epoch} saved!")

    # ==========================================
    # DISCRIMINATOR TRAINING
    # ==========================================

    @torch.no_grad()
    def _collect_real_atoms(self, count):
        """Collect `count` real atoms randomly from the dataset. Returns [N, 128, T_frames]."""
        n = min(count, len(self.dataset))
        indices = torch.randperm(len(self.dataset))[:n].tolist()
        atoms = []
        for idx in indices:
            raw = self.dataset[idx]
            atoms.append(raw["target_latent"].unsqueeze(0).cpu())
        return torch.cat(atoms, dim=0)

    def _train_discriminator(self, max_epochs, lr=1e-4):
        """Train AtomDiscriminator from scratch using the X_hat buffer vs real atoms.

        Real → 0, Fake → 1 (discriminator outputs a "fakeness" score).
        Fake samples are weighted by (1 + (2s-1)**3) to focus on late-flow artifacts.
        The generator minimizes this fakeness score.
        """
        if len(self.x_hat_buffer) == 0:
            raise RuntimeError("Cannot train discriminator: X_hat buffer is empty")

        all_fake = torch.cat([x for x, _ in self.x_hat_buffer], dim=0).to(self.device)
        all_s = torch.cat([s for _, s in self.x_hat_buffer], dim=0).to(self.device)
        n_fake = all_fake.shape[0]

        n_real = min(n_fake, len(self.dataset))
        real_atoms = self._collect_real_atoms(n_real).to(self.device)
        n_total = min(n_fake, n_real)

        all_fake = all_fake[:n_total]
        all_s = all_s[:n_total]
        real_atoms = real_atoms[:n_total]

        disc = AtomDiscriminator().to(self.device)
        opt = torch.optim.AdamW(disc.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        fake_weight = (1.0 + (2.0 * all_s.squeeze(-1) - 1.0) ** 3)
        batch_size = min(128, n_total)

        for ep in range(max_epochs):
            disc.train()
            total_d_loss = 0
            n_batches = 0
            perm = torch.randperm(n_total, device=self.device)
            pbar = tqdm(range(0, n_total, batch_size), desc=f"  Disc ep {ep+1}/{max_epochs}", leave=False)
            for i in pbar:
                idx = perm[i:i + batch_size]
                real_batch = real_atoms[idx]
                fake_batch = all_fake[idx]
                w_batch = fake_weight[idx]

                opt.zero_grad()
                logits_real = disc(real_batch)
                logits_fake = disc(fake_batch)

                loss_real = loss_fn(logits_real, torch.zeros_like(logits_real)).mean()
                loss_fake = (loss_fn(logits_fake, torch.ones_like(logits_fake)).squeeze(-1) * w_batch).mean()
                d_loss = loss_real + loss_fake

                d_loss.backward()
                opt.step()
                total_d_loss += d_loss.item()
                n_batches += 1
                pbar.set_postfix({"L": f"{total_d_loss / n_batches:.4f}"})

        disc = disc.eval()
        del real_atoms, all_fake, all_s
        torch.cuda.empty_cache()
        return disc

    # ==========================================
    # MAIN TRAINING LOOP (2-stage)
    # ==========================================


    def _trainer_state(self, global_epoch):
        return {
            'epoch': global_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    # Validation pass (skipped without a val loader, or if it already ran at this step): logs the per-file losses and
    # saves the best checkpoint by the deterministic val loss. Returns the val loss or None.
    def _validate(self):
        if self.val_loader is None:
            return None
        if self._last_val_step == self.global_step:
            return self._last_val_metric
        val_out = self.val_epoch()
        self._last_val_step = self.global_step
        if self.spectral_mode:
            avg_v_total, avg_v_mag, avg_v_cos, avg_v_sin, avg_v_scale = val_out
            self.val_losses["total"].append(avg_v_total)
            self.val_losses["mag"].append(avg_v_mag)
            self.val_losses["cos"].append(avg_v_cos)
            self.val_losses["sin"].append(avg_v_sin)
            self.val_losses["scale"].append(avg_v_scale)
            print(f"Val   | Total: {avg_v_total:.4f} (Mag: {avg_v_mag:.4f}, Cos: {avg_v_cos:.4f}, Sin: {avg_v_sin:.4f}, Sca: {avg_v_scale:.4f})")
        else:
            avg_v_total, avg_v_lat, avg_v_scale = val_out
            self.val_losses["total"].append(avg_v_total)
            self.val_losses["latent"].append(avg_v_lat)
            self.val_losses["scale"].append(avg_v_scale)
            print(f"Val   | step {self.global_step} | Total: {avg_v_total:.4f} (Lat: {avg_v_lat:.4f}, Sca: {avg_v_scale:.4f})")
            self.logger.log_val(self.global_step, self._last_val_per_file)

        self._last_val_metric = avg_v_total
        if avg_v_total < self.best_metric:
            self.best_metric = avg_v_total
            torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / "best_flow_model.pt")
            torch.save({'model_state_dict': self.local_encoder.state_dict()}, self.ckpt_dir / "best_local_encoder.pt")
            print("🌟 Saved new best models!")
        return avg_v_total

    def _record_train(self, train_out):
        if self.spectral_mode:
            (avg_t_total, avg_t_mag, avg_t_cos, avg_t_sin,
             avg_t_scale, avg_t_adv, avg_t_reg) = train_out
            self.train_losses["total"].append(avg_t_total)
            self.train_losses["mag"].append(avg_t_mag)
            self.train_losses["cos"].append(avg_t_cos)
            self.train_losses["sin"].append(avg_t_sin)
            self.train_losses["scale"].append(avg_t_scale)
            self.train_losses.setdefault("adv", []).append(avg_t_adv)
            self.train_losses.setdefault("reg", []).append(avg_t_reg)
            parts = [f"Train | Total: {avg_t_total:.4f} (Mag: {avg_t_mag:.4f}, Cos: {avg_t_cos:.4f}, Sin: {avg_t_sin:.4f}, Sca: {avg_t_scale:.4f}"]
        else:
            avg_t_total, avg_t_lat, avg_t_scale, avg_t_adv, avg_t_reg = train_out
            self.train_losses["total"].append(avg_t_total)
            self.train_losses["latent"].append(avg_t_lat)
            self.train_losses["scale"].append(avg_t_scale)
            self.train_losses.setdefault("adv", []).append(avg_t_adv)
            self.train_losses.setdefault("reg", []).append(avg_t_reg)
            parts = [f"Train | Total: {avg_t_total:.4f} (Lat: {avg_t_lat:.4f}, Sca: {avg_t_scale:.4f}"]
        if avg_t_reg > 0:
            parts.append(f"Reg: {avg_t_reg:.6f}")
        if avg_t_adv > 0:
            parts.append(f"Adv: {avg_t_adv:.6f}")
        print(", ".join(parts) + ")")

    def _max_steps_reached(self):
        return self.max_steps is not None and self.global_step >= self.max_steps

    def _epoch_bookkeeping(self, global_epoch, audio_val_freq, val_nfe):
        """Shared checkpoint/save/log logic after each epoch (the best checkpoint is saved in _validate)."""
        trainer_state = self._trainer_state(global_epoch) if self.save_resume_states else None

        ckpt_freq = self.checkpoint_freq
        if ckpt_freq > 0 and global_epoch % ckpt_freq == 0:
            torch.save({'model_state_dict': self.model.state_dict()},
                       self.ckpt_dir / f"epoch_{global_epoch}_flow_model.pt")
            torch.save({'model_state_dict': self.local_encoder.state_dict()},
                       self.ckpt_dir / f"epoch_{global_epoch}_local_encoder.pt")
            if self.save_resume_states:
                torch.save(trainer_state, self.ckpt_dir / f"epoch_{global_epoch}_trainer_state.pt")

        if (global_epoch) % audio_val_freq == 0:
            self.generate_validation_audio(global_epoch, NFE=val_nfe)

        torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / "last_flow_model.pt")
        torch.save({'model_state_dict': self.local_encoder.state_dict()}, self.ckpt_dir / "last_local_encoder.pt")
        if self.save_resume_states:
            torch.save(trainer_state, self.ckpt_dir / "last_trainer_state.pt")

        history_path = self.loss_dir / "loss_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "train": self.train_losses,
                "val": self.val_losses if self.val_loader else {}
            }, f, indent=4)

        self._plot_and_save_losses(global_epoch)

    # One epoch of stage 1 or 2: train, validate (if there is a val loader), save, count patience.
    # Returns True if training must stop (max_steps reached, or patience exhausted).
    def _run_epoch(self, global_epoch, audio_val_freq, val_nfe, patience, best_ref, discriminator=None, collect_adv=False):
        train_out = self.train_epoch(discriminator=discriminator, collect_adv=collect_adv)
        self._record_train(train_out)
        metric = self._validate()
        self._epoch_bookkeeping(global_epoch, audio_val_freq, val_nfe)
        self.start_epoch += 1

        if self._max_steps_reached():
            print(f"  Stopped at max_steps = {self.max_steps}")
            return True
        # Patience needs a val loss and is off when max_steps is set
        if self.max_steps is None and metric is not None:
            if metric < best_ref["value"]:
                best_ref["value"] = metric
                best_ref["no_improve"] = 0
            else:
                best_ref["no_improve"] += 1
                if best_ref["no_improve"] >= patience:
                    print(f"  Early stopping at epoch {global_epoch} (no improvement for {patience} epochs)")
                    return True
        return False

    def train(self, epochs, audio_val_freq=5, val_nfe=32, patience=10):
        # ─── Stage 1: Standard Flow Matching ───
        print(f"\n{'='*60}")
        print(f"🔥 Stage 1 — Standard Flow Matching ({epochs} epochs, patience={patience}, max_steps={self.max_steps})")
        print(f"{'='*60}")

        self.x_hat_buffer = []
        best_ref = {"value": self.best_metric, "no_improve": self.resume_no_improve}
        stop = False

        for epoch in range(1, epochs + 1):
            global_epoch = self.start_epoch
            print(f"\n=== Stage 1 — Epoch {global_epoch} ===")
            collect_adv = self.use_discriminator and (epoch == epochs)
            stop = self._run_epoch(global_epoch, audio_val_freq, val_nfe, patience, best_ref, collect_adv=collect_adv)
            if stop:
                break

        if stop or not self.use_discriminator:
            self.logger.finish()
            print(f"\n✅ Training complete!")
            return

        if len(self.x_hat_buffer) == 0:
            raise RuntimeError(
                "Buffer is empty but use_discriminator=True. "
                "This means Stage 1 completed without collecting adversarial data."
            )

        # Freeze LocalEncoder for Stage 2
        self.local_encoder.eval()
        for p in self.local_encoder.parameters():
            p.requires_grad_(False)

        print(f"\n{'='*60}")
        print(f"🔥 Stage 2 — Adversarial Fine-tuning ({self.stage2_epochs} epochs, patience={patience})")
        print(f"   Buffer has {sum(x.shape[0] for x, _ in self.x_hat_buffer)} fake samples")
        print(f"{'='*60}")

        # Stage 2 has its own patience tracking — the loss landscape differs from stage 1
        best_ref = {"value": float('inf'), "no_improve": 0}

        for epoch in range(1, self.stage2_epochs + 1):
            global_epoch = self.start_epoch
            print(f"\n=== Stage 2 — Epoch {global_epoch} ===")
            print(f"  ─── Part A: Train discriminator ───")

            disc = self._train_discriminator(max_epochs=self.disc_epochs, lr=1e-4)

            print(f"  ─── Part B: Train generator (adversarial) ───")

            self.x_hat_buffer = []
            stop = self._run_epoch(global_epoch, audio_val_freq, val_nfe, patience, best_ref,
                                   discriminator=disc, collect_adv=True)

            # Clean up discriminator
            del disc
            torch.cuda.empty_cache()
            if stop:
                break

        # Unfreeze LocalEncoder for any downstream use
        self.local_encoder.train()
        for p in self.local_encoder.parameters():
            p.requires_grad_(True)

        self.x_hat_buffer = []
        self.logger.finish()
        print(f"\n✅ Training complete!")
