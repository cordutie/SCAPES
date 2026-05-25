import os
import json
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
import math
import matplotlib.pyplot as plt

from SCAPES.auxiliar.losses_flow import flow_matching_loss
from SCAPES.data.config_loader import TrainingConfig


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

        # Data-derived params (read from the dataset, not from gin)
        self.frame_dim = 129
        self.context_vector_dim = 1024
        self.num_past_atoms = dataset.memory_buffer_atoms
        self.atom_frames = dataset.atoms_frames
        self.atoms_hop_frames = dataset.atoms_hop_frames
        self.crossfade_frames = dataset.crossfade_frames
        self.structure_dim = dataset.structure_feature_dimension

        # Build config dicts for JSON serialization
        self.model_config = {
            "frame_dim": self.frame_dim,
            "context_vector_dim": self.context_vector_dim,
            "num_past_atoms": self.num_past_atoms,
            "frames_per_atom": self.atom_frames,
            "atoms_hop_frames": self.atoms_hop_frames,
            "crossfade_frames": self.crossfade_frames,
            "context_seconds": dataset.context_seconds,
            "semantic_random_extension": dataset.semantic_random_extension,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_layers": config.num_layers,
            "dim_feedforward": config.dim_feedforward,
            "structure_dim": self.structure_dim,
        }
        self.encoder_config = {
            "in_channels": self.frame_dim,
            "hidden_dim": config.local_encoder_hidden_dim,
            "out_channels": config.d_model,
            "time_entanglement": config.local_encoder_time_entanglement,
            "temporal_compression": config.local_encoder_temporal_compression,
        }

        self.use_structure = dataset.structure_feature_dimension > 0

        if self.use_structure and "target_structure" not in self.dataset.requested_keys:
            raise ValueError("Model expects structure but dataset does not request it")

        if self.use_structure:
            self.model_config["structure_feature_names"] = getattr(self.dataset, "structure_feature_names", None)

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

        self._save_inference_gin()

        self.train_losses = {"total": [], "latent": [], "scale": []}
        self.val_losses   = {"total": [], "latent": [], "scale": []}

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
        self.best_metric = state['best_metric']

        self.train_losses = state.get('train_losses', self.train_losses)
        self.val_losses = state.get('val_losses', self.val_losses)

        print(f"✅ Successfully resumed! Starting at Epoch {self.start_epoch} (Best Metric so far: {self.best_metric:.4f})")

    def _save_inference_gin(self):
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
            "# ─── Atom geometry (from dataprep) ───",
            f"atoms.frames = {self.model_config['frames_per_atom']}",
            f"atoms.hop_frames = {self.model_config['atoms_hop_frames']}",
            f"atoms.crossfade_frames = {self.model_config['crossfade_frames']}",
            f"dataset.memory_buffer_atoms = {self.model_config['num_past_atoms']}",
            f"dataset.context_seconds = {self.model_config['context_seconds']}",
            f"dataset.semantic_random_extension = {self.model_config['semantic_random_extension']}",
        ]

        struct_names = self.model_config.get("structure_feature_names")
        if struct_names:
            lines.append("")
            lines.append("# ─── Structure features ───")
            lines.append(f"structure.features = {struct_names}")

        lines.extend([
            "",
            "# ─── Inference defaults ───",
            "inference.cfg_scale = 3.0",
        ])

        gin_path = self.ckpt_dir / "inference.gin"
        with open(gin_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _prepare_batch(self, batch):
        past_latent = batch["memory_buffer_latent"].to(self.device)
        past_scale = batch["memory_buffer_scale"].to(self.device)

        past_scale_exp = past_scale.unsqueeze(-1).expand(
            -1, -1, -1, self.atom_frames
        )

        past_memory = torch.cat(
            [past_latent, past_scale_exp],
            dim=2
        )

        present_latent = batch["target_latent"].to(self.device)
        present_scale = batch["target_scale"].to(self.device)

        present_scale_exp = present_scale.unsqueeze(-1).expand(
            -1, -1, self.atom_frames
        )

        present_target = torch.cat(
            [present_latent, present_scale_exp],
            dim=1
        ).transpose(1, 2)

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

    def _plot_and_save_losses(self, current_epoch):
        has_val = self.val_loader is not None

        rows = 2 if has_val else 1
        cols = 3

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

        if rows == 1:
            axes = axes.reshape(1, -1)

        epochs_x = range(1, current_epoch + 1)

        loss_types = [
            ("total", "Total Loss", "tab:purple"),
            ("latent", "Latent Loss", "tab:blue"),
            ("scale", "Scale Loss", "tab:green")
        ]

        for col_idx, (key, title, color) in enumerate(loss_types):
            axes[0, col_idx].plot(epochs_x, self.train_losses[key], label=f"Train {title}", color=color, linewidth=2)
            axes[0, col_idx].set_title(f"Train {title}")
            axes[0, col_idx].set_xlabel("Epoch")
            axes[0, col_idx].set_ylabel("Loss")
            axes[0, col_idx].grid(True, linestyle='--', alpha=0.6)
            axes[0, col_idx].legend()

            if has_val:
                axes[1, col_idx].plot(epochs_x, self.val_losses[key], label=f"Val {title}", color=color, linewidth=2, linestyle="--")
                axes[1, col_idx].set_title(f"Val {title}")
                axes[1, col_idx].set_xlabel("Epoch")
                axes[1, col_idx].set_ylabel("Loss")
                axes[1, col_idx].grid(True, linestyle='--', alpha=0.6)
                axes[1, col_idx].legend()

        plt.tight_layout()

        save_path = self.loss_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def train_epoch(self):
        self.model.train()
        self.local_encoder.train()
        total_loss = 0
        total_lat_loss = 0
        total_scale_loss = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            past_memory, present_target, context, structure = self._prepare_batch(batch)

            if self.conditioning_dropout > 0.0:
                mask = torch.rand(context.shape[0], 1, device=self.device) < self.conditioning_dropout
                context = torch.where(mask, torch.zeros_like(context), context)

            self.optimizer.zero_grad()

            encoded_past = self.local_encoder(past_memory)

            if self.past_dropout > 0.0:
                B, N_past, T_frames, d_model = encoded_past.shape

                mask = torch.zeros((B, N_past, 1, 1), dtype=torch.bool, device=self.device)

                for b in range(B):
                    if torch.rand(1).item() < self.past_dropout:
                        num_drop = torch.randint(1, N_past + 1, (1,)).item()
                        mask[b, :num_drop] = True

                encoded_past = torch.where(mask, self.model.null_past_embed, encoded_past)

            noise = torch.randn_like(present_target)
            loss, l_lat, l_scale = flow_matching_loss(
                self.model,
                noise,
                present_target,
                context,
                encoded_past,
                structure_vector=structure,
                scale_weight=1.0
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) +
                list(self.local_encoder.parameters()),
                1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_lat_loss += l_lat.item()
            total_scale_loss += l_scale.item()

            pbar.set_postfix({
                "L": f"{loss.item():.4f}",
                "Lat": f"{l_lat.item():.4f}",
                "Sca": f"{l_scale.item():.4f}"
            })

        n = len(self.train_loader)
        return total_loss / n, total_lat_loss / n, total_scale_loss / n

    @torch.no_grad()
    def val_epoch(self):
        if self.val_loader is None:
            return 0.0, 0.0, 0.0

        self.model.eval()
        self.local_encoder.eval()
        total_loss = 0
        total_lat = 0
        total_scale = 0

        for batch in self.val_loader:
            past_memory, present_target, context, structure = self._prepare_batch(batch)
            encoded_past = self.local_encoder(past_memory)
            noise = torch.randn_like(present_target)

            loss, l_lat, l_scale = flow_matching_loss(
                self.model,
                noise,
                present_target,
                context,
                encoded_past,
                structure_vector=structure,
                scale_weight=1.0
            )

            total_loss += loss.item()
            total_lat += l_lat.item()
            total_scale += l_scale.item()

        return (
            total_loss / len(self.val_loader),
            total_lat / len(self.val_loader),
            total_scale / len(self.val_loader)
        )

    @torch.no_grad()
    def generate_validation_audio(self, epoch, NFE=32):
        if not self.val_audio_files:
            return

        self.model.eval()
        self.local_encoder.eval()

        hop_time = self.atoms_hop_frames / 150.0
        num_atoms = int(self.val_duration // hop_time)

        macro_overlap_frames = self.atom_frames - self.atoms_hop_frames

        samples_per_frame = self.dataset.samples_per_frame

        segment_samples = self.atom_frames * samples_per_frame
        hop_samples     = self.atoms_hop_frames * samples_per_frame
        crossfade_samples = self.crossfade_frames * samples_per_frame

        zeros_frames = macro_overlap_frames - self.crossfade_frames
        zeros = torch.zeros(zeros_frames * samples_per_frame, device=self.device)

        hann = torch.hann_window(crossfade_samples * 2, device=self.device)

        ones_frames = self.atoms_hop_frames - self.crossfade_frames
        ones = torch.ones(ones_frames * samples_per_frame, device=self.device)

        window = torch.cat([
            zeros,
            hann[:crossfade_samples],
            ones,
            hann[crossfade_samples:]
        ]).view(1, 1, -1)

        alpha_smooth = 0.6
        max_jump = 1.15
        max_drop = 0.85

        for target_file in self.val_audio_files:
            file_indices = [i for i, (fname, _) in enumerate(self.dataset.all_indices) if fname == target_file]
            if not file_indices:
                continue

            seq_indices = file_indices[:num_atoms]

            total_samples = (len(seq_indices) - 1) * hop_samples + segment_samples
            tf_out_audio = torch.zeros(1, 2, total_samples, device=self.device)

            prev_scale = None

            for i, idx in enumerate(tqdm(seq_indices, desc=f"Generating {target_file} (TF)")):
                raw_batch = self.dataset[idx]
                for k in raw_batch:
                    if isinstance(raw_batch[k], torch.Tensor):
                        raw_batch[k] = raw_batch[k].unsqueeze(0)

                gt_past, _, context, structure = self._prepare_batch(raw_batch)

                x0 = torch.randn(1, self.atom_frames, 129, device=self.device)

                enc_tf = self.local_encoder(gt_past)
                tf_pred = self.model.generate(
                    x0,
                    enc_tf,
                    context,
                    structure_vector=structure,
                    max_nfe=NFE
                ).transpose(1, 2)

                tf_pred_smooth = tf_pred.clone()
                raw_scale = torch.abs(tf_pred[:, 128, :]).mean(dim=-1, keepdim=True)

                if prev_scale is None:
                    smoothed_scale = raw_scale
                else:
                    target_scale = torch.clamp(raw_scale, prev_scale * max_drop, prev_scale * max_jump)
                    smoothed_scale = (alpha_smooth * target_scale) + ((1.0 - alpha_smooth) * prev_scale)

                prev_scale = smoothed_scale
                tf_pred_smooth[:, 128, :] = smoothed_scale.expand_as(tf_pred_smooth[:, 128, :])

                latents = tf_pred_smooth[:, :128, :]
                meta = {
                    "audio_scales": [smoothed_scale.squeeze(0).float()],
                    "padding_mask": torch.ones((1, self.atom_frames * samples_per_frame),
                                             dtype=torch.bool, device=self.device)
                }

                audio = self.processor.decode_latents_audio(latents, metadata=meta)
                audio = audio * window

                start = i * hop_samples
                tf_out_audio[:, :, start : start + segment_samples] += audio

            file_stem = Path(target_file).stem
            sf.write(self.val_dir / f"epoch_{epoch}_{file_stem}_TF.wav",
                     tf_out_audio.squeeze(0).T.cpu().numpy(), 48000)

        print(f"✅ Validation audio for epoch {epoch} saved!")

    def train(self, epochs, audio_val_freq=5, val_nfe=32):
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")

            avg_t_total, avg_t_lat, avg_t_scale = self.train_epoch()

            self.train_losses["total"].append(avg_t_total)
            self.train_losses["latent"].append(avg_t_lat)
            self.train_losses["scale"].append(avg_t_scale)

            print(f"Train | Total: {avg_t_total:.4f} (Lat: {avg_t_lat:.4f}, Sca: {avg_t_scale:.4f})")

            if self.val_loader is not None:
                avg_v_total, avg_v_lat, avg_v_scale = self.val_epoch()

                self.val_losses["total"].append(avg_v_total)
                self.val_losses["latent"].append(avg_v_lat)
                self.val_losses["scale"].append(avg_v_scale)

                print(f"Val   | Total: {avg_v_total:.4f} (Lat: {avg_v_lat:.4f}, Sca: {avg_v_scale:.4f})")
                current_metric = avg_v_total
            else:
                current_metric = avg_t_total

            trainer_state = None
            if self.save_resume_states:
                trainer_state = {
                    'epoch': epoch,
                    'best_metric': min(current_metric, self.best_metric),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }

            if current_metric < self.best_metric:
                self.best_metric = current_metric
                torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / "best_flow_model.pt")
                torch.save({'model_state_dict': self.local_encoder.state_dict()}, self.ckpt_dir / "best_local_encoder.pt")

                if self.save_resume_states:
                    torch.save(trainer_state, self.ckpt_dir / "best_trainer_state.pt")

                print("🌟 Saved new best models!")

            if self.checkpoint_freq > 0 and epoch % self.checkpoint_freq == 0:
                torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / f"epoch_{epoch}_flow_model.pt")
                torch.save({'model_state_dict': self.local_encoder.state_dict()}, self.ckpt_dir / f"epoch_{epoch}_local_encoder.pt")

                if self.save_resume_states:
                    torch.save(trainer_state, self.ckpt_dir / f"epoch_{epoch}_trainer_state.pt")

            if epoch % audio_val_freq == 0:
                self.generate_validation_audio(epoch, NFE=val_nfe)

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

            self._plot_and_save_losses(epoch)
