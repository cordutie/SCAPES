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

def get_model_configs(
    size, 
    segment_length, 
    frame_dim=129,
    frames_per_atom=48,       # <--- Updated Default
    atoms_hop_frames=15,      # <--- NEW
    crossfade_frames=3,       # <--- NEW
    context_vector_dim=1024,
    LocalEncoder_in_channels=129,
    LocalEncoder_hidden_dim=256,
    LocalEncoder_time_entanglement=True,
    LocalEncoder_temporal_compression=1,
    structure_dim=0
):
    configs = {
        "small": {
            "d_model": 512,
            "num_layers": 6,
            "nhead": 8,
            "dim_feedforward": 2048,
            "max_seq_len": 2048,
        }, # 30 M parameters
        "medium": {
            "d_model": 768,
            "num_layers": 8,
            "nhead": 12,
            "dim_feedforward": 2048,
            "max_seq_len": 2048,
        }, # 90 M parameters
        "large": {
            "d_model": 1024,
            "num_layers": 12,
            "nhead": 16,
            "dim_feedforward": 2048,
            "max_seq_len": 2048,
        }, # 200 M parameters
        "extra_large": {
            "d_model": 1280,
            "num_layers": 16,
            "nhead": 20,
            "dim_feedforward": 2048,
            "max_seq_len": 20
        }, # 400 M parameters
        "audiobox": {
            "d_model": 1024,
            "num_layers": 24,
            "nhead": 16,
            "dim_feedforward": 4096,
            "max_seq_len": 6000,
        } # ? parameters
    }
    
    if size not in configs:
        raise ValueError(f"Invalid size '{size}'. Choose from: {list(configs.keys())}")
    
    # Extract config values
    cfg = configs[size]
    d_model = cfg["d_model"]
    nhead = cfg["nhead"]
    num_layers = cfg["num_layers"]
    dim_feedforward = cfg["dim_feedforward"]
    
    # LocalEncoder configuration
    LocalEncoder_config = {
        "in_channels": LocalEncoder_in_channels,
        "hidden_dim": LocalEncoder_hidden_dim,
        "out_channels": d_model,
        "time_entanglement": LocalEncoder_time_entanglement,
        "temporal_compression": LocalEncoder_temporal_compression
    }

    # FlowModel configuration
    FlowModel_config = {
        "frame_dim": frame_dim,
        "context_vector_dim": context_vector_dim,
        "num_past_atoms": segment_length,
        "frames_per_atom": frames_per_atom,
        "atoms_hop_frames": atoms_hop_frames, 
        "crossfade_frames": crossfade_frames, 
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "structure_dim": structure_dim
    }

    return LocalEncoder_config, FlowModel_config


class FlowTrainer:
    def __init__(
            self, 
            model, 
            local_encoder, 
            train_loader, 
            dataset, 
            processor, 
            optimizer,
            model_config: dict,      
            encoder_config: dict,    
            val_loader=None,         
            model_path="checkpoints/flow_model",
            context_source="clap", 
            val_audio_files=None, 
            device="cuda",
            past_dropout=0.1,
            conditioning_dropout=0.2,
            save_resume_states=False,     
            resume_from=None              # <--- CHANGED: Smart resume argument
        ):
        self.model = model.to(device)
        self.local_encoder = local_encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset = dataset
        self.processor = processor
        self.optimizer = optimizer
        self.device = device
        self.context_source = context_source
        self.past_dropout = past_dropout
        self.conditioning_dropout = conditioning_dropout
        self.model_config = model_config
        self.encoder_config = encoder_config
        self.atom_frames = model_config.get("frames_per_atom", 39) 
        self.use_structure = dataset.structure_feature_dimension > 0

        if self.use_structure and "target_structure" not in self.dataset.requested_keys:
            raise ValueError("Model expects structure but dataset does not request it")
        
        # ---> NEW: Inject feature names directly into the config dictionary <---
        if self.use_structure:
            # We use getattr just to be safe, defaulting to None if it somehow doesn't exist
            self.model_config["structure_feature_names"] = getattr(self.dataset, "structure_feature_names", None)

        # State Variables
        self.save_resume_states = save_resume_states
        self.start_epoch = 1
        self.best_metric = float('inf')
        
        if isinstance(val_audio_files, str):
            self.val_audio_files = [val_audio_files]
        else:
            self.val_audio_files = val_audio_files if val_audio_files else []
        
        # Setup Directories
        self.model_path = Path(model_path)
        self.ckpt_dir = self.model_path / "checkpoints"
        self.loss_dir = self.model_path / "loss"
        self.val_dir = self.model_path / "validation"
        
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        # <--- NEW: OVERWRITE PROTECTION
        existing_ckpts = list(self.ckpt_dir.glob("*.pt"))
        if existing_ckpts and not resume_from:
            raise FileExistsError(
                f"⚠️ Model directory '{self.ckpt_dir}' already contains checkpoints! "
                "To prevent accidentally overwriting your trained models, please either specify a different `model_path`, "
                "or set `resume_from='latest'` (or a specific epoch number) to continue training."
            )

        with open(self.ckpt_dir / "config_flow_model.json", "w") as f:
            json.dump(self.model_config, f, indent=4)
            
        with open(self.ckpt_dir / "config_local_encoder.json", "w") as f:
            json.dump(self.encoder_config, f, indent=4)

        self.train_losses = {"total": [], "latent": [], "scale": []}
        self.val_losses   = {"total": [], "latent": [], "scale": []}

        # <--- NEW: Smart Resume Trigger
        if resume_from is not None and resume_from is not False:
            resolved_path = self._resolve_resume_path(resume_from)
            if resolved_path:
                self._resume_from_state(resolved_path)
            else:
                print("⚠️ Could not resolve a valid trainer state to resume from.")

    def _resolve_resume_path(self, resume_from):
        """Intelligently resolves the user's intent into a concrete file path."""
        # 1. User wants the absolute last checkpoint (Colab crash recovery)
        if resume_from is True or resume_from in ["latest", "last"]:
            target_path = self.ckpt_dir / "last_trainer_state.pt"
            if target_path.exists():
                return target_path
            return None

        # 2. User wants a specific epoch (e.g., resume_from=45)
        elif isinstance(resume_from, int):
            target_path = self.ckpt_dir / f"epoch_{resume_from}_trainer_state.pt"
            if target_path.exists():
                return target_path
            raise FileNotFoundError(f"Requested to resume from epoch {resume_from}, but {target_path} does not exist.")

        # 3. User wants the best historical model
        elif resume_from == "best":
            target_path = self.ckpt_dir / "best_trainer_state.pt"
            if target_path.exists():
                return target_path
            raise FileNotFoundError(f"Requested 'best' resume, but {target_path} does not exist.")

        # 4. User provided a manual string/Path (Fallback)
        elif isinstance(resume_from, (str, Path)):
            target_path = Path(resume_from)
            if target_path.exists():
                return target_path
            raise FileNotFoundError(f"Manual resume path {target_path} does not exist.")
        
        return None

    def _resume_from_state(self, state_path):
        """Loads model weights, optimizer, and training history to resume seamlessly."""
        state_path = Path(state_path)
        if not state_path.exists():
            raise FileNotFoundError(f"Resume state not found at {state_path}")
        
        print(f"🔄 Resuming training from {state_path.name}...")
        
        # 1. Derive corresponding model paths by directly swapping the target names
        state_name = state_path.stem
        flow_name = state_name.replace("trainer_state", "flow_model")
        enc_name = state_name.replace("trainer_state", "local_encoder")
        
        flow_path = state_path.parent / f"{flow_name}.pt"
        enc_path = state_path.parent / f"{enc_name}.pt"
        
        if not flow_path.exists() or not enc_path.exists():
            raise FileNotFoundError(f"Could not find accompanying model files for {state_name}. Looked for {flow_name}.pt")

        # 2. Load Weights
        self.model.load_state_dict(torch.load(flow_path, map_location=self.device)['model_state_dict'])
        self.local_encoder.load_state_dict(torch.load(enc_path, map_location=self.device)['model_state_dict'])
        
        # 3. Load Optimizer & Training States
        state = torch.load(state_path, map_location=self.device)
        try:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        except Exception as e:
            print(f"⚠️ Could not load optimizer state: {e}")
        
        self.start_epoch = state['epoch'] + 1  # Start at the next epoch
        self.best_metric = state['best_metric']
        
        # Restore loss histories so plots don't start from scratch
        self.train_losses = state.get('train_losses', self.train_losses)
        self.val_losses = state.get('val_losses', self.val_losses)
        
        print(f"✅ Successfully resumed! Starting at Epoch {self.start_epoch} (Best Metric so far: {self.best_metric:.4f})")

    def _prepare_batch(self, batch):
        """Assembles the 129th Dimension (Latent + Scale)"""

        # -------------------------------------------------
        # 1. Past Memory
        # -------------------------------------------------
        past_latent = batch["memory_buffer_latent"].to(self.device)
        past_scale = batch["memory_buffer_scale"].to(self.device)

        past_scale_exp = past_scale.unsqueeze(-1).expand(
            -1, -1, -1, self.atom_frames
        )

        past_memory = torch.cat(
            [past_latent, past_scale_exp],
            dim=2
        )

        # -------------------------------------------------
        # 2. Present Target
        # -------------------------------------------------
        present_latent = batch["target_latent"].to(self.device)
        present_scale = batch["target_scale"].to(self.device)

        present_scale_exp = present_scale.unsqueeze(-1).expand(
            -1, -1, self.atom_frames
        )

        present_target = torch.cat(
            [present_latent, present_scale_exp],
            dim=1
        ).transpose(1, 2)

        # -------------------------------------------------
        # 3. Semantic Context
        # -------------------------------------------------
        if self.context_source not in ["clap", None]:
            raise ValueError(
                "Only 'clap' target_semantic is supported."
            )

        context = batch["target_semantic"].to(self.device)

        # -------------------------------------------------
        # 4. Optional Structure Conditioning
        # -------------------------------------------------
        structure = batch.get("target_structure", None)
        if structure is not None:
            structure = structure.to(self.device)

        if self.use_structure and structure is None:
            raise ValueError("Model expects structure_vector but dataset did not provide it")

        return past_memory, present_target, context, structure

    def _plot_and_save_losses(self, current_epoch):
        """Generates and saves a grid plot of the loss history."""
        
        has_val = self.val_loader is not None
        
        # Decide matrix layout
        rows = 2 if has_val else 1
        cols = 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        
        # Ensure axes is always a 2D numpy array for easy indexing
        if rows == 1:
            axes = axes.reshape(1, -1)
            
        epochs_x = range(1, current_epoch + 1)
        
        # Define the 3 loss types and their friendly titles
        loss_types = [
            ("total", "Total Loss", "tab:purple"),
            ("latent", "Latent Loss", "tab:blue"),
            ("scale", "Scale Loss", "tab:green")
        ]
        
        for col_idx, (key, title, color) in enumerate(loss_types):
            # --- Train Plot (Row 0) ---
            axes[0, col_idx].plot(epochs_x, self.train_losses[key], label=f"Train {title}", color=color, linewidth=2)
            axes[0, col_idx].set_title(f"Train {title}")
            axes[0, col_idx].set_xlabel("Epoch")
            axes[0, col_idx].set_ylabel("Loss")
            axes[0, col_idx].grid(True, linestyle='--', alpha=0.6)
            axes[0, col_idx].legend()
            
            # --- Validation Plot (Row 1) ---
            if has_val:
                axes[1, col_idx].plot(epochs_x, self.val_losses[key], label=f"Val {title}", color=color, linewidth=2, linestyle="--")
                axes[1, col_idx].set_title(f"Val {title}")
                axes[1, col_idx].set_xlabel("Epoch")
                axes[1, col_idx].set_ylabel("Loss")
                axes[1, col_idx].grid(True, linestyle='--', alpha=0.6)
                axes[1, col_idx].legend()

        plt.tight_layout()
        
        # Save to the loss directory
        save_path = self.loss_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # Prevent it from displaying in the terminal/notebook output

    def train_epoch(self):
        self.model.train()
        self.local_encoder.train()
        total_loss = 0
        total_lat_loss = 0
        total_scale_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            past_memory, present_target, context, structure = self._prepare_batch(batch)
            
            # Drop the CLAP embedding 10% of the time by replacing it with zeros
            if self.conditioning_dropout > 0.0:
                # Create a boolean mask of shape [Batch, 1]
                mask = torch.rand(context.shape[0], 1, device=self.device) < self.conditioning_dropout
                # Where mask is True, replace CLAP with zeros. Otherwise, keep CLAP.
                context = torch.where(mask, torch.zeros_like(context), context)

            self.optimizer.zero_grad()
            
            # 1. Encode memory
            encoded_past = self.local_encoder(past_memory)
            
            # 2. Dropout logic
            if self.past_dropout > 0.0:
                B, N_past, T_frames, d_model = encoded_past.shape
                
                # option 1: Randomly drop frames within atoms (less aggressive, more noisy)
                # mask = torch.zeros((B, N_past, T_frames, d_model), dtype=torch.bool, device=self.device)
                # for b in range(B):
                #     if torch.rand(1).item() < self.past_dropout:
                #         num_drop = torch.randint(1, N_past + 1, (1,)).item()
                #         mask[b, :num_drop, :, :] = True 
                # encoded_past = torch.where(mask, self.model.null_past_embed, encoded_past)

                # option 2: Drop entire atoms instead of random frames within atoms (more aggressive but cleaner)
                mask = torch.zeros((B, N_past, 1, 1), dtype=torch.bool, device=self.device)

                for b in range(B):
                    if torch.rand(1).item() < self.past_dropout:
                        num_drop = torch.randint(1, N_past + 1, (1,)).item()
                        mask[b, :num_drop] = True

                encoded_past = torch.where(mask, self.model.null_past_embed, encoded_past)
            
            # 3. Flow Loss with Scale Weighting
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
        # <--- NEW: Early exit if no validation loader exists
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
            
            # Use the same weighted loss as training
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
        """Generates TF validation audio using Asymmetric OLA Smooth Padding."""
        if not self.val_audio_files: 
            return
            
        self.model.eval()
        self.local_encoder.eval()

        # --- DYNAMIC ASYMMETRIC OLA SETUP ---
        atoms_frames = self.model_config.get("frames_per_atom", 39)
        hop_frames = self.model_config.get("atoms_hop_frames", 18)

        hop_time = hop_frames / 150.0
        time = 5 # seconds
        num_atoms = int(time // hop_time)

        crossfade_frames = self.model_config.get("crossfade_frames", 3)
        macro_overlap_frames = atoms_frames - hop_frames
        
        samples_per_frame = self.dataset.samples_per_frame
        
        segment_samples = atoms_frames * samples_per_frame
        hop_samples     = hop_frames * samples_per_frame
        crossfade_samples = crossfade_frames * samples_per_frame
        
        # Build the Asymmetric Window
        zeros_frames = macro_overlap_frames - crossfade_frames
        zeros = torch.zeros(zeros_frames * samples_per_frame, device=self.device)
        
        hann = torch.hann_window(crossfade_samples * 2, device=self.device)
        
        ones_frames = hop_frames - crossfade_frames
        ones = torch.ones(ones_frames * samples_per_frame, device=self.device)
        
        window = torch.cat([
            zeros, 
            hann[:crossfade_samples], 
            ones, 
            hann[crossfade_samples:]
        ]).view(1, 1, -1)

        # --- OLA Smooth Hyperparameters ---
        alpha_smooth = 0.6      # EMA factor
        max_jump = 1.15         # Max 15% growth
        max_drop = 0.85         # Max 15% drop

        for target_file in self.val_audio_files:
            file_indices = [i for i, (fname, _) in enumerate(self.dataset.all_indices) if fname == target_file]
            if not file_indices: 
                continue
                
            seq_indices = file_indices[:num_atoms]
            
            # Initialize Audio Buffer
            total_samples = (len(seq_indices) - 1) * hop_samples + segment_samples
            tf_out_audio = torch.zeros(1, 2, total_samples, device=self.device)
            
            prev_scale = None

            for i, idx in enumerate(tqdm(seq_indices, desc=f"Generating {target_file} (TF)")):
                # Prepare single-item batch
                raw_batch = self.dataset[idx]
                for k in raw_batch: 
                    if isinstance(raw_batch[k], torch.Tensor): 
                        raw_batch[k] = raw_batch[k].unsqueeze(0)
                
                gt_past, _, context, structure = self._prepare_batch(raw_batch)
                
                # Sample noise: [1, T_frames, 129]
                x0 = torch.randn(1, atoms_frames, 129, device=self.device)
                
                # --- Teacher Forcing Generation ---
                enc_tf = self.local_encoder(gt_past)
                tf_pred = self.model.generate(
                    x0, 
                    enc_tf, 
                    context, 
                    structure_vector=structure, # <--- Structure Injected
                    max_nfe=NFE
                ).transpose(1, 2)
                
                # --- OLA Smooth Processing ---
                tf_pred_smooth = tf_pred.clone()
                raw_scale = torch.abs(tf_pred[:, 128, :]).mean(dim=-1, keepdim=True)
                
                if prev_scale is None:
                    smoothed_scale = raw_scale
                else:
                    target_scale = torch.clamp(raw_scale, prev_scale * max_drop, prev_scale * max_jump)
                    smoothed_scale = (alpha_smooth * target_scale) + ((1.0 - alpha_smooth) * prev_scale)
                    
                prev_scale = smoothed_scale
                tf_pred_smooth[:, 128, :] = smoothed_scale.expand_as(tf_pred_smooth[:, 128, :])
                
                # --- Internal Decoder ---
                latents = tf_pred_smooth[:, :128, :]
                meta = {
                    "audio_scales": [smoothed_scale.squeeze(0).float()],
                    "padding_mask": torch.ones((1, atoms_frames * samples_per_frame), 
                                             dtype=torch.bool, device=self.device)
                }
                
                audio = self.processor.decode_latents_audio(latents, metadata=meta)
                audio = audio * window
                
                start = i * hop_samples
                tf_out_audio[:, :, start : start + segment_samples] += audio

            # Save results to validation directory (Only TF now)
            file_stem = Path(target_file).stem
            sf.write(self.val_dir / f"epoch_{epoch}_{file_stem}_TF.wav", 
                     tf_out_audio.squeeze(0).T.cpu().numpy(), 48000)
            
        print(f"✅ Validation audio for epoch {epoch} saved!")

    def train(self, epochs, audio_val_freq=5, val_nfe=32):
        
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            
            # Run Training 
            avg_t_total, avg_t_lat, avg_t_scale = self.train_epoch()
            
            self.train_losses["total"].append(avg_t_total)
            self.train_losses["latent"].append(avg_t_lat)
            self.train_losses["scale"].append(avg_t_scale)
            
            print(f"Train | Total: {avg_t_total:.4f} (Lat: {avg_t_lat:.4f}, Sca: {avg_t_scale:.4f})")

            # Conditional Validation Logic
            if self.val_loader is not None:
                avg_v_total, avg_v_lat, avg_v_scale = self.val_epoch()
                
                self.val_losses["total"].append(avg_v_total)
                self.val_losses["latent"].append(avg_v_lat)
                self.val_losses["scale"].append(avg_v_scale)
                
                print(f"Val   | Total: {avg_v_total:.4f} (Lat: {avg_v_lat:.4f}, Sca: {avg_v_scale:.4f})")
                current_metric = avg_v_total
            else:
                current_metric = avg_t_total
            
            # ----------------------------------------------------
            # STATE PACKAGING (If enabled)
            # ----------------------------------------------------
            trainer_state = None
            if self.save_resume_states:
                trainer_state = {
                    'epoch': epoch,
                    'best_metric': min(current_metric, self.best_metric),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }

            # 1. Save "Best" Checkpoint (No changes here)
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / "best_flow_model.pt")
                torch.save({'model_state_dict': self.local_encoder.state_dict()}, self.ckpt_dir / "best_local_encoder.pt")
                
                if self.save_resume_states:
                    torch.save(trainer_state, self.ckpt_dir / "best_trainer_state.pt")
                    
                print("🌟 Saved new best models!")
                
            # 2. Save Periodic Checkpoints & Generate Audio
            if epoch % audio_val_freq == 0:
                torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / f"epoch_{epoch}_flow_model.pt")
                torch.save({'model_state_dict': self.local_encoder.state_dict()}, self.ckpt_dir / f"epoch_{epoch}_local_encoder.pt")
                
                if self.save_resume_states:
                    torch.save(trainer_state, self.ckpt_dir / f"epoch_{epoch}_trainer_state.pt")
                
                # Audio generation
                self.generate_validation_audio(epoch, NFE=val_nfe)

            # 3. Always save the "Last" Checkpoint (Overwrites every epoch)
            torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / "last_flow_model.pt")
            torch.save({'model_state_dict': self.local_encoder.state_dict()}, self.ckpt_dir / "last_local_encoder.pt")
            if self.save_resume_states:
                torch.save(trainer_state, self.ckpt_dir / "last_trainer_state.pt")
                
            # 4. Save Detailed Loss History
            history_path = self.loss_dir / "loss_history.json"
            with open(history_path, "w") as f:
                json.dump({
                    "train": self.train_losses,
                    "val": self.val_losses if self.val_loader else {} 
                }, f, indent=4)
                
            # 5. Generate and Save Loss Plots
            self._plot_and_save_losses(epoch)