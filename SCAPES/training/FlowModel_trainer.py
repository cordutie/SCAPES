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
    frames_per_atom=39,       # <--- Updated Default
    atoms_hop_frames=18,      # <--- NEW
    crossfade_frames=3,       # <--- NEW
    context_vector_dim=1024,
    LocalEncoder_in_channels=129,
    LocalEncoder_hidden_dim=256,
    LocalEncoder_time_entanglement=True,
    LocalEncoder_temporal_compression=1
):
    configs = {
        "small": {
            "d_model": 512,
            "num_layers": 6,
            "nhead": 8,
            "dim_feedforward": 2048,
            "max_seq_len": 2048,
        },
        "small_new": {
            "d_model": 512,
            "num_layers": 12,
            "nhead": 16,
            "dim_feedforward": 2048,
            "max_seq_len": 2048,
        },
        "medium": {
            "d_model": 768,
            "num_layers": 8,
            "nhead": 12,
            "dim_feedforward": 2048,
            "max_seq_len": 2048,
        },
        "large": {
            "d_model": 1024,
            "num_layers": 12,
            "nhead": 16,
            "dim_feedforward": 2048,
            "max_seq_len": 2048,
        },
        "extra_large": {
            "d_model": 1024,
            "num_layers": 24,
            "nhead": 16,
            "dim_feedforward": 4096,
            "max_seq_len": 6000,
        }
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
        "dim_feedforward": dim_feedforward
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
            val_loader=None,         # <--- Default to None
            model_path="checkpoints/flow_model",
            context_source="clap", 
            val_audio_files=None, 
            device="cuda",
            past_dropout=0.1
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
        self.model_config = model_config
        self.encoder_config = encoder_config
        self.atom_frames = model_config.get("frames_per_atom", 39) 
        
        # Ensure it's a list for iteration
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

        # Save the JSON configs immediately
        with open(self.ckpt_dir / "flow_model_config.json", "w") as f:
            json.dump(self.model_config, f, indent=4)
            
        with open(self.ckpt_dir / "local_encoder_config.json", "w") as f:
            json.dump(self.encoder_config, f, indent=4)

        # History updated for split losses
        self.train_losses = {"total": [], "latent": [], "scale": []}
        self.val_losses   = {"total": [], "latent": [], "scale": []}

    def _prepare_batch(self, batch):
        """Assembles the 129th Dimension (Latent + Scale)"""
        # 1. Past Memory: [B, N, 128, T] + [B, N, 1] -> [B, N, 129, T]
        past_latent = batch["latent_past"].to(self.device)
        past_scale = batch["scale_past"].to(self.device)
        past_scale_exp = past_scale.unsqueeze(-1).expand(-1, -1, -1, self.atom_frames)
        past_memory = torch.cat([past_latent, past_scale_exp], dim=2)

        # 2. Present Target: [B, 128, T] + [B, 1] -> [B, T, 129]
        present_latent = batch["latent_present"].to(self.device)
        present_scale = batch["scale_present"].to(self.device)
        present_scale_exp = present_scale.unsqueeze(-1).expand(-1, -1, self.atom_frames)
        present_target = torch.cat([present_latent, present_scale_exp], dim=1).transpose(1, 2)

        # 3. Context Toggle
        ctx_key = "clap_context_win" if self.context_source == "clap" else "ctx_emb_context_win"
        context = batch[ctx_key].to(self.device)

        return past_memory, present_target, context

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
            past_memory, present_target, context = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            
            # 1. Encode memory
            encoded_past = self.local_encoder(past_memory)
            
            # 2. Dropout logic
            if self.past_dropout > 0.0:
                B, N_past, T_frames, d_model = encoded_past.shape
                mask = torch.zeros((B, N_past, T_frames, d_model), dtype=torch.bool, device=self.device)
                for b in range(B):
                    if torch.rand(1).item() < self.past_dropout:
                        num_drop = torch.randint(1, N_past + 1, (1,)).item()
                        mask[b, :num_drop, :, :] = True 
                encoded_past = torch.where(mask, self.model.null_past_embed, encoded_past)
            
            # 3. Flow Loss with Scale Weighting
            noise = torch.randn_like(present_target)
            loss, l_lat, l_scale = flow_matching_loss(
                self.model, noise, present_target, context, encoded_past, 
                scale_weight=1.0 
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
            past_memory, present_target, context = self._prepare_batch(batch)
            encoded_past = self.local_encoder(past_memory)
            noise = torch.randn_like(present_target)
            
            # Use the same weighted loss as training
            loss, l_lat, l_scale = flow_matching_loss(
                self.model, noise, present_target, context, encoded_past, scale_weight=1.0
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
    def generate_validation_audio(self, epoch, num_atoms=50, NFE=32):
        """Generates audio using Asymmetric OLA Prefix Padding."""
        if not self.val_audio_files: 
            return
            
        self.model.eval()
        self.local_encoder.eval()

        # --- DYNAMIC ASYMMETRIC OLA SETUP ---
        # 1. Pull atom geometry from model_config
        atoms_frames = self.model_config.get("frames_per_atom", 39)
        hop_frames = self.model_config.get("atoms_hop_frames", 18)
        crossfade_frames = self.model_config.get("crossfade_frames", 3)
        macro_overlap_frames = atoms_frames - hop_frames
        
        # 2. Pull sample rate math from dataset
        samples_per_frame = self.dataset.samples_per_frame
        
        segment_samples = atoms_frames * samples_per_frame
        hop_samples     = hop_frames * samples_per_frame
        crossfade_samples = crossfade_frames * samples_per_frame
        macro_overlap_samples = macro_overlap_frames * samples_per_frame
        
        # 3. Build the Asymmetric Window
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

        for target_file in self.val_audio_files:
            # Find indices for this specific file
            file_indices = [i for i, (fname, _) in enumerate(self.dataset.all_indices) if fname == target_file]
            if not file_indices: 
                continue
                
            seq_indices = file_indices[:num_atoms]
            ar_buffer = None # Will initialize from first batch
            
            # Initialize Audio Buffers for OLA
            total_samples = (len(seq_indices) - 1) * hop_samples + segment_samples
            tf_out_audio = torch.zeros(1, 2, total_samples, device=self.device)
            ar_out_audio = torch.zeros(1, 2, total_samples, device=self.device)

            for i, idx in enumerate(tqdm(seq_indices, desc=f"Generating {target_file}")):
                # Prepare single-item batch
                raw_batch = self.dataset[idx]
                for k in raw_batch: 
                    if isinstance(raw_batch[k], torch.Tensor): 
                        raw_batch[k] = raw_batch[k].unsqueeze(0)
                
                gt_past, _, context = self._prepare_batch(raw_batch)
                
                # Initialize AR buffer at step 0
                if ar_buffer is None: 
                    ar_buffer = torch.zeros_like(gt_past)
                
                # Sample noise: [1, T_frames, 129]
                x0 = torch.randn(1, atoms_frames, 129, device=self.device)
                
                # --- 1. Teacher Forcing Step ---
                enc_tf = self.local_encoder(gt_past)
                tf_pred = self.model.generate(x0, enc_tf, context, max_nfe=NFE).transpose(1, 2)
                
                # --- 2. Autoregressive Step (with NULL cold-start) ---
                enc_ar = self.local_encoder(ar_buffer)
                num_nulls = max(0, gt_past.shape[1] - i)
                if num_nulls > 0:
                    enc_ar[:, :num_nulls, :, :] = self.model.null_past_embed
                
                ar_pred = self.model.generate(x0, enc_ar, context, max_nfe=NFE).transpose(1, 2)
                
                # --- 3. Internal Decoder + OLA Function ---
                def decode_and_add(pred, buffer, index):
                    # Robust Scale Handling: Force positive and average over the atom
                    latents = pred[:, :128, :]
                    scale = torch.abs(pred[:, 128, :]).mean(dim=-1, keepdim=True)
                    
                    meta = {
                        "audio_scales": [scale.squeeze(0).float()],
                        "padding_mask": torch.ones((1, atoms_frames * samples_per_frame), 
                                                 dtype=torch.bool, device=self.device)
                    }
                    
                    # Decode to audio [1, 2, T]
                    audio = self.processor.decode_latents_audio(latents, metadata=meta)
                    # Apply asymmetric window and OLA
                    audio = audio * window
                    
                    start = index * hop_samples
                    buffer[:, :, start : start + segment_samples] += audio

                # Process both streams
                decode_and_add(tf_pred, tf_out_audio, i)
                decode_and_add(ar_pred, ar_out_audio, i)

                # Update AR Buffer for next step: Roll left, insert new prediction as latest atom
                new_ar_atom = ar_pred.unsqueeze(1)
                ar_buffer = torch.cat([ar_buffer[:, 1:], new_ar_atom], dim=1)

            # Save results to validation directory
            file_stem = Path(target_file).stem
            sf.write(self.val_dir / f"epoch_{epoch}_{file_stem}_TF.wav", 
                     tf_out_audio.squeeze(0).T.cpu().numpy(), 48000)
            sf.write(self.val_dir / f"epoch_{epoch}_{file_stem}_AR.wav", 
                     ar_out_audio.squeeze(0).T.cpu().numpy(), 48000)
            
        print(f"✅ Validation audio for epoch {epoch} saved!")

    def train(self, epochs, audio_val_freq=5, val_nfe=32):
        best_metric = float('inf') # <--- NEW: renamed to be generic for train or val
        
        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            
            # Run Training 
            avg_t_total, avg_t_lat, avg_t_scale = self.train_epoch()
            
            self.train_losses["total"].append(avg_t_total)
            self.train_losses["latent"].append(avg_t_lat)
            self.train_losses["scale"].append(avg_t_scale)
            
            print(f"Train | Total: {avg_t_total:.4f} (Lat: {avg_t_lat:.4f}, Sca: {avg_t_scale:.4f})")

            # <--- NEW: Conditional Validation Logic
            if self.val_loader is not None:
                avg_v_total, avg_v_lat, avg_v_scale = self.val_epoch()
                
                self.val_losses["total"].append(avg_v_total)
                self.val_losses["latent"].append(avg_v_lat)
                self.val_losses["scale"].append(avg_v_scale)
                
                print(f"Val   | Total: {avg_v_total:.4f} (Lat: {avg_v_lat:.4f}, Sca: {avg_v_scale:.4f})")
                current_metric = avg_v_total
            else:
                # If no validation set, use training loss to track the "best" model
                current_metric = avg_t_total
            
            # 1. Save "Best" Checkpoint
            if current_metric < best_metric:
                best_metric = current_metric
                torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / "best_flow_model.pt")
                torch.save({'model_state_dict': self.local_encoder.state_dict()}, self.ckpt_dir / "best_local_encoder.pt")
                print("🌟 Saved new best models!")
                
            # 2. Save Periodic Checkpoints & Generate Audio
            if epoch % audio_val_freq == 0:
                torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / f"flow_model_epoch_{epoch}.pt")
                torch.save({'model_state_dict': self.local_encoder.state_dict()}, self.ckpt_dir / f"local_encoder_epoch_{epoch}.pt")
                
                # Audio generation still runs based purely on self.val_audio_files
                self.generate_validation_audio(epoch, NFE=val_nfe)
                
            # 3. Save Detailed Loss History
            history_path = self.loss_dir / "loss_history.json"
            with open(history_path, "w") as f:
                json.dump({
                    "train": self.train_losses,
                    "val": self.val_losses if self.val_loader else {} # Skip val dict if empty
                }, f, indent=4)
                
            # 4. Generate and Save Loss Plots
            self._plot_and_save_losses(epoch)