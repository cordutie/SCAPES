import os
import json
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

def get_global_model_configs(
    frames_per_atom=39, 
    latent_dim=128, 
    cnn_hidden=256, 
    transformer_dim=256, 
    num_heads=4, 
    num_layers=4, 
    clap_dim=1024
):
    """Generates the config dictionary for the GlobalEncoder."""
    config = {
        "latent_dim": latent_dim,
        "frames_per_atom": frames_per_atom,
        "cnn_hidden": cnn_hidden,
        "transformer_dim": transformer_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "clap_dim": clap_dim
    }
    return config


class GlobalTrainer:
    def __init__(
            self, 
            model, 
            train_loaders, 
            val_loaders, 
            optimizer,
            scheduler,
            model_config: dict,      
            model_path="checkpoints/global_model",
            device="cuda"
        ):
        self.model = model.to(device)
        self.train_loaders = train_loaders if isinstance(train_loaders, list) else [train_loaders]
        self.val_loaders = val_loaders if isinstance(val_loaders, list) else [val_loaders]
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_config = model_config
        
        # Setup Directories
        self.model_path = Path(model_path)
        self.ckpt_dir = self.model_path / "checkpoints"
        self.loss_dir = self.model_path / "loss"
        
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)

        # Save the JSON config immediately
        with open(self.ckpt_dir / "global_encoder_config.json", "w") as f:
            json.dump(self.model_config, f, indent=4)

        # History tracking (supports multiple sequence lengths)
        self.train_loss_history = {}
        self.val_loss_history = {}

    def _prepare_batch(self, batch):
        latent = batch["latent_context_win"].to(self.device)
        scale = batch["scale_context_win"].to(self.device)
        target_clap = batch["clap_context_win"].to(self.device)
        return latent, scale, target_clap

    def train_epoch(self):
        self.model.train()
        total_train_steps = sum(len(loader) for loader in self.train_loaders)
        
        epoch_train_losses = {}
        train_pbar = tqdm(total=total_train_steps, desc="Training")
        
        # Iterators for randomly picking batches from different seq lengths
        train_iters = [iter(loader) for loader in self.train_loaders]
        active_iters = list(range(len(train_iters)))

        while active_iters:
            idx = random.choice(active_iters)
            try:
                batch = next(train_iters[idx])
            except StopIteration:
                active_iters.remove(idx)
                continue

            latent, scale, target_clap = self._prepare_batch(batch)
            seq_len = latent.shape[1] 
            
            self.optimizer.zero_grad()
            pred_clap = self.model(latent, scale)
            
            # Cosine Loss
            sim = F.cosine_similarity(pred_clap, target_clap, dim=-1)
            loss = (1.0 - sim).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track loss by sequence length
            if seq_len not in epoch_train_losses:
                epoch_train_losses[seq_len] = []
            epoch_train_losses[seq_len].append(loss.item())
            
            train_pbar.update(1)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'len': seq_len})
            
        train_pbar.close()
        return epoch_train_losses

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        total_val_steps = sum(len(loader) for loader in self.val_loaders)
        
        epoch_val_losses = {}
        val_pbar = tqdm(total=total_val_steps, desc="Validation")
        
        for loader in self.val_loaders:
            for batch in loader:
                latent, scale, target_clap = self._prepare_batch(batch)
                seq_len = latent.shape[1]
                
                pred_clap = self.model(latent, scale)
                sim = F.cosine_similarity(pred_clap, target_clap, dim=-1)
                loss = (1.0 - sim).mean()
                
                if seq_len not in epoch_val_losses:
                    epoch_val_losses[seq_len] = []
                epoch_val_losses[seq_len].append(loss.item())
                
                val_pbar.update(1)
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'len': seq_len})
                
        val_pbar.close()
        return epoch_val_losses

    def _aggregate_and_store(self, epoch_losses, history_dict):
        """Averages the losses for the epoch and appends to the history."""
        global_loss = 0.0
        total_batches = 0
        
        for seq_len, losses in epoch_losses.items():
            if seq_len not in history_dict:
                history_dict[seq_len] = []
            
            avg_len_loss = sum(losses) / len(losses)
            history_dict[seq_len].append(avg_len_loss)
            
            global_loss += sum(losses)
            total_batches += len(losses)
            
        return global_loss / total_batches if total_batches > 0 else 0.0

    def train(self, epochs):
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            
            # Run passes
            epoch_t_losses = self.train_epoch()
            epoch_v_losses = self.val_epoch()
            
            # Aggregate and store histories
            avg_train_loss = self._aggregate_and_store(epoch_t_losses, self.train_loss_history)
            avg_val_loss = self._aggregate_and_store(epoch_v_losses, self.val_loss_history)
            
            if self.scheduler:
                self.scheduler.step()
                
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # 1. Save Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt_dir / "best_global_encoder.pt")
                print("🌟 Saved new best model!")
                
            # 2. Save Periodic Checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_loss': best_val_loss,
                    'train_loss_history': self.train_loss_history,
                    'val_loss_history': self.val_loss_history,
                }, self.ckpt_dir / f"global_encoder_epoch_{epoch}.resume")
                
            # 3. Save History to JSON
            history_path = self.loss_dir / "loss_history.json"
            with open(history_path, "w") as f:
                json.dump({
                    "train": self.train_loss_history,
                    "val": self.val_loss_history
                }, f, indent=4)