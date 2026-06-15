"""
reflow_trainer.py — ReFlow Trainer for SCAPES.

Subclasses FlowTrainer to use paired noise from the dataset (saved during
data_generation.py) instead of randomly sampling noise at each training step.

Usage:
    trainer = ReFlowTrainer(model, local_encoder, train_loader, dataset,
                            processor, optimizer, config, ...)
    trainer.train(epochs=75, ...)
"""

import torch
from tqdm import tqdm

from SCAPES.auxiliar.losses_flow import flow_matching_loss, time_phase_regularizer, fft_phase_regularizer
from SCAPES.training.FlowModel_trainer import FlowTrainer


class ReFlowTrainer(FlowTrainer):
    """ReFlow trainer: uses noise from dataset instead of random sampling.

    All parameters and behavior are identical to FlowTrainer except:
    - ``_prepare_batch`` also returns ``noise`` from ``batch["target_noise"]``
    - ``train_epoch`` uses dataset noise instead of ``torch.randn_like``
    - ``val_epoch`` uses dataset noise instead of ``torch.randn_like``
    """

    def _prepare_batch(self, batch):
        past_memory, present_target, context, structure = super()._prepare_batch(batch)
        noise = batch["target_noise"].to(self.device)
        return past_memory, present_target, context, structure, noise

    def train_epoch(self, discriminator=None, collect_adv=False):
        self.model.train()
        self.local_encoder.train()
        total_loss = 0
        total_lat_loss = 0
        total_scale_loss = 0
        total_adv_loss = 0
        total_reg_loss = 0

        active_regularizers = []
        if self.regularizers_and_weights:
            REG_MAP = {
                "time_phase": time_phase_regularizer,
                "fft_phase": fft_phase_regularizer,
            }
            for name, weight in self.regularizers_and_weights:
                fn = REG_MAP.get(name)
                if fn is None:
                    print(f"Unknown regularizer '{name}', skipping.")
                else:
                    active_regularizers.append((fn, weight))

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            past_memory, present_target, context, structure, noise = self._prepare_batch(batch)

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

            loss, l_lat, l_scale, X_hat, s = flow_matching_loss(
                self.model, noise, present_target, context, encoded_past,
                structure_vector=structure
            )

            reg_loss = torch.tensor(0.0, device=self.device)
            for fn, weight in active_regularizers:
                reg_loss += weight * fn(X_hat, present_target, s)

            combined_loss = loss + reg_loss

            if discriminator is not None:
                X_hat_latent = X_hat[:, :, :128].transpose(1, 2).contiguous()
                logits = discriminator(X_hat_latent).squeeze(-1)
                adv_loss = self.adv_loss_fn(logits, torch.zeros_like(logits)).mean()
                combined_loss = combined_loss + adv_loss
                total_adv_loss += adv_loss.item()
            else:
                adv_loss = None

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.local_encoder.parameters()), 1.0
            )
            self.optimizer.step()

            total_loss += combined_loss.item()
            total_lat_loss += l_lat.item()
            total_scale_loss += l_scale.item()
            total_reg_loss += reg_loss.item()

            if collect_adv:
                remaining = self.adv_buffer_max - len(self.x_hat_buffer)
                if remaining > 0:
                    X_hat_latent = X_hat[:, :, :128].transpose(1, 2).contiguous()
                    B_curr = X_hat_latent.shape[0]
                    take = min(remaining, B_curr)
                    self.x_hat_buffer.append((
                        X_hat_latent[:take].detach().cpu(),
                        s[:take].detach().cpu().reshape(take, 1)
                    ))

            postfix = {
                "L": f"{combined_loss.item():.4f}",
                "Lat": f"{l_lat.item():.4f}",
                "Sca": f"{l_scale.item():.4f}",
            }
            if reg_loss.item() > 0:
                postfix["Reg"] = f"{reg_loss.item():.4f}"
            if adv_loss is not None:
                postfix["Adv"] = f"{adv_loss.item():.4f}"
            pbar.set_postfix(postfix)

        n = len(self.train_loader)
        return total_loss / n, total_lat_loss / n, total_scale_loss / n, total_adv_loss / n, total_reg_loss / n

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
            past_memory, present_target, context, structure, noise = self._prepare_batch(batch)
            encoded_past = self.local_encoder(past_memory)

            loss, l_lat, l_scale, _, _ = flow_matching_loss(
                self.model,
                noise,
                present_target,
                context,
                encoded_past,
                structure_vector=structure,
            )

            total_loss += loss.item()
            total_lat += l_lat.item()
            total_scale += l_scale.item()

        return (
            total_loss / len(self.val_loader),
            total_lat / len(self.val_loader),
            total_scale / len(self.val_loader),
        )
