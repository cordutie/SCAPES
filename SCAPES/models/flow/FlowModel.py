from typing import Optional

import torch
import torch.nn as nn
from SCAPES.models.flow.PosEnc import MemoryPositionalEncoding, RotaryEmbedding
from SCAPES.auxiliar.ode_utils import sample_with_ode_capped
from SCAPES.data.config_loader import TrainingConfig
import json


def load_flow_model(checkpoint_path, json_path, device="cpu"):
    with open(json_path, 'r') as f:
        config = json.load(f)

    model = FlowModel(
        frame_dim=config.get("frame_dim", 129),
        context_vector_dim=config.get("context_vector_dim", 1024),
        num_past_atoms=config.get("num_past_atoms", 5),
        frames_per_atom=config.get("frames_per_atom", 21),
        d_model=config.get("d_model", 256),
        nhead=config.get("nhead", 8),
        num_layers=config.get("num_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 1024),
        structure_dim=config.get("structure_dim", 0),
        device=device
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model.to(device)


class AdaLN(nn.Module):
    def __init__(self, d_model, cond_dim, device=None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, device=device)
        self.proj = nn.Linear(cond_dim, d_model * 2, device=device)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return self.norm(x) * (1 + gamma) + beta


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, cond_dim: int, dropout: float = 0.1, device=None):
        super().__init__()

        self.norm1 = AdaLN(d_model, cond_dim, device=device)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, device=device)

        self.norm2 = AdaLN(d_model, cond_dim, device=device)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, device=device)

        self.norm3 = AdaLN(d_model, cond_dim, device=device)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, device=device),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, device=device),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x, cond)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.norm2(x, cond)
        cross_out, _ = self.cross_attn(x_norm, memory, memory)
        x = x + cross_out

        x_norm = self.norm3(x, cond)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class VectorField(nn.Module):
    def __init__(
        self,
        frame_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        context_dim: int = 1024,
        structure_dim: int = 0,
        max_atom_frames: int = 21,
        device=None
    ):
        super().__init__()
        self.d_model = d_model

        self.target_proj = nn.Linear(frame_dim, d_model, device=device)
        self.target_rope = RotaryEmbedding(d_model, max_position=max_atom_frames+1)

        if structure_dim > 0:
            self.use_structure = True
        else:
            self.use_structure = False

        if self.use_structure:
            self.structure_proj = nn.Sequential(
                nn.BatchNorm1d(structure_dim, device=device),
                nn.Linear(structure_dim, context_dim, device=device),
                nn.GELU(),
                nn.Linear(context_dim, context_dim, device=device),
            )

        cond_dim = d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model, device=device),
            nn.GELU(),
            nn.Linear(d_model, cond_dim, device=device)
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, d_model, device=device),
            nn.GELU(),
            nn.Linear(d_model, cond_dim, device=device)
        )

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward, cond_dim, device=device)
            for _ in range(num_layers)
        ])

        self.final_norm = AdaLN(d_model, cond_dim, device=device)
        self.output_proj = nn.Linear(d_model, frame_dim, device=device)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_target: torch.Tensor,
        precomputed_memory: torch.Tensor,
        context_vector: torch.Tensor,
        s: torch.Tensor,
        structure_vector: torch.Tensor = None
    ) -> torch.Tensor:
        B, T_new, _ = noisy_target.shape

        t_emb = self.time_mlp(s)
        if self.use_structure:
            if structure_vector is None:
                raise ValueError("structure_vector must be provided when structure_dim > 0")
            structure_emb = self.structure_proj(structure_vector)
            context_vector = context_vector + structure_emb
        c_emb = self.context_mlp(context_vector)
        cond = t_emb + c_emb

        x = self.target_proj(noisy_target)
        x_fake_4d = x.unsqueeze(1)
        x_fake_4d = self.target_rope.apply_rotary(x_fake_4d)
        x = x_fake_4d.squeeze(1)

        for layer in self.layers:
            x = layer(x, memory=precomputed_memory, cond=cond)

        x = self.final_norm(x, cond)
        velocity_field = self.output_proj(x)
        return velocity_field


class FlowModel(nn.Module):
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        *,
        frame_dim: int = 128,
        context_vector_dim: int = 1024,
        num_past_atoms: int = 20,
        frames_per_atom: int = 21,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        structure_dim: int = 0,
        device=None,
        **kwargs
    ):
        if config is not None:
            d_model = config.d_model
            nhead = config.nhead
            num_layers = config.num_layers
            dim_feedforward = config.dim_feedforward

        super().__init__()

        self.null_past_embed = nn.Parameter(torch.randn(1, 1, frames_per_atom, d_model) * 0.02)

        self.memory_pos_enc = MemoryPositionalEncoding(
            d_model=d_model,
            n_atoms_max=num_past_atoms + 5,
            max_atom_frames=frames_per_atom + 5
        )

        self.transformer = VectorField(
            frame_dim=frame_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            context_dim=context_vector_dim,
            max_atom_frames=frames_per_atom,
            structure_dim=structure_dim,
            device=device
        )

    def prepare_memory(self, encoded_past: torch.Tensor) -> torch.Tensor:
        return self.memory_pos_enc(encoded_past)

    def forward(self, x_t, s, context_vector, encoded_past, structure_vector=None):
        precomputed_memory = self.prepare_memory(encoded_past)
        return self.transformer(x_t, precomputed_memory, context_vector, s, structure_vector)

    def vector_field(self, x, s, context_dict):
        precomputed_memory = context_dict['memory']
        clap_context       = context_dict['clap']
        structure_vector   = context_dict.get('structure', None)
        cfg_scale          = context_dict.get('cfg_scale', 3.0)

        if cfg_scale == 1.0:
            return self.transformer(x, precomputed_memory, clap_context, s, structure_vector)

        v_cond = self.transformer(x, precomputed_memory, clap_context, s, structure_vector)

        null_clap = torch.zeros_like(clap_context)
        v_uncond = self.transformer(x, precomputed_memory, null_clap, s, structure_vector)

        return v_uncond + cfg_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def generate(self, x0, encoded_past, clap_context, structure_vector=None, max_nfe=16, cfg_scale=1.0):
        memory = self.prepare_memory(encoded_past)

        context_dict = {
            'memory': memory,
            'clap': clap_context,
            'structure': structure_vector,
            'cfg_scale': cfg_scale
        }

        x1 = sample_with_ode_capped(
            u=self.vector_field,
            x0=x0,
            context=context_dict,
            max_nfe=max_nfe,
            method='euler',
            step_size=1.0/max_nfe
        )
        return x1
