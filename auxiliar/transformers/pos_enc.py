import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=64):
        super().__init__()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position).float()

        freqs = torch.einsum("i,j->ij", t, inv_freq)

        self.register_buffer("sin", freqs.sin())
        self.register_buffer("cos", freqs.cos())

    def apply_rotary(self, x):
        """
        x: (B, seq_len, d_model)
        """

        sin = self.sin[: x.size(1)]
        cos = self.cos[: x.size(1)]

        sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_rot = torch.stack(
            (x1 * cos - x2 * sin,
             x1 * sin + x2 * cos),
            dim=-1
        )

        return x_rot.flatten(-2)
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n_atoms_max, d_model):
        super().__init__()
        self.embedding = nn.Embedding(n_atoms_max, d_model)

    def forward(self, x):
        """
        x: (B, N_atoms, T_atom, d_model)
        """

        B, N_atoms, T_atom, d = x.shape

        atom_ids = torch.arange(N_atoms, device=x.device)
        atom_ids = atom_ids.unsqueeze(0).expand(B, -1)

        atom_emb = self.embedding(atom_ids)  # (B, N_atoms, d)
        atom_emb = atom_emb.unsqueeze(2)     # (B, N_atoms, 1, d)

        x = x + atom_emb

        return x