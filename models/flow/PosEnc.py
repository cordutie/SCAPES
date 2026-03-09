import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=64):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("sin", freqs.sin())
        self.register_buffer("cos", freqs.cos())

    def apply_rotary(self, x):
        seq_len = x.size(2)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return x_rot.flatten(-2)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n_atoms_max, d_model):
        super().__init__()
        self.embedding = nn.Embedding(n_atoms_max, d_model)

    def forward(self, x):
        B, N_atoms, _, _ = x.shape
        atom_ids = torch.arange(N_atoms, device=x.device).unsqueeze(0).expand(B, -1)
        atom_emb = self.embedding(atom_ids).unsqueeze(2)
        return x + atom_emb

class MemoryPositionalEncoding(nn.Module):
    def __init__(self, d_model, n_atoms_max=20, max_atom_frames=32):
        super().__init__()
        self.atom_embedding = LearnablePositionalEncoding(n_atoms_max, d_model)
        self.rope = RotaryEmbedding(d_model, max_position=max_atom_frames)

    def forward(self, x):
        # x shape: (B, N_atoms, T_atom, d_model)
        x = self.rope.apply_rotary(x) # Micro-time
        x = self.atom_embedding(x)    # Macro-time
        B, N_atoms, T_atom, d = x.shape
        return x.view(B, N_atoms * T_atom, d)