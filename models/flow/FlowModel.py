import torch
import torch.nn as nn
from models.flow.PosEnc import MemoryPositionalEncoding, RotaryEmbedding
from auxiliar.ode_util_v2 import sample_with_ode_capped

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    Injects the Time (s) and Context (CLAP) into the network by modulating the normalization.
    """
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, d_model * 2)
        
        # Initialize projection to output exactly 0 so it starts as a standard LayerNorm
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1) 
        beta = beta.unsqueeze(1)
        return self.norm(x) * (1 + gamma) + beta

class TransformerLayer(nn.Module):
    """
    A standard Flow Matching Transformer Layer using AdaLN.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = AdaLN(d_model, cond_dim)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm2 = AdaLN(d_model, cond_dim)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm3 = AdaLN(d_model, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
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
    """
    The Core Neural Network for Flow Matching.
    Expects memory to be pre-computed and passed in as `precomputed_memory`.
    """
    def __init__(
        self,
        frame_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        context_dim: int = 1024,
        max_atom_frames: int = 21,
    ):
        super().__init__()
        self.d_model = d_model
        
        # --- 1. Target Encoding Pipeline (The Present/Future) ---
        self.target_proj = nn.Linear(frame_dim, d_model)
        self.target_rope = RotaryEmbedding(d_model, max_position=max_atom_frames+1)
        
        # --- 2. Global Conditioning (Time + Timbre) ---
        cond_dim = d_model 
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, cond_dim)
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, cond_dim)
        )
        
        # --- 3. The Transformer ---
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_feedforward, cond_dim)
            for _ in range(num_layers)
        ])
        
        # --- 4. Output Projection ---
        self.final_norm = AdaLN(d_model, cond_dim)
        self.output_proj = nn.Linear(d_model, frame_dim)
        
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self, 
        noisy_target: torch.Tensor, 
        precomputed_memory: torch.Tensor, 
        context_vector: torch.Tensor, 
        s: torch.Tensor
    ) -> torch.Tensor:
        """
        noisy_target: (B, 21, 128) - White noise (or partially solved data) for Atom 21
        precomputed_memory: (B, N_past * 21, d_model) - The pre-encoded, pos-encoded past atoms
        context_vector: (B, 1024) - The CLAP timbre embedding
        s: (B, 1) - The current ODE time step in [0, 1]
        """
        B, T_new, _ = noisy_target.shape
        
        # 1. Process Global Conditioning
        t_emb = self.time_mlp(s)                  
        c_emb = self.context_mlp(context_vector)  
        cond = t_emb + c_emb                      
        
        # 2. Process Noisy Target
        x = self.target_proj(noisy_target) 
        
        x_fake_4d = x.unsqueeze(1)         
        x_fake_4d = self.target_rope.apply_rotary(x_fake_4d)
        x = x_fake_4d.squeeze(1)           
        
        # 3. Flow through Transformer
        for layer in self.layers:
            x = layer(x, memory=precomputed_memory, cond=cond)
            
        # 4. Output Velocity Vector
        x = self.final_norm(x, cond)
        velocity_field = self.output_proj(x)
        
        return velocity_field

class FlowModel(nn.Module):
    """
    Wrapper for the Flow Matching Generator.
    Expects the past atoms to ALREADY be encoded into d_model dimension by an external LocalEncoder.
    """
    def __init__(
        self, 
        frame_dim: int = 128,                  
        context_vector_dim: int = 1024,        
        num_past_atoms: int = 20,              
        frames_per_atom: int = 21,
        d_model: int = 256,                    
        nhead: int = 8,                        
        num_layers: int = 6,                   
        dim_feedforward: int = 1024,           
    ):
        super().__init__()
        
        # 1. Positional Encoding (Applies Macro & Micro time to the already-encoded past)
        self.memory_pos_enc = MemoryPositionalEncoding(
            d_model=d_model, 
            n_atoms_max=num_past_atoms + 5, 
            max_atom_frames=frames_per_atom + 5
        )
        
        # 2. The Core Flow Transformer
        self.transformer = VectorField(
            frame_dim=frame_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            context_dim=context_vector_dim,
            max_atom_frames=frames_per_atom
        )

    def prepare_memory(self, encoded_past: torch.Tensor) -> torch.Tensor:
        """
        encoded_past: (B, N_past, T_frames, d_model)
        Returns flattened memory ready for cross-attention.
        """
        return self.memory_pos_enc(encoded_past)

    def forward(self, x_t: torch.Tensor, s: torch.Tensor, context_vector: torch.Tensor, encoded_past: torch.Tensor) -> torch.Tensor:
        """
        TRAINING PASS.
        x_t: The blended noisy data (B, T_frames, frame_dim)
        s: The time step in [0, 1] (B, 1)
        context_vector: CLAP embedding (B, context_dim)
        encoded_past: Output of your LocalEncoder (B, N_past, T_frames, d_model)
        """
        precomputed_memory = self.prepare_memory(encoded_past)
        return self.transformer(x_t, precomputed_memory, context_vector, s)

    def vector_field(self, x, s, context_dict):
        """Used by the ODE Solver during inference."""
        precomputed_memory = context_dict['memory']
        clap_context = context_dict['clap']
        return self.transformer(x, precomputed_memory, clap_context, s)
    
    @torch.no_grad()
    def generate(self, x0: torch.Tensor, encoded_past: torch.Tensor, clap_context: torch.Tensor, max_nfe: int = 16) -> torch.Tensor:
        """
        INFERENCE PASS.
        x0: The starting white noise tensor of shape (B, T_frames, frame_dim)
        encoded_past: The past audio memory of shape (B, N_past, T_frames, d_model)
        clap_context: The timbre target of shape (B, context_dim)
        """
        # Prepare context dictionary
        memory = self.prepare_memory(encoded_past)
        context_dict = {'memory': memory, 'clap': clap_context}
        
        # Run the ODE solver using the user-provided x0
        x1 = sample_with_ode_capped(
            u=self.vector_field, 
            x0=x0, 
            context=context_dict, 
            max_nfe=max_nfe, 
            method='euler', 
            step_size=1.0/max_nfe
        )
        return x1