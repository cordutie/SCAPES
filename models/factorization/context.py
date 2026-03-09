import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalEncoder(nn.Module):
    def __init__(self, latent_dim=128, frames_per_atom=21, cnn_hidden=256, 
                 transformer_dim=512, num_heads=8, num_layers=4, clap_dim=1024):
        super().__init__()
        
        # 1. Intra-Atom CNN (Local)
        # Input channels = latent_dim (128) + 1 (scale) = 129
        self.intra_cnn = nn.Sequential(
            nn.Conv1d(latent_dim + 1, cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.GELU(),
            nn.Conv1d(cnn_hidden, transformer_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(transformer_dim),
            nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1) # Squashes the 21 frames to 1
        
        # 2. Inter-Atom Transformer (Global)
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        # We assume a max of 50 atoms just to be safe (your sequences are 10 or 20)
        self.pos_embed = nn.Parameter(torch.randn(1, 50, transformer_dim)) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=num_heads, 
            dim_feedforward=transformer_dim * 4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. CLAP Projection Head
        self.projection = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, clap_dim)
        )

    def forward(self, latent, scale):
        """
        latent: [Batch, N, 128, 21]
        scale:  [Batch, N, 1]
        """
        B, N, C, T = latent.shape
        
        # --- 1. Scale Injection ---
        # Expand scale to match the time dimension: [Batch, N, 1, 21]
        scale_expanded = scale.unsqueeze(-1).expand(-1, -1, -1, T)
        
        # Concat along the channel dimension (C): [Batch, N, 129, 21]
        x = torch.cat([latent, scale_expanded], dim=2)
        
        # --- 2. Intra-Atom CNN ---
        # Flatten Batch and N to process all atoms independently: [B*N, 129, 21]
        x = x.view(B * N, C + 1, T)
        
        x = self.intra_cnn(x)           # -> [B*N, transformer_dim, 21]
        x = self.pool(x)                # -> [B*N, transformer_dim, 1]
        x = x.squeeze(-1)               # -> [B*N, transformer_dim]
        
        # Reshape back to sequence: [B, N, transformer_dim]
        x = x.view(B, N, -1)
        
        # --- 3. Inter-Atom Transformer ---
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # -> [B, 1, transformer_dim]
        x = torch.cat([cls_tokens, x], dim=1)         # -> [B, N+1, transformer_dim]
        
        # Add Positional Embedding (slice to match current sequence length N+1)
        x = x + self.pos_embed[:, :N+1, :]
        
        # Pass through Transformer
        x = self.transformer(x) # -> [B, N+1, transformer_dim]
        
        # Extract the CLS token's output state
        cls_output = x[:, 0, :] # -> [B, transformer_dim]
        
        # --- 4. Projection to CLAP space ---
        out = self.projection(cls_output) # -> [B, 1024]
        
        # Optional: CLAP embeddings are highly directional, normalizing helps cosine loss
        out = F.normalize(out, p=2, dim=-1)
        
        return out