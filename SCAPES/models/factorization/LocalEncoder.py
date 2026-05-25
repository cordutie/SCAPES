from typing import Optional

import torch
import torch.nn as nn
import json

from SCAPES.data.config_loader import TrainingConfig


def load_local_encoder(checkpoint_path, json_path, device="cpu"):
    with open(json_path, 'r') as f:
        config = json.load(f)

    model = LocalEncoder(
        in_channels=config.get("in_channels", 129),
        hidden_dim=config.get("hidden_dim", 256),
        out_channels=config.get("out_channels", 256),
        time_entanglement=config.get("time_entanglement", True),
        temporal_compression=config.get("temporal_compression", 1)
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model.to(device)


class ConvLNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return self.act(x)


class LocalEncoder(nn.Module):
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        *,
        in_channels: int = 128,
        hidden_dim: int = 256,
        out_channels: int = 256,
        time_entanglement: bool = True,
        temporal_compression: int = 1
    ):
        if config is not None:
            hidden_dim = config.local_encoder_hidden_dim
            out_channels = config.d_model
            time_entanglement = config.local_encoder_time_entanglement
            temporal_compression = config.local_encoder_temporal_compression

        super().__init__()

        if not time_entanglement:
            k1, s1, p1 = 1, 1, 0
            k2, p2 = 1, 0
            k3, p3 = 1, 0
            if temporal_compression > 1:
                print("Warning: time_entanglement=False but compression > 1. This will just blindly drop frames.")
                s1 = temporal_compression
        else:
            if temporal_compression == 3:
                k1, s1, p1 = 9, 3, 4
            else:
                k1, s1, p1 = 5, temporal_compression, 2
            k2, p2 = 5, 2
            k3, p3 = 3, 1

        self.net = nn.Sequential(
            ConvLNBlock(in_channels, hidden_dim, kernel_size=k1, stride=s1, padding=p1),
            ConvLNBlock(hidden_dim, hidden_dim, kernel_size=k2, stride=1, padding=p2),
            ConvLNBlock(hidden_dim, out_channels, kernel_size=k3, stride=1, padding=p3),
        )

    def forward(self, atoms):
        B, N, C, T = atoms.shape
        x = atoms.reshape(B * N, C, T)
        x = self.net(x)
        _, C_out, T_new = x.shape
        x = x.view(B, N, C_out, T_new)
        x = x.transpose(2, 3)
        return x
