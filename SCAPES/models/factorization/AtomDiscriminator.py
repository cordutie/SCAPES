import torch
import torch.nn as nn


class AtomDiscriminator(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(128, hidden_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.head(x)
        return x
