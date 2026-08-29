from __future__ import annotations

import torch
import torch.nn as nn


class SpatialDuelingQ(nn.Module):
    def __init__(self, obs_dim: int, board_size: int, tile_channels: int, global_dim: int,
                 unit_dim: int, unit_id_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.board_size = board_size
        self.tile_channels = tile_channels
        self.global_dim = global_dim
        self.unit_dim = unit_dim
        self.unit_id_dim = unit_id_dim
        spatial_dim = tile_channels * board_size * board_size
        expected = global_dim + spatial_dim + unit_dim + unit_id_dim
        if obs_dim != expected:
            raise ValueError(f"obs_dim={obs_dim}, expected={expected}")

        self.conv = nn.Sequential(
            nn.Conv2d(tile_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        conv_out = 64 * 4
        vec_dim = global_dim + unit_dim + unit_id_dim
        self.trunk = nn.Sequential(
            nn.Linear(conv_out + vec_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        s = self.tile_channels * self.board_size * self.board_size
        g = x[:, : self.global_dim]
        tile = x[:, self.global_dim : self.global_dim + s].reshape(-1, self.tile_channels, self.board_size, self.board_size)
        u = x[:, self.global_dim + s :]
        z = self.conv(tile).flatten(1)
        z = self.trunk(torch.cat([g, u, z], dim=1))
        a = self.advantage(z)
        q = self.value(z) + a - a.mean(dim=1, keepdim=True)
        if mask is not None:
            q = q.masked_fill(~mask.bool(), -1e9)
        return q
