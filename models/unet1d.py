"""
1D UNet for Toy Diffusion Experiments

Handles D-dimensional vector data by treating it as 1D sequence.
Input shape: (batch_size, 1, D) where D is the embedding dimension.
"""

import torch
import torch.nn as nn
from .embedding import TimeEmbedding, TimeEmbeddingType


def get_num_groups(channels: int, preferred: int = 8) -> int:
    """Get number of groups for GroupNorm."""
    for g in [preferred, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ResBlock1d(nn.Module):
    """1D Residual block with time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        num_groups_1 = get_num_groups(in_channels)
        num_groups_2 = get_num_groups(out_channels)

        self.norm1 = nn.GroupNorm(num_groups_1, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups_2, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb.unsqueeze(-1)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class DownBlock1d(nn.Module):
    """1D Downsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock1d(ch_in, out_channels, time_emb_dim))

        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        skips = []
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
            skips.append(x)

        x = self.downsample(x)
        return x, skips


class UpBlock1d(nn.Module):
    """1D Upsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            ch_in = in_channels + out_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock1d(ch_in, out_channels, time_emb_dim))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, skip: torch.Tensor = None):
        x = self.upsample(x)

        # Handle size mismatch
        if skip is not None:
            if x.shape[-1] != skip.shape[-1]:
                x = nn.functional.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        for res_block in self.res_blocks:
            x = res_block(x, time_emb)

        return x


class UNet1d(nn.Module):
    """
    1D UNet for toy diffusion experiments.

    Input: (batch_size, in_channels, length)
    Output: (batch_size, out_channels, length)

    For toy experiments:
    - Input is (batch_size, 1, D) where D is embedding dimension
    - Treats D as spatial dimension
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        num_res_blocks: int = 2,
        channel_multiples: tuple = (1, 2, 4),
        time_embedding_type: TimeEmbeddingType = "continuous",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        # Time embedding
        self.time_embedding = TimeEmbedding(
            time_emb_dim,
            time_emb_dim * 4,
            embedding_type=time_embedding_type,
        )

        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels_in = base_channels
        for i, mult in enumerate(channel_multiples[:-1]):
            channels_out = base_channels * channel_multiples[i + 1]
            self.down_blocks.append(
                DownBlock1d(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    time_emb_dim=time_emb_dim * 4,
                    num_res_blocks=num_res_blocks,
                )
            )
            channels_in = channels_out

        # Middle
        middle_channels = base_channels * channel_multiples[-1]
        self.middle_res1 = ResBlock1d(middle_channels, middle_channels, time_emb_dim * 4)
        self.middle_res2 = ResBlock1d(middle_channels, middle_channels, time_emb_dim * 4)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i in range(len(channel_multiples) - 1):
            idx = len(channel_multiples) - 2 - i
            channels_in = base_channels * channel_multiples[idx + 1]
            channels_out = base_channels * channel_multiples[idx]

            self.up_blocks.append(
                UpBlock1d(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    time_emb_dim=time_emb_dim * 4,
                    num_res_blocks=num_res_blocks + 1,
                )
            )

        # Output
        num_groups_out = get_num_groups(base_channels)
        self.norm_out = nn.GroupNorm(num_groups_out, base_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch_size, in_channels, length)
            t: Timesteps (batch_size,)

        Returns:
            Output (batch_size, out_channels, length)
        """
        # Time embedding
        time_emb = self.time_embedding(t)

        # Initial conv
        h = self.conv_in(x)

        # Downsampling
        skips = []
        for down_block in self.down_blocks:
            h, block_skips = down_block(h, time_emb)
            skips.extend(block_skips)

        # Middle
        h = self.middle_res1(h, time_emb)
        h = self.middle_res2(h, time_emb)

        # Upsampling
        for up_block in self.up_blocks:
            skip = skips.pop() if skips else None
            h = up_block(h, time_emb, skip)

        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        return h
