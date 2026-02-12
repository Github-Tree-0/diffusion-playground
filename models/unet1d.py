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

        time_emb = self.time_mlp(time_emb)
        h = h + time_emb.unsqueeze(-1)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        return self.conv(x)


class UNet1d(nn.Module):
    """
    1D UNet for toy diffusion experiments.

    Input: (batch_size, in_channels, length)
    Output: (batch_size, out_channels, length)

    Architecture follows the standard diffusion UNet pattern:
    - Down path: ResBlocks -> Downsample, saving skip connections
    - Middle: ResBlocks
    - Up path: Upsample -> concat skip -> ResBlocks
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

        emb_dim = time_emb_dim * 4

        # Time embedding
        self.time_embedding = TimeEmbedding(
            time_emb_dim,
            emb_dim,
            embedding_type=time_embedding_type,
        )

        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # Build channel list per level
        # e.g. channel_multiples=(1,2,4) -> channels = [64, 128, 256]
        channels = [base_channels * m for m in channel_multiples]
        num_levels = len(channel_multiples)

        # ============ Down path ============
        self.down_res_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch_in = base_channels
        for level in range(num_levels):
            ch_out = channels[level]
            # ResBlocks at this level
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                level_blocks.append(ResBlock1d(ch_in, ch_out, emb_dim))
                ch_in = ch_out
            self.down_res_blocks.append(level_blocks)

            # Downsample (except last level)
            if level < num_levels - 1:
                self.downsamples.append(Downsample1d(ch_out))
            else:
                self.downsamples.append(nn.Identity())

        # ============ Middle ============
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock1d(mid_ch, mid_ch, emb_dim)
        self.mid_block2 = ResBlock1d(mid_ch, mid_ch, emb_dim)

        # ============ Up path ============
        self.up_res_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(num_levels)):
            ch_out = channels[level]

            # Upsample (except first = deepest level)
            if level < num_levels - 1:
                self.upsamples.append(Upsample1d(ch_in))
            else:
                self.upsamples.append(nn.Identity())

            # ResBlocks: each one takes concat(h, skip) as input
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                # First block at this level concatenates with skip
                block_in = ch_in + ch_out  # h_channels + skip_channels
                level_blocks.append(ResBlock1d(block_in, ch_out, emb_dim))
                ch_in = ch_out
            self.up_res_blocks.append(level_blocks)

        # Output
        num_groups_out = get_num_groups(channels[0])
        self.norm_out = nn.GroupNorm(num_groups_out, channels[0])
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv1d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch_size, in_channels, length)
            t: Timesteps (batch_size,)

        Returns:
            Output (batch_size, out_channels, length)
        """
        time_emb = self.time_embedding(t)
        h = self.conv_in(x)

        # Down path: save skip at each res block output
        skips = []
        for level, (level_blocks, downsample) in enumerate(
            zip(self.down_res_blocks, self.downsamples)
        ):
            for res_block in level_blocks:
                h = res_block(h, time_emb)
                skips.append(h)
            h = downsample(h)

        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_block2(h, time_emb)

        # Up path: pop skips in reverse
        for level_blocks, upsample in zip(self.up_res_blocks, self.upsamples):
            h = upsample(h)
            for res_block in level_blocks:
                skip = skips.pop()
                # Align spatial size
                if h.shape[-1] != skip.shape[-1]:
                    h = nn.functional.interpolate(
                        h, size=skip.shape[-1], mode='linear', align_corners=False
                    )
                h = torch.cat([h, skip], dim=1)
                h = res_block(h, time_emb)

        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        return h
