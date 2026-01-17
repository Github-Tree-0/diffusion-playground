import torch
import torch.nn as nn
from .embedding import TimeEmbedding
from .blocks3d import DownBlock3d, UpBlock3d, ResBlock3d, Attention3d


class UNet3d(nn.Module):
    """
    3D UNet用于视频生成。
    
    输入: (batch_size, in_channels, frames, height, width)
    输出: (batch_size, out_channels, frames, height, width)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 256,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        channel_multiples: tuple = (1, 2, 4, 8),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.channel_multiples = channel_multiples

        # 时间嵌入
        self.time_embedding = TimeEmbedding(time_emb_dim, time_emb_dim * 4)

        # 初始卷积
        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        # 下采样层
        self.down_blocks = nn.ModuleList()
        channels_in = base_channels
        for i, mult in enumerate(channel_multiples[:-1]):
            channels_out = base_channels * channel_multiples[i + 1]
            use_attention = base_channels * mult in [base_channels * m for m in attention_resolutions]
            
            self.down_blocks.append(
                DownBlock3d(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    time_emb_dim=time_emb_dim * 4,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attention,
                )
            )
            channels_in = channels_out

        # 中间块
        middle_channels = base_channels * channel_multiples[-1]
        self.middle_res_block1 = ResBlock3d(
            middle_channels, middle_channels, time_emb_dim * 4
        )
        self.middle_attention = Attention3d(middle_channels)
        self.middle_res_block2 = ResBlock3d(
            middle_channels, middle_channels, time_emb_dim * 4
        )

        # 上采样层
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_multiples[1:])):
            channels_out = base_channels * mult
            channels_in = base_channels * channel_multiples[len(channel_multiples) - 1 - i]
            use_attention = channels_out in [base_channels * m for m in attention_resolutions]

            self.up_blocks.append(
                UpBlock3d(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    time_emb_dim=time_emb_dim * 4,
                    num_res_blocks=num_res_blocks + 1,
                    use_attention=use_attention,
                )
            )

        # 输出层
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入视频 (batch_size, in_channels, frames, height, width)
            t: 时间步 (batch_size,)
        
        Returns:
            output: (batch_size, out_channels, frames, height, width)
        """
        # 时间嵌入
        time_emb = self.time_embedding(t)

        # 初始卷积
        h = self.conv_in(x)

        # 下采样
        skips = []
        for down_block in self.down_blocks:
            h, block_skips = down_block(h, time_emb)
            skips.extend(block_skips)

        # 中间块
        h = self.middle_res_block1(h, time_emb)
        h = self.middle_attention(h)
        h = self.middle_res_block2(h, time_emb)

        # 上采样
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, time_emb, skip)

        # 输出
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        return h
