import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import ConditionalBatchNorm3d


class ResBlock3d(nn.Module):
    """3D残差块，含时间条件"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = ConditionalBatchNorm3d(out_channels, time_emb_dim)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = ConditionalBatchNorm3d(out_channels, time_emb_dim)
        self.act2 = nn.SiLU()

        # 时间嵌入投影
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        # 跳连接
        if in_channels != out_channels:
            self.skip_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, in_channels, frames, height, width)
            time_emb: shape (batch_size, time_emb_dim)
        Returns:
            output: shape (batch_size, out_channels, frames, height, width)
        """
        h = self.conv1(x)
        h = self.norm1(h, time_emb)
        h = self.act1(h)

        # 添加时间条件
        time_emb_proj = self.time_proj(time_emb)
        time_emb_proj = time_emb_proj.view(
            time_emb_proj.shape[0], time_emb_proj.shape[1], 1, 1, 1
        )
        h = h + time_emb_proj

        h = self.conv2(h)
        h = self.norm2(h, time_emb)
        h = self.act2(h)

        # 跳连接
        return h + self.skip_proj(x)


class Attention3d(nn.Module):
    """3D多头自注意力"""
    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv3d(channels, inner_dim, kernel_size=1)

        self.to_q = nn.Conv3d(inner_dim, inner_dim, kernel_size=1)
        self.to_k = nn.Conv3d(inner_dim, inner_dim, kernel_size=1)
        self.to_v = nn.Conv3d(inner_dim, inner_dim, kernel_size=1)

        self.proj_out = nn.Conv3d(inner_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, channels, frames, height, width)
        Returns:
            output: shape (batch_size, channels, frames, height, width)
        """
        batch, channels, frames, height, width = x.shape

        # 归一化和投影
        h = self.norm(x)
        h = self.proj_in(h)

        # 计算Q, K, V
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)

        # 重塑用于多头注意力
        q = q.view(batch, self.num_heads, self.head_dim, frames * height * width)
        k = k.view(batch, self.num_heads, self.head_dim, frames * height * width)
        v = v.view(batch, self.num_heads, self.head_dim, frames * height * width)

        # 计算注意力权重
        dots = torch.einsum("bhdi,bhdj->bhij", q, k) * (self.head_dim ** -0.5)
        weights = torch.softmax(dots, dim=-1)

        # 应用注意力
        out = torch.einsum("bhij,bhdj->bhdi", weights, v)

        # 重塑回原始形状
        out = out.view(batch, self.num_heads * self.head_dim, frames, height, width)
        out = self.proj_out(out)

        return out + x


class DownBlock3d(nn.Module):
    """下采样块"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.use_attention = use_attention

        for i in range(num_res_blocks):
            channels = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResBlock3d(channels, out_channels, time_emb_dim)
            )
            if use_attention:
                self.attention_blocks.append(Attention3d(out_channels))

        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple:
        """
        Returns:
            output, skip_connections
        """
        skip = None
        for res_block, *att_block in zip(
            self.res_blocks,
            self.attention_blocks if self.use_attention else [None] * len(self.res_blocks),
        ):
            x = res_block(x, time_emb)
            if self.use_attention and att_block[0] is not None:
                x = att_block[0](x)
            skip = x  # 只保存最后一个res_block的输出作为skip

        x = self.downsample(x)
        return x, [skip] if skip is not None else []


class UpBlock3d(nn.Module):
    """上采样块"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.use_attention = use_attention

        for i in range(num_res_blocks):
            channels = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResBlock3d(channels, out_channels, time_emb_dim)
            )
            if use_attention:
                self.attention_blocks.append(Attention3d(out_channels))

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # 处理skip连接后的通道调整
        # upsample后: out_channels, skip来自down_block: in_channels
        # concat后: out_channels + in_channels，需要投影回in_channels以匹配第一个res_block的输入
        self.skip_proj = nn.Conv3d(out_channels + in_channels, in_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        skip: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.skip_proj(x)

        for res_block, *att_block in zip(
            self.res_blocks,
            self.attention_blocks if self.use_attention else [None] * len(self.res_blocks),
        ):
            x = res_block(x, time_emb)
            if self.use_attention and att_block[0] is not None:
                x = att_block[0](x)

        return x
