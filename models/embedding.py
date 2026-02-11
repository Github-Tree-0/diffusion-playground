import math
import torch
import torch.nn as nn
from typing import Literal


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码

    对于离散timestep t ∈ {0, 1, ..., T-1}
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (batch_size,), 离散timestep
        Returns:
            embeddings: shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FourierTimeEmbedding(nn.Module):
    """
    Fourier features for continuous time t ∈ [0, 1].

    Uses random Fourier features: embed(t) = [sin(2π * freq * t), cos(2π * freq * t)]
    where frequencies are sampled from N(0, scale^2).

    This is more suitable for continuous time than discrete sinusoidal encoding.
    """
    def __init__(self, dim: int, scale: float = 16.0, learnable: bool = False):
        super().__init__()
        self.dim = dim
        self.scale = scale
        half_dim = dim // 2

        # Random frequencies (can be learnable or fixed)
        freqs = torch.randn(half_dim) * scale
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (batch_size,), continuous time in [0, 1]
        Returns:
            embeddings: shape (batch_size, dim)
        """
        # t: (batch_size,) -> (batch_size, 1)
        t = t.unsqueeze(-1)
        # freqs: (half_dim,) -> (1, half_dim)
        freqs = self.freqs.unsqueeze(0)
        # (batch_size, half_dim)
        args = 2 * math.pi * t * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ContinuousTimeEmbedding(nn.Module):
    """
    Continuous time embedding using scaled sinusoidal encoding.

    For t ∈ [0, 1], we scale it to [0, max_period] to get meaningful frequencies.
    This is equivalent to the discrete version but handles float inputs properly.
    """
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (batch_size,), continuous time in [0, 1]
        Returns:
            embeddings: shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        # Create frequency bands
        # freqs go from 1 to max_period
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / (half_dim - 1)
        )

        # t: (batch_size,) -> (batch_size, 1)
        # freqs: (half_dim,) -> (1, half_dim)
        args = t[:, None] * freqs[None, :] * self.max_period
        return torch.cat([args.sin(), args.cos()], dim=-1)


TimeEmbeddingType = Literal["discrete", "continuous", "fourier"]


class TimeEmbedding(nn.Module):
    """
    时间嵌入层，将时间步映射到高维向量

    支持三种模式：
    - "discrete": 离散timestep (int)，使用标准sinusoidal encoding
    - "continuous": 连续时间 t ∈ [0,1] (float)，使用scaled sinusoidal
    - "fourier": 连续时间，使用random Fourier features
    """
    def __init__(
        self,
        time_dim: int,
        out_dim: int,
        embedding_type: TimeEmbeddingType = "continuous",
        fourier_scale: float = 16.0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.out_dim = out_dim
        self.embedding_type = embedding_type

        # 选择encoding类型
        if embedding_type == "discrete":
            self.pos_encoding = SinusoidalPositionalEncoding(time_dim)
        elif embedding_type == "continuous":
            self.pos_encoding = ContinuousTimeEmbedding(time_dim)
        elif embedding_type == "fourier":
            self.pos_encoding = FourierTimeEmbedding(time_dim, scale=fourier_scale)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        # MLP projection
        self.fc1 = nn.Linear(time_dim, out_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (batch_size,)
               - 如果是 "discrete": 整数 timestep
               - 如果是 "continuous" 或 "fourier": float in [0, 1]
        Returns:
            embeddings: shape (batch_size, out_dim)
        """
        # Ensure t is float for all embedding types
        t = t.float()
        embeddings = self.pos_encoding(t)
        embeddings = self.fc1(embeddings)
        embeddings = self.act(embeddings)
        embeddings = self.fc2(embeddings)
        return embeddings


class ConditionalBatchNorm3d(nn.Module):
    """条件批归一化 - 根据时间嵌入调整BN参数"""
    def __init__(self, num_features: int, time_emb_dim: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(num_features)
        self.time_fc_gamma = nn.Linear(time_emb_dim, num_features)
        self.time_fc_beta = nn.Linear(time_emb_dim, num_features)

        # 初始化为接近identity
        nn.init.zeros_(self.time_fc_gamma.weight)
        nn.init.ones_(self.time_fc_gamma.bias)
        nn.init.zeros_(self.time_fc_beta.weight)
        nn.init.zeros_(self.time_fc_beta.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, channels, frames, height, width)
            time_emb: shape (batch_size, time_emb_dim)
        Returns:
            output: shape (batch_size, channels, frames, height, width)
        """
        out = self.batch_norm(x)
        gamma = self.time_fc_gamma(time_emb)
        beta = self.time_fc_beta(time_emb)

        # 调整形状以匹配广播
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1, 1)

        return gamma * out + beta
