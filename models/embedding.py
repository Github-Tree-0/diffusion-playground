import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (batch_size,)
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


class TimeEmbedding(nn.Module):
    """时间嵌入层，将时间步映射到高维向量"""
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.out_dim = out_dim
        
        self.pos_encoding = SinusoidalPositionalEncoding(time_dim)
        self.fc1 = nn.Linear(time_dim, out_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (batch_size,)
        Returns:
            embeddings: shape (batch_size, out_dim)
        """
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
