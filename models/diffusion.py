import torch
import torch.nn as nn
import numpy as np
from .unet3d import UNet3d


class DDPMScheduler:
    """DDPM采样时间表"""
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 计算beta
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "quadratic":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # 计算相关系数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # 计算方差相关的系数
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # 用于采样
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", 
                            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区"""
        setattr(self, name, tensor)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple:
        """
        向视频添加噪声 (前向过程)
        
        Args:
            x_0: 原始视频 (batch_size, channels, frames, height, width)
            t: 时间步 (batch_size,)
            noise: 高斯噪声，如果为None则随机生成
        
        Returns:
            x_t: 噪声版本 (batch_size, channels, frames, height, width)
            noise: 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 确保t在CPU上以便索引
        t_cpu = t.cpu() if t.is_cuda else t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_cpu]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_cpu]
        
        # 移到正确的设备
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(x_0.device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(x_0.device)

        # 重塑以匹配x_0的维度
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDPM反向采样步骤
        
        Args:
            x_t: 当前噪声版本
            t: 当前时间步
            predicted_noise: 模型预测的噪声
        
        Returns:
            x_{t-1}: 前一时间步的版本
        """
        posterior_mean_coef1 = self.posterior_mean_coef1[t]
        posterior_mean_coef2 = self.posterior_mean_coef2[t]
        posterior_variance = self.posterior_variance[t]

        # 重塑系数
        while len(posterior_mean_coef1.shape) < len(x_t.shape):
            posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
            posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)
            posterior_variance = posterior_variance.unsqueeze(-1)

        # 计算均值
        mean = posterior_mean_coef1 * x_t - posterior_mean_coef2 * predicted_noise

        # 采样噪声
        noise = torch.randn_like(x_t)
        x_t_minus_1 = mean + torch.sqrt(posterior_variance) * noise

        return x_t_minus_1


class VideoGenerationDDPM(nn.Module):
    """基于DDPM的视频生成模型"""
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_timesteps: int = 1000,
        **unet_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps

        # UNet模型
        self.unet = UNet3d(
            in_channels=in_channels,
            out_channels=out_channels,
            **unet_kwargs,
        )

        # 调度器
        self.scheduler = DDPMScheduler(num_timesteps=num_timesteps)

    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        训练时的前向传播
        
        Args:
            x_0: 原始视频 (batch_size, channels, frames, height, width)
            t: 时间步 (batch_size,)
        
        Returns:
            predicted_noise: 预测的噪声
        """
        # 添加噪声
        x_t, noise = self.scheduler.add_noise(x_0, t)

        # 预测噪声
        predicted_noise = self.unet(x_t, t)

        return predicted_noise

    def loss(self, x_0: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        计算训练损失
        
        Args:
            x_0: 原始视频
            noise: 可选的固定噪声
        
        Returns:
            loss: MSE损失
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # 前向传播
        predicted_noise = self.forward(x_0, t)

        # 实际噪声
        if noise is None:
            _, noise = self.scheduler.add_noise(x_0, t)

        # MSE损失
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: int = None,
        progress_bar: bool = False,
    ) -> torch.Tensor:
        """
        从模型生成视频
        
        Args:
            shape: 输出形状 (batch_size, channels, frames, height, width)
            num_steps: 去噪步数，默认为num_timesteps
            progress_bar: 是否显示进度条
        
        Returns:
            video: 生成的视频
        """
        device = next(self.parameters()).device
        
        if num_steps is None:
            num_steps = self.num_timesteps

        # 从高斯噪声开始
        x_t = torch.randn(shape, device=device)

        # 去噪循环
        timesteps = np.linspace(0, self.num_timesteps - 1, num_steps, dtype=int)
        
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(reversed(timesteps), total=len(timesteps))
            except ImportError:
                iterator = reversed(timesteps)
        else:
            iterator = reversed(timesteps)

        for step, t_step in enumerate(iterator):
            t = torch.tensor([t_step] * shape[0], device=device, dtype=torch.long)

            # 预测噪声
            predicted_noise = self.unet(x_t, t)

            # 去噪一步
            x_t = self.scheduler.denoise_step(x_t, t, predicted_noise)

        return x_t

    @torch.no_grad()
    def interpolate(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        num_steps: int = 50,
        num_interp_frames: int = 10,
    ) -> torch.Tensor:
        """
        在两个视频之间插值
        
        Args:
            x_start: 起始视频
            x_end: 结束视频
            num_steps: 去噪步数
            num_interp_frames: 插值帧数
        
        Returns:
            interpolated: 插值视频
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        results = []
        for alpha in np.linspace(0, 1, num_interp_frames):
            # 线性插值
            x_interp = (1 - alpha) * x_start + alpha * x_end

            # 加噪
            t = torch.full((batch_size,), self.num_timesteps // 2, device=device, dtype=torch.long)
            x_t, _ = self.scheduler.add_noise(x_interp, t)

            # 去噪
            timesteps = np.linspace(self.num_timesteps // 2, 0, num_steps, dtype=int)
            for t_step in reversed(timesteps):
                t = torch.tensor([t_step] * batch_size, device=device, dtype=torch.long)
                predicted_noise = self.unet(x_t, t)
                x_t = self.scheduler.denoise_step(x_t, t, predicted_noise)

            results.append(x_t)

        return torch.cat(results, dim=2)  # 沿时间维度连接
