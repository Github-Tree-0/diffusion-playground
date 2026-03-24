"""
Diffusion Model with Multiple Prediction Strategies

Based on "Back to Basics: Let Denoising Generative Models Denoise" (arXiv:2511.13720)

Uses continuous time t ∈ [0, 1] with flow matching formulation:
    z_t = t * x_0 + (1-t) * ε

where:
    - t=0: pure noise (z_0 = ε)
    - t=1: clean data (z_1 = x_0)

Velocity is defined as:
    v = dz_t/dt = x_0 - ε

All time values are continuous float t ∈ [0, 1].
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Literal, Optional, Tuple
from .unet3d import UNet3d
from .unet1d import UNet1d
from .embedding import TimeEmbeddingType


PredictionType = Literal["epsilon", "x", "v"]
LossType = Literal["epsilon", "x", "v"]
TimeSampler = Literal["uniform", "logit_normal", "beta"]


def expand_t(t: torch.Tensor, ndim: int = 5) -> torch.Tensor:
    """
    Expand t from (batch_size,) to (batch_size, 1, 1, ...) for broadcasting.

    Args:
        t: Tensor of shape (batch_size,)
        ndim: Target number of dimensions (default 5 for video: B,C,T,H,W)

    Returns:
        Expanded tensor of shape (batch_size, 1, 1, ...)
    """
    return t.view(-1, *([1] * (ndim - 1)))


class FlowMatchingScheduler:
    """
    Flow Matching Scheduler with continuous time t ∈ [0, 1].

    Forward process (interpolation):
        z_t = t * x_0 + (1-t) * ε

    This gives us:
        - t=0: z_0 = ε (pure noise)
        - t=1: z_1 = x_0 (clean data)

    Velocity (flow):
        v = dz_t/dt = x_0 - ε
    """

    def __init__(
        self,
        prediction_type: PredictionType = "epsilon",
        time_sampler: TimeSampler = "logit_normal",
        time_sampler_params: Optional[dict] = None,
    ):
        self.prediction_type = prediction_type
        self.time_sampler = time_sampler

        default_params = {
            "logit_normal": {"mu": -0.8, "sigma": 0.8},
            "beta": {"alpha": 1.0, "beta": 1.0},
            "uniform": {},
        }
        self.time_sampler_params = time_sampler_params or default_params.get(time_sampler, {})

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample continuous timesteps t ∈ [0, 1].

        Returns:
            t: Tensor of shape (batch_size,) with float values in [0, 1]
        """
        if self.time_sampler == "uniform":
            t = torch.rand(batch_size, device=device)

        elif self.time_sampler == "logit_normal":
            mu = self.time_sampler_params.get("mu", -0.8)
            sigma = self.time_sampler_params.get("sigma", 0.8)
            u = torch.randn(batch_size, device=device) * sigma + mu
            t = torch.sigmoid(u)

        elif self.time_sampler == "beta":
            alpha = self.time_sampler_params.get("alpha", 1.0)
            beta = self.time_sampler_params.get("beta", 1.0)
            t_np = np.random.beta(alpha, beta, size=batch_size)
            t = torch.from_numpy(t_np).float().to(device)

        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")

        t = t.clamp(1e-5, 1 - 1e-5)
        return t

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between noise and data (forward process).

        z_t = t * x_0 + (1-t) * ε

        Args:
            x_0: Clean data (batch_size, C, T, H, W)
            t: Continuous timesteps in [0, 1], shape (batch_size,)
            noise: Optional pre-generated noise

        Returns:
            z_t: Noisy data
            noise: The noise used
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        t_exp = expand_t(t, x_0.ndim)
        z_t = t_exp * x_0 + (1 - t_exp) * noise
        return z_t, noise

    def compute_velocity(self, x_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity (flow direction).

        v = dz_t/dt = x_0 - ε
        """
        return x_0 - noise

    def predict_x0_from_epsilon(
        self,
        z_t: torch.Tensor,
        epsilon: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        x_0 = (z_t - (1-t) * ε) / t
        """
        t_exp = expand_t(t, z_t.ndim)
        return (z_t - (1 - t_exp) * epsilon) / t_exp.clamp(min=1e-5)

    def predict_epsilon_from_x0(
        self,
        z_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        ε = (z_t - t * x_0) / (1-t)
        """
        t_exp = expand_t(t, z_t.ndim)
        return (z_t - t_exp * x_0) / (1 - t_exp).clamp(min=1e-5)

    def predict_x0_from_v(
        self,
        z_t: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        x_0 = z_t + (1-t) * v
        """
        t_exp = expand_t(t, z_t.ndim)
        return z_t + (1 - t_exp) * v

    def predict_epsilon_from_v(
        self,
        z_t: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        ε = z_t - t * v
        """
        t_exp = expand_t(t, z_t.ndim)
        return z_t - t_exp * v

    def get_all_predictions(
        self,
        z_t: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        prediction_type: Optional[PredictionType] = None
    ) -> dict:
        """
        Convert model output to all prediction types (x, epsilon, v).
        """
        pred_type = prediction_type or self.prediction_type

        if pred_type == "epsilon":
            epsilon = model_output
            x_0 = self.predict_x0_from_epsilon(z_t, epsilon, t)
            v = self.compute_velocity(x_0, epsilon)

        elif pred_type == "x":
            x_0 = model_output
            epsilon = self.predict_epsilon_from_x0(z_t, x_0, t)
            v = self.compute_velocity(x_0, epsilon)

        elif pred_type == "v":
            v = model_output
            x_0 = self.predict_x0_from_v(z_t, v, t)
            epsilon = self.predict_epsilon_from_v(z_t, v, t)

        else:
            raise ValueError(f"Unknown prediction type: {pred_type}")

        return {"x": x_0, "epsilon": epsilon, "v": v}

    def euler_step(
        self,
        z_t: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Euler step for ODE sampling.

        z_{t_next} = z_t + (t_next - t) * v

        Args:
            z_t: Current state
            v: Velocity at current state
            t: Current timesteps (batch_size,)
            t_next: Next timesteps (batch_size,)

        Returns:
            z_{t_next}: State at next timestep
        """
        t_exp = expand_t(t, z_t.ndim)
        t_next_exp = expand_t(t_next, z_t.ndim)
        dt = t_next_exp - t_exp
        return z_t + dt * v


class VideoGenerationDDPM(nn.Module):
    """
    Video Generation model using Flow Matching.

    All time values are continuous float t ∈ [0, 1].
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_timesteps: int = 1000,
        prediction_type: PredictionType = "epsilon",
        loss_type: LossType = "epsilon",
        time_sampler: TimeSampler = "logit_normal",
        time_sampler_params: Optional[dict] = None,
        time_embedding_type: TimeEmbeddingType = "continuous",
        beta_schedule: str = "linear",  # Backward compatibility
        **unet_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.time_embedding_type = time_embedding_type

        self.unet = UNet3d(
            in_channels=in_channels,
            out_channels=out_channels,
            time_embedding_type=time_embedding_type,
            **unet_kwargs,
        )

        self.scheduler = FlowMatchingScheduler(
            prediction_type=prediction_type,
            time_sampler=time_sampler,
            time_sampler_params=time_sampler_params,
        )

    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x_0: Clean video (batch_size, C, T, H, W)
            t: Continuous timesteps in [0, 1], shape (batch_size,)

        Returns:
            Model output
        """
        z_t, _ = self.scheduler.add_noise(x_0, t)
        return self.unet(z_t, t)

    def loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss.
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample t ∈ [0, 1]
        t = self.scheduler.sample_timesteps(batch_size, device)

        # Forward: z_t = t * x_0 + (1-t) * ε
        z_t, noise = self.scheduler.add_noise(x_0, t)

        # Model prediction
        model_output = self.unet(z_t, t)

        # Target
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x":
            target = x_0
        elif self.prediction_type == "v":
            target = self.scheduler.compute_velocity(x_0, noise)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Loss
        if self.loss_type == self.prediction_type:
            return nn.functional.mse_loss(model_output, target)

        # Convert to loss space
        predictions = self.scheduler.get_all_predictions(z_t, model_output, t)

        if self.loss_type == "epsilon":
            pred = predictions["epsilon"]
            target = noise
        elif self.loss_type == "x":
            pred = predictions["x"]
            target = x_0
        elif self.loss_type == "v":
            pred = predictions["v"]
            target = self.scheduler.compute_velocity(x_0, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return nn.functional.mse_loss(pred, target)

    def _make_timesteps(
        self,
        t_start: float,
        t_end: float,
        num_steps: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create timestep schedule.

        Returns:
            timesteps: (num_steps + 1, batch_size) tensor
        """
        t_vals = torch.linspace(t_start, t_end, num_steps + 1, device=device)
        return t_vals.unsqueeze(1).expand(-1, batch_size)  # (num_steps+1, batch_size)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: Optional[int] = None,
        progress_bar: bool = False,
    ) -> torch.Tensor:
        """
        Generate videos using Euler ODE solver.

        Sampling: t=0 (noise) → t=1 (data)
        """
        device = next(self.parameters()).device
        batch_size = shape[0]

        if num_steps is None:
            num_steps = 50

        z_t = torch.randn(shape, device=device)
        t_start = 1.0 / num_steps
        timesteps = self._make_timesteps(t_start, 1.0, num_steps, batch_size, device)

        iterator = range(num_steps)
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=num_steps)
            except ImportError:
                pass

        for i in iterator:
            t = timesteps[i]          # (batch_size,)
            t_next = timesteps[i + 1]  # (batch_size,)

            model_output = self.unet(z_t, t)
            predictions = self.scheduler.get_all_predictions(z_t, model_output, t)
            v = predictions["v"]

            z_t = self.scheduler.euler_step(z_t, v, t, t_next)

        return z_t

    @torch.no_grad()
    def sample_heun(
        self,
        shape: tuple,
        num_steps: Optional[int] = None,
        progress_bar: bool = False,
    ) -> torch.Tensor:
        """
        Generate videos using Heun's method (2nd order ODE solver).
        """
        device = next(self.parameters()).device
        batch_size = shape[0]

        if num_steps is None:
            num_steps = 50

        z_t = torch.randn(shape, device=device)
        t_start = 1.0 / num_steps
        timesteps = self._make_timesteps(t_start, 1.0, num_steps, batch_size, device)

        iterator = range(num_steps)
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=num_steps)
            except ImportError:
                pass

        for i in iterator:
            t = timesteps[i]          # (batch_size,)
            t_next = timesteps[i + 1]  # (batch_size,)

            # First evaluation at t
            model_output_1 = self.unet(z_t, t)
            predictions_1 = self.scheduler.get_all_predictions(z_t, model_output_1, t)
            v_1 = predictions_1["v"]

            # Euler step to intermediate point
            z_intermediate = self.scheduler.euler_step(z_t, v_1, t, t_next)

            # Second evaluation at t_next
            model_output_2 = self.unet(z_intermediate, t_next)
            predictions_2 = self.scheduler.get_all_predictions(z_intermediate, model_output_2, t_next)
            v_2 = predictions_2["v"]

            # Heun: average velocity
            v_avg = (v_1 + v_2) / 2
            z_t = self.scheduler.euler_step(z_t, v_avg, t, t_next)

        return z_t

    @torch.no_grad()
    def interpolate(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        num_steps: int = 50,
        num_interp_frames: int = 10,
    ) -> torch.Tensor:
        """
        Interpolate between two videos.
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        results = []
        for alpha in np.linspace(0, 1, num_interp_frames):
            x_interp = (1 - alpha) * x_start + alpha * x_end

            # Add noise (go to t=0.5)
            t_start_val = torch.full((batch_size,), 0.5, device=device)
            z_t, _ = self.scheduler.add_noise(x_interp, t_start_val)

            # Denoise: t=0.5 → t=1.0
            timesteps = self._make_timesteps(0.5, 1.0, num_steps, batch_size, device)
            for i in range(num_steps):
                t = timesteps[i]
                t_next = timesteps[i + 1]

                model_output = self.unet(z_t, t)
                predictions = self.scheduler.get_all_predictions(z_t, model_output, t)
                v = predictions["v"]

                z_t = self.scheduler.euler_step(z_t, v, t, t_next)

            results.append(z_t)

        return torch.cat(results, dim=2)

    def get_strategy_info(self) -> str:
        """Return strategy description."""
        return (
            f"prediction_type={self.prediction_type}, "
            f"loss_type={self.loss_type}, "
            f"time_sampler={self.scheduler.time_sampler}, "
            f"time_embedding={self.time_embedding_type}"
        )


class ToyDiffusion(nn.Module):
    """
    Toy Diffusion model using Flow Matching with 1D UNet.

    For D-dimensional vector data embedded from d-dimensional intrinsic space.
    Input shape: (batch_size, D) -> internally (batch_size, 1, D)

    All time values are continuous float t ∈ [0, 1].
    """

    def __init__(
        self,
        input_dim: int = 64,
        prediction_type: PredictionType = "epsilon",
        loss_type: LossType = "epsilon",
        time_sampler: TimeSampler = "logit_normal",
        time_sampler_params: Optional[dict] = None,
        time_embedding_type: TimeEmbeddingType = "continuous",
        base_channels: int = 64,
        time_emb_dim: int = 128,
        num_res_blocks: int = 2,
        channel_multiples: tuple = (1, 2, 4),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.time_embedding_type = time_embedding_type

        self.unet = UNet1d(
            in_channels=1,
            out_channels=1,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
            num_res_blocks=num_res_blocks,
            channel_multiples=channel_multiples,
            time_embedding_type=time_embedding_type,
        )

        self.scheduler = FlowMatchingScheduler(
            prediction_type=prediction_type,
            time_sampler=time_sampler,
            time_sampler_params=time_sampler_params,
        )

    def _to_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (batch, D) to (batch, 1, D) for 1D conv."""
        return x.unsqueeze(1)

    def _to_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (batch, 1, D) back to (batch, D)."""
        return x.squeeze(1)

    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x_0: Clean data (batch_size, D)
            t: Continuous timesteps in [0, 1], shape (batch_size,)

        Returns:
            Model output (batch_size, D)
        """
        x_0_3d = self._to_3d(x_0)
        z_t, _ = self.scheduler.add_noise(x_0_3d, t)
        output = self.unet(z_t, t)
        return self._to_2d(output)

    def loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            x_0: Clean data (batch_size, D)
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Convert to 3D for UNet
        x_0_3d = self._to_3d(x_0)

        # Sample t ∈ [0, 1]
        t = self.scheduler.sample_timesteps(batch_size, device)

        # Forward: z_t = t * x_0 + (1-t) * ε
        z_t, noise = self.scheduler.add_noise(x_0_3d, t)

        # Model prediction
        model_output = self.unet(z_t, t)

        # Target
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x":
            target = x_0_3d
        elif self.prediction_type == "v":
            target = self.scheduler.compute_velocity(x_0_3d, noise)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Loss
        if self.loss_type == self.prediction_type:
            return nn.functional.mse_loss(model_output, target)

        # Convert to loss space
        predictions = self.scheduler.get_all_predictions(z_t, model_output, t)

        if self.loss_type == "epsilon":
            pred = predictions["epsilon"]
            target = noise
        elif self.loss_type == "x":
            pred = predictions["x"]
            target = x_0_3d
        elif self.loss_type == "v":
            pred = predictions["v"]
            target = self.scheduler.compute_velocity(x_0_3d, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return nn.functional.mse_loss(pred, target)

    def _make_timesteps(
        self,
        t_start: float,
        t_end: float,
        num_steps: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create timestep schedule."""
        t_vals = torch.linspace(t_start, t_end, num_steps + 1, device=device)
        return t_vals.unsqueeze(1).expand(-1, batch_size)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        progress_bar: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using Euler ODE solver.

        Sampling: t=0 (noise) → t=1 (data)

        Args:
            num_samples: Number of samples to generate
            num_steps: Number of integration steps
            device: Device for generation
            progress_bar: Show progress bar

        Returns:
            Generated samples (num_samples, D)
        """
        if device is None:
            device = next(self.parameters()).device

        if num_steps is None:
            num_steps = 100

        # Start from pure noise (batch, 1, D)
        z_t = torch.randn(num_samples, 1, self.input_dim, device=device)
        # Start from 1/num_steps to avoid t=0 which causes division-by-zero
        # when converting epsilon/v predictions to x0 (x0 = (z_t - (1-t)*eps) / t)
        t_start = 1.0 / num_steps
        timesteps = self._make_timesteps(t_start, 1.0, num_steps, num_samples, device)

        iterator = range(num_steps)
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=num_steps)
            except ImportError:
                pass

        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]

            model_output = self.unet(z_t, t)
            predictions = self.scheduler.get_all_predictions(z_t, model_output, t)
            v = predictions["v"]

            z_t = self.scheduler.euler_step(z_t, v, t, t_next)

        return self._to_2d(z_t)

    @torch.no_grad()
    def sample_heun(
        self,
        num_samples: int,
        num_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        progress_bar: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using Heun's method (2nd order ODE solver).
        """
        if device is None:
            device = next(self.parameters()).device

        if num_steps is None:
            num_steps = 100

        z_t = torch.randn(num_samples, 1, self.input_dim, device=device)
        t_start = 1.0 / num_steps
        timesteps = self._make_timesteps(t_start, 1.0, num_steps, num_samples, device)

        iterator = range(num_steps)
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=num_steps)
            except ImportError:
                pass

        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # First evaluation
            model_output_1 = self.unet(z_t, t)
            predictions_1 = self.scheduler.get_all_predictions(z_t, model_output_1, t)
            v_1 = predictions_1["v"]

            # Euler step to intermediate
            z_intermediate = self.scheduler.euler_step(z_t, v_1, t, t_next)

            # Second evaluation
            model_output_2 = self.unet(z_intermediate, t_next)
            predictions_2 = self.scheduler.get_all_predictions(z_intermediate, model_output_2, t_next)
            v_2 = predictions_2["v"]

            # Heun: average velocity
            v_avg = (v_1 + v_2) / 2
            z_t = self.scheduler.euler_step(z_t, v_avg, t, t_next)

        return self._to_2d(z_t)

    def get_strategy_info(self) -> str:
        """Return strategy description."""
        return (
            f"prediction_type={self.prediction_type}, "
            f"loss_type={self.loss_type}, "
            f"time_sampler={self.scheduler.time_sampler}, "
            f"time_embedding={self.time_embedding_type}"
        )


# Backward compatibility
DDPMScheduler = FlowMatchingScheduler
