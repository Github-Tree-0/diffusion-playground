# Quick Start Guide

## Installation

```bash
pip install torch
```

## Training

```python
from models import VideoGenerationDDPM
import torch

# Create model
model = VideoGenerationDDPM(
    in_channels=3,
    out_channels=3,
    num_timesteps=1000,
    base_channels=64,
    time_emb_dim=256,
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dummy data
x = torch.randn(4, 3, 8, 32, 32).to(device)  # (batch, channels, frames, height, width)

# Loss
loss = model.loss(x)
loss.backward()
optimizer.step()
```

## Sampling

```python
# Generate new videos
videos = model.sample(shape=(2, 3, 8, 32, 32), num_steps=100)

# Interpolate between two videos
x_start = torch.randn(1, 3, 8, 32, 32)
x_end = torch.randn(1, 3, 8, 32, 32)
interpolated = model.interpolate(x_start, x_end, num_interp_frames=10)
```

## Key Components

- **embedding.py**: Time embeddings and sinusoidal positional encoding
- **blocks3d.py**: 3D residual blocks, attention layers, up/down sampling
- **unet3d.py**: Full 3D UNet architecture with skip connections
- **diffusion.py**: DDPM scheduler and VideoGenerationDDPM model

## Model Architecture

```
Input (B, 3, T, H, W)
    ↓
Initial Conv (B, 64, T, H, W)
    ↓
Downsampling (4 levels)
    ↓
Middle blocks + Attention
    ↓
Upsampling (4 levels) with skip connections
    ↓
Output Conv (B, 3, T, H, W)
```

## Training Tips

1. Start with small resolution (32x32) and short sequences (8 frames)
2. Use learning rate around 1e-4
3. Monitor loss - should decrease smoothly
4. Use GPU for efficient training

## Parameters

- `num_timesteps`: Number of diffusion steps (default 1000)
- `base_channels`: Base channel count (default 64)
- `time_emb_dim`: Time embedding dimension (default 256)
- `num_res_blocks`: Residual blocks per level (default 2)
- `channel_multiples`: Channel multipliers for different levels
