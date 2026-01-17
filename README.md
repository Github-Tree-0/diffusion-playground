# Video Generation Model with DDPM

A simple yet comprehensive video generation model using Denoising Diffusion Probabilistic Models (DDPM) with 3D UNet architecture.

## Overview

This project implements a complete video generation pipeline:

- **Model Architecture**: 3D UNet with skip connections and attention mechanisms
- **Diffusion Process**: DDPM-based denoising for generation
- **Features**:
  - Multi-head self-attention in 3D
  - Conditional batch normalization
  - Time-dependent positional encoding
  - Support for video interpolation

## Project Structure

```
DiffusionPlayground/
├── models/
│   ├── __init__.py
│   ├── embedding.py         # Time embedding and position encoding
│   ├── blocks3d.py          # 3D convolutional blocks and attention
│   ├── unet3d.py            # 3D UNet architecture
│   └── diffusion.py         # DDPM scheduler and training loop
├── train_video.py           # Training script
├── generate_video.py        # Inference and video generation
├── config.py                # Configuration file
└── README.md
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy tqdm
pip install imageio imageio-ffmpeg  # Optional: for video saving
```

## Usage

### Training

Basic training with dummy data:

```bash
python train_video.py
```

The script will:
1. Create a dummy video dataset
2. Initialize the model
3. Train for the specified number of epochs
4. Save checkpoints to `checkpoints/` directory
5. Generate sample videos during training

Customize training by modifying `config.py`:
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimizer
- `DATASET_CONFIG`: Video dimensions and normalization

### Generation

Generate videos using a trained model:

```bash
python generate_video.py
```

This will:
1. Load the best trained model checkpoint
2. Generate 2 sample videos
3. Save them as MP4 files (if imageio is installed)

### Advanced Usage

#### Custom Dataset

Replace `DummyVideoDataset` in `train_video.py` with your own dataset:

```python
class CustomVideoDataset(Dataset):
    def __init__(self, video_dir):
        self.video_paths = list(Path(video_dir).glob("*.mp4"))
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess video
        video = load_video(self.video_paths[idx])  # (C, T, H, W)
        return normalize_to_range(video, -1, 1)
```

#### Video Interpolation

Interpolate between two videos:

```python
from generate_video import VideoGenerator

generator = VideoGenerator("checkpoints/best_model.pt")

video1 = torch.randn(1, 3, 8, 64, 64)
video2 = torch.randn(1, 3, 8, 64, 64)

interpolated = generator.interpolate(
    video1, 
    video2, 
    num_steps=50,
    num_interp_frames=20
)
```

#### Controlling Generation

Control sampling quality and speed:

```python
# Faster generation (fewer steps, lower quality)
videos = generator.generate(
    num_samples=4,
    num_frames=8,
    height=64,
    width=64,
    num_steps=20  # Fewer steps = faster
)

# Higher quality generation (more steps, slower)
videos = generator.generate(
    num_samples=4,
    num_frames=8,
    height=64,
    width=64,
    num_steps=100  # More steps = higher quality but slower
)
```

## Model Details

### Architecture

The 3D UNet consists of:

1. **Time Embedding**: Sinusoidal positional encoding → MLP
2. **Encoder (Down blocks)**: Progressive downsampling with residual connections
3. **Middle blocks**: Deepest features with self-attention
4. **Decoder (Up blocks)**: Progressive upsampling with skip connections
5. **Attention layers**: Multi-head self-attention for capturing temporal and spatial dependencies

### Diffusion Process

- **Forward process**: Gradually add Gaussian noise to clean videos
- **Reverse process**: Learn to denoise videos back to clean samples
- **Training objective**: Predict added noise given a noisy video and timestep

## Configuration

Edit `config.py` to customize:

- **Model capacity**: `channel_multiples`, `base_channels`
- **Training**: `learning_rate`, `batch_size`, `num_epochs`
- **Video quality**: `DATASET_CONFIG` height/width
- **Inference speed**: `SAMPLING_CONFIG` num_steps

## Results

The model learns to:
- Generate realistic video sequences
- Interpolate smoothly between videos
- Capture temporal dynamics through 3D convolutions
- Use attention to focus on important spatial-temporal regions

## Performance Tips

1. **For faster training on CPU/small GPU**:
   - Reduce `base_channels` to 16
   - Use `channel_multiples=(1, 2)` instead of `(1, 2, 4, 8)`
   - Reduce `batch_size`

2. **For better quality**:
   - Increase `base_channels` to 64
   - Use `channel_multiples=(1, 2, 4, 8)`
   - Increase training duration
   - Collect more diverse training data

3. **For memory efficiency**:
   - Reduce video resolution (height, width)
   - Reduce `num_frames`
   - Use gradient accumulation

## Troubleshooting

**Out of Memory (OOM)**:
- Reduce batch size
- Reduce video resolution
- Reduce `base_channels`

**Training is slow**:
- Ensure CUDA is being used (check with `torch.cuda.is_available()`)
- Reduce model complexity
- Increase batch size (if memory allows)

**Generated videos are noisy**:
- Train for more epochs
- Use more denoising steps during inference
- Check that training loss is decreasing

## References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- [Video Diffusion Models](https://arxiv.org/abs/2204.03461)

## License

MIT License

## Author

Created as a video generation playground for exploring diffusion models.
