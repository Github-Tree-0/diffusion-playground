# DiffusionPlayground

Flow Matching diffusion framework for video generation and toy experiments, based on [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720).

## Framework

**Core formulation** (Flow Matching):
- Forward: `z_t = t * x_0 + (1 - t) * epsilon`, continuous `t in [0, 1]`
- Velocity: `v = x_0 - epsilon`
- 9 strategy combinations: 3 prediction types (epsilon, x, v) x 3 loss types

**Project structure:**

```
DiffusionPlayground/
├── models/
│   ├── diffusion.py          # FlowMatchingScheduler, VideoGenerationDDPM, ToyDiffusion
│   ├── embedding.py          # Time embeddings (sinusoidal, continuous, learned)
│   ├── unet3d.py             # 3D UNet for video (Conv3d, spatiotemporal attention)
│   ├── unet1d.py             # 1D UNet for toy vector data (Conv1d)
│   └── blocks3d.py           # 3D conv blocks and attention modules
├── toy_experiments/
│   ├── generate_data.py      # Generate checkerboard data embedded in D-dim space
│   ├── toy_dataset.py        # ToyDataset (PyTorch Dataset)
│   └── utils/
│       └── visualization.py  # 2D scatter plots, training curves, comparison charts
├── src/
│   └── video_dataset.py      # VideoDataset for frame sequences
├── configs/
│   ├── toy_exp_D64_*.json    # Toy experiment configs (epsilon/x/v prediction)
│   ├── toy_exp_template.json # Template with all options documented
│   └── config_*.json         # Video generation configs
├── train.py                  # Unified training script (toy + video), wandb logging
├── tests/                    # Unit tests
└── extract_all.sh            # Data extraction utility
```

**Key design decisions:**
- All training goes through `train.py` with config-driven dispatch (`experiment_type: "toy" | "video"`)
- `ToyDiffusion` and `VideoGenerationDDPM` share `FlowMatchingScheduler`, live in the same `diffusion.py`
- Toy data: 2D checkerboard (8x8) embedded in D-dim space via random orthogonal projection
- Periodic visualization during training: generate samples, project back to 2D, save scatter plots

## Quick Start

```bash
# Generate toy data
python -m toy_experiments.generate_data --embed-dims 2 8 16 64 512

# Train (toy experiment)
python train.py --config configs/toy_exp_D64_v.json

# Train (video generation)
python train.py --config configs/config_example.json
```

## Experiment Plan (TODO)

### Toy Experiments (checkerboard, D=64)
- [x] Infrastructure: data generation, dataset, 1D UNet, training loop, visualization
- [x] Checkerboard data generation (8x8 black/white pattern)
- [ ] Baseline runs: epsilon-prediction, x-prediction, v-prediction (compare convergence)
- [ ] Sweep across embedding dimensions D = {2, 8, 16, 64, 512}
- [ ] Compare loss landscapes across prediction types at high D
- [ ] Analyze sample quality (coverage, mode collapse) per strategy

### Video Generation
- [x] 3D UNet architecture with spatiotemporal attention
- [x] Video dataset pipeline (frame extraction, normalization)
- [ ] Train on real video data
- [ ] Evaluate prediction type impact on video quality (FVD, SSIM)
- [ ] Scale up model and resolution

### Framework Improvements
- [ ] Learning rate scheduler support
- [ ] EMA (Exponential Moving Average) for model weights
- [ ] Higher-order ODE solvers (Heun already implemented, add DPM-Solver++)
- [ ] Classifier-free guidance support