"""
训练脚本 - 支持视频生成模型和Toy实验

Usage:
    # Video generation
    python train.py --config configs/config_example.json

    # Toy experiment
    python train.py --config configs/toy_exp_D64_epsilon.json
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import io

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import VideoGenerationDDPM, ToyDiffusion
from video_dataset import VideoDataset, VideoDatasetConfig
from toy_experiments import ToyDataset, ToyDatasetConfig

# 尝试导入wandb（可选依赖）
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Run 'pip install wandb' to enable logging.")

# 导入matplotlib（用于toy实验可视化，使用Agg后端支持无显示器的服务器）
try:
    import matplotlib
    matplotlib.use('Agg')  # 必须在import pyplot之前设置
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """
    加载并验证配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 设置默认值以保证向后兼容
    config.setdefault('experiment_type', 'video')  # 'video' or 'toy'
    config.setdefault('model', {})
    config.setdefault('dataloader', {})
    config.setdefault('optimizer', {})
    config.setdefault('training', {})
    config.setdefault('wandb', {'enabled': False})
    config.setdefault('diffusion', {})
    config.setdefault('sampling', {})
    config['training'].setdefault('gradient_accumulation_steps', 1)
    config['training'].setdefault('max_grad_norm', 1.0)
    config['training'].setdefault('mixed_precision', {'enabled': False})
    config['training'].setdefault('checkpoint', {})
    config['training'].setdefault('logging', {})
    # Diffusion strategy defaults (based on "Back to Basics" paper)
    config['diffusion'].setdefault('prediction_type', 'epsilon')
    config['diffusion'].setdefault('loss_type', 'epsilon')
    config['diffusion'].setdefault('time_sampler', 'logit_normal')
    config['diffusion'].setdefault('time_sampler_params', {'mu': -0.8, 'sigma': 0.8})
    config['diffusion'].setdefault('time_embedding_type', 'continuous')

    return config


def create_optimizer(model: nn.Module, optimizer_config: dict) -> optim.Optimizer:
    """从配置创建优化器"""
    opt_type = optimizer_config.get('type', 'Adam')
    lr = optimizer_config.get('lr', 1e-4)

    if opt_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=optimizer_config.get('weight_decay', 0.0),
            amsgrad=optimizer_config.get('amsgrad', False)
        )
    elif opt_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            amsgrad=optimizer_config.get('amsgrad', False)
        )
    elif opt_type.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0.0),
            nesterov=optimizer_config.get('nesterov', False)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


# ============================================================
# Video-specific functions
# ============================================================

def create_video_model(config: dict) -> VideoGenerationDDPM:
    """创建视频生成模型"""
    model_config = config.get('model', {})
    diffusion_config = config.get('diffusion', {})

    return VideoGenerationDDPM(
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 3),
        num_timesteps=model_config.get('num_timesteps', 1000),
        prediction_type=diffusion_config.get('prediction_type', 'epsilon'),
        loss_type=diffusion_config.get('loss_type', 'epsilon'),
        time_sampler=diffusion_config.get('time_sampler', 'logit_normal'),
        time_sampler_params=diffusion_config.get('time_sampler_params', {'mu': -0.8, 'sigma': 0.8}),
        time_embedding_type=diffusion_config.get('time_embedding_type', 'continuous'),
        base_channels=model_config.get('base_channels', 64),
        time_emb_dim=model_config.get('time_emb_dim', 256),
        num_res_blocks=model_config.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_config.get('attention_resolutions', [16, 8])),
        channel_multiples=tuple(model_config.get('channel_multiples', [1, 2, 4, 8])),
    )


def create_video_dataloader(config: dict) -> DataLoader:
    """创建视频数据加载器"""
    dataset_config = VideoDatasetConfig(
        data_dir=config['dataset'].get('data_dir', 'data'),
        scenes=config['dataset']['scenes'],
        num_frames=config['dataset']['num_frames'],
        image_size=config['dataset']['image_size'],
        seed=config['dataset'].get('seed', 42),
    )

    dataset = VideoDataset(dataset_config)

    dataloader_config = config.get('dataloader', {})
    return DataLoader(
        dataset,
        batch_size=dataloader_config.get('batch_size', 1),
        shuffle=dataloader_config.get('shuffle', True),
        num_workers=dataloader_config.get('num_workers', 0),
        pin_memory=dataloader_config.get('pin_memory', True),
        drop_last=dataloader_config.get('drop_last', False),
        persistent_workers=dataloader_config.get('persistent_workers', False) and dataloader_config.get('num_workers', 0) > 0,
    )


def generate_sample_video(
    model: VideoGenerationDDPM,
    device: torch.device,
    num_frames: int = 16,
    image_size: int = 128,
    num_sampling_steps: int = 50,
) -> np.ndarray:
    """生成样本视频用于可视化"""
    model.eval()
    with torch.no_grad():
        shape = (1, 3, num_frames, image_size, image_size)
        generated = model.sample(shape, num_steps=num_sampling_steps, progress_bar=False)
        video = generated[0].cpu().permute(1, 2, 3, 0).numpy()
        video = np.clip(video, 0, 1)
        video = (video * 255).astype(np.uint8)
    model.train()
    return video


def create_gif_from_video(video: np.ndarray, duration: int = 100) -> io.BytesIO:
    """从视频numpy数组创建GIF"""
    frames = [Image.fromarray(frame) for frame in video]
    gif_buffer = io.BytesIO()
    frames[0].save(
        gif_buffer,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    gif_buffer.seek(0)
    return gif_buffer


# ============================================================
# Toy experiment-specific functions
# ============================================================

def create_toy_model(config: dict) -> ToyDiffusion:
    """创建Toy实验模型"""
    model_config = config.get('model', {})
    diffusion_config = config.get('diffusion', {})

    return ToyDiffusion(
        input_dim=model_config.get('input_dim', 64),
        prediction_type=diffusion_config.get('prediction_type', 'epsilon'),
        loss_type=diffusion_config.get('loss_type', 'epsilon'),
        time_sampler=diffusion_config.get('time_sampler', 'logit_normal'),
        time_sampler_params=diffusion_config.get('time_sampler_params', {'mu': -0.8, 'sigma': 0.8}),
        time_embedding_type=diffusion_config.get('time_embedding_type', 'continuous'),
        base_channels=model_config.get('base_channels', 64),
        time_emb_dim=model_config.get('time_emb_dim', 128),
        num_res_blocks=model_config.get('num_res_blocks', 2),
        channel_multiples=tuple(model_config.get('channel_multiples', [1, 2, 4])),
    )


def create_toy_dataloader(config: dict) -> DataLoader:
    """创建Toy数据加载器"""
    dataset_config = ToyDatasetConfig(
        data_dir=config['dataset'].get('data_dir', 'toy_data'),
        embed_dim=config['dataset'].get('embed_dim', 64),
        seed=config['dataset'].get('seed', 42),
    )

    dataset = ToyDataset(dataset_config, split='train')

    dataloader_config = config.get('dataloader', {})
    return DataLoader(
        dataset,
        batch_size=dataloader_config.get('batch_size', 256),
        shuffle=dataloader_config.get('shuffle', True),
        num_workers=dataloader_config.get('num_workers', 0),
        pin_memory=dataloader_config.get('pin_memory', True),
        drop_last=dataloader_config.get('drop_last', True),
    )


def generate_toy_samples_plot(
    model: ToyDiffusion,
    dataset: ToyDataset,
    device: torch.device,
    num_samples: int = 1000,
    num_steps: int = 100,
) -> io.BytesIO:
    """
    生成Toy样本并创建2D可视化图

    Returns:
        BytesIO对象包含PNG图像数据
    """
    model.eval()
    with torch.no_grad():
        # 生成样本
        samples = model.sample(num_samples, num_steps=num_steps, device=device)
        samples = samples.cpu().numpy()

        # 获取projection matrix
        projection = dataset.get_projection().numpy()

        # Project to 2D
        if samples.shape[1] > 2:
            samples_2d = samples @ projection
        else:
            samples_2d = samples

        # 获取参考数据
        ref_indices = np.random.choice(len(dataset), min(1000, len(dataset)), replace=False)
        ref_data = np.stack([dataset[i]['data'].numpy() for i in ref_indices])
        if ref_data.shape[1] > 2:
            ref_2d = ref_data @ projection
        else:
            ref_2d = ref_data

    model.train()

    # 创建图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ref_2d[:, 0], ref_2d[:, 1], c='gray', alpha=0.3, s=10, label='Reference')
    ax.scatter(samples_2d[:, 0], samples_2d[:, 1], c='blue', alpha=0.5, s=10, label='Generated')
    ax.set_title(f'Generated Samples (D={dataset.get_embed_dim()})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # 保存到BytesIO
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


# ============================================================
# Unified training function
# ============================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict,
    save_dir: str = "checkpoints",
    dataset: object = None,  # For toy experiment visualization
):
    """
    统一训练函数 - 支持视频和Toy实验

    Args:
        model: 模型 (VideoGenerationDDPM 或 ToyDiffusion)
        train_loader: 数据加载器
        optimizer: 优化器
        device: 训练设备
        config: 完整配置
        save_dir: 检查点保存目录
        dataset: 数据集对象（用于toy实验可视化）
    """
    experiment_type = config.get('experiment_type', 'video')
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # 训练配置
    epochs = config['training'].get('epochs', 10)
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)
    log_every = config['training'].get('logging', {}).get('log_every_n_steps', 10)
    save_every = config['training'].get('checkpoint', {}).get('save_every_n_epochs', 1)

    # 混合精度设置
    use_amp = config['training'].get('mixed_precision', {}).get('enabled', False)
    amp_dtype = config['training'].get('mixed_precision', {}).get('dtype', 'float16')
    dtype = torch.float16 if amp_dtype == 'float16' else torch.bfloat16

    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # WandB设置
    use_wandb = config.get('wandb', {}).get('enabled', False) and WANDB_AVAILABLE

    # Visualization设置
    sampling_config = config.get('sampling', {})
    visualize_every = sampling_config.get('visualize_every_n_epochs', 0)  # 0 = disabled

    print(f"\n📊 Training Configuration:")
    print(f"   Experiment Type: {experiment_type}")
    print(f"   Epochs: {epochs}")
    print(f"   Gradient Accumulation: {gradient_accumulation_steps} steps")
    print(f"   Effective Batch Size: {train_loader.batch_size * gradient_accumulation_steps}")
    print(f"   Mixed Precision: {use_amp}")
    print(f"   Max Grad Norm: {max_grad_norm}")
    print(f"   WandB Logging: {use_wandb}")

    model.to(device)
    model.train()

    for epoch in range(epochs):
        loss_accumulator = []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # 获取数据 - 根据实验类型
            if experiment_type == 'toy':
                if isinstance(batch, dict):
                    data = batch['data'].to(device)
                else:
                    data = batch.to(device)
            else:  # video
                if isinstance(batch, dict):
                    data = batch['video'].to(device)
                else:
                    data = batch.to(device)

            # 混合精度前向传播
            with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda', dtype=dtype):
                loss = model.loss(data)
                loss = loss / gradient_accumulation_steps

            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 每gradient_accumulation_steps更新一次权重
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

            # 累积loss
            loss_accumulator.append((loss.detach() * gradient_accumulation_steps).cpu())

            # WandB step logging
            if use_wandb and (batch_idx + 1) % gradient_accumulation_steps == 0:
                recent_loss = loss_accumulator[-1].item()
                global_step = epoch * len(train_loader) + batch_idx
                wandb.log({
                    'train/loss': recent_loss,
                    'train/epoch': epoch + 1,
                }, step=global_step)

            if (batch_idx + 1) % log_every == 0:
                avg_loss = torch.stack(loss_accumulator).mean().item()
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.6f}"
                )

        # Epoch结束
        avg_loss = torch.stack(loss_accumulator).mean().item()
        print(f"✅ Epoch [{epoch + 1}/{epochs}] Average loss: {avg_loss:.6f}")

        # WandB epoch logging
        epoch_metrics = {}
        if use_wandb:
            epoch_metrics = {
                'train/epoch_loss': avg_loss,
                'train/epoch': epoch + 1,
            }

        # 可视化（不依赖wandb，保存到磁盘）
        should_visualize = visualize_every > 0 and (epoch + 1) % visualize_every == 0
        if should_visualize:
            if experiment_type == 'toy':
                print(f"📊 Generating toy samples visualization...")
                plot_buf = generate_toy_samples_plot(
                    model, dataset, device,
                    num_samples=sampling_config.get('num_samples', 1000),
                    num_steps=sampling_config.get('num_steps', 100),
                )
                # 保存到磁盘
                vis_dir = save_dir / "visualizations"
                vis_dir.mkdir(exist_ok=True)
                vis_path = vis_dir / f"samples_epoch_{epoch + 1}.png"
                with open(vis_path, 'wb') as f:
                    f.write(plot_buf.getvalue())
                print(f"✅ Visualization saved to {vis_path}")

                # 同时上传到wandb
                if use_wandb:
                    plot_buf.seek(0)
                    epoch_metrics['samples/generated'] = wandb.Image(
                        Image.open(plot_buf),
                        caption=f"Epoch {epoch + 1}"
                    )
            else:  # video
                print(f"🎬 Generating sample video...")
                wandb_config = config.get('wandb', {})
                image_size = config['dataset'].get('image_size', 128)
                if isinstance(image_size, list):
                    image_size = 128

                sample_video = generate_sample_video(
                    model, device,
                    num_frames=wandb_config.get('num_sample_frames', 16),
                    image_size=image_size,
                    num_sampling_steps=wandb_config.get('sample_steps', 50)
                )

                if use_wandb:
                    epoch_metrics['samples/generated_video'] = wandb.Video(
                        np.array([sample_video.transpose(0, 3, 1, 2)]),
                        fps=10, format="gif"
                    )
                print(f"✅ Sample video generated")

        if use_wandb and epoch_metrics:
            # 用global_step保持单调递增
            global_step = (epoch + 1) * len(train_loader) - 1
            wandb.log(epoch_metrics, step=global_step)

        # 保存检查点
        if (epoch + 1) % save_every == 0:
            checkpoint_path = save_dir / f"model_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model (video or toy)")
    parser.add_argument("--config", type=str, default="configs/config_example.json",
                       help="Config file path")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, default: auto)")

    args = parser.parse_args()

    # ============================================================
    # 1. 加载配置
    # ============================================================
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(args.config)
    experiment_type = config.get('experiment_type', 'video')

    print(f"✅ Config loaded from {config_path}")
    print(f"📋 Experiment Type: {experiment_type}")

    # 检查可视化依赖
    visualize_every = config.get('sampling', {}).get('visualize_every_n_epochs', 0)
    if visualize_every > 0 and experiment_type == 'toy' and not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for toy experiment visualization. "
            "Install with: pip install matplotlib"
        )

    # ============================================================
    # 2. 设置设备
    # ============================================================
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    if device.type == 'cuda' and config.get('hardware', {}).get('allow_tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"   TF32 enabled for faster training")

    # ============================================================
    # 3. 创建模型和数据加载器 (根据实验类型)
    # ============================================================
    diffusion_config = config.get('diffusion', {})

    print(f"\n📋 Diffusion Strategy (Flow Matching):")
    print(f"   Prediction type: {diffusion_config.get('prediction_type', 'epsilon')}")
    print(f"   Loss type: {diffusion_config.get('loss_type', 'epsilon')}")
    print(f"   Time sampler: {diffusion_config.get('time_sampler', 'logit_normal')}")
    print(f"   Formulation: z_t = t*x_0 + (1-t)*ε, v = x_0 - ε")

    if experiment_type == 'toy':
        print("\n🧪 Creating toy experiment model...")
        model = create_toy_model(config)
        train_loader = create_toy_dataloader(config)
        # 获取dataset用于可视化
        dataset = train_loader.dataset

        print(f"\n📋 Model Configuration:")
        print(f"   Input dim: {config['model'].get('input_dim', 64)}")
        print(f"   Hidden dim: {config['model'].get('hidden_dim', 256)}")
        print(f"   Num layers: {config['model'].get('num_layers', 5)}")
        print(f"\n📋 Dataset Configuration:")
        print(f"   Embed dim: {config['dataset'].get('embed_dim', 64)}")
        print(f"   Samples: {len(dataset)}")
    else:
        print("\n🎬 Creating video generation model...")
        model = create_video_model(config)
        train_loader = create_video_dataloader(config)
        dataset = train_loader.dataset

        print(f"\n📋 Model Configuration:")
        print(f"   Base Channels: {config['model'].get('base_channels', 64)}")
        print(f"   Channel Multiples: {config['model'].get('channel_multiples', [1,2,4,8])}")
        print(f"\n📋 Dataset Configuration:")
        print(f"   Scenes: {config['dataset'].get('scenes', [])}")
        print(f"   Num frames: {config['dataset'].get('num_frames', 32)}")

    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")

    print(f"\n📋 DataLoader Configuration:")
    print(f"   Batch size: {config['dataloader'].get('batch_size', 1)}")
    print(f"   Num workers: {config['dataloader'].get('num_workers', 0)}")

    # ============================================================
    # 4. 创建优化器
    # ============================================================
    optimizer_config = config.get('optimizer', {})
    optimizer = create_optimizer(model, optimizer_config)
    print(f"\n✅ Optimizer created: {optimizer_config.get('type', 'Adam')}")
    print(f"   Learning rate: {optimizer_config.get('lr', 1e-4)}")
    print(f"   Weight decay: {optimizer_config.get('weight_decay', 0.0)}")

    # ============================================================
    # 5. 初始化WandB
    # ============================================================
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', False):
        if not WANDB_AVAILABLE:
            print("⚠️  WandB logging requested but wandb is not installed!")
            print("   Install with: pip install wandb")
        else:
            print("\n🔗 Initializing WandB...")
            wandb.init(
                project=wandb_config.get('project', 'diffusion'),
                entity=wandb_config.get('entity', None),
                name=wandb_config.get('run_name', None),
                config={
                    'experiment_type': experiment_type,
                    'model': config['model'],
                    'diffusion': config['diffusion'],
                    'dataset': config['dataset'],
                    'dataloader': config['dataloader'],
                    'optimizer': config['optimizer'],
                    'training': config['training'],
                    'total_params': total_params,
                },
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
            )
            if wandb_config.get('watch_model', True):
                wandb.watch(model, log='all', log_freq=100)
            print("✅ WandB initialized")

    # ============================================================
    # 6. 训练
    # ============================================================
    save_dir = config['training'].get('checkpoint', {}).get('save_dir', 'checkpoints')
    print(f"\n🚀 Starting training...")

    try:
        train(
            model,
            train_loader,
            optimizer,
            device,
            config,
            save_dir,
            dataset=dataset,
        )
        print("\n✅ Training completed!")
    finally:
        if wandb_config.get('enabled', False) and WANDB_AVAILABLE:
            wandb.finish()
            print("🔗 WandB run finished")


if __name__ == "__main__":
    main()
