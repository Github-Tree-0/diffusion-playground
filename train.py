"""
训练脚本 - 用于训练视频生成模型
"""

import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import io

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import VideoGenerationDDPM
from video_dataset import VideoDataset, VideoDatasetConfig

# 尝试导入wandb（可选依赖）
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Run 'pip install wandb' to enable logging.")


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
    config.setdefault('model', {})
    config.setdefault('dataloader', {})
    config.setdefault('optimizer', {})
    config.setdefault('training', {})
    config.setdefault('wandb', {'enabled': False})
    config['training'].setdefault('gradient_accumulation_steps', 1)
    config['training'].setdefault('max_grad_norm', 1.0)
    config['training'].setdefault('mixed_precision', {'enabled': False})
    config['training'].setdefault('checkpoint', {})
    config['training'].setdefault('logging', {})

    return config


def create_optimizer(model: torch.nn.Module, optimizer_config: dict) -> optim.Optimizer:
    """
    从配置创建优化器

    Args:
        model: 要优化的模型
        optimizer_config: 优化器配置

    Returns:
        optimizer: 优化器实例
    """
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


def generate_sample_video(
    model: VideoGenerationDDPM,
    device: torch.device,
    num_frames: int = 16,
    image_size: int = 128,
    num_sampling_steps: int = 50,
) -> np.ndarray:
    """
    生成样本视频用于可视化

    Args:
        model: 训练的模型
        device: 设备
        num_frames: 生成的帧数
        image_size: 图像尺寸
        num_sampling_steps: 采样步数

    Returns:
        video: numpy数组 (T, H, W, C) 范围[0, 255]
    """
    model.eval()
    with torch.no_grad():
        # 生成视频 (1, 3, T, H, W)
        shape = (1, 3, num_frames, image_size, image_size)
        generated = model.sample(shape, num_steps=num_sampling_steps, progress_bar=False)

        # 转换为numpy: (1, 3, T, H, W) -> (T, H, W, 3)
        video = generated[0].cpu().permute(1, 2, 3, 0).numpy()

        # Clip到[0, 1]并转换为[0, 255]
        video = np.clip(video, 0, 1)
        video = (video * 255).astype(np.uint8)

    model.train()
    return video


def create_gif_from_video(video: np.ndarray, duration: int = 100) -> io.BytesIO:
    """
    从视频numpy数组创建GIF

    Args:
        video: (T, H, W, C) numpy数组
        duration: 每帧持续时间（毫秒）

    Returns:
        gif_buffer: BytesIO对象包含GIF数据
    """
    frames = [Image.fromarray(frame) for frame in video]

    # 保存为GIF到BytesIO
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


def train(
    model: VideoGenerationDDPM,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict,
    save_dir: str = "checkpoints",
):
    """
    训练模型 with gradient accumulation and mixed precision

    Args:
        model: 视频生成DDPM模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 训练设备
        config: 完整训练配置
        save_dir: 检查点保存目录
    """
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

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # WandB设置
    use_wandb = config.get('wandb', {}).get('enabled', False) and WANDB_AVAILABLE
    sample_every_epoch = config.get('wandb', {}).get('sample_every_epoch', True)
    num_sample_frames = config.get('wandb', {}).get('num_sample_frames', 16)
    sample_steps = config.get('wandb', {}).get('sample_steps', 50)

    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Gradient Accumulation: {gradient_accumulation_steps} steps")
    print(f"   Effective Batch Size: {train_loader.batch_size * gradient_accumulation_steps}")
    print(f"   Mixed Precision: {use_amp} ({amp_dtype if use_amp else 'N/A'})")
    print(f"   Max Grad Norm: {max_grad_norm}")
    print(f"   WandB Logging: {use_wandb}")

    model.to(device)
    model.train()

    for epoch in range(epochs):
        # 使用tensor累积loss，避免频繁的GPU-CPU同步（.item()调用）
        loss_accumulator = []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # 获取视频数据
            if isinstance(batch, dict):
                videos = batch['video'].to(device)
            else:
                videos = batch.to(device)

            # 混合精度前向传播
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
                loss = model.loss(videos)
                # 为梯度累积缩放loss
                loss = loss / gradient_accumulation_steps

            # 反向传播
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 每gradient_accumulation_steps更新一次权重
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    # 取消缩放并梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

            # 累积loss tensor，避免.item()同步（性能优化）
            loss_accumulator.append((loss.detach() * gradient_accumulation_steps).cpu())

            # WandB logging每个accumulation cycle
            if use_wandb and (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 只在需要log时才同步最近的loss
                recent_loss = loss_accumulator[-1].item()
                step = epoch * len(train_loader) + batch_idx
                wandb.log({
                    'train/loss': recent_loss,
                    'train/epoch': epoch + 1,
                    'train/step': step,
                }, step=step)

            if (batch_idx + 1) % log_every == 0:
                # 计算累积的平均loss（一次性同步）
                avg_loss = torch.stack(loss_accumulator).mean().item()
                current_step = batch_idx + 1
                total_steps = len(train_loader)
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{current_step}/{total_steps}] "
                    f"Loss: {avg_loss:.6f}"
                )

                # Log GPU memory if enabled
                if config['training'].get('logging', {}).get('log_gpu_memory', False):
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(device) / 1024**3
                        reserved = torch.cuda.memory_reserved(device) / 1024**3
                        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Epoch结束：计算平均loss
        avg_loss = torch.stack(loss_accumulator).mean().item()
        print(f"✅ Epoch [{epoch + 1}/{epochs}] Average loss: {avg_loss:.6f}")

        # WandB logging每个epoch
        if use_wandb:
            epoch_metrics = {
                'train/epoch_loss': avg_loss,
                'train/epoch': epoch + 1,
            }

            # 生成样本视频
            if sample_every_epoch:
                print(f"🎬 Generating sample video...")
                try:
                    # 获取图像尺寸
                    image_size = config['dataset']['image_size']
                    if isinstance(image_size, list):
                        image_size = 128  # 使用固定的128x128用于生成

                    # 生成视频
                    sample_video = generate_sample_video(
                        model,
                        device,
                        num_frames=num_sample_frames,
                        image_size=image_size,
                        num_sampling_steps=sample_steps
                    )

                    # 创建GIF
                    gif_buffer = create_gif_from_video(sample_video, duration=100)

                    # 上传到wandb
                    epoch_metrics['samples/generated_video'] = wandb.Video(
                        np.array([sample_video.transpose(0, 3, 1, 2)]),  # (1, T, C, H, W)
                        fps=10,
                        format="gif"
                    )
                    print(f"✅ Sample video generated and logged to wandb")
                except Exception as e:
                    print(f"⚠️  Failed to generate sample: {e}")

            wandb.log(epoch_metrics, step=epoch)

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
    parser = argparse.ArgumentParser(description="Train video generation model")
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

    print(f"✅ Config loaded from {config_path}")
    print(f"\n📋 Model Configuration:")
    print(f"   Base Channels: {config['model'].get('base_channels', 64)}")
    print(f"   Channel Multiples: {config['model'].get('channel_multiples', [1,2,4,8])}")
    print(f"   Num Timesteps: {config['model'].get('num_timesteps', 1000)}")
    print(f"\n📋 Dataset Configuration:")
    print(f"   Scenes: {config['dataset']['scenes']}")
    print(f"   Num frames: {config['dataset']['num_frames']}")
    print(f"   Image size: {config['dataset']['image_size']}")
    print(f"\n📋 DataLoader Configuration:")
    print(f"   Batch size: {config['dataloader'].get('batch_size', 1)}")
    print(f"   Num workers: {config['dataloader'].get('num_workers', 0)}")

    # ============================================================
    # 2. 设置设备
    # ============================================================
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # Enable TF32 for faster training on Ampere GPUs
    if device.type == 'cuda' and config.get('hardware', {}).get('allow_tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"   TF32 enabled for faster training")

    # ============================================================
    # 3. 创建模型 (从配置中读取所有参数)
    # ============================================================
    print("\n🧠 Creating model...")
    model_config = config.get('model', {})

    model = VideoGenerationDDPM(
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 3),
        num_timesteps=model_config.get('num_timesteps', 1000),
        base_channels=model_config.get('base_channels', 64),
        time_emb_dim=model_config.get('time_emb_dim', 256),
        num_res_blocks=model_config.get('num_res_blocks', 2),
        attention_resolutions=tuple(model_config.get('attention_resolutions', [16, 8])),
        channel_multiples=tuple(model_config.get('channel_multiples', [1, 2, 4, 8])),
    )

    # Calculate and print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")

    # ============================================================
    # 4. 创建数据加载器
    # ============================================================
    print("\n📦 Creating dataset...")
    dataset_config = VideoDatasetConfig(
        data_dir=config['dataset'].get('data_dir', 'data'),
        scenes=config['dataset']['scenes'],
        num_frames=config['dataset']['num_frames'],
        image_size=config['dataset']['image_size'],
        seed=config['dataset'].get('seed', 42),
    )

    dataset = VideoDataset(dataset_config)
    print(f"✅ Dataset created with {len(dataset)} samples")

    dataloader_config = config.get('dataloader', {})
    train_loader = DataLoader(
        dataset,
        batch_size=dataloader_config.get('batch_size', 1),
        shuffle=dataloader_config.get('shuffle', True),
        num_workers=dataloader_config.get('num_workers', 0),
        pin_memory=dataloader_config.get('pin_memory', True),  # Default True for faster PCIe transfers
        drop_last=dataloader_config.get('drop_last', False),
        persistent_workers=dataloader_config.get('persistent_workers', False) and dataloader_config.get('num_workers', 0) > 0,
    )
    print(f"✅ DataLoader created")

    # ============================================================
    # 5. 创建优化器 (从配置中读取)
    # ============================================================
    optimizer_config = config.get('optimizer', {})
    optimizer = create_optimizer(model, optimizer_config)
    print(f"✅ Optimizer created: {optimizer_config.get('type', 'Adam')}")
    print(f"   Learning rate: {optimizer_config.get('lr', 1e-4)}")
    print(f"   Weight decay: {optimizer_config.get('weight_decay', 0.0)}")

    # ============================================================
    # 6. 初始化WandB
    # ============================================================
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', False):
        if not WANDB_AVAILABLE:
            print("⚠️  WandB logging requested but wandb is not installed!")
            print("   Install with: pip install wandb")
        else:
            print("\n🔗 Initializing WandB...")
            wandb.init(
                project=wandb_config.get('project', 'video-diffusion'),
                entity=wandb_config.get('entity', None),
                name=wandb_config.get('run_name', None),
                config={
                    'model': config['model'],
                    'dataset': config['dataset'],
                    'dataloader': config['dataloader'],
                    'optimizer': config['optimizer'],
                    'training': config['training'],
                    'total_params': total_params,
                },
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
            )
            # Watch model (track gradients and parameters)
            if wandb_config.get('watch_model', True):
                wandb.watch(model, log='all', log_freq=100)
            print("✅ WandB initialized")

    # ============================================================
    # 7. 训练
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
            save_dir
        )
        print("\n✅ Training completed!")
    finally:
        # 确保wandb properly关闭
        if wandb_config.get('enabled', False) and WANDB_AVAILABLE:
            wandb.finish()
            print("🔗 WandB run finished")


if __name__ == "__main__":
    main()
