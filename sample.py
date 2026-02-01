"""
采样脚本 - 用于生成视频
"""

import torch
from models import VideoGenerationDDPM
import argparse
from pathlib import Path


def sample_videos(
    model: VideoGenerationDDPM,
    batch_size: int = 2,
    num_frames: int = 8,
    img_size: int = 32,
    num_steps: int = 100,
    device: torch.device = None,
):
    """生成视频"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    shape = (batch_size, 3, num_frames, img_size, img_size)

    print(f"Sampling videos with shape {shape}...")
    with torch.no_grad():
        videos = model.sample(shape, num_steps=num_steps, progress_bar=True)

    return videos


def interpolate_videos(
    model: VideoGenerationDDPM,
    num_frames: int = 8,
    img_size: int = 32,
    num_interp_frames: int = 10,
    device: torch.device = None,
):
    """在两个随机视频之间插值"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # 生成两个随机视频
    x_start = torch.randn(1, 3, num_frames, img_size, img_size, device=device)
    x_end = torch.randn(1, 3, num_frames, img_size, img_size, device=device)

    print(f"Interpolating between two videos...")
    with torch.no_grad():
        interpolated = model.interpolate(
            x_start,
            x_end,
            num_steps=50,
            num_interp_frames=num_interp_frames,
        )

    return interpolated


def main():
    parser = argparse.ArgumentParser(description="Sample videos from the model")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames")
    parser.add_argument("--img-size", type=int, default=32, help="Image size")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of denoising steps")
    parser.add_argument("--mode", type=str, default="sample", choices=["sample", "interpolate"])
    parser.add_argument("--num-interp-frames", type=int, default=10, help="Number of interpolation frames")
    parser.add_argument("--output", type=str, default="samples", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # 创建模型
    print("Creating model...")

    # 如果提供了检查点，先加载以获取配置
    model_config = {}
    if args.model:
        print(f"Loading checkpoint from {args.model}...")
        checkpoint = torch.load(args.model, map_location="cpu")

        # 尝试从检查点中读取模型配置
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            if 'model' in checkpoint['config']:
                model_config = checkpoint['config']['model']
                print("✅ Loaded model config from checkpoint")
            else:
                print("⚠️  No model config in checkpoint, using default parameters")
        else:
            print("⚠️  Old checkpoint format, using default parameters")

    # 使用配置或默认值创建模型
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
    print("✅ Model created")

    # 加载模型权重
    if args.model:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model weights from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # 旧格式检查点
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights from checkpoint")

    device = torch.device(args.device)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # 采样
    if args.mode == "sample":
        videos = sample_videos(
            model,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_steps=args.num_steps,
            device=device,
        )
        output_path = output_dir / "samples.pt"
        torch.save(videos, output_path)
        print(f"Saved samples to {output_path}")
        print(f"Videos shape: {videos.shape}")
        print(f"Videos range: [{videos.min():.3f}, {videos.max():.3f}]")

    elif args.mode == "interpolate":
        interpolated = interpolate_videos(
            model,
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_interp_frames=args.num_interp_frames,
            device=device,
        )
        output_path = output_dir / "interpolated.pt"
        torch.save(interpolated, output_path)
        print(f"Saved interpolated video to {output_path}")
        print(f"Interpolated shape: {interpolated.shape}")


if __name__ == "__main__":
    main()
