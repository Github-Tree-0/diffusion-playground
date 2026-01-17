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
    model = VideoGenerationDDPM(
        in_channels=3,
        out_channels=3,
        num_timesteps=1000,
        base_channels=64,
        time_emb_dim=256,
        num_res_blocks=2,
        channel_multiples=(1, 2, 4, 8),
    )

    # 加载检查点
    if args.model:
        print(f"Loading model from {args.model}...")
        checkpoint = torch.load(args.model, map_location="cpu")
        model.load_state_dict(checkpoint)

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
