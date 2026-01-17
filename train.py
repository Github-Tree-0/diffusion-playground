"""
训练脚本 - 用于训练视频生成模型
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models import VideoGenerationDDPM
import argparse
from pathlib import Path


class DummyVideoDataset(Dataset):
    """虚拟数据集用于演示"""
    def __init__(self, num_samples: int = 100, num_frames: int = 8, img_size: int = 32):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回随机视频张量 (3, num_frames, height, width)
        video = torch.randn(3, self.num_frames, self.img_size, self.img_size)
        return video


def train(
    model: VideoGenerationDDPM,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    save_dir: str = "checkpoints",
):
    """训练模型"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, videos in enumerate(train_loader):
            videos = videos.to(device)

            # 计算损失
            loss = model.loss(videos)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}, "
                    f"Loss: {avg_loss:.6f}"
                )

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.6f}")

        # 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = save_dir / f"model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train video generation model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames per video")
    parser.add_argument("--img-size", type=int, default=32, help="Image size")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
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

    # 创建数据加载器
    print("Creating dataset...")
    dataset = DummyVideoDataset(
        num_samples=100,
        num_frames=args.num_frames,
        img_size=args.img_size,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    print(f"Starting training on {args.device}...")
    device = torch.device(args.device)
    train(model, train_loader, optimizer, device, args.epochs, args.save_dir)

    print("Training completed!")


if __name__ == "__main__":
    main()
