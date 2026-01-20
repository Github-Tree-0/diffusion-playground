"""
最小化测试脚本：
1. 创建dataloader
2. 取一帧数据
3. 加噪
4. 过model
5. 预测噪声
6. 计算loss
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_dataset import VideoDataset, VideoDatasetConfig
from torch.utils.data import DataLoader
from models.diffusion import VideoGenerationDDPM


def main():
    """主测试函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================================
    # 1. 加载配置
    # ============================================================
    config_path = "configs/config_example.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n📋 Config loaded from {config_path}")
    print(f"   Scenes: {config['dataset']['scenes']}")
    print(f"   Num frames: {config['dataset']['num_frames']}")
    print(f"   Image size: {config['dataset']['image_size']}")
    
    # ============================================================
    # 2. 创建数据加载器
    # ============================================================
    dataset_config = VideoDatasetConfig(
        data_dir="data",
        scenes=config['dataset']['scenes'],
        num_frames=config['dataset']['num_frames'],
        image_size=config['dataset']['image_size'],
        seed=config['dataset'].get('seed', 42),
    )
    
    dataset = VideoDataset(dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 只取一帧
        shuffle=False,
        num_workers=0,
    )
    
    print(f"\n✅ DataLoader created")
    print(f"   Total samples: {len(dataset)}")
    
    # ============================================================
    # 3. 获取一个batch
    # ============================================================
    batch = next(iter(dataloader))
    videos = batch['video'].to(device)  # (B, C, T, H, W)
    print(f"\n🎬 Batch shape: {videos.shape}")
    print(f"   B={videos.shape[0]}, C={videos.shape[1]}, T={videos.shape[2]}, H={videos.shape[3]}, W={videos.shape[4]}")
    
    # ============================================================
    # 4. 创建模型
    # ============================================================
    model = VideoGenerationDDPM(
        in_channels=3,
        out_channels=3,
        num_timesteps=1000,
        base_channels=64,
        time_emb_dim=256,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        channel_multiples=(1, 2, 4, 8),
    )
    model.to(device)
    print(f"\n🧠 Model created and moved to {device}")
    
    # ============================================================
    # 5. 计算loss (这已经包含了加噪、过model、预测噪声的过程)
    # ============================================================
    print(f"\n⚙️ Computing loss...")
    loss = model.loss(videos)
    
    print(f"✅ Loss computed!")
    print(f"   Loss value: {loss.item():.6f}")

    loss.backward()  # 反向传播测试

    # ============================================================
    # 6. 详细步骤（可选，用于理解过程）
    # ============================================================
    print(f"\n📊 Detailed breakdown:")
    print(f"   1. Input video shape: {videos.shape}")
    
    # 手动执行前向过程以展示细节
    batch_size = videos.shape[0]
    t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
    print(f"   2. Random timestep: {t.item()}")
    
    # 添加噪声
    x_t, noise = model.scheduler.add_noise(videos, t)
    print(f"   3. After adding noise - shape: {x_t.shape}")
    
    # 过model预测噪声
    predicted_noise = model.unet(x_t, t)
    print(f"   4. Predicted noise shape: {predicted_noise.shape}")
    
    # 计算MSE loss
    mse_loss = nn.functional.mse_loss(predicted_noise, noise)
    print(f"   5. MSE Loss: {mse_loss.item():.6f}")
    
    print(f"\n✨ Test completed successfully!")


if __name__ == "__main__":
    main()
