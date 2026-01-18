"""
简单示例：如何快速使用视频数据加载器
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from video_dataset import VideoDataset, VideoDatasetConfig
from torch.utils.data import DataLoader
import torch


def simple_example():
    """最简单的使用方式"""
    
    # 步骤 1: 指定场景
    scenes = [
        "55_RZ_2464601_Aug-11-10-18-09",
        "58_RZ_2489381_Aug-11-17-37-10",
        "60_RZ_2724114_Aug-14-10-26-09",
        "62_RZ_2740959_Aug-14-15-03-32",
        "64_RZ_2809807_Aug-15-10-11-14",
    ]
    
    # 步骤 2: 创建配置
    config = VideoDatasetConfig(
        data_dir="data",
        scenes=scenes,
        num_frames=40,
        image_size=256,
        seed=42,
    )
    
    # 步骤 3: 创建数据集
    dataset = VideoDataset(config)
    print(f"✓ Dataset created with {len(dataset.scene_indices)} valid scenes")
    
    # 步骤 4: 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # 设置 > 0 以启用多进程加载
        pin_memory=True,
    )
    
    # 步骤 5: 使用数据
    print("\nIterating through batches:")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # 只显示前3个batch
            break
        
        videos = batch['video']              # (B, C, T, H, W)
        scene_names = batch['scene_name']
        frame_indices = batch['frame_indices']
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Video shape: {videos.shape}")
        print(f"  Dtype: {videos.dtype}")
        print(f"  Value range: [{videos.min():.3f}, {videos.max():.3f}]")
        print(f"  Scenes: {scene_names}")
        print(f"  Frame ranges: {[f[0]}-{f[-1]} for f in frame_indices]}")


def training_example():
    """如何在训练中使用"""
    
    scenes = [
        "55_RZ_2464601_Aug-11-10-18-09",
        "58_RZ_2489381_Aug-11-17-37-10",
        "60_RZ_2724114_Aug-14-10-26-09",
    ]
    
    config = VideoDatasetConfig(
        data_dir="data",
        scenes=scenes,
        num_frames=40,
        image_size=256,
    )
    
    dataset = VideoDataset(config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 模拟训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 2
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            videos = batch['video'].to(device)  # 移到 GPU
            
            # 这里放你的模型和损失计算
            # output = model(videos)
            # loss = criterion(output, target)
            # total_loss += loss.item()
            
            if batch_idx % 2 == 0:
                print(f"  Batch {batch_idx}: videos {videos.shape}")


def config_file_example():
    """使用 JSON 配置文件"""
    
    import json
    
    # 创建配置文件
    config_path = Path("config/my_dataset_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "scenes": [
            "55_RZ_2464601_Aug-11-10-18-09",
            "58_RZ_2489381_Aug-11-17-37-10",
            "60_RZ_2724114_Aug-14-10-26-09",
        ]
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created config file: {config_path}")
    
    # 使用配置文件
    dataset_config = VideoDatasetConfig(
        data_dir="data",
        config_path=str(config_path),
        num_frames=40,
        image_size=256,
    )
    
    dataset = VideoDataset(dataset_config)
    print(f"✓ Dataset created with {len(dataset.scene_indices)} scenes")


def check_scenes():
    """检查可用的场景"""
    
    data_dir = Path("data")
    
    # 扫描所有场景文件夹
    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(scenes)} scene folders:")
    
    # 显示每个场景的帧数
    for scene_name in scenes[:10]:  # 只显示前10个
        scene_path = data_dir / scene_name
        num_frames = len(list(scene_path.glob("*.png")))
        print(f"  {scene_name}: {num_frames} frames")
    
    if len(scenes) > 10:
        print(f"  ... and {len(scenes) - 10} more scenes")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    else:
        cmd = "simple"
    
    if cmd == "simple":
        print("=" * 60)
        print("Simple Example - Basic Usage")
        print("=" * 60)
        simple_example()
    
    elif cmd == "training":
        print("=" * 60)
        print("Training Example - Integration with Training Loop")
        print("=" * 60)
        training_example()
    
    elif cmd == "config":
        print("=" * 60)
        print("Config File Example - Using JSON Configuration")
        print("=" * 60)
        config_file_example()
    
    elif cmd == "check":
        print("=" * 60)
        print("Check Available Scenes")
        print("=" * 60)
        check_scenes()
    
    else:
        print(f"Unknown command: {cmd}")
        print("Available commands: simple, training, config, check")
