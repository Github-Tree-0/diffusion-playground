"""
视频数据加载器使用示例和训练脚本
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch
from torch.utils.data import DataLoader
from video_dataset import VideoDataset, VideoDatasetConfig, create_default_config


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_dataloader(config_dict: dict, data_dir: str = "data"):
    """
    设置数据加载器
    
    Args:
        config_dict: 配置字典
        data_dir: 数据目录
        
    Returns:
        DataLoader 对象
    """
    dataset_config_dict = config_dict['dataset']
    training_config_dict = config_dict['training']
    
    # 创建数据集配置
    dataset_config = VideoDatasetConfig(
        data_dir=data_dir,
        config_path=None,  # 我们将手动设置场景
        num_frames=dataset_config_dict.get('num_frames', 40),
        image_size=dataset_config_dict.get('image_size', 256),
        seed=dataset_config_dict.get('seed'),
    )
    
    # 手动设置配置中的场景列表（为了避免需要config_path）
    dataset_config._scenes = dataset_config_dict.get('scenes', [])
    
    # 创建数据集（需要修改数据集类以支持直接传入场景列表）
    # 这里我们先使用原始方式
    
    # 创建数据加载器配置
    dataloader_config = {
        'batch_size': training_config_dict.get('batch_size', 4),
        'shuffle': training_config_dict.get('shuffle', True),
        'num_workers': training_config_dict.get('num_workers', 4),
        'pin_memory': training_config_dict.get('pin_memory', True),
    }
    
    return dataloader_config


def create_dataset_config_file(
    config_path: str,
    scenes: list,
    num_frames: int = 40,
    image_size: int = 256,
):
    """
    创建数据集配置文件
    
    Args:
        config_path: 配置文件路径
        scenes: 场景列表
        num_frames: 每个视频的帧数
        image_size: 图像大小
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        'scenes': scenes,
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    """主程序：演示数据加载器的使用"""
    
    # 配置参数
    data_dir = "data"
    config_path = "config_example.json"
    dataset_config_path = "config/dataset_config.json"
    
    # 加载主配置文件
    config = load_config(config_path)
    
    scenes = config['dataset']['scenes']
    num_frames = config['dataset']['num_frames']
    image_size = config['dataset']['image_size']
    
    # 创建数据集配置文件
    create_dataset_config_file(dataset_config_path, scenes, num_frames, image_size)
    
    print(f"✓ Created dataset config with {len(scenes)} scenes")
    print(f"✓ Scenes: {scenes[:3]}... (showing first 3)")
    
    # 创建数据集配置对象
    dataset_config = VideoDatasetConfig(
        data_dir=data_dir,
        config_path=dataset_config_path,
        num_frames=num_frames,
        image_size=image_size,
        seed=config['dataset'].get('seed', 42),
    )
    
    print(f"\n✓ Dataset config created:")
    print(f"  - Data dir: {dataset_config.data_dir}")
    print(f"  - Num frames: {dataset_config.num_frames}")
    print(f"  - Image size: {dataset_config.image_size}x{dataset_config.image_size}")
    
    # 创建数据集
    try:
        dataset = VideoDataset(dataset_config)
        print(f"\n✓ Dataset created with {len(dataset.scene_indices)} valid scenes")
        print(f"  - Total samples: {len(dataset)}")
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        return
    
    # 创建数据加载器
    training_config = config['training']
    dataloader = DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        shuffle=training_config['shuffle'],
        num_workers=training_config['num_workers'],
        pin_memory=training_config['pin_memory'],
    )
    
    print(f"\n✓ DataLoader created:")
    print(f"  - Batch size: {training_config['batch_size']}")
    print(f"  - Shuffle: {training_config['shuffle']}")
    print(f"  - Num workers: {training_config['num_workers']}")
    
    # 测试：遍历几个batch
    print(f"\n{'='*60}")
    print("Testing DataLoader - iterating through batches:")
    print(f"{'='*60}")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:
            break
        
        print(f"\nBatch {batch_idx}:")
        print(f"  └─ Video shape: {batch['video'].shape}")  # (B, C, T, H, W)
        print(f"  └─ Scene names: {batch['scene_name']}")
        print(f"  └─ Frame ranges:")
        
        for scene_name, frame_indices in zip(batch['scene_name'], batch['frame_indices']):
            first_frame = frame_indices[0]
            last_frame = frame_indices[-1]
            num_frames_loaded = len(frame_indices)
            print(f"      {scene_name}: frames {first_frame} ~ {last_frame} ({num_frames_loaded} frames)")


def example_training_loop():
    """
    示例：如何在训练循环中使用数据加载器
    """
    
    # 设置
    data_dir = "data"
    dataset_config_path = "config/dataset_config.json"
    config_path = "config_example.json"
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建数据集配置文件
    scenes = config['dataset']['scenes']
    create_dataset_config_file(dataset_config_path, scenes)
    
    # 创建数据集和数据加载器
    dataset_config = VideoDatasetConfig(
        data_dir=data_dir,
        config_path=dataset_config_path,
        num_frames=config['dataset']['num_frames'],
        image_size=config['dataset']['image_size'],
    )
    
    dataset = VideoDataset(dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['training']['shuffle'],
        num_workers=config['training']['num_workers'],
    )
    
    # 训练循环示例
    num_epochs = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            videos = batch['video'].to(device)  # (B, C, T, H, W)
            scene_names = batch['scene_name']
            frame_indices = batch['frame_indices']
            
            # 这里放你的训练代码
            # loss = model(videos)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: video shape {videos.shape}, scenes: {scene_names[:2]}")
            
            if batch_idx >= 10:  # 只示例前10个batch
                break


if __name__ == "__main__":
    print("Video Dataset Loader Example\n")
    print(f"{'='*60}\n")
    
    # 运行主程序
    main()
    
    # 取消注释下面这行来查看训练循环示例
    # example_training_loop()
