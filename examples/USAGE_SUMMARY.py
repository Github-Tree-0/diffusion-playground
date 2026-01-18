#!/usr/bin/env python3
"""
视频数据加载器 - 使用摘要
"""

# ============================================================================
# 核心概念
# ============================================================================

"""
数据结构:
  data/
  ├── scene_name_1/
  │   ├── prefix_1.png       (frame 1)
  │   ├── prefix_50.png      (frame 50)
  │   └── ...
  └── scene_name_2/
      └── ...

文件名格式: <prefix>_<frame_number>.png
例如: RZ_2464601_8150.png -> 帧号是 8150

主要流程:
  1. 指定要使用的场景 -> scenes list
  2. 创建配置 -> VideoDatasetConfig
  3. 创建数据集 -> VideoDataset
  4. 创建加载器 -> DataLoader
  5. 遍历数据 -> for batch in dataloader
"""

# ============================================================================
# 最小工作示例
# ============================================================================

def minimal_example():
    from video_dataset import VideoDataset, VideoDatasetConfig
    from torch.utils.data import DataLoader
    
    # 1. 场景列表
    scenes = [
        "55_RZ_2464601_Aug-11-10-18-09",
        "58_RZ_2489381_Aug-11-17-37-10",
    ]
    
    # 2. 配置
    config = VideoDatasetConfig(
        data_dir="data",
        scenes=scenes,
        num_frames=40,
        image_size=256,
    )
    
    # 3. 数据集
    dataset = VideoDataset(config)
    
    # 4. 加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 5. 使用
    for batch in dataloader:
        videos = batch['video']                # (B, C, T, H, W)
        scene_names = batch['scene_name']      # 场景名称
        frame_indices = batch['frame_indices']  # 帧号列表
        print(f"Batch shape: {videos.shape}")


# ============================================================================
# 在训练中使用
# ============================================================================

def training_loop_example():
    import torch
    from video_dataset import VideoDataset, VideoDatasetConfig
    from torch.utils.data import DataLoader
    
    config = VideoDatasetConfig(
        data_dir="data",
        scenes=["scene1", "scene2"],
        num_frames=40,
    )
    
    dataset = VideoDataset(config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            videos = batch['video'].to(device)  # (B, C, T, H, W)
            
            # 你的模型前向传播
            # output = model(videos)
            # loss = criterion(output, target)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()


# ============================================================================
# 检查数据
# ============================================================================

def inspect_data():
    from pathlib import Path
    from video_dataset import VideoFrameIndex
    
    # 1. 扫描所有可用的场景
    data_dir = Path("data")
    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Total scenes: {len(scenes)}")
    
    # 2. 检查某个场景的帧
    scene_dir = data_dir / scenes[0]
    frame_index = VideoFrameIndex(scene_dir)
    print(f"Scene: {scenes[0]}")
    print(f"  Frames: {len(frame_index.frames)}")
    print(f"  First frame: {frame_index.frames[0][0]}")
    print(f"  Last frame: {frame_index.frames[-1][0]}")


# ============================================================================
# 关键参数说明
# ============================================================================

"""
VideoDatasetConfig 参数:
  
  data_dir: str
    - 数据根目录
    - 例: "data"
  
  scenes: List[str] (可选)
    - 场景名称列表
    - 优先级比 config_path 高
    - 例: ["scene1", "scene2"]
  
  config_path: str (可选)
    - JSON 配置文件路径
    - 只在不提供 scenes 时使用
    - 例: "config/dataset_config.json"
  
  num_frames: int (默认 40)
    - 每个视频包含多少帧
    - 例: 16, 32, 40, 64
  
  image_size: int (默认 256)
    - 图像尺寸（正方形）
    - 自动缩放所有图像
    - 例: 128, 256, 512
  
  seed: int (可选)
    - 随机种子
    - 用于可重复性
    - 例: 42
"""

# ============================================================================
# DataLoader 参数说明
# ============================================================================

"""
DataLoader(dataset, 
  batch_size=4,           # 批次大小，通常 4-16
  shuffle=True,           # 是否打乱顺序
  num_workers=4,          # 多进程数，通常等于 CPU 核数
  pin_memory=True,        # GPU 内存固定（使用 GPU 时）
  drop_last=False,        # 是否丢弃最后一个不完整 batch
)
"""

# ============================================================================
# 输出格式
# ============================================================================

"""
batch = {
    'video': tensor,                    # (B, C, T, H, W)
                                        # B: batch size
                                        # C: 3 (RGB)
                                        # T: num_frames
                                        # H, W: image_size
    
    'scene_name': list[str],           # 每个样本的场景名
    
    'frame_indices': list[list[int]],  # 每个样本的帧号
}

示例:
  batch['video'].shape = torch.Size([4, 3, 40, 256, 256])
  batch['scene_name'] = ['scene1', 'scene2', 'scene1', 'scene3']
  batch['frame_indices'] = [[1, 2, 3, ..., 40], [50, 51, 52, ..., 89], ...]
"""

# ============================================================================
# JSON 配置文件格式
# ============================================================================

"""
config/dataset_config.json:

{
  "scenes": [
    "55_RZ_2464601_Aug-11-10-18-09",
    "58_RZ_2489381_Aug-11-17-37-10",
    "60_RZ_2724114_Aug-14-10-26-09"
  ]
}

然后使用:
  config = VideoDatasetConfig(
    data_dir="data",
    config_path="config/dataset_config.json",
  )
"""

# ============================================================================
# 自动获取所有场景
# ============================================================================

def get_all_scenes():
    from pathlib import Path
    
    data_dir = Path("data")
    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    return scenes


# ============================================================================
# 过滤场景（只保留有足够帧的）
# ============================================================================

def get_valid_scenes(min_frames=40):
    from pathlib import Path
    from video_dataset import VideoFrameIndex
    
    data_dir = Path("data")
    valid_scenes = []
    
    for scene_dir in sorted(data_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        frame_index = VideoFrameIndex(scene_dir)
        if len(frame_index.frames) >= min_frames:
            valid_scenes.append(scene_dir.name)
    
    return valid_scenes


# ============================================================================
# 常用配置模板
# ============================================================================

templates = {
    # 小数据集用于测试
    "small": {
        "scenes": ["scene1", "scene2", "scene3"],
        "num_frames": 16,
        "image_size": 128,
    },
    
    # 标准配置
    "standard": {
        "scenes": get_all_scenes() if __name__ != "__main__" else [],
        "num_frames": 40,
        "image_size": 256,
    },
    
    # 高分辨率
    "hires": {
        "scenes": get_all_scenes() if __name__ != "__main__" else [],
        "num_frames": 32,
        "image_size": 512,
    },
}

# ============================================================================
# 常见问题解答
# ============================================================================

"""
Q: 如何获取特定场景的数据？
A: 在 scenes 列表中只包含该场景
   scenes = ["specific_scene"]

Q: 如何确保数据不重复？
A: DataLoader 会自动处理，设置 shuffle=True
   dataloader = DataLoader(dataset, shuffle=True)

Q: 如何加快加载速度？
A: 1. 增加 num_workers
   2. 启用 pin_memory=True
   3. 减小 image_size
   4. 使用 SSD 而不是 HDD

Q: 如何调试数据加载问题？
A: 1. 运行 test_dataloader.py
   2. 设置 num_workers=0 看是否有多进程问题
   3. 检查文件名格式

Q: 如何在分布式训练中使用？
A: 使用 DistributedSampler
   from torch.utils.data import DistributedSampler
   sampler = DistributedSampler(dataset)
   dataloader = DataLoader(dataset, sampler=sampler)
"""

# ============================================================================
# 文件和目录
# ============================================================================

"""
重要文件:
  - video_dataset.py              主要实现
  - simple_example.py             简单示例
  - example_usage.py              详细示例
  - test_dataloader.py            测试脚本
  - config_example.json           配置文件示例
  - DATALOADER_QUICKSTART.md      快速开始指南
  - VIDEO_DATALOADER_README.md    完整文档

运行示例:
  python simple_example.py simple         # 简单例子
  python simple_example.py training       # 训练例子
  python simple_example.py config         # 配置文件例子
  python simple_example.py check          # 检查场景
  python test_dataloader.py               # 运行测试
"""

# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n可用函数:")
    print("  - minimal_example()      最小化示例")
    print("  - training_loop_example() 训练循环示例")
    print("  - inspect_data()         检查数据")
    print("  - get_all_scenes()       获取所有场景")
    print("  - get_valid_scenes()     获取有效场景")
