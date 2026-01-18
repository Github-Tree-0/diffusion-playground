"""
Video Dataset Loader - 从磁盘加载连续视频帧
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class VideoDatasetConfig:
    """视频数据集配置"""
    
    def __init__(
        self,
        data_dir: str,
        config_path: Optional[str] = None,
        scenes: Optional[List[str]] = None,
        num_frames: int = 40,
        image_size: int = 256,
        seed: Optional[int] = None,
    ):
        """
        Args:
            data_dir: 数据根目录，包含各个场景文件夹
            config_path: JSON配置文件路径，指定使用哪些场景（可选）
            scenes: 场景列表（如果提供则优先使用，不需要config_path）
            num_frames: 每个视频包含的帧数
            image_size: 图像大小（假设为正方形）
            seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path) if config_path else None
        self.scenes_list = scenes
        self.num_frames = num_frames
        self.image_size = image_size
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)


class VideoFrameIndex:
    """管理单个场景中的帧索引"""
    
    def __init__(self, scene_dir: Path):
        """
        Args:
            scene_dir: 场景目录路径
        """
        self.scene_dir = scene_dir
        self.scene_name = scene_dir.name
        
        # 从文件名提取帧编号
        self.frames = self._load_frames()
    
    def _load_frames(self) -> List[Tuple[int, Path]]:
        """
        从场景目录加载所有帧文件，并提取帧编号
        
        Returns:
            [(frame_number, file_path), ...] 按帧编号排序
        """
        frames = []
        for img_path in self.scene_dir.glob("*.png"):
            # 文件名格式: RZ_2464601_8150.png -> frame_num = 8150
            # 提取最后一个下划线后的数字
            try:
                frame_num = int(img_path.stem.split("_")[-1])
                frames.append((frame_num, img_path))
            except (ValueError, IndexError):
                continue
        
        # 按帧编号排序
        frames.sort(key=lambda x: x[0])
        return frames
    
    def get_frames(self, start_idx: int, num_frames: int) -> Optional[List[Path]]:
        """
        获取连续的帧序列
        
        Args:
            start_idx: 开始帧的索引（在排序的帧列表中）
            num_frames: 需要的帧数
            
        Returns:
            帧文件路径列表，如果不足则返回None
        """
        if start_idx + num_frames > len(self.frames):
            return None
        
        frame_paths = [path for _, path in self.frames[start_idx:start_idx + num_frames]]
        return frame_paths
    
    def get_random_sequence(self, num_frames: int) -> Optional[List[Path]]:
        """
        随机选择一个连续的帧序列
        
        Args:
            num_frames: 需要的帧数
            
        Returns:
            帧文件路径列表
        """
        if len(self.frames) < num_frames:
            return None
        
        max_start = len(self.frames) - num_frames
        start_idx = random.randint(0, max_start)
        return self.get_frames(start_idx, num_frames)


class VideoDataset(Dataset):
    """视频数据集"""
    
    def __init__(
        self,
        config: VideoDatasetConfig,
        transform=None,
    ):
        """
        Args:
            config: VideoDatasetConfig 对象
            transform: 图像变换函数
        """
        self.config = config
        self.transform = transform
        
        # 读取配置文件
        self.scenes = self._load_config()
        
        # 初始化场景索引
        self.scene_indices = self._init_scene_indices()
        
        if not self.scene_indices:
            raise ValueError(f"No valid scenes found in {config.data_dir}")
    
    def _load_config(self) -> List[str]:
        """
        从JSON配置文件或直接使用提供的场景列表读取场景
        
        Returns:
            场景名称列表
        """
        # 如果直接提供了场景列表，使用它
        if self.config.scenes_list:
            return self.config.scenes_list
        
        # 否则从配置文件读取
        if self.config.config_path and self.config.config_path.exists():
            with open(self.config.config_path, 'r') as f:
                config_data = json.load(f)
                scenes = config_data.get('scenes', [])
                return scenes
        else:
            raise ValueError(
                "Either provide scenes list directly or provide a valid config_path"
            )
    
    def _init_scene_indices(self) -> Dict[str, VideoFrameIndex]:
        """
        初始化所有场景的帧索引
        
        Returns:
            {scene_name: VideoFrameIndex}
        """
        scene_indices = {}
        
        for scene_name in self.scenes:
            scene_dir = self.config.data_dir / scene_name
            if scene_dir.exists() and scene_dir.is_dir():
                try:
                    frame_index = VideoFrameIndex(scene_dir)
                    if len(frame_index.frames) >= self.config.num_frames:
                        scene_indices[scene_name] = frame_index
                except Exception as e:
                    print(f"Warning: Failed to load scene {scene_name}: {e}")
        
        return scene_indices
    
    def __len__(self) -> int:
        """
        数据集大小（可以设置为一个大数字用于无限采样）
        """
        # 返回场景数 * 每个场景的最大可能序列数
        # 或者简单地返回一个固定的大数字用于 epoch-based 训练
        total = 0
        for scene_index in self.scene_indices.values():
            if len(scene_index.frames) >= self.config.num_frames:
                total += len(scene_index.frames) - self.config.num_frames + 1
        return max(total, len(self.scene_indices) * 100)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本：随机选择场景和起始帧，读取连续帧序列
        
        Args:
            idx: 样本索引（主要用于确定性采样，实际采样通常是随机的）
            
        Returns:
            {
                'video': tensor (C, T, H, W),
                'scene_name': str,
                'frame_indices': list of int,
            }
        """
        # 随机选择场景
        scene_name = random.choice(list(self.scene_indices.keys()))
        scene_index = self.scene_indices[scene_name]
        
        # 随机选择起始帧和连续序列
        frame_paths = scene_index.get_random_sequence(self.config.num_frames)
        
        if frame_paths is None:
            # 如果随机选择失败，尝试从开始选择
            frame_paths = scene_index.get_frames(0, self.config.num_frames)
        
        if frame_paths is None:
            raise RuntimeError(f"Cannot get frame sequence for scene {scene_name}")
        
        # 读取并处理图像
        frames = self._load_frames(frame_paths)
        
        # 转换为 tensor (C, T, H, W)
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        
        # 提取帧编号
        frame_indices = [int(path.stem.split("_")[-1]) for path in frame_paths]
        
        return {
            'video': video_tensor,
            'scene_name': scene_name,
            'frame_indices': frame_indices,
        }
    
    def _load_frames(self, frame_paths: List[Path]) -> np.ndarray:
        """
        加载帧图像
        
        Args:
            frame_paths: 帧文件路径列表
            
        Returns:
            np.ndarray (T, H, W, C)
        """
        frames = []
        
        for path in frame_paths:
            img = Image.open(path)
            
            # 调整大小
            if img.size != (self.config.image_size, self.config.image_size):
                img = img.resize(
                    (self.config.image_size, self.config.image_size),
                    Image.BILINEAR
                )
            
            # 转为 RGB（如果需要）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            frames.append(np.array(img))
        
        return np.stack(frames, axis=0)  # (T, H, W, C)


def create_default_config(
    data_dir: str,
    output_path: str,
    scenes: Optional[List[str]] = None,
) -> None:
    """
    创建默认的配置文件
    
    Args:
        data_dir: 数据目录
        output_path: 输出配置文件路径
        scenes: 场景列表，如果为None则自动扫描
    """
    if scenes is None:
        # 自动扫描数据目录中的所有场景文件夹
        data_path = Path(data_dir)
        scenes = [
            d.name for d in data_path.iterdir()
            if d.is_dir() and (d / "*.png")
        ]
        scenes.sort()
    
    config = {
        'scenes': scenes,
        'num_frames': 40,
        'image_size': 256,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file with {len(scenes)} scenes: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 1. 创建配置文件（如果还没有）
    config_path = "config/dataset_config.json"
    data_dir = "data"
    
    # 指定要使用的场景（或留空自动扫描）
    scenes_to_use = [
        "55_RZ_2464601_Aug-11-10-18-09",
        "58_RZ_2489381_Aug-11-17-37-10",
        "60_RZ_2724114_Aug-14-10-26-09",
        # 添加更多场景...
    ]
    
    # 创建配置文件
    # create_default_config(data_dir, config_path, scenes_to_use)
    
    # 2. 创建数据集
    config = VideoDatasetConfig(
        data_dir=data_dir,
        config_path=config_path,
        num_frames=40,
        image_size=256,
        seed=42,
    )
    
    dataset = VideoDataset(config)
    
    # 3. 使用数据加载器
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # 测试：遍历几个batch
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:
            break
        
        print(f"Batch {batch_idx}:")
        print(f"  Video shape: {batch['video'].shape}")  # (B, C, T, H, W)
        print(f"  Scene names: {batch['scene_name']}")
        print(f"  Frame indices: {batch['frame_indices']}")
