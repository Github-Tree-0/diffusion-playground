# 视频数据加载器系统

## 📋 概述

完整的视频数据加载器实现，用于从本地磁盘加载连续的视频帧序列进行深度学习训练。

### ✨ 主要特性

- **灵活的场景管理**：支持 JSON 配置文件或直接场景列表
- **随机采样**：随机选择场景和起始帧，支持不重复采样
- **连续帧加载**：自动加载指定数量的连续帧
- **PyTorch 兼容**：完全兼容 PyTorch DataLoader 和多进程加载
- **自动缩放**：自动缩放图像到指定大小
- **格式转换**：自动处理不同的图像格式

## 📁 文件结构

```
DiffusionPlayground/
├── video_dataset.py              # 主要数据加载器类
├── simple_example.py             # 简单使用示例
├── example_usage.py              # 详细使用示例
├── config_example.json           # 配置文件示例
├── DATALOADER_QUICKSTART.md      # 快速开始指南
├── data/
│   ├── 55_RZ_2464601_Aug-11-10-18-09/
│   │   ├── RZ_2464601_1.png
│   │   ├── RZ_2464601_50.png
│   │   └── ...
│   ├── 58_RZ_2489381_Aug-11-17-37-10/
│   │   └── ...
│   └── ...
└── config/
    └── dataset_config.json       # 可选：使用配置文件时
```

## 🚀 快速开始

### 最简单的方式

```python
from video_dataset import VideoDataset, VideoDatasetConfig
from torch.utils.data import DataLoader

# 1. 指定场景
scenes = [
    "55_RZ_2464601_Aug-11-10-18-09",
    "58_RZ_2489381_Aug-11-17-37-10",
]

# 2. 创建配置
config = VideoDatasetConfig(
    data_dir="data",
    scenes=scenes,
    num_frames=40,
    image_size=256,
)

# 3. 创建数据集
dataset = VideoDataset(config)

# 4. 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# 5. 使用
for batch in dataloader:
    videos = batch['video']        # (B, C, T, H, W)
    scene_names = batch['scene_name']
    frame_indices = batch['frame_indices']
```

## 📖 使用示例

### 运行快速示例

```bash
# 简单例子
python simple_example.py simple

# 训练集成例子
python simple_example.py training

# 配置文件例子
python simple_example.py config

# 检查可用场景
python simple_example.py check
```

### 详细示例

```bash
python example_usage.py
```

## 🔧 核心类说明

### VideoDatasetConfig
配置类，用于设置数据加载器的参数。

```python
config = VideoDatasetConfig(
    data_dir="data",              # 数据目录
    scenes=[...],                 # 场景列表（可选）
    config_path="config.json",    # 或使用配置文件（可选）
    num_frames=40,                # 每个视频的帧数
    image_size=256,               # 图像大小
    seed=42,                      # 随机种子（可选）
)
```

### VideoFrameIndex
管理单个场景中的帧索引，支持随机采样和连续加载。

```python
frame_index = VideoFrameIndex(scene_dir)
frames = frame_index.get_random_sequence(num_frames=40)
```

### VideoDataset
PyTorch Dataset 类，实现了 `__len__` 和 `__getitem__` 方法。

```python
dataset = VideoDataset(config)

# 获取一个样本
sample = dataset[0]
# {
#     'video': tensor (C, T, H, W),
#     'scene_name': str,
#     'frame_indices': list[int],
# }
```

## 📊 输出格式

每个 batch 包含：

| 键 | 类型 | 形状 | 说明 |
|-----|------|------|------|
| `video` | Tensor | (B, C, T, H, W) | 视频帧张量，值范围 [0, 1] |
| `scene_name` | List[str] | (B,) | 每个样本的场景名称 |
| `frame_indices` | List[List[int]] | (B, T) | 每个样本的帧号列表 |

其中：
- **B**: batch size
- **C**: channels (3 for RGB)
- **T**: num_frames
- **H, W**: image height and width

## ⚙️ 配置选项

### 通过直接列表（推荐）

```python
config = VideoDatasetConfig(
    data_dir="data",
    scenes=["scene1", "scene2", ...],
    num_frames=40,
    image_size=256,
)
```

### 通过 JSON 文件

`config/dataset_config.json`:
```json
{
  "scenes": [
    "55_RZ_2464601_Aug-11-10-18-09",
    "58_RZ_2489381_Aug-11-17-37-10"
  ]
}
```

然后：
```python
config = VideoDatasetConfig(
    data_dir="data",
    config_path="config/dataset_config.json",
)
```

## 🎯 常见用例

### 训练循环集成

```python
import torch
from torch.utils.data import DataLoader
from video_dataset import VideoDataset, VideoDatasetConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = VideoDatasetConfig(
    data_dir="data",
    scenes=[...],
    num_frames=40,
)

dataset = VideoDataset(config)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for epoch in range(num_epochs):
    for batch in dataloader:
        videos = batch['video'].to(device)  # (B, C, T, H, W)
        
        # 你的训练代码
        # output = model(videos)
        # loss = criterion(output, ...)
        # ...
```

### 数据检查和可视化

```python
# 获取所有可用的场景
from pathlib import Path
data_dir = Path("data")
scenes = [d.name for d in data_dir.iterdir() if d.is_dir()]

# 创建小规模数据集用于测试
config = VideoDatasetConfig(
    data_dir="data",
    scenes=scenes[:5],  # 只使用前5个场景
    num_frames=40,
)

dataset = VideoDataset(config)
sample = dataset[0]

# 检查视频张量
print(f"Video shape: {sample['video'].shape}")
print(f"Scene: {sample['scene_name']}")
print(f"Frames: {sample['frame_indices']}")
```

## 🔍 数据结构要求

### 文件名格式

**必须** 使用格式：`<prefix>_<frame_number>.png`

例如：
- ✅ `RZ_2464601_8150.png` → frame 8150
- ✅ `JAW_2679477_100.png` → frame 100
- ❌ `frame_001.png` → 无法正确解析
- ❌ `frame8150.png` → 无法正确解析

### 目录结构

```
data/
├── scene1_folder/
│   ├── prefix_1.png
│   ├── prefix_50.png
│   ├── prefix_100.png
│   └── ...
├── scene2_folder/
│   └── ...
└── ...
```

## ⚡ 性能优化

### 多进程加载

```python
dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,  # 设置为 CPU 核心数
    pin_memory=True,  # GPU 内存固定
)
```

### 调整参数

- **batch_size**: 根据 GPU 显存调整，通常 4-16
- **num_workers**: 设置为 CPU 核心数或 2x CPU 核心数
- **pin_memory**: 使用 GPU 时设置为 True
- **image_size**: 更小的图像加载更快

## 🐛 常见问题

### Q: 无法找到场景？
A: 检查：
1. 数据目录路径正确
2. 场景文件夹确实存在
3. 文件夹中有 PNG 文件

### Q: 帧号提取失败？
A: 检查文件名格式是否为 `<prefix>_<frame_number>.png`

### Q: DataLoader 加载缓慢？
A: 尝试：
1. 增加 `num_workers`
2. 启用 `pin_memory=True`
3. 检查磁盘 I/O 是否是瓶颈

### Q: 内存不足？
A: 减小 `batch_size` 或 `image_size`

## 📝 API 参考

### VideoDatasetConfig

```python
class VideoDatasetConfig:
    def __init__(
        self,
        data_dir: str,
        config_path: Optional[str] = None,
        scenes: Optional[List[str]] = None,
        num_frames: int = 40,
        image_size: int = 256,
        seed: Optional[int] = None,
    )
```

### VideoDataset

```python
class VideoDataset(Dataset):
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Dict
```

### VideoFrameIndex

```python
class VideoFrameIndex:
    def get_random_sequence(self, num_frames: int) -> Optional[List[Path]]
    def get_frames(self, start_idx: int, num_frames: int) -> Optional[List[Path]]
```

## 🔗 相关文件

- `DATALOADER_QUICKSTART.md` - 快速开始指南（中文）
- `simple_example.py` - 简单使用示例
- `example_usage.py` - 详细使用示例
- `config_example.json` - 配置文件示例

## 📄 许可证

根据项目许可证

## 🤝 贡献

欢迎改进和建议！

---

**最后更新**: 2026年1月

**版本**: 1.0
