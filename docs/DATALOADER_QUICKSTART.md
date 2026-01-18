# 视频数据加载器 (Video DataLoader) - 快速开始指南

## 概述

这是一个为您的视频数据集设计的数据加载器。它支持：
- ✅ 从 JSON 配置文件或直接列表中读取场景
- ✅ 随机选择场景和起始帧
- ✅ 加载连续的视频帧序列
- ✅ 支持 PyTorch DataLoader 和多进程数据加载
- ✅ 自动图像缩放和格式转换

## 数据结构

假设您的数据组织如下：
```
data/
├── 55_RZ_2464601_Aug-11-10-18-09/
│   ├── RZ_2464601_1.png       (frame 1)
│   ├── RZ_2464601_50.png      (frame 50)
│   ├── RZ_2464601_100.png     (frame 100)
│   └── ...
├── 58_RZ_2489381_Aug-11-17-37-10/
│   ├── RZ_2489381_1.png
│   └── ...
└── ...
```

**重要**：文件名格式必须是 `<prefix>_<frame_number>.png`，这样可以正确提取帧号。

## 快速使用

### 方法 1: 使用直接场景列表（推荐）

```python
from video_dataset import VideoDataset, VideoDatasetConfig
from torch.utils.data import DataLoader

# 指定要使用的场景
scenes = [
    "55_RZ_2464601_Aug-11-10-18-09",
    "58_RZ_2489381_Aug-11-17-37-10",
    "60_RZ_2724114_Aug-14-10-26-09",
    # 添加更多场景...
]

# 创建配置（不需要配置文件）
config = VideoDatasetConfig(
    data_dir="data",
    scenes=scenes,              # 直接传入场景列表
    num_frames=40,              # 每个视频 40 帧
    image_size=256,             # 调整到 256x256
    seed=42                     # 可重复性
)

# 创建数据集
dataset = VideoDataset(config)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# 使用 DataLoader
for batch in dataloader:
    videos = batch['video']                    # (B, C, T, H, W)
    scene_names = batch['scene_name']          # 场景名称列表
    frame_indices = batch['frame_indices']     # 帧号列表
    
    print(f"Video shape: {videos.shape}")
    print(f"Scenes: {scene_names}")
    break
```

### 方法 2: 使用 JSON 配置文件

1. 创建配置文件 `config/dataset_config.json`:

```json
{
  "scenes": [
    "55_RZ_2464601_Aug-11-10-18-09",
    "58_RZ_2489381_Aug-11-17-37-10",
    "60_RZ_2724114_Aug-14-10-26-09"
  ]
}
```

2. 在代码中使用：

```python
from video_dataset import VideoDataset, VideoDatasetConfig
from torch.utils.data import DataLoader

config = VideoDatasetConfig(
    data_dir="data",
    config_path="config/dataset_config.json",
    num_frames=40,
    image_size=256,
)

dataset = VideoDataset(config)
dataloader = DataLoader(dataset, batch_size=4)
```

## 输出格式

每个 batch 包含以下内容：

```python
{
    'video': tensor,           # shape (B, C, T, H, W)
                              # B: batch size
                              # C: channels (3 for RGB)
                              # T: num_frames (e.g., 40)
                              # H, W: image height and width
    
    'scene_name': list[str],  # 每个样本所属的场景名
                              # e.g., ['55_RZ_2464601_Aug-11-10-18-09', ...]
    
    'frame_indices': list[list[int]],  # 每个样本的帧号
                                       # e.g., [[1, 2, 3, ..., 40], ...]
}
```

## 集成到训练循环

```python
import torch
from torch.utils.data import DataLoader
from video_dataset import VideoDataset, VideoDatasetConfig

# 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scenes = [...]  # 您的场景列表

# 创建数据集
config = VideoDatasetConfig(
    data_dir="data",
    scenes=scenes,
    num_frames=40,
    image_size=256,
)
dataset = VideoDataset(config)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# 训练循环
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        videos = batch['video'].to(device)           # (B, C, T, H, W)
        scene_names = batch['scene_name']
        frame_indices = batch['frame_indices']
        
        # 送入模型
        # output = model(videos)
        # loss = criterion(output, target)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        print(f"Epoch {epoch}, Batch {batch_idx}: videos shape {videos.shape}")
```

## 创建场景列表

### 方法 1: 手动指定

```python
scenes = [
    "55_RZ_2464601_Aug-11-10-18-09",
    "58_RZ_2489381_Aug-11-17-37-10",
    # ... 更多场景
]
```

### 方法 2: 自动扫描目录

```python
from pathlib import Path

data_dir = Path("data")
scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
print(f"Found {len(scenes)} scenes")
```

### 方法 3: 使用 create_default_config

```python
from video_dataset import create_default_config

create_default_config(
    data_dir="data",
    output_path="config/dataset_config.json",
    scenes=["55_RZ_2464601_Aug-11-10-18-09", ...]  # 可选
)
```

## 重要参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `data_dir` | str | 数据根目录 | - |
| `config_path` | str | 配置文件路径（可选） | None |
| `scenes` | list | 场景列表（可选） | None |
| `num_frames` | int | 每个视频的帧数 | 40 |
| `image_size` | int | 图像大小（正方形） | 256 |
| `seed` | int | 随机种子 | None |

## 常见问题

### Q: 文件名格式是什么？
A: 格式为 `<prefix>_<frame_number>.png`，例如 `RZ_2464601_8150.png`。代码会自动从文件名的最后一部分提取帧号。

### Q: 如何处理不同大小的图像？
A: 配置中的 `image_size` 参数会自动将所有图像缩放到该大小。默认是 256x256。

### Q: 可以使用多进程加载吗？
A: 可以。在 DataLoader 中设置 `num_workers > 0`。推荐值是 CPU 核心数。

### Q: 数据集中没有足够的帧怎么办？
A: 数据集会自动跳过不足 `num_frames` 的场景。在初始化时检查 `len(dataset.scene_indices)` 来看有多少个有效的场景。

### Q: 如何确保可重复性？
A: 设置 `seed` 参数，例如 `seed=42`。

## 性能优化建议

1. **使用多进程加载**：设置 `num_workers` 为 CPU 核心数
2. **启用 pin_memory**：在 DataLoader 中设置 `pin_memory=True`（如果使用 GPU）
3. **调整 batch_size**：根据显存大小调整，通常 4-16 是合理的
4. **预处理**：如果需要额外的预处理，在 `VideoDataset` 中的 `__getitem__` 方法中实现

## 扩展功能

### 添加图像变换

```python
from torchvision import transforms

# 定义变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 传入数据集
dataset = VideoDataset(config, transform=transform)
```

### 自定义采样策略

修改 `VideoDataset.__getitem__()` 中的采样逻辑：

```python
def __getitem__(self, idx: int) -> Dict:
    # 自定义逻辑：例如倾向于选择特定场景
    if random.random() < 0.5:
        scene_name = self.preferred_scene
    else:
        scene_name = random.choice(list(self.scene_indices.keys()))
    # ...
```

## 运行示例

```bash
# 运行快速测试
python example_usage.py
```

这将：
1. 加载配置文件
2. 创建数据集
3. 打印数据集信息
4. 遍历几个 batch 并显示信息

## 文件列表

- `video_dataset.py` - 主要的数据加载器类
- `config_example.json` - 配置文件示例
- `example_usage.py` - 使用示例和快速测试
- `QUICKSTART.md` - 本文件

## 许可证

根据项目许可证

## 支持

有问题？检查以下内容：
1. 数据目录路径是否正确
2. 文件名格式是否符合 `<prefix>_<frame_number>.png`
3. 场景目录中是否有 PNG 文件
4. 是否有足够的帧（至少 `num_frames` 个）
