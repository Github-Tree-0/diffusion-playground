# 📁 视频数据加载器项目结构

## 文件组织

```
DiffusionPlayground/
│
├── src/                          # ⭐ 核心代码
│   └── video_dataset.py         # 主要实现文件
│
├── examples/                      # 📝 使用示例
│   ├── simple_example.py        # 快速入门示例
│   ├── example_usage.py         # 完整使用演示
│   └── USAGE_SUMMARY.py         # 代码参考手册
│
├── tests/                         # ✅ 测试
│   └── test_dataloader.py       # 完整测试套件
│
├── docs/                          # 📚 文档
│   ├── DATALOADER_QUICKSTART.md         # 中文快速指南
│   ├── VIDEO_DATALOADER_README.md       # 英文完整文档
│   ├── IMPLEMENTATION_SUMMARY.md        # 实现总结
│   ├── INDEX.md                         # 文件导航
│   └── COMPLETION_SUMMARY.md            # 项目完成总结
│
├── configs/                       # 🎛️ 配置文件
│   └── config_example.json      # 配置示例
│
├── data/                          # 📦 数据目录（你的数据）
│   ├── scene_1/
│   ├── scene_2/
│   └── ...
│
├── VIDEO_DATALOADER_GUIDE.md      # 📖 本文件
├── README.md                      # 原始项目 README
└── ...
```

## 🚀 快速开始

### 1️⃣ 查看快速指南
```bash
# 中文快速入门
cat docs/DATALOADER_QUICKSTART.md

# 英文完整文档
cat docs/VIDEO_DATALOADER_README.md
```

### 2️⃣ 运行测试
```bash
# 验证系统是否正常
python tests/test_dataloader.py
```

### 3️⃣ 查看示例
```bash
# 最简单的示例（推荐第一次查看）
python examples/simple_example.py simple

# 列出可用的场景
python examples/simple_example.py check

# 查看训练集成示例
python examples/simple_example.py training

# 完整示例
python examples/example_usage.py
```

### 4️⃣ 在你的代码中使用
```python
import sys
from pathlib import Path
sys.path.insert(0, "src")  # 添加 src 到路径

from video_dataset import VideoDataset, VideoDatasetConfig
from torch.utils.data import DataLoader

# 配置
config = VideoDatasetConfig(
    data_dir="data",
    scenes=["scene_1", "scene_2"],
    num_frames=40,
    image_size=256,
)

# 创建数据加载器
dataset = VideoDataset(config)
dataloader = DataLoader(dataset, batch_size=4)

# 使用
for batch in dataloader:
    videos = batch['video']  # (B, C, T, H, W)
```

## 📂 各文件夹说明

### `src/`
- **包含内容**: 核心实现代码
- **主要文件**: `video_dataset.py`
- **包括**: VideoDatasetConfig、VideoFrameIndex、VideoDataset 类

### `examples/`
- **包含内容**: 使用示例和参考代码
- **文件列表**:
  - `simple_example.py` - 快速入门（推荐先读这个）
  - `example_usage.py` - 完整使用演示
  - `USAGE_SUMMARY.py` - 代码参考手册

### `tests/`
- **包含内容**: 测试和验证代码
- **主要文件**: `test_dataloader.py`
- **测试项目**: 6 个完整测试函数

### `docs/`
- **包含内容**: 完整的文档
- **文件列表**:
  - 快速开始指南（中文）
  - 完整参考文档（英文）
  - 实现总结
  - 项目完成总结
  - 文件导航索引

### `configs/`
- **包含内容**: 配置文件示例
- **主要文件**: `config_example.json`
- **用途**: 参考配置格式创建自己的配置

## 🎯 常见任务

### 任务 1: 快速测试数据加载
```bash
python examples/simple_example.py simple
```

### 任务 2: 检查可用的场景
```bash
python examples/simple_example.py check
```

### 任务 3: 运行完整测试
```bash
python tests/test_dataloader.py
```

### 任务 4: 查看可用命令
```bash
python examples/simple_example.py help
```

### 任务 5: 使用配置文件
```bash
python examples/example_usage.py
```

## 📖 学习顺序建议

1. **初学者** (5-10 分钟)
   - 阅读: `docs/DATALOADER_QUICKSTART.md`
   - 运行: `python examples/simple_example.py simple`

2. **中级用户** (30 分钟)
   - 阅读: `docs/VIDEO_DATALOADER_README.md`
   - 运行: `python examples/example_usage.py`
   - 查看: `examples/simple_example.py` 源代码

3. **高级用户** (1-2 小时)
   - 阅读: `src/video_dataset.py` 源代码
   - 查看: `tests/test_dataloader.py` 测试用例
   - 自定义修改代码

## 🔍 导入路径说明

所有示例和测试文件已配置自动路径，所以可以直接运行：

```bash
# 从任何位置都可以运行
python examples/simple_example.py simple
python tests/test_dataloader.py
```

内部使用的导入路径：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from video_dataset import VideoDataset, VideoDatasetConfig
```

## 💡 主要特性

✅ 灵活的配置方式（JSON 或直接列表）
✅ 随机场景选择 + 随机起始帧
✅ 连续帧加载（可指定帧数）
✅ 自动图像缩放和格式转换
✅ PyTorch DataLoader 完全兼容
✅ 多进程加载支持
✅ 完整的测试和文档
✅ 丰富的使用示例

## ❓ 常见问题

### Q: 为什么要分这么多文件夹？
**A**: 这样可以保持代码有序，便于维护和使用：
- `src/` 核心代码
- `examples/` 学习参考
- `tests/` 验证功能
- `docs/` 完整文档

### Q: 怎样在训练脚本中使用？
**A**: 复制这两行到你的脚本顶部：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from video_dataset import VideoDataset, VideoDatasetConfig
```

### Q: 能否修改数据加载的方式？
**A**: 完全可以，源代码在 `src/video_dataset.py`，可自由修改。

### Q: 如何配置数据目录路径？
**A**: 在 `VideoDatasetConfig` 中指定 `data_dir` 参数。

## 📞 获取帮助

### 查看快速开始
```bash
cat docs/DATALOADER_QUICKSTART.md
```

### 查看完整文档
```bash
cat docs/VIDEO_DATALOADER_README.md
```

### 查看文件导航
```bash
cat docs/INDEX.md
```

### 查看实现细节
```bash
cat docs/IMPLEMENTATION_SUMMARY.md
```

### 运行测试
```bash
python tests/test_dataloader.py
```

## 📊 统计信息

| 类别 | 文件数 | 说明 |
|------|--------|------|
| 核心代码 | 1 | video_dataset.py |
| 示例代码 | 3 | 不同使用场景 |
| 测试代码 | 1 | 6 个测试函数 |
| 文档 | 5 | 完整文档集合 |
| 配置 | 1 | 配置示例 |

## ✨ 项目完成

- ✅ 所有核心功能已实现
- ✅ 所有文件已整理归类
- ✅ 导入路径已自动配置
- ✅ 所有示例可直接运行
- ✅ 完整的文档和测试

**现在你可以直接使用它们了！** 🎉

---

**有任何问题？** 查看 [docs/INDEX.md](docs/INDEX.md) 了解更多信息。
