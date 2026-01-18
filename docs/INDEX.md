# 📚 视频数据加载器 - 文件索引和快速导航

## 📋 快速导航

### 🚀 我是新手，应该从哪里开始？
1. **首先阅读**: [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) - 中文快速指南
2. **然后运行**: 
   ```bash
   python simple_example.py simple
   ```
3. **查看代码**: [simple_example.py](simple_example.py) - 10 分钟快速上手

---

### 📖 完整文档

| 文件 | 说明 | 适合人群 |
|------|------|---------|
| [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) | 快速开始指南（中文） | ⭐⭐⭐ 所有用户 |
| [VIDEO_DATALOADER_README.md](VIDEO_DATALOADER_README.md) | 完整文档和 API | ⭐⭐ 中高级用户 |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | 实现总结 | ⭐⭐⭐ 参考手册 |
| [USAGE_SUMMARY.py](USAGE_SUMMARY.py) | 使用示例代码 | ⭐⭐⭐ 代码参考 |

---

### 💻 代码文件

#### 核心实现
- [video_dataset.py](video_dataset.py) - 主要实现（500+ 行代码）
  - `VideoDatasetConfig` - 配置类
  - `VideoFrameIndex` - 帧索引管理
  - `VideoDataset` - PyTorch Dataset
  - `create_default_config()` - 工具函数

#### 示例代码
- [simple_example.py](simple_example.py) - 简单示例（推荐从这里开始）
  - `simple_example()` - 基础用法
  - `training_example()` - 训练集成
  - `config_file_example()` - 配置文件用法
  - `check_scenes()` - 检查可用场景

- [example_usage.py](example_usage.py) - 详细示例
  - `setup_dataloader()` - 数据加载器设置
  - `example_training_loop()` - 训练循环示例

#### 测试和调试
- [test_dataloader.py](test_dataloader.py) - 完整测试套件
  - 6 个测试函数
  - 全面的错误检查
  - 详细的输出报告

---

### ⚙️ 配置文件

- [config_example.json](config_example.json) - 配置文件示例
  - 包含 20 个示例场景
  - 完整的参数设置

---

## 🎯 使用场景导航

### 📌 场景 1: "我想立即开始使用"
**所需时间**: 5 分钟

```bash
# 方法 1: 运行简单示例
python simple_example.py simple

# 方法 2: 运行测试
python test_dataloader.py

# 方法 3: 查看配置
cat config_example.json
```

**参考文件**:
- [simple_example.py](simple_example.py) - 第 10-40 行
- [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) - "快速使用" 部分

---

### 📌 场景 2: "我想集成到我的训练代码中"
**所需时间**: 30 分钟

**步骤**:
1. 复制 [video_dataset.py](video_dataset.py) 到你的项目
2. 导入并创建配置:
   ```python
   from video_dataset import VideoDataset, VideoDatasetConfig
   ```
3. 创建数据集:
   ```python
   config = VideoDatasetConfig(data_dir="data", scenes=[...])
   dataset = VideoDataset(config)
   ```
4. 在训练循环中使用

**参考文件**:
- [simple_example.py](simple_example.py) - `training_example()`
- [VIDEO_DATALOADER_README.md](VIDEO_DATALOADER_README.md) - "使用示例" 部分

---

### 📌 场景 3: "我想自定义和扩展"
**所需时间**: 1-2 小时

**主要修改点**:
1. 修改 `VideoDataset.__getitem__()` - 自定义采样
2. 修改 `_load_frames()` - 添加数据增强
3. 继承 `VideoDataset` - 实现自己的版本

**参考文件**:
- [video_dataset.py](video_dataset.py) - 源代码
- [VIDEO_DATALOADER_README.md](VIDEO_DATALOADER_README.md) - "扩展功能" 部分

---

### 📌 场景 4: "我想调试问题"
**所需时间**: 10-30 分钟

**步骤**:
1. 运行测试套件:
   ```bash
   python test_dataloader.py
   ```
2. 检查问题类型
3. 查阅常见问题

**参考文件**:
- [test_dataloader.py](test_dataloader.py) - 所有测试
- [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) - "常见问题" 部分

---

## 🔍 功能查询

### "我想要..."

- **随机加载视频**: 查看 [simple_example.py](simple_example.py) 第 25-35 行
- **使用 JSON 配置**: 查看 [simple_example.py](simple_example.py) 的 `config_file_example()`
- **集成训练循环**: 查看 [simple_example.py](simple_example.py) 的 `training_example()`
- **检查数据**: 运行 `python simple_example.py check`
- **调整批大小**: 查看 [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) 的 "性能优化"
- **多进程加载**: 查看 [simple_example.py](simple_example.py) 第 40 行
- **处理不同大小**: 查看 [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) 的 "常见问题"
- **分布式训练**: 查看 [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) 的 "常见问题"

---

## 📊 文件大小和复杂度

| 文件 | 行数 | 复杂度 | 说明 |
|------|------|--------|------|
| video_dataset.py | 500+ | ⭐⭐⭐ | 核心实现 |
| simple_example.py | 150+ | ⭐ | 简单示例 |
| test_dataloader.py | 250+ | ⭐⭐⭐ | 完整测试 |
| example_usage.py | 200+ | ⭐⭐ | 详细示例 |
| USAGE_SUMMARY.py | 200+ | ⭐⭐ | 代码参考 |

---

## 🚀 命令快速参考

```bash
# 运行简单示例
python simple_example.py simple

# 运行训练示例
python simple_example.py training

# 运行配置文件示例
python simple_example.py config

# 检查可用场景
python simple_example.py check

# 运行完整测试
python test_dataloader.py

# 运行详细示例
python example_usage.py

# 查看使用摘要
cat USAGE_SUMMARY.py
```

---

## 📝 快速参考

### 最简洁的代码

```python
from video_dataset import VideoDataset, VideoDatasetConfig
from torch.utils.data import DataLoader

config = VideoDatasetConfig("data", scenes=["scene1"], num_frames=40)
dataset = VideoDataset(config)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    videos = batch['video']  # (B, C, T, H, W)
```

### 完整的训练循环

```python
import torch
from torch.utils.data import DataLoader
from video_dataset import VideoDataset, VideoDatasetConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = VideoDatasetConfig(
    data_dir="data",
    scenes=["scene1", "scene2"],
    num_frames=40,
    seed=42,
)

dataset = VideoDataset(config)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for epoch in range(num_epochs):
    for batch in dataloader:
        videos = batch['video'].to(device)
        
        # 你的训练代码
        output = model(videos)
        loss = criterion(output, target)
        # ...
```

---

## 🎓 学习建议

### 第 1 天：了解基础
- [ ] 读 [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md)
- [ ] 运行 `python simple_example.py simple`
- [ ] 运行 `python simple_example.py check`

### 第 2 天：深入理解
- [ ] 读 [VIDEO_DATALOADER_README.md](VIDEO_DATALOADER_README.md)
- [ ] 运行所有示例
- [ ] 运行 `python test_dataloader.py`

### 第 3 天：实践应用
- [ ] 集成到你的项目
- [ ] 修改参数进行实验
- [ ] 查看 [video_dataset.py](video_dataset.py) 源代码

---

## 🆘 需要帮助？

### 问题分类

| 问题类型 | 查看文件 | 部分 |
|---------|---------|------|
| 快速开始 | [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) | "快速使用" |
| 常见问题 | [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) | "常见问题" |
| API 参考 | [VIDEO_DATALOADER_README.md](VIDEO_DATALOADER_README.md) | "API 参考" |
| 代码示例 | [simple_example.py](simple_example.py) | 任何函数 |
| 错误排查 | [test_dataloader.py](test_dataloader.py) | 运行完整测试 |
| 配置帮助 | [USAGE_SUMMARY.py](USAGE_SUMMARY.py) | "JSON 配置文件格式" |

---

## ✅ 检查清单

完成以下检查确保一切正常：

- [ ] 数据目录存在: `data/`
- [ ] 有场景文件夹: `data/scene_1/`, `data/scene_2/`, ...
- [ ] 场景中有 PNG 文件
- [ ] 文件名格式正确: `prefix_frame_number.png`
- [ ] 运行 `python test_dataloader.py` 所有测试通过

---

## 💡 提示和技巧

1. **快速检查**: 运行 `python simple_example.py check` 查看所有可用场景
2. **测试小数据集**: 在 `scenes` 中只使用前 3-5 个场景
3. **调试多进程**: 将 `num_workers` 设置为 0 以禁用多进程
4. **性能基准**: 测试不同的 `batch_size` 和 `num_workers` 组合
5. **可重复性**: 总是设置 `seed` 参数

---

## 📚 相关资源

- PyTorch Dataset: https://pytorch.org/docs/stable/data.html
- DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
- 数据加载最佳实践: https://pytorch.org/tutorials/recipes/recipes/

---

## 📞 支持

有问题？

1. 检查 [DATALOADER_QUICKSTART.md](DATALOADER_QUICKSTART.md) 的常见问题
2. 运行 `python test_dataloader.py` 诊断问题
3. 查看 [VIDEO_DATALOADER_README.md](VIDEO_DATALOADER_README.md) 的完整文档
4. 查看 [simple_example.py](simple_example.py) 的示例代码

---

最后更新: 2026年1月  
版本: 1.0

---

## 🎉 开始使用

**现在就开始**: 
```bash
python simple_example.py simple
```

祝您使用愉快！🚀
