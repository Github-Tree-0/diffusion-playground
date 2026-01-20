#!/usr/bin/env python3
"""
视频数据加载器测试脚本
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("Test 1: Importing modules...")
    print("=" * 60)
    
    try:
        from video_dataset import (
            VideoDataset,
            VideoDatasetConfig,
            VideoFrameIndex,
            create_default_config,
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_scene_scanning():
    """测试场景扫描"""
    print("\n" + "=" * 60)
    print("Test 2: Scanning scenes...")
    print("=" * 60)
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    if not scenes:
        print("❌ No scene folders found")
        return False
    
    print(f"✅ Found {len(scenes)} scene folders")
    print(f"   First 5 scenes:")
    for scene in scenes[:5]:
        scene_path = data_dir / scene
        num_frames = len(list(scene_path.glob("*.png")))
        print(f"   - {scene}: {num_frames} frames")
    
    return True


def test_frame_indexing():
    """测试帧索引"""
    print("\n" + "=" * 60)
    print("Test 3: Testing frame indexing...")
    print("=" * 60)
    
    try:
        from video_dataset import VideoFrameIndex
    except ImportError:
        print("❌ Cannot import VideoFrameIndex")
        return False
    
    data_dir = Path("data")
    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    if not scenes:
        print("❌ No scenes found")
        return False
    
    # 测试第一个有足够帧的场景
    for scene_name in scenes[:5]:
        scene_path = data_dir / scene_name
        frame_index = VideoFrameIndex(scene_path)
        
        num_frames = len(frame_index.frames)
        
        if num_frames >= 40:
            print(f"✅ Scene '{scene_name}' has {num_frames} frames")
            
            # 尝试获取连续帧序列
            frames = frame_index.get_random_sequence(40)
            if frames and len(frames) == 40:
                print(f"✅ Successfully loaded 40-frame sequence")
                print(f"   First frame: {frames[0].name}")
                print(f"   Last frame: {frames[-1].name}")
                return True
    
    print("❌ No scene with enough frames found")
    return False


def test_dataset_creation():
    """测试数据集创建"""
    print("\n" + "=" * 60)
    print("Test 4: Creating dataset...")
    print("=" * 60)
    
    try:
        from video_dataset import VideoDataset, VideoDatasetConfig
    except ImportError:
        print("❌ Cannot import classes")
        return False
    
    data_dir = Path("data")
    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])[:5]
    
    if not scenes:
        print("❌ No scenes found")
        return False
    
    try:
        config = VideoDatasetConfig(
            data_dir="data",
            scenes=scenes,
            num_frames=40,
            image_size=[160, 210],
        )
        print(f"✅ Config created")
        
        dataset = VideoDataset(config)
        print(f"✅ Dataset created")
        print(f"   Valid scenes: {len(dataset.scene_indices)}")
        print(f"   Total samples: {len(dataset)}")
        
        if len(dataset.scene_indices) == 0:
            print("⚠️  Warning: No valid scenes in dataset")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_loading():
    """测试批次加载"""
    print("\n" + "=" * 60)
    print("Test 5: Loading batches...")
    print("=" * 60)
    
    try:
        from video_dataset import VideoDataset, VideoDatasetConfig
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    data_dir = Path("data")
    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])[:3]
    
    try:
        config = VideoDatasetConfig(
            data_dir="data",
            scenes=scenes,
            num_frames=40,
            image_size=[160, 210],
            seed=42,
        )
        
        dataset = VideoDataset(config)
        
        if len(dataset.scene_indices) == 0:
            print("❌ No valid scenes")
            return False
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        
        print(f"✅ DataLoader created")
        
        # 获取一个batch
        batch = next(iter(dataloader))
        
        print(f"✅ Successfully loaded batch")
        print(f"   Video shape: {batch['video'].shape}")
        print(f"   Expected shape: (2, 3, 40, 210, 160)")
        print(f"   Scene names: {batch['scene_name']}")        
        # 验证形状
        if batch['video'].shape == (2, 3, 40, 210, 160):
            print("✅ Batch shape is correct")
            return True
        else:
            print(f"❌ Batch shape mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_file():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("Test 6: Testing config file...")
    print("=" * 60)
    
    try:
        from video_dataset import VideoDataset, VideoDatasetConfig
    except ImportError:
        print("❌ Cannot import classes")
        return False
    
    # 创建配置文件
    config_path = Path("config/test_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path("data")
    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])[:3]
    
    config_data = {"scenes": scenes}
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    print(f"✅ Created config file: {config_path}")
    
    try:
        config = VideoDatasetConfig(
            data_dir="data",
            config_path=str(config_path),
            num_frames=40,
            image_size=[160, 210],
        )
        
        dataset = VideoDataset(config)
        print(f"✅ Dataset loaded from config file")
        print(f"   Scenes: {len(dataset.scene_indices)}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # 清理
        config_path.unlink()


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  视频数据加载器测试套件".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    tests = [
        ("导入模块", test_imports),
        ("场景扫描", test_scene_scanning),
        ("帧索引", test_frame_indexing),
        ("数据集创建", test_dataset_creation),
        ("批次加载", test_batch_loading),
        ("配置文件", test_config_file),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结".center(60))
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("-" * 60)
    print(f"总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
