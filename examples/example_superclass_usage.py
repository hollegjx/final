#!/usr/bin/env python3
"""
CIFAR-100超类功能使用示例
展示如何在GCD项目中使用15个超类数据划分
"""

import os
import sys
from torchvision import transforms

# 添加项目根目录路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cifar100_superclass import (
    get_single_superclass_datasets,
    get_superclass_splits,
    SUPERCLASS_NAMES,
    CIFAR100_SUPERCLASSES
)


def demo_superclass_info():
    """演示超类信息查看"""
    print("🌟 CIFAR-100 15个超类信息")
    print("=" * 80)

    superclass_splits = get_superclass_splits()

    for i, (name, info) in enumerate(superclass_splits.items()):
        print(f"\n超类 {i}: {name}")
        print(f"  包含类别总数: {len(CIFAR100_SUPERCLASSES[name])}")
        print(f"  原始类别ID: {CIFAR100_SUPERCLASSES[name]}")
        print(f"  已知类 (< 80): {info['known_classes']} (共{len(info['known_classes'])}个)")
        print(f"  未知类 (>= 80): {info['unknown_classes']} (共{len(info['unknown_classes'])}个)")

        # 检查是否适合GCD训练
        if len(info['known_classes']) == 0:
            print(f"  ⚠️  警告: 没有已知类，不适合GCD训练")
        elif len(info['unknown_classes']) == 0:
            print(f"  ⚠️  警告: 没有未知类，不适合GCD训练")
        else:
            print(f"  ✅ 适合GCD训练")


def demo_single_superclass_dataset():
    """演示单个超类数据集加载"""
    print("\n🎯 单个超类数据集加载演示")
    print("=" * 80)

    # 选择一个有已知类和未知类的超类进行演示
    superclass_name = 'mammals'  # 哺乳动物类，通常包含较多样本
    print(f"演示超类: {superclass_name}")

    # 定义简单的数据变换
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.ToTensor()

    try:
        # 获取超类数据集
        datasets = get_single_superclass_datasets(
            superclass_name=superclass_name,
            train_transform=train_transform,
            test_transform=test_transform,
            prop_train_labels=0.8,  # 80%有标签样本
            split_train_val=False,
            seed=42
        )

        print(f"\n📊 超类 '{superclass_name}' 数据集统计:")
        for split_name, dataset in datasets.items():
            if dataset is not None:
                print(f"  {split_name}: {len(dataset)} 样本")

                # 显示标签分布
                if hasattr(dataset, 'targets'):
                    unique_labels = set(dataset.targets)
                    print(f"    包含标签: {sorted(unique_labels)}")
            else:
                print(f"  {split_name}: None")

        # 测试数据加载
        if datasets['train_labelled'] is not None:
            sample_img, sample_label, sample_idx = datasets['train_labelled'][0]
            print(f"\n🔍 样本检查:")
            print(f"  图像形状: {sample_img.shape}")
            print(f"  标签: {sample_label}")
            print(f"  唯一索引: {sample_idx}")

    except Exception as e:
        print(f"❌ 加载超类数据集时出错: {e}")
        print("💡 请确保CIFAR-100数据集已下载")


def demo_data_split_usage():
    """演示数据划分使用"""
    print("\n📊 数据划分使用演示")
    print("=" * 80)

    # 检查是否存在数据划分文件
    splits_dir = './data_splits'
    if os.path.exists(splits_dir):
        print(f"✅ 找到数据划分目录: {splits_dir}")

        # 列出所有划分文件
        split_files = [f for f in os.listdir(splits_dir) if f.startswith('superclass_') and f.endswith('.json')]
        print(f"📁 找到 {len(split_files)} 个超类划分文件:")

        for file in sorted(split_files)[:5]:  # 只显示前5个
            print(f"  - {file}")

        if len(split_files) > 5:
            print(f"  ... 还有 {len(split_files) - 5} 个文件")

        # 读取总结文件
        summary_file = os.path.join(splits_dir, 'gcd_superclass_splits_summary.json')
        if os.path.exists(summary_file):
            import json
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            print(f"\n📈 数据划分总结:")
            total_stats = summary['gcd_superclass_splits_summary']['total_statistics']
            print(f"  总训练样本: {total_stats['total_train_samples']}")
            print(f"  总验证样本: {total_stats['total_val_samples']}")
            print(f"  总测试样本: {total_stats['total_test_samples']}")
            print(f"  总计样本: {total_stats['grand_total']}")

    else:
        print(f"❌ 未找到数据划分目录: {splits_dir}")
        print("💡 请先运行 data_split_generator.py 生成数据划分")


def demo_training_commands():
    """演示训练命令"""
    print("\n🚀 训练命令示例")
    print("=" * 80)

    print("1. 训练单个超类:")
    print("   python train_superclass.py --superclass_name mammals --epochs 20")
    print("   python train_superclass.py --superclass_name trees --epochs 20")

    print("\n2. 训练所有超类:")
    print("   python train_superclass.py --train_all_superclasses --epochs 20")

    print("\n3. 自定义参数训练:")
    print("   python train_superclass.py --superclass_name flowers \\")
    print("                               --epochs 50 \\")
    print("                               --batch_size 64 \\")
    print("                               --lr 0.05 \\")
    print("                               --prop_train_labels 0.8")

    print("\n4. 生成数据划分:")
    print("   python data_split_generator.py --output_dir ./data_splits")


def demo_integration_check():
    """演示集成检查"""
    print("\n🔧 集成检查")
    print("=" * 80)

    # 检查必要的文件
    required_files = [
        'data/cifar100_superclass.py',
        'train_superclass.py',
        'data_split_generator.py'
    ]

    print("📋 检查必要文件:")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (缺失)")

    # 检查数据目录
    data_dirs = ['./data', '../data']
    print(f"\n📁 检查CIFAR-100数据目录:")
    found_data = False
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"  ✅ {data_dir}")
            cifar_dir = os.path.join(data_dir, 'cifar-100-python')
            if os.path.exists(cifar_dir):
                print(f"    ✅ CIFAR-100数据已下载")
                found_data = True
            else:
                print(f"    ⚠️  CIFAR-100数据未下载")
        else:
            print(f"  ❌ {data_dir} (不存在)")

    if not found_data:
        print("💡 如果CIFAR-100数据未下载，首次运行时会自动下载")

    # 检查依赖库
    print(f"\n📦 检查关键依赖:")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError:
        print(f"  ❌ PyTorch (未安装)")

    try:
        import torchvision
        print(f"  ✅ torchvision {torchvision.__version__}")
    except ImportError:
        print(f"  ❌ torchvision (未安装)")

    try:
        from sklearn.cluster import KMeans
        print(f"  ✅ scikit-learn")
    except ImportError:
        print(f"  ❌ scikit-learn (未安装)")


def main():
    """主演示函数"""
    print("🌟 CIFAR-100超类功能演示")
    print("基于DCCL项目的15个超类划分，适配GCD项目")
    print("=" * 80)

    try:
        # 1. 超类信息演示
        demo_superclass_info()

        # 2. 数据集加载演示
        demo_single_superclass_dataset()

        # 3. 数据划分使用演示
        demo_data_split_usage()

        # 4. 训练命令演示
        demo_training_commands()

        # 5. 集成检查
        demo_integration_check()

        print("\n🎉 演示完成!")
        print("现在您可以使用超类功能进行GCD训练了。")

    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()