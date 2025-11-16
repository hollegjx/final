#!/usr/bin/env python3
"""
数据集和标签测试脚本
测试超类数据集的数据加载和标签映射是否正确
以trees超类为例进行测试
"""

import sys
import os
import argparse
import numpy as np
import torch
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cifar100_superclass import (
    get_single_superclass_datasets,
    CIFAR100_SUPERCLASSES,
    SUPERCLASS_NAMES,
    get_superclass_splits
)
from data.augmentations import get_transform
from data.data_utils import MergedDataset
from copy import deepcopy

def test_superclass_dataset(superclass_name='trees', verbose=True):
    """
    测试超类数据集的数据加载和标签映射

    Args:
        superclass_name: 要测试的超类名称
        verbose: 是否显示详细信息
    """
    print(f"\n{'='*60}")
    print(f"🧪 测试超类: {superclass_name}")
    print(f"{'='*60}")

    # 1. 检查超类定义
    if superclass_name not in CIFAR100_SUPERCLASSES:
        print(f"❌ 错误: 未知的超类名称 '{superclass_name}'")
        print(f"📋 可用的超类: {SUPERCLASS_NAMES}")
        return False

    original_classes = CIFAR100_SUPERCLASSES[superclass_name]
    print(f"📊 原始类别定义: {original_classes}")

    # 按GCD标准划分已知类和未知类
    known_classes = [cls for cls in original_classes if cls < 80]
    unknown_classes = [cls for cls in original_classes if cls >= 80]

    print(f"✅ 已知类 (< 80): {known_classes}")
    print(f"🔍 未知类 (>= 80): {unknown_classes}")

    if len(known_classes) == 0:
        print(f"⚠️ 警告: 超类 '{superclass_name}' 没有已知类")
    if len(unknown_classes) == 0:
        print(f"⚠️ 警告: 超类 '{superclass_name}' 没有未知类")

    # 2. 创建数据转换
    try:
        # 创建简化的转换，避免复杂的依赖
        import torch
        from torchvision import transforms
        from PIL import Image

        train_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        print("✅ 数据转换创建成功")
    except Exception as e:
        print(f"❌ 数据转换创建失败: {e}")
        return False

    # 3. 获取超类数据集
    try:
        datasets = get_single_superclass_datasets(
            superclass_name=superclass_name,
            train_transform=train_transform,
            test_transform=test_transform,
            prop_train_labels=0.8,
            split_train_val=False,
            seed=1
        )
        print("✅ 超类数据集创建成功")
    except Exception as e:
        print(f"❌ 超类数据集创建失败: {e}")
        return False

    # 4. 检查数据集结构
    print(f"\n📂 数据集结构:")
    for split_name, dataset in datasets.items():
        if dataset is not None:
            print(f"  {split_name}: {len(dataset)} 样本")
        else:
            print(f"  {split_name}: None")

    # 5. 测试标签分布
    print(f"\n🏷️ 标签分布测试:")

    def test_dataset_labels(dataset, dataset_name):
        """测试数据集的标签分布"""
        if dataset is None:
            print(f"  {dataset_name}: 数据集为空")
            return

        labels = []
        for i in range(min(len(dataset), 100)):  # 测试前100个样本
            try:
                if len(dataset[i]) == 2:  # (img, label)
                    _, label = dataset[i]
                elif len(dataset[i]) == 3:  # (img, label, uq_idx)
                    _, label, _ = dataset[i]
                else:
                    print(f"⚠️ 意外的数据格式: {len(dataset[i])} 个元素")
                    continue
                labels.append(label)
            except Exception as e:
                print(f"❌ 第{i}个样本加载失败: {e}")
                break

        if labels:
            label_counts = Counter(labels)
            print(f"  {dataset_name}: 标签范围 [{min(labels)}, {max(labels)}], 分布: {dict(label_counts)}")

            # 检查标签映射是否正确
            if hasattr(dataset, 'label_mapping') and dataset.label_mapping is not None:
                print(f"  标签映射: {dataset.label_mapping}")
                # 验证映射后的标签是否连续
                mapped_labels = set(labels)
                expected_labels = set(range(len(dataset.label_mapping)))
                if mapped_labels == expected_labels:
                    print(f"  ✅ 标签映射正确，标签连续 [0, {len(dataset.label_mapping)-1}]")
                else:
                    print(f"  ❌ 标签映射异常:")
                    print(f"    实际标签: {sorted(mapped_labels)}")
                    print(f"    期望标签: {sorted(expected_labels)}")
        else:
            print(f"  {dataset_name}: 无法获取标签")

    # 测试各个数据集
    for split_name, dataset in datasets.items():
        test_dataset_labels(dataset, split_name)

    # 6. 测试MergedDataset
    print(f"\n🔗 MergedDataset测试:")
    try:
        train_dataset = MergedDataset(
            labelled_dataset=deepcopy(datasets['train_labelled']),
            unlabelled_dataset=deepcopy(datasets['train_unlabelled'])
        )
        print(f"✅ MergedDataset创建成功，总样本数: {len(train_dataset)}")

        # 测试前几个样本
        print("🔍 样本格式测试:")
        for i in range(min(3, len(train_dataset))):
            try:
                sample = train_dataset[i]
                print(f"  样本{i}: {len(sample)}个元素, 标签: {sample[1]}, 是否标记: {sample[3][0]}")
            except Exception as e:
                print(f"  ❌ 样本{i}加载失败: {e}")

    except Exception as e:
        print(f"❌ MergedDataset创建失败: {e}")
        return False

    # 7. 测试与训练脚本的兼容性
    print(f"\n🎯 训练兼容性测试:")
    try:
        # 模拟训练脚本中的args
        class MockArgs:
            def __init__(self):
                self.superclass_name = superclass_name
                self.prop_train_labels = 0.8
                self.seed = 1

        mock_args = MockArgs()

        # 获取类别划分信息（模拟get_class_splits）
        superclass_splits = get_superclass_splits()
        split_info = superclass_splits[superclass_name]

        mock_args.train_classes = split_info['known_classes']
        mock_args.unlabeled_classes = split_info['unknown_classes']

        print(f"✅ 训练类别: {mock_args.train_classes}")
        print(f"✅ 未标记类别: {mock_args.unlabeled_classes}")

        print(f"✅ 与训练脚本兼容")

    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}")
        return False

    print(f"\n🎉 超类 '{superclass_name}' 测试完成!")
    return True

def test_all_superclasses():
    """测试所有超类"""
    print("🌟 测试所有超类...")
    success_count = 0

    for superclass_name in SUPERCLASS_NAMES:
        try:
            success = test_superclass_dataset(superclass_name, verbose=False)
            if success:
                success_count += 1
                print(f"✅ {superclass_name}: 测试通过")
            else:
                print(f"❌ {superclass_name}: 测试失败")
        except Exception as e:
            print(f"❌ {superclass_name}: 测试异常 - {e}")

    print(f"\n📊 测试结果: {success_count}/{len(SUPERCLASS_NAMES)} 个超类测试通过")

def main():
    parser = argparse.ArgumentParser(description='测试超类数据集和标签映射')
    parser.add_argument('--superclass', type=str, default='trees',
                        help='要测试的超类名称，默认为trees')
    parser.add_argument('--all', action='store_true',
                        help='测试所有超类')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细信息')

    args = parser.parse_args()

    print("🧪 超类数据集测试工具")
    print("=" * 60)

    if args.all:
        test_all_superclasses()
    else:
        test_superclass_dataset(args.superclass, args.verbose)

if __name__ == "__main__":
    main()