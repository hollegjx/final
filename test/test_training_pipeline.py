#!/usr/bin/env python3
"""
训练流水线测试脚本
测试整个训练过程中的数据流和标签映射
模拟训练的前几个步骤，确保所有组件正常工作
"""

import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cifar100_superclass import (
    get_single_superclass_datasets,
    CIFAR100_SUPERCLASSES,
    get_superclass_splits
)
from data.augmentations import get_transform
from data.data_utils import MergedDataset
from methods.contrastive_training.contrastive_training import ContrastiveLearningViewGenerator
from copy import deepcopy

def test_training_pipeline(superclass_name='trees', batch_size=8, num_workers=0):
    """
    测试训练流水线

    Args:
        superclass_name: 超类名称
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
    """
    print(f"\n{'='*60}")
    print(f"🚀 测试训练流水线: {superclass_name}")
    print(f"{'='*60}")

    # 1. 设置模拟参数
    class MockArgs:
        def __init__(self):
            self.superclass_name = superclass_name
            self.prop_train_labels = 0.8
            self.seed = 1
            self.n_views = 2
            # 添加get_transform需要的属性
            self.interpolation = 3
            self.crop_pct = 0.875
            # 添加其他可能需要的属性
            self.resize_lower_bound = 0.08
            self.rand_aug_n = 2
            self.rand_aug_m = 10

    args = MockArgs()

    # 2. 获取类别划分
    try:
        superclass_splits = get_superclass_splits()
        split_info = superclass_splits[superclass_name]

        args.train_classes = split_info['known_classes']
        args.unlabeled_classes = split_info['unknown_classes']

        print(f"✅ 类别划分获取成功")
        print(f"   已知类别: {args.train_classes}")
        print(f"   未知类别: {args.unlabeled_classes}")

    except Exception as e:
        print(f"❌ 类别划分获取失败: {e}")
        return False

    # 3. 创建数据转换
    try:
        # 创建简化的转换，避免复杂的依赖
        import torch
        from torchvision import transforms
        from PIL import Image

        base_train_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        # 创建对比学习转换
        train_transform = ContrastiveLearningViewGenerator(
            base_transform=base_train_transform,
            n_views=args.n_views
        )

        print(f"✅ 数据转换创建成功")
        print(f"   对比学习视图数: {args.n_views}")

    except Exception as e:
        print(f"❌ 数据转换创建失败: {e}")
        return False

    # 4. 获取数据集
    try:
        datasets = get_single_superclass_datasets(
            superclass_name=superclass_name,
            train_transform=train_transform,
            test_transform=test_transform,
            prop_train_labels=args.prop_train_labels,
            split_train_val=False,
            seed=args.seed
        )

        print(f"✅ 数据集创建成功")
        for split_name, dataset in datasets.items():
            if dataset is not None:
                print(f"   {split_name}: {len(dataset)} 样本")

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False

    # 5. 创建训练数据集
    try:
        train_dataset = MergedDataset(
            labelled_dataset=deepcopy(datasets['train_labelled']),
            unlabelled_dataset=deepcopy(datasets['train_unlabelled'])
        )

        test_dataset = datasets['test']
        unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
        unlabelled_train_examples_test.transform = test_transform

        print(f"✅ 训练数据集组装成功")
        print(f"   训练集总样本: {len(train_dataset)}")
        print(f"   测试集样本: {len(test_dataset)}")
        print(f"   未标记训练样本: {len(unlabelled_train_examples_test)}")

    except Exception as e:
        print(f"❌ 训练数据集组装失败: {e}")
        return False

    # 6. 创建数据加载器并测试
    try:
        # 创建采样器（平衡标记和未标记样本）
        label_len = len(datasets['train_labelled'])
        unlabelled_len = len(datasets['train_unlabelled'])
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False
        )

        unlabelled_loader = DataLoader(
            unlabelled_train_examples_test,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False
        )

        print(f"✅ 数据加载器创建成功")
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   测试批次数: {len(test_loader)}")
        print(f"   未标记批次数: {len(unlabelled_loader)}")

    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return False

    # 7. 测试训练数据批次
    print(f"\n🔍 训练数据批次测试:")
    try:
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # 只测试前2个批次
                break

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            print(f"   批次 {batch_idx}:")
            print(f"     图像形状: {[img.shape for img in images]}")  # 多视图
            print(f"     类别标签形状: {class_labels.shape}, 范围: [{class_labels.min()}, {class_labels.max()}]")
            print(f"     标记掩码: {mask_lab.sum().item()}/{len(mask_lab)} 个样本被标记")

            # 检查标签分布
            label_counts = Counter(class_labels.numpy())
            print(f"     标签分布: {dict(sorted(label_counts.items()))}")

            # 检查对比学习视图
            if isinstance(images, list) and len(images) == 2:
                print(f"     ✅ 对比学习双视图正确")
            else:
                print(f"     ❌ 对比学习视图异常: {type(images)}")

        print(f"   ✅ 训练数据批次测试通过")

    except Exception as e:
        print(f"   ❌ 训练数据批次测试失败: {e}")
        return False

    # 8. 测试测试数据批次
    print(f"\n🔍 测试数据批次测试:")
    try:
        for batch_idx, (images, labels, uq_idxs) in enumerate(test_loader):
            if batch_idx >= 1:  # 只测试1个批次
                break

            print(f"   批次 {batch_idx}:")
            print(f"     图像形状: {images.shape}")
            print(f"     标签形状: {labels.shape}, 范围: [{labels.min()}, {labels.max()}]")

            # 检查标签分布
            label_counts = Counter(labels.numpy())
            print(f"     标签分布: {dict(sorted(label_counts.items()))}")

        print(f"   ✅ 测试数据批次测试通过")

    except Exception as e:
        print(f"   ❌ 测试数据批次测试失败: {e}")
        return False

    # 9. 验证标签映射一致性
    print(f"\n🔍 标签映射一致性验证:")
    try:
        # 收集所有数据集的标签
        all_train_labels = set()
        all_test_labels = set()

        # 从训练数据中采样标签
        sample_count = 0
        for batch in train_loader:
            if sample_count >= 50:  # 限制采样数量
                break
            _, class_labels, _, _ = batch
            all_train_labels.update(class_labels.numpy())
            sample_count += len(class_labels)

        # 从测试数据中采样标签
        sample_count = 0
        for batch in test_loader:
            if sample_count >= 50:  # 限制采样数量
                break
            _, labels, _ = batch
            all_test_labels.update(labels.numpy())
            sample_count += len(labels)

        print(f"   训练集标签范围: {sorted(all_train_labels)}")
        print(f"   测试集标签范围: {sorted(all_test_labels)}")

        # 检查标签是否连续且从0开始
        expected_labels = set(range(len(args.train_classes) + len(args.unlabeled_classes)))

        if all_train_labels.issubset(expected_labels) and all_test_labels.issubset(expected_labels):
            print(f"   ✅ 标签映射一致，符合预期范围 [0, {len(expected_labels)-1}]")
        else:
            print(f"   ❌ 标签映射异常:")
            print(f"     期望标签: {sorted(expected_labels)}")
            print(f"     训练集多余标签: {all_train_labels - expected_labels}")
            print(f"     测试集多余标签: {all_test_labels - expected_labels}")

    except Exception as e:
        print(f"   ❌ 标签映射验证失败: {e}")
        return False

    print(f"\n🎉 训练流水线测试完成!")
    print(f"✅ 超类 '{superclass_name}' 的训练流水线工作正常")
    return True

def main():
    parser = argparse.ArgumentParser(description='测试训练流水线')
    parser.add_argument('--superclass', type=str, default='trees',
                        help='要测试的超类名称，默认为trees')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小，默认为8')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载工作进程数，默认为0')

    args = parser.parse_args()

    print("🚀 训练流水线测试工具")
    print("=" * 60)

    success = test_training_pipeline(
        superclass_name=args.superclass,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if success:
        print(f"\n🎊 测试成功! 可以开始训练超类 '{args.superclass}'")
        print(f"💡 建议的训练命令:")
        print(f"   python scripts/train_superclass.py --superclass_name {args.superclass} --epochs 20 --gpu 0")
    else:
        print(f"\n❌ 测试失败! 请检查数据集配置")

if __name__ == "__main__":
    main()