#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集加载模块
处理CIFAR-100超类数据集的加载
注意：此模块需要依赖data/目录（无法完全避免）
"""

import os
import sys
from torch.utils.data import DataLoader
from copy import deepcopy

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from data.augmentations import get_transform
    from data.get_datasets import get_single_superclass_datasets, MergedDataset
except ImportError:
    print("⚠️  警告: 无法导入data模块")
    print("   数据集加载功能需要项目根目录的data/目录")
    get_transform = None
    get_single_superclass_datasets = None
    MergedDataset = None


class DatasetLoader:
    """
    数据集加载器类
    负责加载CIFAR-100超类数据集并创建DataLoader
    """

    def __init__(self, superclass_name, image_size=224, batch_size=64,
                 num_workers=4, prop_train_labels=0.8, seed=0):
        """
        初始化数据集加载器

        Args:
            superclass_name: 超类名称
            image_size: 图像尺寸（默认224）
            batch_size: 批次大小（默认64）
            num_workers: 数据加载线程数（默认4）
            prop_train_labels: 训练集中有标签样本的比例（默认0.8）
            seed: 随机种子（默认0）
        """
        self.superclass_name = superclass_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prop_train_labels = prop_train_labels
        self.seed = seed

    def load(self, silent=False):
        """
        加载数据集并创建DataLoader

        Args:
            silent: 是否静默模式

        Returns:
            dict: 包含数据加载器的字典
                - 'train_loader': 训练集加载器（合并有标签+无标签）
                - 'test_loader': 测试集加载器
                - 'train_dataset': 训练集数据集
                - 'test_dataset': 测试集数据集

        Raises:
            ImportError: 如果data模块不可用
        """
        if get_transform is None or get_single_superclass_datasets is None:
            raise ImportError("data模块不可用，无法加载数据集")

        if not silent:
            print(f"🔄 加载数据集: {self.superclass_name}")

        # 创建数据转换参数（模拟args）
        class Args:
            def __init__(self, image_size, prop_train_labels, seed):
                self.image_size = image_size
                self.interpolation = 3
                self.crop_pct = 0.875
                self.prop_train_labels = prop_train_labels
                self.seed = seed

        args = Args(self.image_size, self.prop_train_labels, self.seed)

        # 获取数据转换
        train_transform, test_transform = get_transform(
            'imagenet',
            image_size=self.image_size,
            args=args
        )

        # 获取数据集
        datasets = get_single_superclass_datasets(
            superclass_name=self.superclass_name,
            train_transform=train_transform,
            test_transform=test_transform,
            prop_train_labels=self.prop_train_labels,
            split_train_val=False,
            seed=self.seed
        )

        # 创建训练集（合并有标签和无标签）
        train_labelled = datasets['train_labelled']
        train_unlabelled = datasets['train_unlabelled']
        test_dataset = datasets['test']

        merged_train_dataset = MergedDataset(
            labelled_dataset=deepcopy(train_labelled),
            unlabelled_dataset=deepcopy(train_unlabelled)
        )

        # 创建数据加载器
        train_loader = DataLoader(
            merged_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        if not silent:
            print(f"✅ 数据集加载成功")
            print(f"   训练集: {len(merged_train_dataset)} 样本")
            print(f"   测试集: {len(test_dataset)} 样本")

        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'train_dataset': merged_train_dataset,
            'test_dataset': test_dataset
        }
