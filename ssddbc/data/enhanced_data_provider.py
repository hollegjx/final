#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强型数据提供器
提供功能更全面、效率更高的数据读取方案
包含所有需要的信息，避免后续重复计算
"""

import numpy as np
from config import feature_cache_dir
from .data_provider import DataProvider
from .dataset_config import get_superclass_info
from .model_loader import ModelLoader
from .dataset_loader import DatasetLoader


class EnhancedDataset:
    """
    增强型数据集类
    包含所有聚类所需的信息，避免重复计算
    """

    def __init__(self, feature_dict, dataset_name='unknown', use_l2=True, source='unknown'):
        """
        初始化增强型数据集

        Args:
            feature_dict: 特征数据字典
            dataset_name: 数据集名称
            use_l2: 是否使用L2归一化
            source: 数据来源 ('cache' or 'extraction')
        """
        # 基础数据
        self.all_features = feature_dict['all_features']
        self.all_targets = feature_dict['all_targets']
        self.all_known_mask = feature_dict['all_known_mask']
        self.all_labeled_mask = feature_dict['all_labeled_mask']

        self.train_features = feature_dict.get('train_features')
        self.train_targets = feature_dict.get('train_targets')
        self.train_known_mask = feature_dict.get('train_known_mask')
        self.train_labeled_mask = feature_dict.get('train_labeled_mask')

        self.test_features = feature_dict.get('test_features')
        self.test_targets = feature_dict.get('test_targets')
        self.test_known_mask = feature_dict.get('test_known_mask')
        self.test_labeled_mask = feature_dict.get('test_labeled_mask')

        # 元信息
        self.dataset_name = dataset_name
        self.use_l2 = use_l2
        self.source = source  # 'cache' or 'extraction'

        # 预计算的信息（避免重复计算）
        self.n_samples = len(self.all_features)
        self.feat_dim = self.all_features.shape[1] if len(self.all_features.shape) > 1 else self.all_features.shape[0]

        # 训练集/测试集信息
        if self.train_features is not None:
            self.train_size = len(self.train_features)
            self.test_size = len(self.test_features) if self.test_features is not None else 0
            self.test_start_idx = self.train_size  # 测试集在合并数据中的起始索引
            self.has_train_test_split = True
        else:
            self.train_size = 0
            self.test_size = self.n_samples
            self.test_start_idx = 0
            self.has_train_test_split = False

        # 统计信息
        self.n_known = np.sum(self.all_known_mask)
        self.n_unknown = np.sum(~self.all_known_mask)
        self.n_labeled = np.sum(self.all_labeled_mask)
        self.n_unlabeled = np.sum(~self.all_labeled_mask)

        # 类别信息
        self.n_classes = len(np.unique(self.all_targets))
        self.n_known_classes = len(np.unique(self.all_targets[self.all_known_mask]))
        self.n_unknown_classes = len(np.unique(self.all_targets[~self.all_known_mask]))

    def get_test_subset(self, predictions=None):
        """
        获取测试集子集（用于ACC计算）

        Args:
            predictions: 全局预测结果（可选）

        Returns:
            dict: 测试集数据字典
        """
        if not self.has_train_test_split:
            # 没有训练/测试划分，返回全部数据
            result = {
                'features': self.all_features,
                'targets': self.all_targets,
                'known_mask': self.all_known_mask,
                'labeled_mask': self.all_labeled_mask,
                'n_samples': self.n_samples
            }
            if predictions is not None:
                result['predictions'] = predictions
        else:
            # 有训练/测试划分，只返回测试集
            result = {
                'features': self.test_features,
                'targets': self.test_targets,
                'known_mask': self.test_known_mask,
                'labeled_mask': self.test_labeled_mask,
                'n_samples': self.test_size
            }
            if predictions is not None:
                result['predictions'] = predictions[self.test_start_idx:]

        return result

    def get_clustering_input(self):
        """
        获取聚类算法所需的输入数据

        Returns:
            tuple: (X, targets, known_mask, labeled_mask, train_size)
        """
        train_size = self.train_size if self.has_train_test_split else None
        return (
            self.all_features,
            self.all_targets,
            self.all_known_mask,
            self.all_labeled_mask,
            train_size
        )

    def print_summary(self, silent=False):
        """
        打印数据集摘要信息

        Args:
            silent: 是否静默模式
        """
        if silent:
            return

        print(f"📊 数据集信息:")
        print(f"   名称: {self.dataset_name}")
        print(f"   数据来源: {self.source}")
        print(f"   L2归一化: {'是' if self.use_l2 else '否'}")
        print(f"   特征维度: {self.feat_dim}")
        print(f"\n📊 样本统计:")
        print(f"   总样本数: {self.n_samples}")
        if self.has_train_test_split:
            print(f"   训练集: {self.train_size} 样本")
            print(f"   测试集: {self.test_size} 样本")
        print(f"   已知类样本: {self.n_known} ({self.n_known/self.n_samples*100:.1f}%)")
        print(f"   未知类样本: {self.n_unknown} ({self.n_unknown/self.n_samples*100:.1f}%)")
        print(f"   有标签样本: {self.n_labeled} ({self.n_labeled/self.n_samples*100:.1f}%)")
        print(f"   无标签样本: {self.n_unlabeled} ({self.n_unlabeled/self.n_samples*100:.1f}%)")
        print(f"\n📊 类别统计:")
        print(f"   总类别数: {self.n_classes}")
        print(f"   已知类别数: {self.n_known_classes}")
        print(f"   未知类别数: {self.n_unknown_classes}")


class EnhancedDataProvider:
    """
    增强型数据提供器
    在原有DataProvider基础上，返回EnhancedDataset对象
    """

    def __init__(self, cache_base_dir=None):
        """
        初始化增强型数据提供器

        Args:
            cache_base_dir: 缓存基础目录
        """
        cache_dir = cache_base_dir or feature_cache_dir
        self.data_provider = DataProvider(cache_base_dir=cache_dir)
        self.cache_base_dir = cache_dir

    def load_dataset(self, dataset_name, model_path=None, use_l2=True,
                    use_train_and_test=True, silent=False):
        """
        加载数据集（优先使用缓存）

        Args:
            dataset_name: 数据集名称
            model_path: 模型路径（缓存不存在时需要）
            use_l2: 是否使用L2归一化
            use_train_and_test: 是否使用训练+测试集
            silent: 是否静默模式

        Returns:
            EnhancedDataset: 增强型数据集对象
        """
        # 尝试从缓存加载
        feature_dict = self.data_provider.feature_loader.load(
            dataset_name, use_l2=use_l2, silent=silent
        )

        if feature_dict is not None:
            # 缓存命中
            if not silent:
                print(f"✅ 使用缓存特征")
            source = 'cache'
        else:
            # 缓存未命中，需要实时提取
            if model_path is None:
                raise ValueError("缓存不存在且未提供模型路径，无法提取特征")

            if not silent:
                print(f"⚠️  缓存不存在，开始实时特征提取...")

            # 加载模型
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model_loader = ModelLoader(
                model_path=model_path,
                base_model='vit_dino',
                feat_dim=768,
                device=device
            )
            model = model_loader.load(silent=silent)

            # 加载数据集
            dataset_loader = DatasetLoader(
                superclass_name=dataset_name,
                image_size=224,
                batch_size=64,
                prop_train_labels=0.8,
                seed=0
            )
            data_loaders = dataset_loader.load(silent=silent)

            # 提取特征
            feature_dict, source = self.data_provider.get_features(
                dataset_name=dataset_name,
                model=model,
                data_loaders=(data_loaders['train_loader'], data_loaders['test_loader']),
                use_l2=use_l2,
                use_train_and_test=use_train_and_test,
                silent=silent
            )

        # 创建增强型数据集对象
        enhanced_dataset = EnhancedDataset(
            feature_dict=feature_dict,
            dataset_name=dataset_name,
            use_l2=use_l2,
            source=source
        )

        return enhanced_dataset
