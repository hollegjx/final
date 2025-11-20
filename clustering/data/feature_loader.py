#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特征缓存加载模块
负责从缓存文件中加载预提取的特征
"""

import os
import pickle
import numpy as np

from config import feature_cache_dir, feature_cache_dir_nol2


class FeatureLoader:
    """
    特征加载器类
    负责从磁盘缓存加载预先提取好的特征数据
    """

    def __init__(self, cache_base_dir=None, cache_base_dir_nol2=None):
        """
        初始化特征加载器

        Args:
            cache_base_dir: 缓存文件的基础目录
        """
        self.cache_base_dir = cache_base_dir or feature_cache_dir
        if cache_base_dir_nol2 is not None:
            self.cache_base_dir_nol2 = cache_base_dir_nol2
        elif cache_base_dir:
            self.cache_base_dir_nol2 = cache_base_dir.replace('features', 'features_nol2')
        else:
            self.cache_base_dir_nol2 = feature_cache_dir_nol2

    def load(self, dataset_name, use_l2=True, silent=False):
        """
        加载指定数据集的特征缓存

        Args:
            dataset_name: 数据集名称（如超类名称）
            use_l2: 是否使用L2归一化的特征
            silent: 是否静默模式

        Returns:
            feature_dict: 特征数据字典，包含:
                - 'all_features': 所有样本特征 (n_samples, feat_dim)
                - 'all_targets': 所有样本标签 (n_samples,)
                - 'all_known_mask': 已知类掩码 (n_samples,)
                - 'all_labeled_mask': 有标签样本掩码 (n_samples,)
                - 'train_features': 训练集特征
                - 'train_targets': 训练集标签
                - 'train_known_mask': 训练集已知类掩码
                - 'train_labeled_mask': 训练集有标签掩码
                - 'test_features': 测试集特征
                - 'test_targets': 测试集标签
                - 'test_known_mask': 测试集已知类掩码
                - 'test_labeled_mask': 测试集有标签掩码
            如果加载失败返回None
        """
        # 根据use_l2选择缓存目录
        cache_dir = self.cache_base_dir if use_l2 else self.cache_base_dir_nol2
        l2_status = "L2归一化" if use_l2 else "无L2归一化"

        # 构建缓存文件路径
        cache_file = os.path.join(cache_dir, dataset_name, 'features.pkl')

        # 检查缓存文件是否存在
        if not os.path.exists(cache_file):
            if not silent:
                print(f"🔍 未发现缓存特征文件 ({l2_status}): {cache_file}")
            return None

        # 尝试加载缓存
        try:
            if not silent:
                print(f"🔍 发现缓存特征文件 ({l2_status}): {cache_file}")
                print(f"📥 正在加载缓存特征...")

            with open(cache_file, 'rb') as f:
                feature_dict = pickle.load(f)

            # 验证数据完整性
            self._validate_feature_dict(feature_dict)

            if not silent:
                print(f"✅ 缓存特征加载成功 ({l2_status})")
                print(f"   特征形状: {feature_dict['all_features'].shape}")
                print(f"   样本总数: {len(feature_dict['all_features'])}")
                print(f"   训练集: {len(feature_dict['train_features'])} 样本")
                print(f"   测试集: {len(feature_dict['test_features'])} 样本")

            return feature_dict

        except Exception as e:
            if not silent:
                print(f"⚠️  缓存文件损坏或格式不兼容: {e}")
            return None

    def _validate_feature_dict(self, feature_dict):
        """
        验证特征字典的完整性

        Args:
            feature_dict: 特征数据字典

        Raises:
            ValueError: 如果数据格式不正确
        """
        required_keys = [
            'all_features', 'all_targets', 'all_known_mask', 'all_labeled_mask',
            'train_features', 'train_targets', 'train_known_mask', 'train_labeled_mask',
            'test_features', 'test_targets', 'test_known_mask', 'test_labeled_mask'
        ]

        for key in required_keys:
            if key not in feature_dict:
                raise ValueError(f"缓存数据缺少必需字段: {key}")

        # 验证数据一致性
        n_all = len(feature_dict['all_features'])
        n_train = len(feature_dict['train_features'])
        n_test = len(feature_dict['test_features'])

        if n_all != n_train + n_test:
            raise ValueError(f"数据不一致: all({n_all}) != train({n_train}) + test({n_test})")

    def check_cache_exists(self, dataset_name, use_l2=True):
        """
        检查缓存文件是否存在（不加载）

        Args:
            dataset_name: 数据集名称
            use_l2: 是否L2归一化

        Returns:
            bool: 缓存是否存在
        """
        cache_dir = self.cache_base_dir if use_l2 else self.cache_base_dir_nol2
        cache_file = os.path.join(cache_dir, dataset_name, 'features.pkl')
        return os.path.exists(cache_file)

    def get_cache_path(self, dataset_name, use_l2=True):
        """
        获取缓存文件的完整路径

        Args:
            dataset_name: 数据集名称
            use_l2: 是否L2归一化

        Returns:
            str: 缓存文件路径
        """
        cache_dir = self.cache_base_dir if use_l2 else self.cache_base_dir_nol2
        return os.path.join(cache_dir, dataset_name, 'features.pkl')
