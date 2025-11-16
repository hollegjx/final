#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据提供器模块
统一的数据获取接口，优先使用缓存，缓存不存在时使用实时提取
"""

from config import feature_cache_dir
from .feature_loader import FeatureLoader
from .feature_extractor import FeatureExtractor
from .dataset_config import get_superclass_info


class DataProvider:
    """
    数据提供器类
    统一管理特征数据的获取，自动选择缓存或实时提取
    """

    def __init__(self, cache_base_dir=None):
        """
        初始化数据提供器

        Args:
            cache_base_dir: 缓存基础目录
        """
        cache_dir = cache_base_dir or feature_cache_dir
        self.feature_loader = FeatureLoader(cache_base_dir=cache_dir)

    def get_features(self, dataset_name, model=None, data_loaders=None,
                    use_l2=True, use_train_and_test=True, silent=False):
        """
        获取特征数据（优先使用缓存，缓存不存在时实时提取）

        Args:
            dataset_name: 数据集名称（如超类名称）
            model: PyTorch模型（缓存不存在时需要）
            data_loaders: 数据加载器字典或元组（缓存不存在时需要）
                - 字典格式: {'train': train_loader, 'test': test_loader}
                - 元组格式: (train_loader, test_loader)
            use_l2: 是否使用L2归一化特征
            use_train_and_test: 是否使用训练+测试（False则只用测试集）
            silent: 是否静默模式

        Returns:
            feature_dict: 特征数据字典
            source: 数据来源 ('cache' or 'extraction')
        """
        # 1. 优先尝试加载缓存
        feature_dict = self.feature_loader.load(dataset_name, use_l2=use_l2, silent=silent)

        if feature_dict is not None:
            return feature_dict, 'cache'

        # 2. 缓存不存在，进行实时特征提取
        if model is None or data_loaders is None:
            error_msg = "缓存不存在且未提供模型或数据加载器，无法提取特征"
            if not silent:
                print(f"❌ {error_msg}")
            raise ValueError(error_msg)

        if not silent:
            print("🔄 缓存不存在，开始实时特征提取...")

        # 创建特征提取器
        device = next(model.parameters()).device  # 从模型获取设备
        extractor = FeatureExtractor(model=model, device=device, use_l2=use_l2)

        # 解析data_loaders
        if isinstance(data_loaders, dict):
            train_loader = data_loaders.get('train')
            test_loader = data_loaders.get('test')
        elif isinstance(data_loaders, (tuple, list)):
            if len(data_loaders) == 2:
                train_loader, test_loader = data_loaders
            else:
                train_loader = None
                test_loader = data_loaders[0]
        else:
            train_loader = None
            test_loader = data_loaders

        # 获取已知类别信息（用于创建known_mask）
        try:
            superclass_info = get_superclass_info(dataset_name)
            known_classes = superclass_info['known_classes_mapped']
        except:
            # 如果不是超类数据集，使用默认规则
            known_classes = None

        # 提取特征
        if use_train_and_test and train_loader is not None and test_loader is not None:
            feature_dict = extractor.extract_train_test(
                train_loader, test_loader, known_classes=known_classes, silent=silent
            )
        else:
            # 只使用测试集
            loader = test_loader if test_loader is not None else train_loader
            feature_dict = extractor.extract_single_dataset(
                loader, known_classes=known_classes, silent=silent
            )

        return feature_dict, 'extraction'

    def check_cache_available(self, dataset_name, use_l2=True):
        """
        检查缓存是否可用

        Args:
            dataset_name: 数据集名称
            use_l2: 是否L2归一化

        Returns:
            bool: 缓存是否存在
        """
        return self.feature_loader.check_cache_exists(dataset_name, use_l2=use_l2)

    def get_cache_info(self, dataset_name, use_l2=True):
        """
        获取缓存信息

        Args:
            dataset_name: 数据集名称
            use_l2: 是否L2归一化

        Returns:
            dict: 缓存信息字典
                - 'exists': 是否存在
                - 'path': 缓存路径
        """
        exists = self.feature_loader.check_cache_exists(dataset_name, use_l2=use_l2)
        path = self.feature_loader.get_cache_path(dataset_name, use_l2=use_l2)

        return {
            'exists': exists,
            'path': path
        }
