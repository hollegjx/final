#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特征提取模块
负责使用模型从数据集中提取特征
尽量减少对外部训练代码的依赖
"""

import torch
import numpy as np
from tqdm import tqdm


class FeatureExtractor:
    """
    特征提取器类
    使用给定的模型从数据加载器中提取特征
    """

    def __init__(self, model, device='cuda', use_l2=True):
        """
        初始化特征提取器

        Args:
            model: PyTorch模型（已加载权重）
            device: 计算设备 ('cuda' or 'cpu')
            use_l2: 是否对特征进行L2归一化
        """
        self.model = model
        self.device = device
        self.use_l2 = use_l2

        # 将模型设置为评估模式
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_from_loader(self, data_loader, known_classes=None, silent=False):
        """
        从数据加载器中提取特征

        Args:
            data_loader: PyTorch数据加载器
            known_classes: 已知类别的集合（用于创建known_mask）
            silent: 是否静默模式

        Returns:
            features: 特征矩阵 (n_samples, feat_dim)
            targets: 真实标签 (n_samples,)
            known_mask: 已知类掩码 (n_samples,)
            labeled_mask: 有标签样本掩码 (n_samples,)
        """
        l2_status = "L2归一化" if self.use_l2 else "无L2归一化"
        if not silent:
            print(f"🔄 提取特征 ({l2_status})...")

        all_feats = []
        all_targets = []
        all_labeled_mask = []

        with torch.no_grad():
            iterator = tqdm(data_loader, desc="提取特征") if not silent else data_loader

            for batch_data in iterator:
                # 解包数据（支持两种格式）
                if len(batch_data) == 4:
                    # 训练集格式: (images, labels, indices, labeled_or_not)
                    images, labels, indices, labeled_or_not = batch_data
                    labeled_batch = labeled_or_not.numpy().flatten().astype(bool)
                elif len(batch_data) == 3:
                    # 测试集格式: (images, labels, indices)
                    images, labels, indices = batch_data
                    # 测试集默认标记为无标签
                    labeled_batch = np.zeros(len(labels), dtype=bool)
                else:
                    continue

                # 提取特征
                images = images.to(self.device)
                feats = self.model(images)

                # L2归一化（如果需要）
                if self.use_l2:
                    feats = torch.nn.functional.normalize(feats, dim=-1)

                # 收集数据
                all_feats.append(feats.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                all_labeled_mask.append(labeled_batch)

                # 清理GPU内存
                del images, feats
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 拼接所有批次
        features = np.concatenate(all_feats, axis=0)
        targets = np.concatenate(all_targets, axis=0).astype(int)
        labeled_mask = np.concatenate(all_labeled_mask, axis=0).astype(bool)

        # 创建已知类掩码
        if known_classes is not None:
            known_mask = np.array([label in known_classes for label in targets], dtype=bool)
        else:
            # 默认：前80个类别是已知类（CIFAR-100标准）
            known_mask = (targets < 80).astype(bool)

        if not silent:
            print(f"✅ 特征提取完成: {features.shape}")

        return features, targets, known_mask, labeled_mask

    def extract_train_test(self, train_loader, test_loader, known_classes=None, silent=False):
        """
        分别提取训练集和测试集特征，然后合并

        Args:
            train_loader: 训练集数据加载器
            test_loader: 测试集数据加载器
            known_classes: 已知类别集合
            silent: 是否静默模式

        Returns:
            feature_dict: 包含所有特征数据的字典
        """
        if not silent:
            print("📊 提取训练集特征...")
        train_feats, train_targets, train_known_mask, train_labeled_mask = \
            self.extract_from_loader(train_loader, known_classes, silent)

        if not silent:
            print("📊 提取测试集特征...")
        test_feats, test_targets, test_known_mask, test_labeled_mask = \
            self.extract_from_loader(test_loader, known_classes, silent)

        # 合并训练集和测试集
        all_feats = np.concatenate([train_feats, test_feats], axis=0)
        all_targets = np.concatenate([train_targets, test_targets], axis=0)
        all_known_mask = np.concatenate([train_known_mask, test_known_mask], axis=0)
        all_labeled_mask = np.concatenate([train_labeled_mask, test_labeled_mask], axis=0)

        # 构建特征字典（与缓存格式一致）
        feature_dict = {
            'all_features': all_feats,
            'all_targets': all_targets,
            'all_known_mask': all_known_mask,
            'all_labeled_mask': all_labeled_mask,
            'train_features': train_feats,
            'train_targets': train_targets,
            'train_known_mask': train_known_mask,
            'train_labeled_mask': train_labeled_mask,
            'test_features': test_feats,
            'test_targets': test_targets,
            'test_known_mask': test_known_mask,
            'test_labeled_mask': test_labeled_mask
        }

        if not silent:
            print(f"✅ 特征提取完成")
            print(f"   总样本数: {len(all_feats)}")
            print(f"   训练集: {len(train_feats)} 样本")
            print(f"   测试集: {len(test_feats)} 样本")

        return feature_dict

    def extract_single_dataset(self, data_loader, known_classes=None, silent=False):
        """
        提取单个数据集的特征（如仅测试集）

        Args:
            data_loader: 数据加载器
            known_classes: 已知类别集合
            silent: 是否静默模式

        Returns:
            feature_dict: 特征字典（仅包含单个数据集）
        """
        feats, targets, known_mask, labeled_mask = \
            self.extract_from_loader(data_loader, known_classes, silent)

        # 构建特征字典
        feature_dict = {
            'all_features': feats,
            'all_targets': targets,
            'all_known_mask': known_mask,
            'all_labeled_mask': labeled_mask,
            'train_features': None,  # 单数据集模式下训练集为None
            'train_targets': None,
            'train_known_mask': None,
            'train_labeled_mask': None,
            'test_features': feats,
            'test_targets': targets,
            'test_known_mask': known_mask,
            'test_labeled_mask': labeled_mask
        }

        return feature_dict
