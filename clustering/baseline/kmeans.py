#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-means基线对比模块
提供K-means聚类的基线对比实现
"""

import sys
import os
import numpy as np
from sklearn.cluster import KMeans

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_kmeans_baseline(test_features, test_targets, test_known_mask, n_clusters, random_state=0, eval_version='v1',
                        kmeans_merge=False, train_features=None, train_targets=None, train_known_mask=None):
    """
    K-means基线对比方案

    完全参考eval_original_gcd.py的实现，保持一致性

    新增kmeans_merge功能：当为True时，和自适应方案一样合并训练集和测试集进行计算

    Args:
        test_features: 测试集特征 (n_test_samples, feat_dim) - 应该是L2归一化的
        test_targets: 测试集真实标签
        test_known_mask: 测试集已知类掩码
        n_clusters: 聚类数量
        random_state: 随机种子
        eval_version: 评估版本 (v1 或 v2)
        kmeans_merge: 是否合并训练集和测试集进行计算
        train_features: 训练集特征 (当kmeans_merge=True时需要)
        train_targets: 训练集真实标签 (当kmeans_merge=True时需要)
        train_known_mask: 训练集已知类掩码 (当kmeans_merge=True时需要)

    Returns:
        kmeans_results: 包含各种评估指标的字典
    """

    if kmeans_merge and (train_features is not None):
        print(f"\n🔄 运行K-means基线对比 (合并训练+测试集模式)...")
        # 合并训练集和测试集
        all_features = np.concatenate([train_features, test_features], axis=0)
        all_targets = np.concatenate([train_targets, test_targets], axis=0)
        all_known_mask = np.concatenate([train_known_mask, test_known_mask], axis=0)

        print(f"   合并后总样本数: {len(all_features)} (训练:{len(train_features)} + 测试:{len(test_features)})")
        print(f"   聚类数量: {n_clusters}")
        print(f"   随机种子: {random_state}")

        # 在合并的数据上运行K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        all_predictions = kmeans.fit_predict(all_features)

        # 只在测试集部分计算ACC
        test_start_idx = len(train_features)
        test_predictions = all_predictions[test_start_idx:]

        print(f"   在合并数据上训练，在测试集上评估")

    else:
        print(f"\n🔄 运行K-means基线对比 (仅测试集模式)...")
        print(f"   测试集样本数: {len(test_features)}")
        print(f"   聚类数量: {n_clusters}")
        print(f"   随机种子: {random_state}")

        # 只在测试集上运行K-means (原始行为)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        test_predictions = kmeans.fit_predict(test_features)

    # 计算各种ACC指标 (只在测试集上)
    # 使用指定版本的ACC计算方法
    if eval_version == 'v1':
        from project_utils.cluster_and_log_utils import split_cluster_acc_v1
        all_acc, old_acc, new_acc = split_cluster_acc_v1(test_targets, test_predictions, test_known_mask)
    else:  # v2
        from project_utils.cluster_and_log_utils import split_cluster_acc_v2
        all_acc, old_acc, new_acc = split_cluster_acc_v2(test_targets, test_predictions, test_known_mask)

    merge_mode = "合并模式" if kmeans_merge and train_features is not None else "仅测试集"
    print(f"✅ K-means结果 ({merge_mode}):")
    print(f"   All ACC: {all_acc:.4f}")
    print(f"   Old ACC: {old_acc:.4f}")
    print(f"   New ACC: {new_acc:.4f}")

    result = {
        'method': f'K-means ({merge_mode})',
        'n_clusters': n_clusters,
        'all_acc': all_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'predictions': test_predictions  # 添加聚类预测结果（仅测试集）
    }

    # 如果是合并模式，也返回整个合并数据集的预测结果
    if kmeans_merge and (train_features is not None):
        result['all_predictions'] = all_predictions  # 整个合并数据集的预测结果

    return result
