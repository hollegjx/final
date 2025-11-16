#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
有标签样本ACC计算模块
专门用于计算有标签样本的分配准确率，考虑unknown_clusters的惩罚
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_labeled_acc_with_unknown_penalty(
    predictions, targets, labeled_mask, unknown_clusters, silent=False
):
    """
    计算有标签样本的ACC，考虑unknown_clusters的惩罚

    逻辑：
    1. 筛选出所有有标签的样本（训练集中的已知类样本）
    2. 被分配到unknown_clusters的样本：直接算错（因为有标签样本不应该被分到未知类簇）
    3. 被分配到其他簇的样本：用匈牙利算法找最优匹配
    4. 计算总ACC = 正确样本数 / 总有标签样本数

    Args:
        predictions: 全部样本的簇预测 (n_samples,)，簇ID
        targets: 全部样本的真实标签 (n_samples,)
        labeled_mask: 有标签样本掩码 (n_samples,)，True表示该样本在训练时有标签
        unknown_clusters: 被识别为未知类的簇ID集合 (set or list)
        silent: 是否静默模式

    Returns:
        labeled_acc: 有标签样本的准确率 [0, 1]
        metrics: 详细指标字典
    """
    # 筛选有标签样本
    labeled_predictions = predictions[labeled_mask]
    labeled_targets = targets[labeled_mask]
    n_labeled = len(labeled_predictions)

    if n_labeled == 0:
        if not silent:
            print("⚠️  警告: 没有有标签样本，无法计算labeled_acc")
        return 0.0, {
            'n_labeled': 0,
            'n_assigned_to_unknown': 0,
            'n_assigned_to_known': 0,
            'n_correct': 0,
            'accuracy': 0.0
        }

    # 将unknown_clusters转为set方便查找
    unknown_clusters_set = set(unknown_clusters) if unknown_clusters else set()

    # 分离被分配到unknown和known簇的样本
    assigned_to_unknown_mask = np.isin(labeled_predictions, list(unknown_clusters_set))
    n_assigned_to_unknown = assigned_to_unknown_mask.sum()
    n_assigned_to_known = n_labeled - n_assigned_to_unknown

    # 被分配到unknown_clusters的样本都算错（有标签样本不应该在未知类簇中）
    n_correct_from_unknown = 0

    # 对被分配到known簇的样本使用匈牙利算法计算ACC
    if n_assigned_to_known > 0:
        known_predictions = labeled_predictions[~assigned_to_unknown_mask]
        known_targets = labeled_targets[~assigned_to_unknown_mask]

        unique_clusters = np.unique(known_predictions)
        unique_targets = np.unique(known_targets)

        # 构建混淆矩阵：行=簇ID，列=真实标签
        n_clusters = len(unique_clusters)
        n_classes = len(unique_targets)
        confusion_matrix = np.zeros((n_clusters, n_classes), dtype=int)

        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = (known_predictions == cluster_id)
            for j, true_label in enumerate(unique_targets):
                confusion_matrix[i, j] = np.sum(known_targets[cluster_mask] == true_label)

        # 使用匈牙利算法找到簇ID到真实标签的最优一对一匹配
        row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)

        # 计算最优匹配下的正确样本数
        n_correct_from_known = int(confusion_matrix[row_ind, col_ind].sum())
    else:
        n_correct_from_known = 0

    # 总正确样本数和准确率
    n_correct = n_correct_from_unknown + n_correct_from_known
    labeled_acc = n_correct / n_labeled

    # 构建详细指标
    metrics = {
        'n_labeled': n_labeled,
        'n_assigned_to_unknown': int(n_assigned_to_unknown),
        'n_assigned_to_known': int(n_assigned_to_known),
        'n_correct': int(n_correct),
        'accuracy': labeled_acc
    }

    # 打印详细信息
    if not silent:
        print(f"\n{'='*80}")
        print(f"📊 有标签样本ACC计算（考虑unknown_clusters惩罚）")
        print(f"{'='*80}")
        print(f"有标签样本总数: {n_labeled}")
        print(f"  分配到unknown_clusters: {n_assigned_to_unknown} 个（算作错误）")
        print(f"  分配到known_clusters: {n_assigned_to_known} 个")
        if n_assigned_to_known > 0:
            print(f"    其中匹配正确: {n_correct_from_known} 个")
        print(f"总正确样本数: {n_correct}")
        print(f"Labeled ACC: {labeled_acc:.4f} ({labeled_acc*100:.2f}%)")
        print(f"{'='*80}")

    return labeled_acc, metrics
