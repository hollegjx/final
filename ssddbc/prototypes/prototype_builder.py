#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
原型构建模块
基于聚类结果构建类原型
"""

import numpy as np


def build_prototypes(X, clusters, labeled_mask, targets):
    """
    基于partial-clustering结果建立原型

    SS-DDBC步骤: (2) 基于partial-clustering的结果建立原型

    Args:
        X: 特征矩阵
        clusters: 聚类列表
        labeled_mask: 有标签掩码
        targets: 真实标签

    Returns:
        prototypes: 每个聚类的原型 (聚类中心)
        prototype_labels: 每个聚类的主导标签
    """
    print(f"   建立聚类原型...")

    prototypes = []
    prototype_labels = []

    for cluster_id, cluster in enumerate(clusters):
        cluster_indices = list(cluster)

        # 计算聚类中心作为原型
        prototype = np.mean(X[cluster_indices], axis=0)
        prototypes.append(prototype)

        # 确定聚类的主导标签
        cluster_labeled_mask = labeled_mask[cluster_indices]
        if np.any(cluster_labeled_mask):
            # 如果有有标签样本，使用主导标签
            labeled_targets = targets[cluster_indices][cluster_labeled_mask]
            unique_labels, counts = np.unique(labeled_targets, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]
            prototype_labels.append(dominant_label)
        else:
            # 无有标签样本，标记为未知类 (-1)
            prototype_labels.append(-1)

    print(f"   建立原型完成: {len(prototypes)}个原型")
    return np.array(prototypes), np.array(prototype_labels)
