#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
未知类识别模块
识别和处理潜在的未知类聚类
"""

import numpy as np


def identify_unknown_clusters(clusters, labeled_mask):
    """
    识别潜在未知类聚类

    根据SS-DDBC: "簇的类别取决于其有标签样本，不含有标签样本的可能是未知类"

    Args:
        clusters: 聚类列表
        labeled_mask: 有标签掩码

    Returns:
        unknown_clusters: 潜在未知类聚类索引列表
    """
    unknown_clusters = []

    for cluster_id, cluster in enumerate(clusters):
        cluster_indices = list(cluster)
        cluster_labeled_mask = labeled_mask[cluster_indices]

        # 如果聚类中没有有标签样本，标记为潜在未知类
        if not np.any(cluster_labeled_mask):
            unknown_clusters.append(cluster_id)

    return unknown_clusters


def identify_unknown_clusters_from_predictions(predictions, labeled_mask):
    """
    从预测结果中识别潜在未知类聚类

    Args:
        predictions: 聚类预测结果
        labeled_mask: 有标签掩码

    Returns:
        unknown_clusters: 潜在未知类聚类索引列表
    """
    unknown_clusters = []
    unique_clusters = np.unique(predictions)

    for cluster_id in unique_clusters:
        cluster_mask = predictions == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_labeled_mask = labeled_mask[cluster_indices]

        # 如果聚类中没有有标签样本，标记为潜在未知类
        if not np.any(cluster_labeled_mask):
            unknown_clusters.append(cluster_id)

    return unknown_clusters
