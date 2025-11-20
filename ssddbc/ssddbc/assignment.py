#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SS-DDBC稀疏点分配模块
实现稀疏点到聚类的分配算法
"""

import numpy as np



def assign_sparse_points_density_based(
    X,
    clusters,
    cluster_labels,
    densities,
    neighbors,
    labeled_mask,
    targets,
    label_threshold=0.1,
    purity_threshold=0.8,
    train_size=None,
    silent=False,
    voting_k=5,
    assign_model=2,
    label_guide=False,
):
    """
    稀疏点分配算法（支持多种分配策略）？/

    Args:
        X: 特征矩阵 (n_samples, feat_dim)
        clusters: 骨干网络聚类列表 (每个簇是一个set，包含高密度点索引)
        cluster_labels: 高密度点的聚类标签 (低密度点标记为-1)
        densities: 所有样本的密度值
        neighbors: k近邻索引矩阵
        labeled_mask: 有标签掩码
        targets: 真实标签（用于定义簇标签和调试分析）
        label_threshold: 簇被定义为有标签的最小有标签样本比例（默认0.1即10%）
        purity_threshold: 簇被定义为有标签的最小标签纯度（默认0.8即80%）
        train_size: 训练集大小（用于区分训练集和测试集）
        silent: 静默模式（默认False）
        voting_k: KNN投票使用的近邻数量（保留参数占位）
        assign_model: 分配策略模式（当前仅支持3）
            3 - 簇内K近邻平均距离：从每个簇中找3个最近的样本，计算平均距离，分配到平均距离最小的簇

    Returns:
        final_labels: 最终聚类标签
    """
    if assign_model != 3:
        raise ValueError(f"当前实现仅支持 assign_model=3，收到 assign_model={assign_model}")

    final_labels = cluster_labels.copy()
    unassigned_indices = np.where(cluster_labels == -1)[0]

    if len(unassigned_indices) == 0:
        return final_labels

    # 按密度从高到低排序稀疏点
    unassigned_densities = densities[unassigned_indices]
    sorted_indices = np.argsort(-unassigned_densities)
    sorted_unassigned = unassigned_indices[sorted_indices]

    total_unassigned = len(sorted_unassigned)

    # 获取高密度点索引（骨干网络）
    # 注意：当label_guide=True时，传入的cluster_labels已经包含了有标签稀疏点的分配
    # 因此high_density_indices实际上是"核心点空间"（高密度点 + 有标签稀疏点）
    high_density_indices = np.where(cluster_labels != -1)[0]

    # 检测高密度子空间中的孤立点（没有被分配到任何簇的高密度点）
    # 这种情况通常不应该发生，因为cluster_labels != -1就是高密度点
    # 但我们还是做个检查
    isolated_high_density = []
    for idx in high_density_indices:
        if cluster_labels[idx] == -1:
            isolated_high_density.append(idx)

    # 构建每个簇的样本索引列表（用于模式3）
    # 关键：使用传入的clusters参数（已经是updated_clusters），而不是从cluster_labels重建
    cluster_samples_dict = {}
    for cluster_id, cluster in enumerate(clusters):
        cluster_samples_dict[cluster_id] = list(cluster)

    # 分配计数器
    assigned_count = 0
    failed_count = 0

    for point_idx in sorted_unassigned:
        point_features = X[point_idx]

        # 计算到每个簇的平均距离（基于簇内最近的3个样本）
        cluster_avg_distances = []

        for cluster_id, cluster_samples in cluster_samples_dict.items():
            if len(cluster_samples) == 0:
                continue

            # 计算到簇内所有样本的距离
            cluster_features = X[cluster_samples]
            distances = np.linalg.norm(cluster_features - point_features, axis=1)

            # 选择最近的min(3, cluster_size)个样本
            k_in_cluster = min(3, len(distances))
            nearest_distances = np.sort(distances)[:k_in_cluster]

            # 计算平均距离
            avg_distance = np.mean(nearest_distances)
            cluster_avg_distances.append((cluster_id, avg_distance))

        # 选择平均距离最小的簇
        if len(cluster_avg_distances) > 0:
            closest_cluster = min(cluster_avg_distances, key=lambda x: x[1])[0]
            final_labels[point_idx] = closest_cluster
            assigned_count += 1
        else:
            # 所有簇都是空的（理论上不应该发生）
            failed_count += 1

    if failed_count > 0 and not silent:
        print(f"⚠️ 稀疏点分配失败 {failed_count} 个样本，建议检查聚类配置或密度阈值设置。")

    return final_labels
