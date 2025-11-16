#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
密度估计模块
实现基于k近邻的密度计算和高密度点识别
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_median_density(X, k=10):
    """
    使用k近邻中位数距离倒数计算密度（更鲁棒，对极值不敏感）

    与 compute_simple_density 的区别：
    - compute_simple_density: 密度 = 1 / 平均距离
    - compute_median_density: 密度 = 1 / 中位数距离

    优点：中位数对极近邻和极远邻不敏感，更能反映邻居的整体分布

    Args:
        X: 特征矩阵 (n_samples, feat_dim)
        k: k近邻数量

    Returns:
        densities: 每个样本的密度值
        knn_distances: k近邻距离矩阵
        neighbors: k近邻索引矩阵
    """
    print(f"[DENSITY] 计算中位数密度 (k={k})...")

    # 计算k近邻 (确保结果确定性)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto').fit(X)
    knn_distances, neighbors = nbrs.kneighbors(X)

    # 处理距离相等的情况：按索引排序确保一致性
    for i in range(len(neighbors)):
        # 对于距离相同的邻居，按索引排序
        row_distances = knn_distances[i]
        row_neighbors = neighbors[i]

        # 创建(距离, 索引)对并排序
        dist_idx_pairs = list(zip(row_distances, row_neighbors))
        dist_idx_pairs.sort(key=lambda x: (x[0], x[1]))  # 先按距离，再按索引

        knn_distances[i] = np.array([pair[0] for pair in dist_idx_pairs])
        neighbors[i] = np.array([pair[1] for pair in dist_idx_pairs])

    # 去除自己（第一个邻居是自己）
    knn_distances = knn_distances[:, 1:]
    neighbors = neighbors[:, 1:]

    # 中位数密度计算：k近邻中位数距离的倒数
    median_distances = np.median(knn_distances, axis=1)
    densities = 1.0 / (median_distances + 1e-8)  # 避免除零

    print(f"   密度统计: min={densities.min():.3f}, max={densities.max():.3f}, mean={densities.mean():.3f}")

    return densities, knn_distances, neighbors


def analyze_knn_average_distances(knn_distances):
    """
    统计每个样本的k近邻平均距离的分布信息

    对每个样本计算其k近邻的平均距离 d_i，然后统计所有 d_i 的分布。

    Args:
        knn_distances: k近邻距离矩阵 (n_samples, k)

    Returns:
        mean_d: 所有样本k近邻平均距离的平均值（用于作为默认co值）
    """
    # 计算每个样本的k近邻平均距离
    avg_distances = np.mean(knn_distances, axis=1)

    # 统计信息
    mean_d = np.mean(avg_distances)
    median_d = np.median(avg_distances)
    min_d = np.min(avg_distances)
    max_d = np.max(avg_distances)
    std_d = np.std(avg_distances)

    print(f"\n[ANALYSIS] k近邻平均距离 (d_i) 统计分析:")
    print(f"   平均值: {mean_d:.4f}")
    print(f"   中位数: {median_d:.4f}")
    print(f"   最小值: {min_d:.4f}")
    print(f"   最大值: {max_d:.4f}")
    print(f"   标准差: {std_d:.4f}")

    return mean_d


def identify_high_density_points(densities, percentile=75, use_relative=False):
    """
    步骤2: 选择高密度点作为聚类种子（使用百分位数阈值）

    Args:
        densities: 密度值数组（可以是绝对密度或相对密度）
        percentile: 百分位数阈值
        use_relative: 是否使用相对密度（用于打印提示）

    Returns:
        high_density_mask: 高密度点掩码
    """
    density_threshold = np.percentile(densities, percentile)
    high_density_mask = densities >= density_threshold

    density_type = "相对密度" if use_relative else "绝对密度"
    print(f"[HIGH-DENSITY] 识别高密度点 ({density_type}, 百分位数阈值):")
    print(f"   密度阈值: {density_threshold:.3f} (第{percentile}百分位数)")
    print(f"   高密度点数量: {np.sum(high_density_mask)} / {len(densities)} ({np.sum(high_density_mask)/len(densities)*100:.1f}%)")

    return high_density_mask


def compute_relative_density(densities, neighbors, k):
    """
    计算相对密度，用于更鲁棒的高密度点识别

    公式: ρ'(x_i) = (K + ρ̄) / (K + (1/K) Σ_{j∈KNN(i)} ρ(x_j)) × ρ(x_i)

    其中：
    - ρ(x_i) 是点i的绝对密度（由dense_method计算）
    - ρ̄ 是所有点密度的平均值
    - K 是近邻数量
    - Σ_{j∈KNN(i)} ρ(x_j) 是点i的K个近邻的密度之和

    Args:
        densities: 绝对密度值数组 (n_samples,)
        neighbors: k近邻索引矩阵 (n_samples, k)
        k: k近邻数量

    Returns:
        relative_densities: 相对密度值数组 (n_samples,)
    """
    print(f"\n[RELATIVE DENSITY] 计算相对密度...")

    n_samples = len(densities)
    rho_mean = np.mean(densities)  # 所有点密度的平均值 ρ̄

    relative_densities = np.zeros(n_samples)

    for i in range(n_samples):
        # 获取点i的K个近邻索引
        neighbor_indices = neighbors[i]  # shape: (k,)

        # 计算近邻密度之和
        neighbor_densities_sum = np.sum(densities[neighbor_indices])

        # 计算近邻密度的平均值
        neighbor_densities_avg = neighbor_densities_sum / k

        # 计算相对密度
        # ρ'(x_i) = (K + ρ̄) / (K + neighbor_densities_avg) × ρ(x_i)
        numerator = k + rho_mean
        denominator = k + neighbor_densities_avg
        relative_densities[i] = (numerator / denominator) * densities[i]

    print(f"   绝对密度统计: min={densities.min():.3f}, max={densities.max():.3f}, mean={densities.mean():.3f}")
    print(f"   相对密度统计: min={relative_densities.min():.3f}, max={relative_densities.max():.3f}, mean={relative_densities.mean():.3f}")

    return relative_densities

