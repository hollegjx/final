#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
聚类质量评估模块
实现基于簇间分离度和样本局部密度的聚类质量评估指标

公式：
Score = exp(-(1/M) * Σd(Pi,Pj)) - (1/N) * Σ[(1/k) * Σ(I(xi,xj) * exp(-I(xi,xj) * d(xi,xj)))]

其中：
- M: 簇对的数量
- N: 样本数量
- Pi, Pj: 核心簇原型
- d(Pi, Pj): 簇间距离
- xi, xj: 样本点
- I(xi, xj): 指示函数（同簇为1，异簇为-1）
- k: 近邻数量
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def compute_cluster_separation(clusters, X, method='prototype', k=10):
    """
    计算核心簇间分离度（公式第一项）

    Args:
        clusters: 簇列表，每个簇是一个包含样本索引的集合
        X: 特征矩阵 (n_samples, n_features)
        method: 簇距离计算方法
                'prototype' - 原型距离
                'nearest_k' - 最近k对点的平均距离
                'all_pairs' - 所有点对的平均距离
        k: 用于'nearest_k'方法的近邻数量

    Returns:
        separation_score: 簇间分离度分数
        metrics: 详细指标字典
    """
    n_clusters = len(clusters)
    if n_clusters < 2:
        return 0.0, {'n_clusters': n_clusters, 'n_pairs': 0}

    # 簇对数量
    M = n_clusters * (n_clusters - 1) / 2

    # 计算所有簇对的距离
    cluster_distances = []

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i_indices = list(clusters[i])
            cluster_j_indices = list(clusters[j])

            if method == 'prototype':
                # 方案3: 原型距离
                prototype_i = X[cluster_i_indices].mean(axis=0)
                prototype_j = X[cluster_j_indices].mean(axis=0)
                dist = np.linalg.norm(prototype_i - prototype_j)

            elif method == 'nearest_k':
                # 方案1: 最近k对点的平均距离
                X_i = X[cluster_i_indices]
                X_j = X[cluster_j_indices]

                # 计算两个簇之间所有点对的距离
                pairwise_dist = euclidean_distances(X_i, X_j)

                # 找出最近的k对点
                k_actual = min(k, pairwise_dist.size)
                nearest_k_distances = np.partition(pairwise_dist.flatten(), k_actual - 1)[:k_actual]
                dist = nearest_k_distances.mean()

            elif method == 'all_pairs':
                # 方案2: 所有点对的平均距离
                X_i = X[cluster_i_indices]
                X_j = X[cluster_j_indices]

                pairwise_dist = euclidean_distances(X_i, X_j)
                dist = pairwise_dist.mean()

            else:
                raise ValueError(f"未知的method: {method}，支持: 'prototype', 'nearest_k', 'all_pairs'")

            cluster_distances.append(dist)

    # 计算簇间分离度: exp(-(1/M) * Σd(Pi,Pj))
    sum_distances = sum(cluster_distances)
    avg_distance = sum_distances / M if M > 0 else 0.0
    separation_score = np.exp(-avg_distance)

    metrics = {
        'n_clusters': n_clusters,
        'n_pairs': int(M),
        'sum_distances': sum_distances,
        'avg_distance': sum_distances / M if M > 0 else 0.0,
        'method': method
    }

    return separation_score, metrics


def compute_local_density_penalty(X, labels, k, neighbors=None):
    """
    计算所有样本的局部密度惩罚项（公式第二项）

    Args:
        X: 特征矩阵 (n_samples, n_features)
        labels: 聚类标签 (n_samples,)
        k: 近邻数量
        neighbors: 预计算的近邻索引 (n_samples, k)，如果为None则重新计算

    Returns:
        penalty_score: 局部密度惩罚分数
        metrics: 详细指标字典
    """
    n_samples = len(X)

    # 如果没有提供neighbors，则计算
    if neighbors is None:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        _, neighbors = nbrs.kneighbors(X)
        neighbors = neighbors[:, 1:]  # 排除自己

    # 计算局部密度惩罚
    total_penalty = 0.0

    for i in range(n_samples):
        knn_indices = neighbors[i][:k]

        local_sum = 0.0

        for j in knn_indices:
            # I(xi, xj): 同簇为1，异簇为-1
            if labels[i] == labels[j]:
                I_ij = 1.0
            else:
                I_ij = -1.0

            # 计算距离
            d_ij = np.linalg.norm(X[i] - X[j])

            # I(xi,xj) * exp(-I(xi,xj) * d(xi,xj))
            # 当I=1时: 1 * exp(-1 * d_ij) = exp(-d_ij)
            # 当I=-1时: -1 * exp(-(-1) * d_ij) = -exp(d_ij)
            local_sum += I_ij * np.exp(-I_ij * d_ij)

        # (1/k) * Σ(...)
        if k > 0:
            total_penalty += local_sum / k

    # (1/N) * Σ[...]
    penalty_score = total_penalty / n_samples if n_samples > 0 else 0.0

    metrics = {
        'n_samples': n_samples,
        'k': k,
        'total_penalty': total_penalty,
        'avg_penalty_per_sample': total_penalty / n_samples if n_samples > 0 else 0.0
    }

    return penalty_score, metrics


def compute_separation_component(clusters, X, k=10, method='prototype'):
    """
    计算簇间分离度组件并返回分值与详细指标
    """
    separation_score, separation_metrics = compute_cluster_separation(
        clusters=clusters,
        X=X,
        method=method,
        k=k
    )
    metrics = {
        'separation_score': separation_score,
        'details': separation_metrics
    }
    return separation_score, metrics


def compute_penalty_component(X, labels, k, neighbors=None):
    """
    计算局部密度惩罚组件并返回分值与详细指标
    """
    penalty_score, penalty_metrics = compute_local_density_penalty(
        X=X,
        labels=labels,
        k=k,
        neighbors=neighbors
    )
    metrics = {
        'penalty_score': penalty_score,
        'details': penalty_metrics
    }
    return penalty_score, metrics


def compute_cluster_quality_score(clusters, X, labels, k,
                                 cluster_distance_method='prototype',
                                 neighbors=None,
                                 separation_weight=1.0,
                                 penalty_weight=1.0):
    """
    计算完整的聚类质量评估分数

    Score = separation_weight * 簇间分离度 - penalty_weight * 局部密度惩罚

    Args:
        clusters: 核心簇列表（高密度点形成的簇）
        X: 特征矩阵 (n_samples, n_features)
        labels: 所有样本的聚类标签 (n_samples,)
        k: 近邻数量
        cluster_distance_method: 簇距离计算方法 ('prototype', 'nearest_k', 'all_pairs')
        neighbors: 预计算的近邻索引，如果为None则重新计算
        separation_weight: 簇间分离度的权重，默认1.0
        penalty_weight: 局部密度惩罚的权重，默认1.0

    Returns:
        quality_score: 聚类质量分数（越大越好）
        metrics: 详细指标字典
    """
    # 计算组件分值
    separation_score, separation_payload = compute_separation_component(
        clusters=clusters,
        X=X,
        k=k,
        method=cluster_distance_method
    )
    penalty_score, penalty_payload = compute_penalty_component(
        X=X,
        labels=labels,
        k=k,
        neighbors=neighbors
    )

    # 计算总分（应用权重）
    quality_score = separation_weight * separation_score - penalty_weight * penalty_score

    # 整合指标
    metrics = {
        'quality_score': quality_score,
        'separation_score': separation_score,
        'penalty_score': penalty_score,
        'separation_weight': separation_weight,
        'penalty_weight': penalty_weight,
        'weighted_separation': separation_weight * separation_score,
        'weighted_penalty': penalty_weight * penalty_score,
        'separation_metrics': separation_payload['details'],
        'penalty_metrics': penalty_payload['details']
    }

    return quality_score, metrics
