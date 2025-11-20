#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
co参数计算模块
当前仅保留模式2：基于所有点K近邻平均距离的全局co。
"""

import numpy as np


def compute_co_value(co_mode, knn_distances, densities, neighbors, k, co_manual=None, silent=False):
    """
    co 值计算函数（精简版）

    当前仅支持:
        co_mode=2 - 使用所有点K近邻平均距离的平均值，返回标量 co。
    其他模式将直接抛出异常。
    """
    if co_mode != 2:
        raise ValueError(f"当前实现仅支持 co_mode=2，收到 co_mode={co_mode}")

    mean_knn_distance = np.mean(knn_distances)

    return mean_knn_distance


def apply_co_filter(X, high_density_indices, co, neighbors, high_density_mask, silent=False):
    """
    应用co过滤，构建高密度子空间的邻居映射

    Args:
        X (np.ndarray): 特征矩阵，形状 (n_samples, feat_dim)
        high_density_indices (np.ndarray): 高密度点索引数组
        co (float or np.ndarray): co值，可以是标量或每个点的相对co
        neighbors (np.ndarray): K近邻索引矩阵，形状 (n_samples, k)
        high_density_mask (np.ndarray): 高密度点掩码，形状 (n_samples,)
        silent (bool): 是否静默模式

    Returns:
        high_density_neighbors_map (dict): {高密度点索引: [co距离内的高密度邻居索引列表]}
    """
    high_density_neighbors_map = {}
    is_scalar_co = isinstance(co, (int, float, np.number))

    for idx in high_density_indices:
        neighbors_in_co = []

        # 获取该点的co值
        co_value = co if is_scalar_co else co[idx]

        # 遍历该高密度点的所有K近邻
        for neighbor_idx in neighbors[idx]:
            # 只考虑也是高密度点的邻居
            if high_density_mask[neighbor_idx]:
                # 计算距离
                dist = np.linalg.norm(X[idx] - X[neighbor_idx])

                # 如果距离在co范围内，则添加到邻居列表
                if dist <= co_value:
                    neighbors_in_co.append(neighbor_idx)

        high_density_neighbors_map[idx] = neighbors_in_co

    return high_density_neighbors_map


def get_co_mode_description(co_mode):
    """
    获取co_mode的描述信息

    Args:
        co_mode (int): co计算模式

    Returns:
        description (str): 模式描述
    """
    descriptions = {
        2: "K近邻平均距离的平均值（全局固定）",
    }
    return descriptions.get(co_mode, "仅支持模式2")
