#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
co参数计算模块
提供三种co计算模式：手动指定、平均距离、相对自适应距离
"""

import numpy as np


def compute_co_value(co_mode, knn_distances, densities, neighbors, k, co_manual=None, silent=False):
    """
    统一的co值计算函数，支持三种模式

    Args:
        co_mode (int): co计算模式
            1 - 手动指定：使用co_manual参数
            2 - 平均距离：使用所有点到K近邻的平均距离的平均值
            3 - 相对自适应距离：基于密度的相对co
        knn_distances (np.ndarray): K近邻距离矩阵，形状 (n_samples, k)
        densities (np.ndarray): 绝对密度数组，形状 (n_samples,)
        neighbors (np.ndarray): K近邻索引矩阵，形状 (n_samples, k)
        k (int): K近邻数量
        co_manual (float, optional): 手动指定的co值（仅当co_mode=1时需要）
        silent (bool): 是否静默模式

    Returns:
        co (float or np.ndarray):
            - 如果co_mode=1或2，返回标量co值
            - 如果co_mode=3，返回每个点的相对co数组，形状 (n_samples,)

    Raises:
        ValueError: 当co_mode不在[1,2,3]范围内，或co_mode=1但未提供co_manual时
    """
    if co_mode not in [1, 2, 3]:
        raise ValueError(f"co_mode必须为1、2或3，当前值: {co_mode}")

    if co_mode == 1:
        # 模式1: 手动指定
        if co_manual is None:
            raise ValueError("co_mode=1时，必须提供co_manual参数")

        if not silent:
            print(f"\n[CO MODE 1] 手动指定co")
            print(f"   co值: {co_manual:.4f}")

        return co_manual

    elif co_mode == 2:
        # 模式2: K近邻平均距离的平均值
        mean_knn_distance = np.mean(knn_distances)

        if not silent:
            print(f"\n[CO MODE 2] K近邻平均距离")
            print(f"   计算方式: mean(所有点的K近邻平均距离)")
            print(f"   co值: {mean_knn_distance:.4f}")

        return mean_knn_distance

    elif co_mode == 3:
        # 模式3: 相对自适应距离
        # co'(x_i) = (K + ρ̄) / (K + (1/K) Σ ρ(x_j)) × co_base
        # 其中 co_base 是模式2的co值

        co_base = np.mean(knn_distances)
        rho_mean = np.mean(densities)
        n_samples = len(densities)

        relative_co = np.zeros(n_samples)

        for i in range(n_samples):
            # 获取点i的K个近邻索引
            neighbor_indices = neighbors[i]

            # 计算近邻密度的平均值
            neighbor_densities_avg = np.mean(densities[neighbor_indices])

            # 计算相对系数
            # ratio = (K + ρ̄) / (K + neighbor_densities_avg)
            numerator = k + rho_mean
            denominator = k + neighbor_densities_avg
            ratio = numerator / denominator

            # 计算相对co
            relative_co[i] = ratio * co_base

        if not silent:
            print(f"\n[CO MODE 3] 相对自适应距离")
            print(f"   基础co (mode 2): {co_base:.4f}")
            print(f"   全局平均密度 ρ̄: {rho_mean:.4f}")
            print(f"   相对co统计:")
            print(f"      最小值: {relative_co.min():.4f}")
            print(f"      最大值: {relative_co.max():.4f}")
            print(f"      平均值: {relative_co.mean():.4f}")
            print(f"      中位数: {np.median(relative_co):.4f}")

        return relative_co


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

    if not silent:
        if is_scalar_co:
            print(f"\n[CO FILTER] 使用标量co={co:.4f}过滤高密度邻居")
        else:
            print(f"\n[CO FILTER] 使用相对co过滤高密度邻居")
            print(f"   相对co范围: [{co.min():.4f}, {co.max():.4f}]")

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

    if not silent:
        # 统计邻居数量分布
        neighbor_counts = [len(neighs) for neighs in high_density_neighbors_map.values()]
        zero_neighbor_count = sum(1 for count in neighbor_counts if count == 0)

        print(f"   高密度点总数: {len(high_density_indices)}")
        print(f"   无邻居的点数: {zero_neighbor_count} ({zero_neighbor_count/len(high_density_indices)*100:.1f}%)")
        print(f"   平均邻居数: {np.mean(neighbor_counts):.2f}")
        print(f"   中位数邻居数: {np.median(neighbor_counts):.0f}")

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
        1: "手动指定co值",
        2: "K近邻平均距离的平均值（全局固定）",
        3: "相对自适应距离（每个点自适应）"
    }
    return descriptions.get(co_mode, "未知模式")
