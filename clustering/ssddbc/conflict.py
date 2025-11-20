#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SS-DDBC冲突解决模块
处理聚类构建过程中的标签冲突
"""

import numpy as np


def ssddbc_conflict_resolution(xi_idx, xj_idx, xi_cluster_center, xj_cluster_center, X, densities):
    """
    SS-DDBC冲突解决算法 - 用于判断是否合并两个簇

    使用场景：
    - 当xi和xj属于不同簇，且无标签冲突时（si = sj 或有的簇无标签）
    - 通过密度和距离判断是否应该合并两个簇

    关键逻辑：
    - xi正在扩展（按密度从高到低处理）
    - xj已经在另一个簇中
    - 判断是否应该将xj的簇合并到xi的簇中

    新策略：低密度点拉入高密度点的簇
    - 如果xj的密度更高，说明xj的簇更"强"，xi应该加入xj的簇
    - 实现方式：让xi的簇合并到xj的簇（通过返回True）
    - 如果xi的密度更高，xi的簇可以"吸收"xj的簇

    Args:
        xi_idx: 当前扩展点索引（来自簇pi）
        xj_idx: 邻居点索引（当前在簇pj中）
        xi_cluster_center: xi所在聚类pi的中心
        xj_cluster_center: xj所在聚类pj的中心
        X: 特征矩阵
        densities: 密度数组

    Returns:
        should_merge: 是否应该合并两个簇
    """
    # 1. 比较xi和xj的密度
    xi_density = densities[xi_idx]
    xj_density = densities[xj_idx]

    # 2. 计算xj点到两个聚类中心的距离
    xj_pos = X[xj_idx]
    distance_to_xi_cluster = np.linalg.norm(xj_pos - xi_cluster_center)
    distance_to_xj_cluster = np.linalg.norm(xj_pos - xj_cluster_center)

    # 3. 判断条件：xj距离xi簇更近
    # 关键改变：不再要求xi密度必须更高！
    closer_to_xi = distance_to_xi_cluster < distance_to_xj_cluster

    if closer_to_xi:
        # xj距离xi的簇更近，应该合并
        # 无论密度高低，都合并
        return True
    else:
        # xj距离自己的簇更近，不合并
        return False


def compute_cluster_cohesion(X, cluster_indices):
    """
    计算聚类的内部凝聚度

    Args:
        X: 特征矩阵
        cluster_indices: 聚类包含的样本索引

    Returns:
        cohesion: 聚类凝聚度 (值越小表示越紧密)
    """
    if len(cluster_indices) < 2:
        return 0.0

    cluster_points = X[cluster_indices]
    cluster_center = np.mean(cluster_points, axis=0)

    # 计算所有点到聚类中心的平均距离
    distances = [np.linalg.norm(point - cluster_center) for point in cluster_points]
    cohesion = np.mean(distances)

    return cohesion
