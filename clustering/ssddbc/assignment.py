#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SS-DDBC稀疏点分配模块
实现稀疏点到聚类的分配算法
"""

import numpy as np



def assign_sparse_points_density_based(X, clusters, cluster_labels, densities, neighbors, labeled_mask, targets,
                                       label_threshold=0.1, purity_threshold=0.8, train_size=None, silent=False,
                                       prototypes=None, prototype_true_labels=None, voting_k=5, assign_model=2,
                                       label_guide=False):
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
        prototypes: 簇原型特征向量（必须提供）
        prototype_true_labels: 簇原型的真实主导标签（可选，用于错误诊断）
        voting_k: KNN投票使用的近邻数量（默认5）
        assign_model: 分配策略模式（默认2）
            1 - 基于簇原型的就近分配：计算到每个簇原型的距离，分配到最近的簇
            2 - 高密度子空间KNN投票（距离加权）：在高密度点中找K近邻，使用exp(-distance)加权投票
            3 - 簇内K近邻平均距离：从每个簇中找3个最近的样本，计算平均距离，分配到平均距离最小的簇

    Returns:
        final_labels: 最终聚类标签
    """
    # 检查prototypes是否提供
    if prototypes is None:
        raise ValueError("prototypes参数必须提供！请在调用前使用build_prototypes构建簇原型。")

    if not silent:
        print(f"\n[ASSIGNMENT] 稀疏点分配策略: assign_model={assign_model}")
        if assign_model == 1:
            print(f"   模式1: 基于簇原型的就近分配")
        elif assign_model == 2:
            print(f"   模式2: 高密度子空间KNN投票（voting_k={voting_k}, 距离加权: exp(-d)）")
        elif assign_model == 3:
            print(f"   模式3: 簇内K近邻平均距离（每簇选3个最近点）")

    final_labels = cluster_labels.copy()
    unassigned_indices = np.where(cluster_labels == -1)[0]

    if len(unassigned_indices) == 0:
        if not silent:
            print(f"   无需分配稀疏点（所有点已分配）")
        return final_labels

    # 按密度从高到低排序稀疏点
    unassigned_densities = densities[unassigned_indices]
    sorted_indices = np.argsort(-unassigned_densities)
    sorted_unassigned = unassigned_indices[sorted_indices]

    total_unassigned = len(sorted_unassigned)
    if not silent:
        print(f"   待分配稀疏点数量: {total_unassigned}")

    # 获取高密度点索引（骨干网络）
    # 注意：当label_guide=True时，传入的cluster_labels已经包含了有标签稀疏点的分配
    # 因此high_density_indices实际上是"核心点空间"（高密度点 + 有标签稀疏点）
    high_density_indices = np.where(cluster_labels != -1)[0]
    n_high_density = len(high_density_indices)

    if not silent:
        if label_guide:
            print(f"   核心点数量(包含有标签稀疏点): {n_high_density}")
        else:
            print(f"   高密度点数量: {n_high_density}")

    # 检测高密度子空间中的孤立点（没有被分配到任何簇的高密度点）
    # 这种情况通常不应该发生，因为cluster_labels != -1就是高密度点
    # 但我们还是做个检查
    isolated_high_density = []
    for idx in high_density_indices:
        if cluster_labels[idx] == -1:
            isolated_high_density.append(idx)

    if len(isolated_high_density) > 0 and not silent:
        print(f"   [WARNING] 检测到{len(isolated_high_density)}个孤立的高密度点（未分配到簇）")

    # 构建每个簇的样本索引列表（用于模式3）
    # 关键：使用传入的clusters参数（已经是updated_clusters），而不是从cluster_labels重建
    cluster_samples_dict = {}
    for cluster_id, cluster in enumerate(clusters):
        cluster_samples_dict[cluster_id] = list(cluster)

    # 统计每个簇的大小
    cluster_sizes = {cid: len(samples) for cid, samples in cluster_samples_dict.items()}

    # 分配计数器
    assigned_count = 0
    failed_count = 0

    # ========== 根据assign_model选择分配策略 ==========
    if assign_model == 1:
        # 模式1: 基于簇原型的就近分配
        if not silent:
            print(f"   开始分配（按密度从高到低）...")

        for point_idx in sorted_unassigned:
            point_features = X[point_idx]

            # 计算到所有簇原型的距离
            distances_to_clusters = []
            for cluster_id in range(len(prototypes)):
                dist = np.linalg.norm(point_features - prototypes[cluster_id])
                distances_to_clusters.append((cluster_id, dist))

            # 找到最近的簇
            if len(distances_to_clusters) > 0:
                closest_cluster = min(distances_to_clusters, key=lambda x: x[1])[0]
                final_labels[point_idx] = closest_cluster
                assigned_count += 1
            else:
                failed_count += 1

            # 进度显示
            if not silent and (assigned_count + failed_count) % 100 == 0:
                print(f"\r   分配进度: {assigned_count + failed_count}/{total_unassigned}", end='', flush=True)

    elif assign_model == 2:
        # 模式2: 高密度子空间KNN投票（距离加权）
        if not silent:
            print(f"   开始分配（按密度从高到低）...")

        # 预先计算高密度点的特征矩阵
        X_high_density = X[high_density_indices]

        for point_idx in sorted_unassigned:
            point_features = X[point_idx]

            # 计算到所有高密度点的距离
            distances = np.linalg.norm(X_high_density - point_features, axis=1)

            # 找到K个最近的高密度邻居
            k_nearest = min(voting_k, len(distances))
            nearest_indices = np.argsort(distances)[:k_nearest]

            # 加权投票：每个邻居的权重 = exp(-distance)
            cluster_scores = {}
            for idx in nearest_indices:
                neighbor_global_idx = high_density_indices[idx]
                # 关键修复：使用final_labels而不是cluster_labels
                # 因为在稀疏点按密度从高到低分配过程中，已分配的点的簇标签在final_labels中
                neighbor_cluster = final_labels[neighbor_global_idx]
                neighbor_distance = distances[idx]

                # 权重计算：exp(-distance)
                weight = np.exp(-neighbor_distance)

                if neighbor_cluster not in cluster_scores:
                    cluster_scores[neighbor_cluster] = 0.0
                cluster_scores[neighbor_cluster] += weight

            # 选择得分最高的簇
            if len(cluster_scores) > 0:
                winner_cluster = max(cluster_scores, key=cluster_scores.get)
                final_labels[point_idx] = winner_cluster
                assigned_count += 1
            else:
                # 没有找到任何高密度邻居（理论上不应该发生）
                failed_count += 1

            # 进度显示
            if not silent and (assigned_count + failed_count) % 100 == 0:
                print(f"\r   分配进度: {assigned_count + failed_count}/{total_unassigned}", end='', flush=True)

    elif assign_model == 3:
        # 模式3: 簇内K近邻平均距离
        if not silent:
            print(f"   开始分配（按密度从高到低）...")

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

            # 进度显示
            if not silent and (assigned_count + failed_count) % 100 == 0:
                print(f"\r   分配进度: {assigned_count + failed_count}/{total_unassigned}", end='', flush=True)

    else:
        raise ValueError(f"不支持的assign_model值: {assign_model}。支持的值: 1, 2, 3")

    # 完成进度显示
    if not silent:
        print(f"\r   分配完成: {assigned_count}/{total_unassigned} 成功, {failed_count} 失败")

    # 处理失败的情况（如果有）
    if failed_count > 0:
        if not silent:
            print(f"   [WARNING] 有{failed_count}个稀疏点分配失败，将使用兜底策略（分配到最近的簇原型）")

        # 兜底策略：对于失败的点，分配到最近的簇原型
        still_unassigned = np.where(final_labels == -1)[0]
        for point_idx in still_unassigned:
            point_features = X[point_idx]
            distances_to_prototypes = [np.linalg.norm(point_features - proto) for proto in prototypes]
            closest_cluster = np.argmin(distances_to_prototypes)
            final_labels[point_idx] = closest_cluster

    return final_labels
