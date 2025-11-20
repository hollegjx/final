#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SS-DDBC自适应密度聚类主算法
集成完整的SS-DDBC算法流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from .clustering import build_clusters_ssddbc
from .analysis import (analyze_ssddbc_clustering_result,
                       evaluate_high_density_clustering)
from .assignment import assign_sparse_points_density_based
from .merging import merge_isolated_clusters
from ..density.density_estimation import (compute_simple_density, compute_median_density,
                                          compute_normalized_inverse_density, compute_exponential_density,
                                          identify_high_density_points, compute_relative_density)
from ..utils.co_calculation import compute_co_value, get_co_mode_description
from ..information import init_logger, reset_logger
from ..prototypes.prototype_builder import build_prototypes
from ..unknown.detection import identify_unknown_clusters_from_predictions


def apply_label_guided_assignment(X, cluster_labels, clusters, labeled_mask, targets, prototypes, prototype_labels, silent=False):
    """
    标签引导分配：将稀疏点中的已知标签样本直接分配到对应标签的核心簇

    Args:
        cluster_labels: 当前聚类标签（核心点已分配，稀疏点为-1）
        clusters: 核心簇列表
        labeled_mask: 有标签掩码
        targets: 真实标签
        prototypes: 核心簇原型
        prototype_labels: 核心簇标签
        silent: 静默模式

    Returns:
        updated_labels: 更新后的聚类标签
    """
    updated_labels = cluster_labels.copy()

    # 找到所有有标签的样本（包括稀疏点和可能错误分配的核心点）
    all_labeled_points = np.where(labeled_mask)[0]

    # 分离稀疏点和已分配点
    sparse_labeled_points = all_labeled_points[cluster_labels[all_labeled_points] == -1]
    assigned_labeled_points = all_labeled_points[cluster_labels[all_labeled_points] != -1]

    if len(sparse_labeled_points) == 0 and len(assigned_labeled_points) == 0:
        return updated_labels

    # 处理有标签的样本（稀疏点 + 可能错误分配的核心点）
    assigned_count = 0
    reassigned_count = 0

    # 合并处理所有有标签的样本
    all_target_points = np.concatenate([sparse_labeled_points, assigned_labeled_points]) if len(assigned_labeled_points) > 0 else sparse_labeled_points

    for point_idx in all_target_points:
        point_label = targets[point_idx]

        # 找到标签匹配的核心簇
        matching_clusters = []
        for cluster_id, cluster_label in enumerate(prototype_labels):
            if cluster_label == point_label:
                matching_clusters.append(cluster_id)

        if len(matching_clusters) > 0:
            # 如果有多个匹配的簇，选择距离最近的
            if len(matching_clusters) == 1:
                best_cluster = matching_clusters[0]
            else:
                # 计算到各个匹配簇原型的距离，选择最近的
                distances = []
                for cluster_id in matching_clusters:
                    if cluster_id < len(prototypes):
                        dist = np.linalg.norm(prototypes[cluster_id] - X[point_idx])
                        distances.append((dist, cluster_id))
                if distances:
                    best_cluster = min(distances)[1]
                else:
                    best_cluster = matching_clusters[0]

            # 检查是否需要重新分配
            current_cluster = cluster_labels[point_idx]
            if current_cluster == -1:
                # 稀疏点，新分配
                updated_labels[point_idx] = best_cluster
                assigned_count += 1
            elif current_cluster != best_cluster:
                # 已分配点，但簇不匹配，重新分配
                updated_labels[point_idx] = best_cluster
                reassigned_count += 1


    return updated_labels


def adaptive_density_clustering(X, targets, known_mask, labeled_mask,
                               k=10, density_percentile=75, random_state=0, train_size=None,
                               co_mode=2, co_manual=None,
                               eval_dense=False, eval_version='v1',
                               silent=False, dense_method=0, assign_model=2, voting_k=5, detail_dense=False,
                               label_guide=False):
    """
    SS-DDBC风格的自适应密度聚类算法（简化版本）

    Args:
        X: 特征矩阵
        targets: 真实标签 (仅用于评估)
        known_mask: 已知类别掩码
        labeled_mask: 有标签掩码
        k: k近邻参数
        density_percentile: 密度百分位数阈值
        random_state: 随机种子 (与K-means保持一致)
        train_size: 训练集大小（用于区分训练集和测试集）
        co_mode: co计算模式 (1=手动指定, 2=K近邻平均距离, 3=相对自适应距离，默认2)
        co_manual: 手动指定的co值（仅当co_mode=1时使用）
        eval_dense: 是否仅评估高密度点（默认False，如为True则跳过低密度点分配）
        eval_version: 评估版本 ('v1' 或 'v2')
        silent: 静默模式，关闭所有不必要的打印输出（用于网格搜索加速，默认False）
        dense_method: 密度计算方法 (0=平均距离, 1=中位数距离, 2=归一化倒数, 3=指数密度，默认0)
        assign_model: 稀疏点分配策略 (1=簇原型就近, 2=KNN投票加权, 3=簇内K近邻平均距离，默认2)
        voting_k: KNN投票时使用的近邻数量（默认5）
        detail_dense: 是否记录骨干网络聚类详细日志（默认False）
        label_guide: 是否启用标签引导模式（默认False，True时先将已知标签稀疏点分配到对应核心簇）

    Returns:
        predictions: 聚类预测结果
        n_clusters: 聚类数量
        unknown_clusters: 潜在未知类聚类索引
        prototypes: 簇原型（用于错误分析）
        updated_clusters: 更新后的簇成员集合（包含所有点，用于错误分析）
        cluster_category_labels: 簇ID到类别标签的映射字典（只包含有类别标签的簇）
        neighbors: K近邻索引矩阵 (n_samples, k)（用于避免重复计算）
        core_clusters: 核心点簇（只包含高密度点，用于计算聚类质量第一项）
        如果eval_dense=True，还会返回评估指标: (all_acc, old_acc, new_acc, None)
    """
    # 设置局部numpy随机种子 (遵循K-means的设计模式)
    np.random.seed(random_state)

    # 初始化详细日志记录器（如果启用）
    logger = None
    if detail_dense:
        logger = init_logger(enabled=True)
        if not silent:
            print("📝 骨干网络详细日志记录已启用")

    if not silent:
        print("🚀 开始简化SS-DDBC聚类...")

    # 步骤1: 根据dense_method选择密度计算方法
    if dense_method == 0:
        # 方法0: 使用平均距离计算密度
        densities, knn_distances, neighbors = compute_simple_density(X, k)
        if not silent:
            print(f"   密度计算方法: 平均距离倒数 (dense_method=0)")
    elif dense_method == 1:
        # 方法1: 使用中位数距离计算密度
        densities, knn_distances, neighbors = compute_median_density(X, k)
        if not silent:
            print(f"   密度计算方法: 中位数距离倒数 (dense_method=1)")
    elif dense_method == 2:
        # 方法2: 使用归一化倒数密度
        densities, knn_distances, neighbors = compute_normalized_inverse_density(X, k)
        if not silent:
            print(f"   密度计算方法: 归一化倒数密度 (dense_method=2)")
    elif dense_method == 3:
        # 方法3: 使用指数密度
        densities, knn_distances, neighbors = compute_exponential_density(X, k)
        if not silent:
            print(f"   密度计算方法: 指数密度 (dense_method=3)")
    else:
        raise ValueError(f"未知的dense_method值: {dense_method}. 支持的值: 0 (平均距离), 1 (中位数距离), 2 (归一化倒数), 3 (指数密度)")

    # 步骤2: 计算相对密度并识别高密度点（使用百分位数阈值）
    relative_densities = compute_relative_density(densities, neighbors, k)
    high_density_mask = identify_high_density_points(relative_densities, density_percentile, use_relative=True)

    # 步骤2.5: 使用新的co计算逻辑
    if not silent:
        print(f"\n[CO CALCULATION] co计算模式: {co_mode} - {get_co_mode_description(co_mode)}")

    co = compute_co_value(
        co_mode=co_mode,
        knn_distances=knn_distances,
        densities=densities,
        neighbors=neighbors,
        k=k,
        co_manual=co_manual,
        silent=silent
    )

    # 设置日志元数据（如果启用）
    if logger is not None:
        n_high_density = np.sum(high_density_mask)
        # 注意：这里的train_size是原始划分的训练集大小（用户传入）
        # 如果未传入，则默认认为所有样本都是训练集
        actual_train_size = train_size if train_size is not None else len(X)
        actual_test_size = len(X) - actual_train_size if train_size is not None else 0

        logger.set_metadata(
            dataset_name='clustering_dataset',  # 通用名称，无法从这里获取超类名
            total_points=len(X),
            n_high_density=n_high_density,
            train_size=actual_train_size,
            test_size=actual_test_size,
            co_mode=co_mode,
            co_value=co
        )

    # 步骤3: 聚类构建 (集成冲突处理)
    clusters, cluster_labels, high_density_neighbors_map, cluster_category_labels = build_clusters_ssddbc(
        X, high_density_mask, neighbors, labeled_mask, targets, densities, known_mask, k, co=co, silent=silent,
        logger=logger, train_size=train_size
    )

    # 核心子空间孤立簇合并（簇规模<=3）
    merged_core_labels, core_merge_info = merge_isolated_clusters(
        cluster_labels,
        X,
        targets,
        labeled_mask,
        cluster_category_labels=None,
        isolated_threshold=3,
        silent=silent
    )
    cluster_labels = merged_core_labels

    # 按照最新标签重建高密度簇集合
    updated_clusters = []
    unique_core_ids = sorted(np.unique(cluster_labels[cluster_labels >= 0]))
    for cluster_id in unique_core_ids:
        members = np.flatnonzero(cluster_labels == cluster_id)
        updated_clusters.append(set(members.tolist()))
    clusters = updated_clusters

    if not silent and core_merge_info.get('n_merges', 0) > 0:
        print(f"   核心子空间孤立簇合并: {core_merge_info['n_merges']}次")

    # 分析SS-DDBC聚类构建结果（静默模式下跳过）
    if not silent:
        analyze_ssddbc_clustering_result(clusters, cluster_labels, labeled_mask, targets, known_mask)

    # 如果仅评估高密度点，则在此提前返回
    if eval_dense:
        if not silent:
            print(f"\n🎯 仅评估高密度点模式")
            print(f"   跳过步骤4-7（原型构建、低密度点分配、孤立簇合并、未知类识别）")

        # 评估高密度点聚类准确率
        all_acc, old_acc, new_acc, n_clusters, _ = evaluate_high_density_clustering(
            cluster_labels, targets, known_mask, eval_version, X=X, silent=silent
        )

        if not silent:
            print(f"\n✅ 高密度点聚类评估完成: {n_clusters}个聚类")
            print(f"📊 评估结果 (仅高密度点):")
            print(f"   All ACC: {all_acc:.4f}")
            print(f"   Old ACC: {old_acc:.4f}")
            print(f"   New ACC: {new_acc:.4f}")

        # 返回高密度点聚类结果和评估指标
        # eval_dense模式不返回core_clusters（因为没有完整的聚类流程）
        return cluster_labels, n_clusters, [], all_acc, old_acc, new_acc, None, neighbors, None

    # 保存核心点簇（只包含高密度点），用于计算聚类质量第一项
    # 这个core_clusters在整个流程中保持不变，只包含核心点
    core_clusters = [set(cluster) for cluster in clusters]  # 深拷贝保存核心簇

    # 步骤4: 基于partial-clustering结果建立原型
    prototypes, prototype_labels = build_prototypes(X, clusters, labeled_mask, targets)

    # 上帝视角：同时计算原型的真实主导标签（用于调试分析）
    prototype_true_labels = []
    for cluster_id, cluster in enumerate(clusters):
        cluster_indices = list(cluster)
        cluster_true_labels = targets[cluster_indices]  # 上帝视角
        if len(cluster_true_labels) > 0:
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            true_dominant_label = unique_labels[np.argmax(counts)]
            prototype_true_labels.append(true_dominant_label)
        else:
            prototype_true_labels.append(-1)

    # 步骤5: 标签引导分配（可选）+ 稀疏点分配

    # 初始化最终标签为核心点聚类结果
    final_labels = cluster_labels.copy()

    # 重建clusters列表用于稀疏点分配
    # 这个clusters会随着label_guide的分配而更新，确保稀疏点分配时能看到完整的核心点结构
    updated_clusters = [set(cluster) for cluster in clusters]  # 深拷贝

    if label_guide:
        # 标签引导：先将稀疏点中的已知标签样本直接分配到对应标签的核心簇
        final_labels = apply_label_guided_assignment(
            X, final_labels, updated_clusters, labeled_mask, targets, prototypes, prototype_labels, silent
        )

        # 关键：更新clusters，将label_guide分配的有标签稀疏点加入到对应簇中
        # 这样后续的稀疏点分配就能看到这些点了
        if not silent:
            print(f"\n[CORE SPACE UPDATE] 更新核心点空间（加入有标签稀疏点）")

        # 重新构建clusters，包含所有已分配的点
        updated_clusters = [set() for _ in range(len(updated_clusters))]
        for idx in range(len(final_labels)):
            cluster_id = final_labels[idx]
            if cluster_id != -1:
                updated_clusters[cluster_id].add(idx)

        if not silent:
            # 统计有标签稀疏点的数量
            labeled_sparse_added = 0
            for idx in range(len(final_labels)):
                # 如果这个点在原始cluster_labels中是-1（稀疏点），但现在有簇标签了
                if cluster_labels[idx] == -1 and final_labels[idx] != -1 and labeled_mask[idx]:
                    labeled_sparse_added += 1
            print(f"   核心点空间扩展: +{labeled_sparse_added}个有标签稀疏点")
            for cid, cluster in enumerate(updated_clusters):
                if len(cluster) > 0:
                    original_size = len(clusters[cid])
                    new_size = len(cluster)
                    added = new_size - original_size
                    if added > 0:
                        print(f"   簇{cid}: {original_size} → {new_size} (+{added})")

    # 然后分配剩余的稀疏点（无标签或标签引导模式未启用的所有稀疏点）
    # 关键：传递updated_clusters，而不是原始的clusters
    final_labels = assign_sparse_points_density_based(
        X, updated_clusters, final_labels, densities, neighbors, labeled_mask, targets,
        label_threshold=0.1, purity_threshold=0.8, train_size=train_size, silent=silent,
        prototypes=prototypes, prototype_true_labels=prototype_true_labels,
        voting_k=voting_k, assign_model=assign_model,
        label_guide=label_guide  # 传递标签引导模式标志
    )

    # 步骤6: 识别潜在未知类聚类
    unknown_clusters = identify_unknown_clusters_from_predictions(final_labels, labeled_mask)

    n_clusters = len(np.unique(final_labels))
    algorithm_name = "简化SS-DDBC"
    if not silent:
        print(f"✅ {algorithm_name}聚类完成: {n_clusters}个聚类")

        if len(unknown_clusters) > 0:
            print(f"🔍 发现{len(unknown_clusters)}个潜在未知类聚类: {unknown_clusters}")
        else:
            print("🔍 未发现潜在未知类聚类")

    # 写入详细日志（如果启用）
    if logger is not None:
        logger.write_log()
        reset_logger()  # 重置全局日志实例，为下次运行做准备

    # 重新构建簇类别标签（基于最终的簇成员）
    # 应用与build_clusters_ssddbc相同的规则：
    # 1. 簇规模 ≥ 5
    # 2. 已知标签样本占比 > 25% 且数量 ≠ 0
    # 3. 已知样本中，单种标签纯度 ≥ 80%
    final_cluster_category_labels = {}
    for cluster_id, cluster_members in enumerate(updated_clusters):
        cluster_indices = list(cluster_members)
        if len(cluster_indices) < 5:
            continue  # 簇太小，无标签

        # 统计簇中的已知标签样本
        labeled_in_cluster = [idx for idx in cluster_indices if labeled_mask[idx]]
        if len(labeled_in_cluster) == 0:
            continue  # 没有已知标签样本

        # 检查已知标签样本占比
        labeled_ratio = len(labeled_in_cluster) / len(cluster_indices)
        if labeled_ratio <= 0.25:
            continue  # 已知标签样本占比 ≤ 25%

        # 统计已知样本中的标签分布
        label_counts = {}
        for idx in labeled_in_cluster:
            label = targets[idx]
            label_counts[label] = label_counts.get(label, 0) + 1

        # 找出主导标签
        dominant_label = max(label_counts, key=label_counts.get)
        dominant_count = label_counts[dominant_label]
        purity = dominant_count / len(labeled_in_cluster)

        if purity >= 0.8:
            final_cluster_category_labels[cluster_id] = dominant_label  # 纯度 ≥ 80%，簇有类别标签

    # 返回完整的聚类信息（包括原型、更新后的簇、簇类别标签、neighbors和核心簇）
    # updated_clusters包含所有点（核心点+稀疏点），用于错误分析
    # core_clusters只包含核心点（高密度点），用于计算聚类质量第一项
    # neighbors用于避免在compute_local_density_penalty中重复计算KNN
    return final_labels, n_clusters, unknown_clusters, prototypes, updated_clusters, final_cluster_category_labels, neighbors, core_clusters
