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
from ..density.density_estimation import (
    compute_median_density,
    identify_high_density_points,
    compute_relative_density,
)
from ..utils.co_calculation import compute_co_value, get_co_mode_description
from ..unknown.detection import identify_unknown_clusters_from_predictions


def adaptive_density_clustering(X, targets, known_mask, labeled_mask,
                               k=10, density_percentile=75, random_state=0, train_size=None,
                               co_mode=2, co_manual=None,
                               eval_dense=False, eval_version='v1',
                               fast_mode=False, dense_method=0, assign_model=2, voting_k=5, detail_dense=False,
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
        fast_mode: 快速模式，跳过网格搜索不需要的计算（未知簇识别、簇类别标签重建），默认False
        dense_method: 密度计算方法 (0=平均距离, 1=中位数距离, 2=归一化倒数, 3=指数密度，默认0)
        assign_model: 稀疏点分配策略 (当前实现仅支持3：簇内K近邻平均距离)
        voting_k: KNN投票时使用的近邻数量（默认5）
        detail_dense: 是否记录骨干网络聚类详细日志（默认False）
        label_guide: 是否启用标签引导模式（默认False；原型移除后该选项被忽略）

    Returns:
        predictions: 聚类预测结果
        n_clusters: 聚类数量
        unknown_clusters: 潜在未知类聚类索引
        updated_clusters: 更新后的簇成员集合（包含所有点，用于错误分析）
        cluster_category_labels: 簇ID到类别标签的映射字典（只包含有类别标签的簇）
        neighbors: K近邻索引矩阵 (n_samples, k)（用于避免重复计算）
        core_clusters: 核心点簇（只包含高密度点，用于计算聚类质量第一项）
        如果eval_dense=True，还会返回评估指标: (all_acc, old_acc, new_acc, None)
    """
    # 设置局部numpy随机种子 (遵循K-means的设计模式)
    np.random.seed(random_state)

    # 步骤1: 固定使用中位数距离密度（dense_method 已在上游入口限制为 1）
    if dense_method != 1:
        raise ValueError("当前自适应聚类仅支持 dense_method=1（中位数距离密度）。")

    densities, knn_distances, neighbors = compute_median_density(X, k, silent=fast_mode)

    # 步骤2: 计算相对密度并识别高密度点（使用百分位数阈值）
    relative_densities = compute_relative_density(densities, neighbors, k, silent=fast_mode)
    high_density_mask = identify_high_density_points(relative_densities, density_percentile, use_relative=True, silent=fast_mode)

    co = compute_co_value(
        co_mode=co_mode,
        knn_distances=knn_distances,
        densities=densities,
        neighbors=neighbors,
        k=k,
        co_manual=co_manual,
        silent=fast_mode
    )

    # 步骤3: 聚类构建 (集成冲突处理)
    clusters, cluster_labels, high_density_neighbors_map, cluster_category_labels = build_clusters_ssddbc(
        X, high_density_mask, neighbors, labeled_mask, targets, densities, known_mask, k, co=co, silent=fast_mode,
        logger=None, train_size=train_size
    )

    # 核心子空间孤立簇合并（簇规模<=3）
    merged_core_labels, core_merge_info = merge_isolated_clusters(
        cluster_labels,
        X,
        targets,
        labeled_mask,
        cluster_category_labels=None,
        isolated_threshold=3,
        silent=fast_mode
    )
    cluster_labels = merged_core_labels

    # 按照最新标签重建高密度簇集合
    updated_clusters = []
    unique_core_ids = sorted(np.unique(cluster_labels[cluster_labels >= 0]))
    for cluster_id in unique_core_ids:
        members = np.flatnonzero(cluster_labels == cluster_id)
        updated_clusters.append(set(members.tolist()))
    clusters = updated_clusters

    # 如果仅评估高密度点，则在此提前返回
    if eval_dense:
        # 评估高密度点聚类准确率
        all_acc, old_acc, new_acc, n_clusters, _ = evaluate_high_density_clustering(
            cluster_labels, targets, known_mask, eval_version, X=X, silent=fast_mode
        )

        # 返回高密度点聚类结果和评估指标
        # eval_dense模式不返回core_clusters（因为没有完整的聚类流程）
        return cluster_labels, n_clusters, [], all_acc, old_acc, new_acc, None, neighbors, None, densities

    # 保存核心点簇（只包含高密度点），用于计算聚类质量第一项
    # 这个core_clusters在整个流程中保持不变，只包含核心点
    core_clusters = [set(cluster) for cluster in clusters]  # 深拷贝保存核心簇

    # 步骤4: 标签引导分配（已停用）+ 稀疏点分配

    # 初始化最终标签为核心点聚类结果
    final_labels = cluster_labels.copy()

    # 重建clusters列表用于稀疏点分配
    # 这个clusters会随着label_guide的分配而更新，确保稀疏点分配时能看到完整的核心点结构
    updated_clusters = [set(cluster) for cluster in clusters]  # 深拷贝
    if label_guide and not fast_mode:
        print("⚠️ label_guide 模式依赖原型构建，当前已停用该分支，其参数将被忽略。")

    # 然后分配剩余的稀疏点（无标签或标签引导模式未启用的所有稀疏点）
    # 关键：传递updated_clusters，而不是原始的clusters
    final_labels = assign_sparse_points_density_based(
        X, updated_clusters, final_labels, densities, neighbors, labeled_mask, targets,
        label_threshold=0.1, purity_threshold=0.8, train_size=train_size, silent=fast_mode,
        voting_k=voting_k, assign_model=assign_model,
        label_guide=label_guide  # 传递标签引导模式标志
    )

    # 步骤6: 识别潜在未知类聚类（fast_mode时跳过）
    if fast_mode:
        unknown_clusters = []
    else:
        unknown_clusters = identify_unknown_clusters_from_predictions(final_labels, labeled_mask)

    n_clusters = len(np.unique(final_labels))

    # 重新构建簇类别标签（基于最终的簇成员）
    # fast_mode时跳过，网格搜索不需要此计算
    if fast_mode:
        final_cluster_category_labels = {}
    else:
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

    # 返回完整的聚类信息（包括更新后的簇、簇类别标签、neighbors和核心簇）
    # updated_clusters包含所有点（核心点+稀疏点），用于错误分析
    # core_clusters只包含核心点（高密度点），用于计算聚类质量第一项
    # neighbors用于避免在compute_local_density_penalty中重复计算KNN
    return (final_labels, n_clusters, unknown_clusters, updated_clusters,
            final_cluster_category_labels, neighbors, core_clusters, densities)
