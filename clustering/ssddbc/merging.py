#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SS-DDBC聚类合并模块
实现聚类完成后的簇合并策略
"""

import numpy as np
from collections import defaultdict


def merge_clusters_by_category_label(predictions, cluster_category_labels, silent=False):
    """
    根据簇的类别标签合并簇

    功能：
    - 将拥有相同类别标签的簇合并成一个簇
    - 未知类（category_label=None）的簇保持独立，不进行合并
    - 稀疏点分配后可能产生多个相同类别的簇，需要合并避免评估时受孤立点影响

    Args:
        predictions: 聚类预测结果 (n_samples,)
        cluster_category_labels: 簇类别标签字典 {cluster_id: category_label or None}
        silent: 是否静默模式

    Returns:
        merged_predictions: 合并后的聚类预测结果 (n_samples,)
        merge_info: 合并信息字典
            {
                'n_original_clusters': 原始簇数量,
                'n_merged_clusters': 合并后簇数量,
                'n_merges': 合并次数,
                'merge_details': [合并详情列表]
            }

    Example:
        假设有以下簇：
        - 簇0: category_label=5 (50个样本)
        - 簇1: category_label=None (20个样本, 未知类)
        - 簇2: category_label=5 (30个样本)
        - 簇3: category_label=7 (40个样本)
        - 簇4: category_label=None (15个样本, 未知类)

        合并后：
        - 簇0: category_label=5 (80个样本, 合并了簇0和簇2)
        - 簇1: category_label=None (20个样本, 保持独立)
        - 簇2: category_label=7 (40个样本, 原簇3)
        - 簇3: category_label=None (15个样本, 原簇4, 保持独立)
    """
    if not silent:
        print(f"\n根据簇类别标签进行合并...")

    merged_predictions = predictions.copy()
    merge_details = []

    # 统计原始簇信息
    unique_clusters = np.unique(predictions)
    n_original_clusters = len(unique_clusters)

    if not silent:
        print(f"   原始簇数量: {n_original_clusters}")

    # 按类别标签分组簇（未知类除外）
    # category_label -> [cluster_ids]
    category_to_clusters = defaultdict(list)
    unknown_clusters = []  # 未知类簇列表

    for cluster_id in unique_clusters:
        category_label = cluster_category_labels.get(cluster_id, None)

        if category_label is None:
            # 未知类簇，记录但不合并
            unknown_clusters.append(cluster_id)
        else:
            # 有类别标签的簇，按标签分组
            category_to_clusters[category_label].append(cluster_id)

    if not silent:
        print(f"   有类别标签的簇: {len(category_to_clusters)}个不同类别")
        print(f"   未知类簇: {len(unknown_clusters)}个（保持独立）")

    # 执行合并：相同类别标签的簇合并成第一个簇
    n_merges = 0
    for category_label, cluster_ids in category_to_clusters.items():
        if len(cluster_ids) <= 1:
            # 该类别只有一个簇，无需合并
            continue

        # 合并策略：将所有簇合并到第一个簇ID
        target_cluster = cluster_ids[0]
        source_clusters = cluster_ids[1:]

        # 统计合并前的簇大小
        target_size = np.sum(predictions == target_cluster)
        source_sizes = [np.sum(predictions == cid) for cid in source_clusters]

        # 执行合并
        for source_cluster in source_clusters:
            mask = merged_predictions == source_cluster
            merged_predictions[mask] = target_cluster
            n_merges += 1

        # 记录合并详情
        merge_details.append({
            'category_label': int(category_label),
            'target_cluster': int(target_cluster),
            'source_clusters': [int(cid) for cid in source_clusters],
            'target_size_before': int(target_size),
            'source_sizes': [int(s) for s in source_sizes],
            'total_size_after': int(target_size + sum(source_sizes))
        })

        if not silent:
            print(f"   [OK] 类别{category_label}: 合并 {source_clusters} -> {target_cluster} "
                  f"({target_size} + {source_sizes} = {target_size + sum(source_sizes)}个样本)")

    # 重新编号簇标签，消除空隙
    unique_merged_clusters = np.unique(merged_predictions)
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_merged_clusters)}
    final_predictions = np.array([cluster_mapping[pred] for pred in merged_predictions])

    n_merged_clusters = len(unique_merged_clusters)

    # 构建合并信息
    merge_info = {
        'n_original_clusters': int(n_original_clusters),
        'n_merged_clusters': int(n_merged_clusters),
        'n_merges': int(n_merges),
        'n_unknown_clusters': len(unknown_clusters),
        'merge_details': merge_details
    }

    if not silent:
        print(f"   合并完成: {n_original_clusters}个簇 → {n_merged_clusters}个簇 (执行{n_merges}次合并)")

    return final_predictions, merge_info


def get_cluster_category_labels(predictions, targets, labeled_mask,
                                min_cluster_size=5, min_labeled_ratio=0.25,
                                min_purity=0.8, silent=False):
    """
    为每个簇确定类别标签

    规则：
    1. 簇规模 ≥ min_cluster_size
    2. 已知标签样本占比 > min_labeled_ratio
    3. 已知样本中，单种标签纯度 ≥ min_purity

    Args:
        predictions: 聚类预测结果 (n_samples,)
        targets: 真实标签 (n_samples,)
        labeled_mask: 有标签掩码 (n_samples,)
        min_cluster_size: 最小簇规模阈值
        min_labeled_ratio: 最小已标记样本占比
        min_purity: 最小标签纯度
        silent: 是否静默模式

    Returns:
        cluster_category_labels: 簇类别标签字典 {cluster_id: category_label or None}
    """
    if not silent:
        print(f"\n确定簇类别标签...")
        print(f"   规则: 规模>={min_cluster_size}, 标记占比>{min_labeled_ratio:.0%}, 纯度>={min_purity:.0%}")

    cluster_category_labels = {}
    unique_clusters = np.unique(predictions)

    n_labeled_clusters = 0
    n_unknown_clusters = 0

    for cluster_id in unique_clusters:
        cluster_indices = np.where(predictions == cluster_id)[0]
        cluster_size = len(cluster_indices)

        # 条件1：簇规模检查
        if cluster_size < min_cluster_size:
            cluster_category_labels[cluster_id] = None
            n_unknown_clusters += 1
            continue

        # 统计簇中的已标记样本
        labeled_in_cluster = cluster_indices[labeled_mask[cluster_indices]]
        n_labeled = len(labeled_in_cluster)

        # 条件2：已标记样本占比检查
        labeled_ratio = n_labeled / cluster_size
        if labeled_ratio <= min_labeled_ratio:
            cluster_category_labels[cluster_id] = None
            n_unknown_clusters += 1
            continue

        # 条件3：标签纯度检查
        labeled_targets = targets[labeled_in_cluster]
        unique_labels, label_counts = np.unique(labeled_targets, return_counts=True)

        dominant_idx = np.argmax(label_counts)
        dominant_label = unique_labels[dominant_idx]
        dominant_count = label_counts[dominant_idx]
        purity = dominant_count / n_labeled

        if purity >= min_purity:
            cluster_category_labels[cluster_id] = int(dominant_label)
            n_labeled_clusters += 1
            if not silent:
                print(f"   簇{cluster_id}: 类别{dominant_label} "
                      f"(规模:{cluster_size}, 标记:{n_labeled}/{cluster_size}={labeled_ratio:.1%}, "
                      f"纯度:{purity:.1%})")
        else:
            cluster_category_labels[cluster_id] = None
            n_unknown_clusters += 1

    if not silent:
        print(f"   结果: {n_labeled_clusters}个有类别标签, {n_unknown_clusters}个未知类")

    return cluster_category_labels


# 为了兼容旧代码，保留原函数名的别名
def merge_clusters_by_label_consistency(predictions, targets, labeled_mask,
                                       dominance_threshold=0.8, detail_merge=False):
    """
    基于标签一致性合并聚类（兼容旧接口）

    内部调用新的合并流程：
    1. 获取簇类别标签
    2. 按类别标签合并簇

    Args:
        predictions: 聚类预测结果
        targets: 真实标签
        labeled_mask: 有标签掩码
        dominance_threshold: 主导标签的最小占比阈值（映射到min_purity）
        detail_merge: 是否显示详细合并过程

    Returns:
        merged_predictions: 合并后的聚类预测结果
        merge_info: 合并信息（为兼容返回列表格式）
    """
    # 步骤1：获取簇类别标签
    cluster_category_labels = get_cluster_category_labels(
        predictions, targets, labeled_mask,
        min_cluster_size=5,
        min_labeled_ratio=0.25,
        min_purity=dominance_threshold,
        silent=not detail_merge
    )

    # 步骤2：按类别标签合并簇
    merged_predictions, merge_info_dict = merge_clusters_by_category_label(
        predictions, cluster_category_labels,
        silent=not detail_merge
    )

    # 转换为旧格式的merge_info（列表）
    merge_info_list = merge_info_dict.get('merge_details', [])

    return merged_predictions, merge_info_list


def merge_isolated_clusters(predictions, X, targets, labeled_mask,
                            cluster_category_labels=None,
                            isolated_threshold=3,
                            silent=False):
    """
    处理孤立簇（簇大小 ≤ isolated_threshold）

    重要：孤立簇只能合并到非孤立簇（size > isolated_threshold），不会孤立簇之间互相合并

    策略：
    1. 识别孤立簇（size ≤ isolated_threshold）和非孤立簇（size > isolated_threshold）
    2. 对每个孤立簇：
       a. 如果标签纯度=1（所有有标签样本标签相同），直接合并到相同标签的任意非孤立簇
       b. 否则（标签不纯或无标签），计算到所有非孤立簇的距离，合并到平均距离最小的非孤立簇
    3. 距离计算：
       - 孤立簇的代表点（size=1用该点，size>1用簇原型）
       - 到目标簇最近的min(3, cluster_size)个样本的平均距离

    Args:
        predictions: 聚类预测结果 (n_samples,)
        X: 特征矩阵 (n_samples, feat_dim)
        targets: 真实标签 (n_samples,)
        labeled_mask: 有标签掩码 (n_samples,)
        cluster_category_labels: 簇类别标签字典 {cluster_id: category_label or None}
        isolated_threshold: 孤立簇阈值，默认3
        silent: 是否静默模式

    Returns:
        merged_predictions: 合并后的聚类预测结果 (n_samples,)
        merge_info: 合并信息字典
            {
                'n_isolated_clusters': 孤立簇数量,
                'n_isolated_points': 孤立点总数,
                'n_merges': 合并次数,
                'merge_details': [合并详情列表]
            }
    """
    if not silent:
        print(f"\n处理孤立簇（大小<={isolated_threshold}）...")

    merged_predictions = predictions.copy()
    merge_details = []

    # 统计所有簇的信息（排除-1，即未分配的点）
    unique_clusters = np.unique(predictions)
    unique_clusters = unique_clusters[unique_clusters >= 0]  # 排除-1
    cluster_info = {}

    for cluster_id in unique_clusters:
        cluster_indices = np.where(predictions == cluster_id)[0]
        cluster_size = len(cluster_indices)

        # 计算簇原型（中心点）
        cluster_prototype = np.mean(X[cluster_indices], axis=0)

        # 获取簇的类别标签（如果提供）
        category_label = None
        if cluster_category_labels is not None:
            category_label = cluster_category_labels.get(cluster_id, None)

        # 分析簇中的已标记样本
        labeled_in_cluster = cluster_indices[labeled_mask[cluster_indices]]
        unique_label = None
        if len(labeled_in_cluster) > 0:
            cluster_targets = targets[labeled_in_cluster]
            unique_labels, label_counts = np.unique(cluster_targets, return_counts=True)

            # 检查是否有唯一标签（所有已标记样本的标签相同）
            if len(unique_labels) == 1:
                unique_label = int(unique_labels[0])

        cluster_info[cluster_id] = {
            'size': cluster_size,
            'indices': cluster_indices,
            'prototype': cluster_prototype,
            'category_label': category_label,
            'unique_label': unique_label,  # 唯一的已知标签（如果存在）
            'is_isolated': cluster_size <= isolated_threshold
        }

    # 分离孤立簇和非孤立簇
    isolated_clusters = [cid for cid, info in cluster_info.items() if info['is_isolated']]
    non_isolated_clusters = [cid for cid, info in cluster_info.items() if not info['is_isolated']]

    n_isolated_clusters = len(isolated_clusters)
    n_isolated_points = sum([cluster_info[cid]['size'] for cid in isolated_clusters])

    if not silent:
        print(f"   孤立簇数量: {n_isolated_clusters}个")
        print(f"   孤立点总数: {n_isolated_points}个样本")
        print(f"   非孤立簇数量: {len(non_isolated_clusters)}个")

    if n_isolated_clusters == 0:
        if not silent:
            print(f"   无孤立簇，跳过处理")
        return merged_predictions, {
            'n_isolated_clusters': 0,
            'n_isolated_points': 0,
            'n_merges': 0,
            'merge_details': []
        }

    if len(non_isolated_clusters) == 0:
        if not silent:
            print(f"   警告: 所有簇都是孤立簇，无法合并")
        return merged_predictions, {
            'n_isolated_clusters': n_isolated_clusters,
            'n_isolated_points': n_isolated_points,
            'n_merges': 0,
            'merge_details': []
        }

    # 处理每个孤立簇，全部合并到非孤立簇
    n_merges = 0
    for isolated_id in isolated_clusters:
        isolated_info = cluster_info[isolated_id]
        isolated_size = isolated_info['size']
        isolated_indices = isolated_info['indices']
        isolated_prototype = isolated_info['prototype']
        unique_label = isolated_info['unique_label']

        # 策略1：如果有唯一标签，合并到相同标签的非孤立簇
        if unique_label is not None:
            # 找非孤立簇中的同标签簇
            same_label_clusters = [
                cid for cid in non_isolated_clusters
                if (cluster_info[cid]['unique_label'] == unique_label or
                    cluster_info[cid]['category_label'] == unique_label)
            ]

            if len(same_label_clusters) > 0:
                target_cluster = same_label_clusters[0]
                target_size = cluster_info[target_cluster]['size']

                merged_predictions[isolated_indices] = target_cluster
                n_merges += 1

                merge_details.append({
                    'source_cluster': int(isolated_id),
                    'target_cluster': int(target_cluster),
                    'source_size': int(isolated_size),
                    'target_size': int(target_size),
                    'merge_strategy': 'label_match',
                    'shared_label': int(unique_label)
                })

                if not silent:
                    print(f"   [标签匹配] 孤立簇{isolated_id}({isolated_size}个样本, 标签={unique_label}) "
                          f"-> 簇{target_cluster}({target_size}个样本)")

                continue

        # 策略2：按距离合并到最近的非孤立簇
        # 使用最近3个样本的平均距离进行投票
        min_avg_distance = float('inf')
        target_cluster = None

        for non_isolated_id in non_isolated_clusters:
            non_isolated_indices = cluster_info[non_isolated_id]['indices']

            # 计算孤立簇代表点
            if isolated_size == 1:
                isolated_point = X[isolated_indices[0]]
            else:
                isolated_point = isolated_prototype

            # 计算到簇内所有样本的距离
            distances_to_cluster = np.linalg.norm(
                X[non_isolated_indices] - isolated_point, axis=1
            )

            # 选择最近的min(3, cluster_size)个样本
            k_nearest = min(3, len(distances_to_cluster))
            nearest_distances = np.sort(distances_to_cluster)[:k_nearest]

            # 计算平均距离
            avg_distance = np.mean(nearest_distances)

            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                target_cluster = non_isolated_id

        # 执行合并
        if target_cluster is not None:
            target_size = cluster_info[target_cluster]['size']
            merged_predictions[isolated_indices] = target_cluster
            n_merges += 1

            merge_details.append({
                'source_cluster': int(isolated_id),
                'target_cluster': int(target_cluster),
                'source_size': int(isolated_size),
                'target_size': int(target_size),
                'merge_strategy': 'nearest_3_avg_distance',
                'avg_distance': float(min_avg_distance)
            })

            if not silent:
                print(f"   [距离匹配] 孤立簇{isolated_id}({isolated_size}个样本) "
                      f"-> 簇{target_cluster}({target_size}个样本, 平均距离={min_avg_distance:.3f})")

    # 重新编号簇标签，消除空隙
    # 保持未分配样本(-1)不变，仅对非负簇ID重新编号
    final_predictions = merged_predictions.copy()
    unique_non_negative = sorted([cid for cid in np.unique(merged_predictions) if cid >= 0])

    if len(unique_non_negative) > 0:
        cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_non_negative)}
        for old_id, new_id in cluster_mapping.items():
            final_predictions[merged_predictions == old_id] = new_id

    # 构建合并信息
    merge_info = {
        'n_isolated_clusters': int(n_isolated_clusters),
        'n_isolated_points': int(n_isolated_points),
        'n_merges': int(n_merges),
        'merge_details': merge_details
    }

    if not silent:
        print(f"   孤立簇处理完成: 合并了{n_merges}个孤立簇")

    return final_predictions, merge_info
