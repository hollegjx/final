#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试集错误预测样本分析
分析测试集中预测错误的样本的详细信息
"""

import numpy as np
import os
from datetime import datetime

from config import error_analysis_log_dir


def analyze_error_samples(all_predictions, all_targets, all_known_mask,
                          X_all, densities_all, neighbors_all,
                          high_density_neighbors_map, high_density_mask, train_size,
                          dataset_name=None,
                          log_dir=None,
                          core_space_description=None,
                          prototypes=None):
    """
    分析测试集中预测错误的样本

    Args:
        all_predictions: 全部样本的预测簇标签 (n_samples,)
        all_targets: 全部样本的真实标签 (n_samples,)
        all_known_mask: 全部样本的已知类掩码 (n_samples,)
        X_all: 全部样本的特征 (n_samples, n_features)
        densities_all: 全部样本的密度 (n_samples,)
        neighbors_all: 全部样本的k近邻索引 (n_samples, k)
        high_density_neighbors_map: 高密度子空间邻居映射 {idx: [neighbor_indices]}
        high_density_mask: 高密度点掩码 (n_samples,)
        train_size: 训练集大小
        dataset_name: 数据集名称
        log_dir: 日志保存目录
        core_space_description: 核心点空间定义描述
        prototypes: 簇原型数组 (可选，如提供则显示到各簇原型的距离)

    Returns:
        log_file_path: 日志文件路径，如果没有错误样本则返回None
    """
    from project_utils.cluster_utils import linear_assignment

    # 提取测试集数据
    test_predictions = all_predictions[train_size:]
    test_targets = all_targets[train_size:]
    test_known_mask = all_known_mask[train_size:]
    test_size = len(test_predictions)

    if test_size == 0:
        print("[WARNING] 没有测试集数据")
        return None

    # ========== 步骤1: 计算线性分配映射 ==========
    test_targets_int = test_targets.astype(int)
    test_predictions_int = test_predictions.astype(int)

    D = max(test_predictions_int.max(), test_targets_int.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(len(test_predictions_int)):
        w[test_predictions_int[i], test_targets_int[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    # cluster_to_label: {预测簇 -> 真实标签}
    cluster_to_label = {i: j for i, j in ind}

    # ========== 步骤2: 识别错误样本 ==========
    error_indices = []  # 测试集内的索引

    for test_idx in range(test_size):
        pred_cluster = test_predictions_int[test_idx]
        true_label = test_targets_int[test_idx]
        assigned_label = cluster_to_label.get(pred_cluster, -1)

        if assigned_label != true_label:
            error_indices.append(test_idx)

    error_count = len(error_indices)

    if error_count == 0:
        print("[INFO] 测试集没有错误样本")
        return None

    print(f"[INFO] 测试集错误样本: {error_count}/{test_size} ({error_count/test_size:.2%})")

    # ========== 步骤3: 按密度排序错误样本 ==========
    error_global_indices = np.array([train_size + idx for idx in error_indices])
    error_densities = densities_all[error_global_indices]
    sorted_order = np.argsort(error_densities)[::-1]  # 从高到低
    error_global_indices = error_global_indices[sorted_order]

    # 计算全局密度排名
    all_density_ranks = np.argsort(np.argsort(densities_all)[::-1]) + 1

    # ========== 步骤4: 生成报告 ==========
    log_dir = log_dir or error_analysis_log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_str = f"{dataset_name}_" if dataset_name else ""
    filename = f"error_samples_{dataset_str}{timestamp}.txt"
    log_path = os.path.join(log_dir, filename)

    print(f"[INFO] 正在生成错误样本分析报告: {log_path}")

    with open(log_path, 'w', encoding='utf-8') as f:
        # 写入头部
        f.write("=" * 100 + "\n")
        f.write("测试集错误预测样本分析报告\n")
        f.write("=" * 100 + "\n\n")

        # 写入元数据
        f.write("[元数据]\n")
        f.write(f"数据集: {dataset_name or 'N/A'}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练集大小: {train_size}\n")
        f.write(f"测试集大小: {test_size}\n")
        f.write(f"错误样本数: {error_count}\n")
        f.write(f"错误率: {error_count/test_size:.2%}\n")
        if core_space_description:
            f.write(f"核心点空间定义: {core_space_description}\n")
        f.write("\n")

        # 写入线性分配映射
        f.write("[线性分配映射 (预测簇 -> 真实标签)]\n")
        for pred_cluster, true_label in sorted(cluster_to_label.items()):
            f.write(f"  簇 {pred_cluster} -> 标签 {true_label}\n")
        f.write("\n" + "=" * 100 + "\n\n")

        # 写入每个错误样本的详细信息
        f.write("[错误样本详细信息] (按密度从高到低排序)\n")
        f.write("=" * 100 + "\n\n")

        for rank, global_idx in enumerate(error_global_indices, 1):
            test_idx = global_idx - train_size

            true_label = all_targets[global_idx]
            pred_cluster = all_predictions[global_idx]
            assigned_label = cluster_to_label.get(pred_cluster, -1)
            pred_label = assigned_label  # 预测标签（通过簇映射得到）
            density = densities_all[global_idx]
            density_rank = all_density_ranks[global_idx]
            is_known = all_known_mask[global_idx]

            f.write(f"[错误样本 #{rank}]\n")
            f.write(f"-" * 100 + "\n")
            f.write(f"索引信息:\n")
            f.write(f"  全局索引: {global_idx}\n")
            f.write(f"  测试集索引: {test_idx}\n")
            f.write(f"\n")
            f.write(f"基本信息:\n")
            f.write(f"  密度: {density:.6f}\n")
            f.write(f"  密度排名: {density_rank}/{len(densities_all)} (全局)\n")
            f.write(f"  是否高密度点: {'是' if high_density_mask[global_idx] else '否'}\n")
            f.write(f"  真实标签: {true_label}\n")
            f.write(f"  预测标签: {pred_label} (来自簇{pred_cluster})\n")
            f.write(f"  已知/未知: {'已知类' if is_known else '未知类'}\n")
            f.write(f"\n")

            # 到各簇原型的距离
            if prototypes is not None and len(prototypes) > 0:
                f.write("到各簇原型的距离:\n")
                point_features = X_all[global_idx]

                # 计算到每个簇原型的距离
                distances_to_prototypes = []
                for cluster_id in range(len(prototypes)):
                    dist = np.linalg.norm(point_features - prototypes[cluster_id])
                    cluster_label = cluster_to_label.get(cluster_id, -1)
                    distances_to_prototypes.append((cluster_id, cluster_label, dist))

                # 按距离从近到远排序
                distances_to_prototypes.sort(key=lambda x: x[2])

                f.write(f"  {'排名':<6} {'簇ID':<8} {'簇标签':<8} {'距离':<12} {'备注'}\n")
                f.write(f"  {'-' * 50}\n")

                for rank, (cluster_id, cluster_label, dist) in enumerate(distances_to_prototypes, 1):
                    remark = ""
                    if cluster_id == pred_cluster:
                        remark = "← 当前分配的簇"
                    elif cluster_label == true_label:
                        remark = "← 真实标签对应的簇"

                    f.write(f"  {rank:<6} {cluster_id:<8} {cluster_label:<8} {dist:<12.4f} {remark}\n")

                f.write(f"\n")

            # 高密度子空间邻居
            f.write("高密度子空间k近邻:\n")

            # 如果样本本身在高密度子空间中，使用邻居映射
            if global_idx in high_density_neighbors_map:
                hd_neighbors = high_density_neighbors_map[global_idx]
                f.write(f"  该样本在高密度子空间中，共 {len(hd_neighbors)} 个高密度邻居:\n")
            else:
                # 样本不在高密度子空间中，计算到所有高密度点的距离，取最近的k个
                high_density_indices = np.where(high_density_mask)[0]
                if len(high_density_indices) > 0:
                    # 计算当前样本到所有高密度点的距离
                    distances_to_hd = []
                    for hd_idx in high_density_indices:
                        if hd_idx != global_idx:  # 排除自己
                            dist = np.linalg.norm(X_all[global_idx] - X_all[hd_idx])
                            distances_to_hd.append((dist, hd_idx))

                    # 按距离排序，取最近的k个（这里k取邻居数组的长度）
                    k_neighbors = len(neighbors_all[global_idx]) if global_idx < len(neighbors_all) else 10
                    distances_to_hd.sort(key=lambda x: x[0])
                    hd_neighbors = [idx for _, idx in distances_to_hd[:k_neighbors]]

                    f.write(f"  该样本不在高密度子空间中，计算最近的 {len(hd_neighbors)} 个高密度邻居:\n")
                else:
                    hd_neighbors = []
                    f.write("  无高密度点\n")

            if len(hd_neighbors) > 0:
                f.write(f"  {'序号':<4} {'全局索引':<8} {'距离':<10} {'密度':<12} {'密度排名':<10} {'高密度':<6} {'真实标签':<8} {'预测标签':<8} {'来源':<6} {'备注'}\n")
                f.write(f"  {'-' * 100}\n")

                for i, nb_idx in enumerate(hd_neighbors[:20], 1):  # 最多显示前20个
                    if nb_idx < len(X_all):
                        dist = np.linalg.norm(X_all[global_idx] - X_all[nb_idx])
                        nb_density = densities_all[nb_idx]
                        nb_density_rank = all_density_ranks[nb_idx]
                        nb_is_hd = '是' if high_density_mask[nb_idx] else '否'
                        nb_true_label = all_targets[nb_idx]
                        nb_pred_cluster = all_predictions[nb_idx]
                        nb_pred_label = cluster_to_label.get(nb_pred_cluster, -1)  # 转换为预测标签
                        source = '训练集' if nb_idx < train_size else '测试集'
                        remark = '同标签' if nb_true_label == true_label else '异标签'

                        f.write(f"  {i:<4} {nb_idx:<8} {dist:<10.4f} {nb_density:<12.6f} {nb_density_rank:<10} {nb_is_hd:<6} "
                               f"{nb_true_label:<8} {nb_pred_label:<8} {source:<6} {remark}\n")

                if len(hd_neighbors) > 20:
                    f.write(f"  ... 还有 {len(hd_neighbors)-20} 个邻居未显示\n")

            f.write(f"\n")

            # 全空间k近邻
            f.write("全空间k近邻:\n")
            if global_idx < len(neighbors_all):
                all_neighbors = neighbors_all[global_idx]
                f.write(f"  共 {len(all_neighbors)} 个邻居:\n")
                f.write(f"  {'序号':<4} {'全局索引':<8} {'距离':<10} {'密度':<12} {'密度排名':<10} {'真实标签':<8} {'预测标签':<8} {'来源':<6} {'备注'}\n")
                f.write(f"  {'-' * 92}\n")

                for i, nb_idx in enumerate(all_neighbors, 1):
                    if nb_idx < len(X_all):
                        dist = np.linalg.norm(X_all[global_idx] - X_all[nb_idx])
                        nb_density = densities_all[nb_idx]
                        nb_density_rank = all_density_ranks[nb_idx]
                        nb_true_label = all_targets[nb_idx]
                        nb_pred_cluster = all_predictions[nb_idx]
                        nb_pred_label = cluster_to_label.get(nb_pred_cluster, -1)  # 转换为预测标签
                        source = '训练集' if nb_idx < train_size else '测试集'
                        remark = '同标签' if nb_true_label == true_label else '异标签'

                        f.write(f"  {i:<4} {nb_idx:<8} {dist:<10.4f} {nb_density:<12.6f} {nb_density_rank:<10} "
                               f"{nb_true_label:<8} {nb_pred_label:<8} {source:<6} {remark}\n")
            else:
                f.write("  无邻居信息\n")

            f.write(f"\n{'=' * 100}\n\n")

        # 统计信息
        f.write("[统计总结]\n")
        f.write("=" * 100 + "\n")

        from collections import Counter
        error_true_labels = [all_targets[idx] for idx in error_global_indices]
        error_pred_clusters = [all_predictions[idx] for idx in error_global_indices]
        error_pred_labels = [cluster_to_label.get(c, -1) for c in error_pred_clusters]  # 转换为预测标签
        error_known_status = [all_known_mask[idx] for idx in error_global_indices]

        f.write(f"\n按真实标签统计:\n")
        for label, count in sorted(Counter(error_true_labels).items()):
            f.write(f"  标签 {label}: {count} 个 ({count/error_count:.1%})\n")

        f.write(f"\n按预测标签统计:\n")
        for label, count in sorted(Counter(error_pred_labels).items()):
            f.write(f"  标签 {label}: {count} 个 ({count/error_count:.1%})\n")

        f.write(f"\n按已知/未知统计:\n")
        known_count = sum(error_known_status)
        unknown_count = error_count - known_count
        f.write(f"  已知类: {known_count} 个 ({known_count/error_count:.1%})\n")
        f.write(f"  未知类: {unknown_count} 个 ({unknown_count/error_count:.1%})\n")

        f.write("\n" + "=" * 100 + "\n")

    print(f"[INFO] 报告已保存: {log_path}")
    return log_path
