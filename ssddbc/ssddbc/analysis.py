#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SS-DDBC聚类结果分析模块
提供详细的聚类质量分析和可视化
"""

import numpy as np


def analyze_ssddbc_clustering_result(clusters, cluster_labels, labeled_mask, targets, known_mask):
    """
    分析SS-DDBC聚类构建步骤的结果

    Args:
        clusters: 聚类列表
        cluster_labels: 聚类标签
        labeled_mask: 有标签掩码
        targets: 真实标签
        known_mask: 已知类掩码
    """
    print(f"\n📊 SS-DDBC聚类构建结果分析:")
    print("="*80)

    total_samples = len(cluster_labels)
    assigned_samples = np.sum(cluster_labels != -1)
    unassigned_samples = total_samples - assigned_samples

    print(f"总体统计:")
    print(f"   总样本数: {total_samples}")
    print(f"   已分配样本: {assigned_samples}")
    print(f"   未分配样本: {unassigned_samples}")
    print(f"   聚类数量: {len(clusters)}")

    print(f"\n各聚类详细分析:")

    for cluster_id, cluster in enumerate(clusters):
        cluster_indices = list(cluster)
        cluster_size = len(cluster_indices)

        # 统计有标签/无标签样本
        cluster_labeled_mask = labeled_mask[cluster_indices]
        labeled_count = np.sum(cluster_labeled_mask)
        unlabeled_count = cluster_size - labeled_count

        # 统计已知类/未知类样本
        cluster_known_mask = known_mask[cluster_indices]
        known_count = np.sum(cluster_known_mask)
        unknown_count = cluster_size - known_count

        # 分析标签分布（有标签样本）
        label_distribution = {}
        dominant_label = None
        label_purity = 0.0

        if labeled_count > 0:
            labeled_targets = targets[cluster_indices][cluster_labeled_mask]
            unique_labels, counts = np.unique(labeled_targets, return_counts=True)
            label_distribution = dict(zip(unique_labels, counts))
            dominant_label = unique_labels[np.argmax(counts)]
            label_purity = np.max(counts) / labeled_count

        # 分析真实类别分布（所有样本）
        all_targets = targets[cluster_indices]
        all_unique_labels, all_counts = np.unique(all_targets, return_counts=True)
        true_distribution = dict(zip(all_unique_labels, all_counts))

        # 输出聚类信息
        print(f"\n聚类 #{cluster_id} (大小: {cluster_size}):")
        print(f"   样本组成:")
        print(f"     有标签样本: {labeled_count} 个")
        print(f"     无标签样本: {unlabeled_count} 个")
        print(f"     已知类样本: {known_count} 个")
        print(f"     未知类样本: {unknown_count} 个")

        if labeled_count > 0:
            print(f"   有标签样本分布:")
            print(f"     主导标签: {dominant_label} (纯度: {label_purity:.3f})")
            print(f"     详细分布: {label_distribution}")

            # 检查标签冲突
            if len(label_distribution) > 1:
                print(f"     ⚠️  标签冲突: 包含{len(label_distribution)}种不同标签")
        else:
            print(f"   有标签样本分布: 无有标签样本 (潜在未知类)")

        print(f"   真实类别分布 (所有样本): {true_distribution}")

    # 未分配样本分析
    if unassigned_samples > 0:
        print(f"\n未分配样本分析 ({unassigned_samples}个):")
        unassigned_indices = np.where(cluster_labels == -1)[0]
        unassigned_labeled = np.sum(labeled_mask[unassigned_indices])
        unassigned_unlabeled = unassigned_samples - unassigned_labeled
        unassigned_known = np.sum(known_mask[unassigned_indices])
        unassigned_unknown = unassigned_samples - unassigned_known

        print(f"   有标签样本: {unassigned_labeled} 个")
        print(f"   无标签样本: {unassigned_unlabeled} 个")
        print(f"   已知类样本: {unassigned_known} 个")
        print(f"   未知类样本: {unassigned_unknown} 个")

        if unassigned_labeled > 0:
            unassigned_targets = targets[unassigned_indices][labeled_mask[unassigned_indices]]
            unassigned_unique, unassigned_counts = np.unique(unassigned_targets, return_counts=True)
            unassigned_distribution = dict(zip(unassigned_unique, unassigned_counts))
            print(f"   未分配有标签样本分布: {unassigned_distribution}")

    print("="*80)


def analyze_cluster_composition(predictions, targets, known_mask, labeled_mask, unknown_clusters):
    """
    分析每个聚类的内部组成情况

    Args:
        predictions: 聚类预测结果
        targets: 真实标签
        known_mask: 已知类别掩码
        labeled_mask: 有标签掩码
        unknown_clusters: 潜在未知类聚类索引列表
    """
    print(f"\n🔍 聚类内部组成分析:")
    print("="*80)

    unique_clusters = np.unique(predictions)

    for cluster_id in sorted(unique_clusters):
        cluster_mask = predictions == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        # 基本信息
        cluster_size = len(cluster_indices)
        is_unknown_cluster = cluster_id in unknown_clusters

        # 标签信息
        cluster_targets = targets[cluster_indices]
        cluster_known_mask = known_mask[cluster_indices]
        cluster_labeled_mask = labeled_mask[cluster_indices]

        # 统计有标签样本
        labeled_samples = cluster_indices[cluster_labeled_mask]
        unlabeled_samples = cluster_indices[~cluster_labeled_mask]

        # 统计已知类/未知类样本
        known_samples = cluster_indices[cluster_known_mask]
        unknown_samples = cluster_indices[~cluster_known_mask]

        # 分析标签分布
        if len(labeled_samples) > 0:
            labeled_targets = cluster_targets[cluster_labeled_mask]
            unique_labels, label_counts = np.unique(labeled_targets, return_counts=True)
            label_distribution = dict(zip(unique_labels, label_counts))
            dominant_label = unique_labels[np.argmax(label_counts)]
            label_purity = np.max(label_counts) / len(labeled_samples)
        else:
            label_distribution = {}
            dominant_label = None
            label_purity = 0.0

        # 输出聚类信息
        cluster_type = "🔍 潜在未知类" if is_unknown_cluster else "📊 常规聚类"
        print(f"\n{cluster_type} #{cluster_id} - 大小: {cluster_size}")
        print(f"   样本组成:")
        print(f"     有标签样本: {len(labeled_samples)} 个")
        print(f"     无标签样本: {len(unlabeled_samples)} 个")
        print(f"     已知类样本: {len(known_samples)} 个")
        print(f"     未知类样本: {len(unknown_samples)} 个")

        if len(labeled_samples) > 0:
            print(f"   有标签样本分布:")
            print(f"     主导标签: {dominant_label} (纯度: {label_purity:.3f})")
            print(f"     详细分布: {label_distribution}")

            # 检查标签冲突
            if len(unique_labels) > 1:
                print(f"     ⚠️  标签冲突: 包含{len(unique_labels)}种不同标签")
        else:
            print(f"   有标签样本分布: 无有标签样本")

        # 分析真实类别分布 (所有样本，包括无标签的)
        all_targets = cluster_targets  # 所有聚类内样本的真实标签
        all_unique_labels, all_label_counts = np.unique(all_targets, return_counts=True)
        all_label_distribution = dict(zip(all_unique_labels, all_label_counts))

        print(f"   真实类别分布 (所有样本):")
        print(f"     详细分布: {all_label_distribution}")

        # 分析已知类和未知类的真实分布
        if len(known_samples) > 0:
            known_targets = cluster_targets[cluster_known_mask]
            known_unique, known_counts = np.unique(known_targets, return_counts=True)
            known_class_distribution = dict(zip(known_unique, known_counts))
            print(f"   已知类样本分布: {known_class_distribution}")

        if len(unknown_samples) > 0:
            unknown_targets = cluster_targets[~cluster_known_mask]
            unknown_unique, unknown_counts = np.unique(unknown_targets, return_counts=True)
            unknown_class_distribution = dict(zip(unknown_unique, unknown_counts))
            print(f"   未知类样本分布: {unknown_class_distribution}")

        # 分析聚类质量
        if len(labeled_samples) > 1:
            # 计算内部一致性
            same_label_pairs = 0
            total_pairs = 0
            for i in range(len(labeled_samples)):
                for j in range(i+1, len(labeled_samples)):
                    if labeled_targets[i] == labeled_targets[j]:
                        same_label_pairs += 1
                    total_pairs += 1

            if total_pairs > 0:
                consistency = same_label_pairs / total_pairs
                print(f"   质量评估:")
                print(f"     内部一致性: {consistency:.3f}")

                if consistency >= 0.9:
                    print(f"     质量评级: ✅ 优秀")
                elif consistency >= 0.7:
                    print(f"     质量评级: ✅ 良好")
                elif consistency >= 0.5:
                    print(f"     质量评级: ⚠️  一般")
                else:
                    print(f"     质量评级: ❌ 较差")

    # 全局统计
    print(f"\n📊 全局聚类统计:")
    print(f"   总聚类数: {len(unique_clusters)}")
    print(f"   潜在未知类聚类数: {len(unknown_clusters)}")
    print(f"   常规聚类数: {len(unique_clusters) - len(unknown_clusters)}")

    # 分析聚类大小分布
    cluster_sizes = []
    for cluster_id in unique_clusters:
        cluster_size = np.sum(predictions == cluster_id)
        cluster_sizes.append(cluster_size)

    print(f"   聚类大小统计:")
    print(f"     平均大小: {np.mean(cluster_sizes):.1f}")
    print(f"     最大聚类: {np.max(cluster_sizes)} 个样本")
    print(f"     最小聚类: {np.min(cluster_sizes)} 个样本")
    print(f"     大小标准差: {np.std(cluster_sizes):.1f}")


def print_prototype_distance_matrix(X, predictions):
    """
    打印簇间原型距离矩阵

    Args:
        X: 特征矩阵
        predictions: 聚类预测结果
    """
    unique_clusters = np.unique(predictions)
    n_clusters = len(unique_clusters)

    if n_clusters == 0:
        print("   没有聚类")
        return

    # 计算每个簇的原型（中心）
    prototypes = []
    for cluster_id in unique_clusters:
        cluster_mask = predictions == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) > 0:
            prototype = np.mean(X[cluster_indices], axis=0)
            prototypes.append(prototype)
        else:
            prototypes.append(None)

    # 计算簇间距离矩阵
    distance_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if prototypes[i] is not None and prototypes[j] is not None:
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    distance_matrix[i, j] = np.linalg.norm(prototypes[i] - prototypes[j])

    # 打印矩阵
    print(f"   聚类数量: {n_clusters}")

    # 如果聚类数量太多，只显示统计信息
    if n_clusters >= 25:
        print(f"   聚类数量较多({n_clusters}个)，仅显示距离统计:")
        # 提取上三角（不包括对角线）
        upper_tri_indices = np.triu_indices(n_clusters, k=1)
        distances = distance_matrix[upper_tri_indices]

        print(f"   簇间距离统计:")
        print(f"     最小距离: {np.min(distances):.4f}")
        print(f"     最大距离: {np.max(distances):.4f}")
        print(f"     平均距离: {np.mean(distances):.4f}")
        print(f"     中位距离: {np.median(distances):.4f}")

        # 显示距离最近的5对簇
        print(f"   距离最近的5对簇:")
        flat_indices = np.argsort(distances)[:5]
        for idx in flat_indices:
            i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
            cluster_i = unique_clusters[i]
            cluster_j = unique_clusters[j]
            dist = distances[idx]
            size_i = np.sum(predictions == cluster_i)
            size_j = np.sum(predictions == cluster_j)
            print(f"     簇{cluster_i}({size_i}样本) ↔ 簇{cluster_j}({size_j}样本): {dist:.4f}")
    else:
        # 打印完整矩阵
        print(f"   完整距离矩阵:")

        # 打印表头
        header = "       "
        for cluster_id in unique_clusters:
            header += f"  簇{cluster_id:<4}"
        print(header)
        print("   " + "-" * (7 + 7 * n_clusters))

        # 打印每一行
        for i, cluster_i in enumerate(unique_clusters):
            row = f"   簇{cluster_i:<4}│"
            for j in range(n_clusters):
                if i == j:
                    row += "   -   "
                else:
                    row += f" {distance_matrix[i, j]:5.2f} "
            print(row)

        # 打印统计信息
        upper_tri_indices = np.triu_indices(n_clusters, k=1)
        distances = distance_matrix[upper_tri_indices]
        print(f"\n   距离统计: 最小={np.min(distances):.4f}, 最大={np.max(distances):.4f}, 平均={np.mean(distances):.4f}")


def print_prototype_distance_matrix_ground_truth(X, targets):
    """
    打印基于真实标签的原型距离矩阵（上帝视角）

    用于对比：展示如果所有样本的真实标签都已知，理想的簇原型应该是什么样的

    Args:
        X: 特征矩阵
        targets: 真实标签
    """
    unique_classes = np.unique(targets)
    n_classes = len(unique_classes)

    if n_classes == 0:
        print("   没有类别")
        return

    # 计算每个真实类别的原型（中心）
    prototypes = []
    class_sizes = []
    for class_id in unique_classes:
        class_mask = targets == class_id
        class_indices = np.where(class_mask)[0]
        if len(class_indices) > 0:
            prototype = np.mean(X[class_indices], axis=0)
            prototypes.append(prototype)
            class_sizes.append(len(class_indices))
        else:
            prototypes.append(None)
            class_sizes.append(0)

    # 计算类间距离矩阵
    distance_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if prototypes[i] is not None and prototypes[j] is not None:
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    distance_matrix[i, j] = np.linalg.norm(prototypes[i] - prototypes[j])

    # 打印矩阵
    print(f"   真实类别数量: {n_classes}")

    # 如果类别数量太多，只显示统计信息
    if n_classes >= 25:
        print(f"   类别数量较多({n_classes}个)，仅显示距离统计:")
        # 提取上三角（不包括对角线）
        upper_tri_indices = np.triu_indices(n_classes, k=1)
        distances = distance_matrix[upper_tri_indices]

        print(f"   类间距离统计:")
        print(f"     最小距离: {np.min(distances):.4f}")
        print(f"     最大距离: {np.max(distances):.4f}")
        print(f"     平均距离: {np.mean(distances):.4f}")
        print(f"     中位距离: {np.median(distances):.4f}")

        # 显示距离最近的5对类
        print(f"   距离最近的5对类别:")
        flat_indices = np.argsort(distances)[:5]
        for idx in flat_indices:
            i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
            class_i = unique_classes[i]
            class_j = unique_classes[j]
            dist = distances[idx]
            size_i = class_sizes[i]
            size_j = class_sizes[j]
            print(f"     类{class_i}({size_i}样本) ↔ 类{class_j}({size_j}样本): {dist:.4f}")
    else:
        # 打印完整矩阵
        print(f"   完整距离矩阵:")

        # 打印表头
        header = "       "
        for class_id in unique_classes:
            header += f"  类{class_id:<4}"
        print(header)
        print("   " + "-" * (7 + 7 * n_classes))

        # 打印每一行
        for i, class_i in enumerate(unique_classes):
            row = f"   类{class_i:<4}│"
            for j in range(n_classes):
                if i == j:
                    row += "   -   "
                else:
                    row += f" {distance_matrix[i, j]:5.2f} "
            print(row)

        # 打印统计信息
        upper_tri_indices = np.triu_indices(n_classes, k=1)
        distances = distance_matrix[upper_tri_indices]
        print(f"\n   距离统计: 最小={np.min(distances):.4f}, 最大={np.max(distances):.4f}, 平均={np.mean(distances):.4f}")


def analyze_intra_inter_class_distances(X, targets):
    """
    分析类内和类间样本距离（上帝视角）

    对每个类别计算：
    - 类内距离：同类别样本之间的距离统计
    - 类间距离：该类别与其他每个类别之间的样本距离统计

    Args:
        X: 特征矩阵 (n_samples, feat_dim)
        targets: 真实标签 (n_samples,)
    """
    print(f"\n📊 上帝视角：类内和类间样本距离详细分析")
    print("="*80)

    unique_classes = np.unique(targets)
    n_classes = len(unique_classes)

    if n_classes == 0:
        print("   没有类别")
        return

    print(f"   类别数量: {n_classes}")
    print(f"   特征已L2归一化，距离范围 [0, 2]")
    print("="*80)

    # 对每个类别进行分析
    for i, class_i in enumerate(unique_classes):
        # 获取类别i的所有样本索引
        class_i_mask = targets == class_i
        class_i_indices = np.where(class_i_mask)[0]
        n_samples_i = len(class_i_indices)

        print(f"\n📌 类别 {class_i} ({n_samples_i}个样本):")
        print("-"*80)

        # 1. 计算类内距离
        if n_samples_i > 1:
            intra_distances = []
            for idx1 in range(len(class_i_indices)):
                for idx2 in range(idx1+1, len(class_i_indices)):
                    dist = np.linalg.norm(X[class_i_indices[idx1]] - X[class_i_indices[idx2]])
                    intra_distances.append(dist)

            intra_distances = np.array(intra_distances)
            print(f"   🔹 类内距离统计 (共{len(intra_distances)}对):")
            print(f"      平均距离: {np.mean(intra_distances):.4f}")
            print(f"      最小距离: {np.min(intra_distances):.4f}")
            print(f"      最大距离: {np.max(intra_distances):.4f}")
            print(f"      中位距离: {np.median(intra_distances):.4f}")
            print(f"      标准差:   {np.std(intra_distances):.4f}")
        else:
            print(f"   🔹 类内距离统计: 只有1个样本，无法计算")

        # 2. 计算类间距离（与其他每个类别）
        print(f"\n   🔸 类间距离统计 (类{class_i} vs 其他类别):")

        for j, class_j in enumerate(unique_classes):
            if i == j:  # 跳过自己
                continue

            # 获取类别j的所有样本索引
            class_j_mask = targets == class_j
            class_j_indices = np.where(class_j_mask)[0]
            n_samples_j = len(class_j_indices)

            # 计算类i和类j之间所有样本对的距离
            inter_distances = []
            for idx_i in class_i_indices:
                for idx_j in class_j_indices:
                    dist = np.linalg.norm(X[idx_i] - X[idx_j])
                    inter_distances.append(dist)

            inter_distances = np.array(inter_distances)

            print(f"      vs 类{class_j:>2} ({n_samples_j:>3}样本): "
                  f"平均={np.mean(inter_distances):.4f}, "
                  f"最小={np.min(inter_distances):.4f}, "
                  f"最大={np.max(inter_distances):.4f}, "
                  f"中位={np.median(inter_distances):.4f}")

    print("\n" + "="*80)
    print("💡 距离解读 (L2归一化后的欧氏距离):")
    print("   - 0.0~0.5: 非常相似 (余弦相似度 > 0.875)")
    print("   - 0.5~1.0: 比较相似 (余弦相似度 0.5~0.875)")
    print("   - 1.0~1.4: 中等相似 (余弦相似度 0~0.5)")
    print("   - 1.4~2.0: 不相似   (余弦相似度 < 0)")
    print("="*80)


def analyze_high_density_intra_inter_class_distances(X, targets, high_density_mask):
    """
    分析高密度点之间的类内和类间距离（上帝视角）

    仅统计被标记为高密度点的样本之间的距离，区分同类和异类：
    - 类内距离：同类别的高密度点之间的距离统计
    - 类间距离：不同类别的高密度点之间的距离统计

    Args:
        X: 特征矩阵 (n_samples, feat_dim)
        targets: 真实标签 (n_samples,)
        high_density_mask: 高密度点掩码 (n_samples,)
    """
    print(f"\n📊 上帝视角：高密度点类内和类间距离详细分析")
    print("="*80)

    # 过滤出高密度点
    high_density_indices = np.where(high_density_mask)[0]
    n_high_density = len(high_density_indices)

    if n_high_density == 0:
        print("   ⚠️  没有高密度点，无法分析")
        return

    X_high_density = X[high_density_indices]
    targets_high_density = targets[high_density_indices]

    unique_classes = np.unique(targets_high_density)
    n_classes = len(unique_classes)

    print(f"   总样本数: {len(X)}")
    print(f"   高密度点数量: {n_high_density} ({n_high_density/len(X)*100:.1f}%)")
    print(f"   涉及类别数: {n_classes}")
    print(f"   特征已L2归一化，距离范围 [0, 2]")
    print("="*80)

    # 对每个类别进行分析
    for i, class_i in enumerate(unique_classes):
        # 获取类别i的高密度点索引
        class_i_mask = targets_high_density == class_i
        class_i_indices = np.where(class_i_mask)[0]
        n_samples_i = len(class_i_indices)

        print(f"\n📌 类别 {class_i} (高密度点: {n_samples_i}个):")
        print("-"*80)

        # 1. 计算类内距离（高密度点之间）
        if n_samples_i > 1:
            intra_distances = []
            for idx1 in range(len(class_i_indices)):
                for idx2 in range(idx1+1, len(class_i_indices)):
                    dist = np.linalg.norm(
                        X_high_density[class_i_indices[idx1]] - X_high_density[class_i_indices[idx2]]
                    )
                    intra_distances.append(dist)

            intra_distances = np.array(intra_distances)
            print(f"   🔹 类内高密度点距离统计 (共{len(intra_distances)}对):")
            print(f"      平均距离: {np.mean(intra_distances):.4f}")
            print(f"      最小距离: {np.min(intra_distances):.4f}")
            print(f"      最大距离: {np.max(intra_distances):.4f}")
            print(f"      中位距离: {np.median(intra_distances):.4f}")
            print(f"      标准差:   {np.std(intra_distances):.4f}")
        else:
            print(f"   🔹 类内高密度点距离统计: 只有1个高密度点，无法计算")

        # 2. 计算类间距离（高密度点之间）
        print(f"\n   🔸 类间高密度点距离统计 (类{class_i} vs 其他类别):")

        for j, class_j in enumerate(unique_classes):
            if i == j:  # 跳过自己
                continue

            # 获取类别j的高密度点索引
            class_j_mask = targets_high_density == class_j
            class_j_indices = np.where(class_j_mask)[0]
            n_samples_j = len(class_j_indices)

            # 计算类i和类j的高密度点之间的所有距离
            inter_distances = []
            for idx_i in class_i_indices:
                for idx_j in class_j_indices:
                    dist = np.linalg.norm(X_high_density[idx_i] - X_high_density[idx_j])
                    inter_distances.append(dist)

            inter_distances = np.array(inter_distances)

            print(f"      vs 类{class_j:>2} ({n_samples_j:>3}个高密度点): "
                  f"平均={np.mean(inter_distances):.4f}, "
                  f"最小={np.min(inter_distances):.4f}, "
                  f"最大={np.max(inter_distances):.4f}, "
                  f"中位={np.median(inter_distances):.4f}")

    print("\n" + "="*80)
    print("💡 距离解读 (L2归一化后的欧氏距离):")
    print("   - 0.0~0.5: 非常相似 (余弦相似度 > 0.875)")
    print("   - 0.5~1.0: 比较相似 (余弦相似度 0.5~0.875)")
    print("   - 1.0~1.4: 中等相似 (余弦相似度 0~0.5)")
    print("   - 1.4~2.0: 不相似   (余弦相似度 < 0)")
    print("="*80)




def evaluate_high_density_clustering(cluster_labels, targets, known_mask, eval_version='v1', X=None, silent=False):
    """
    评估高密度点的聚类准确率（仅评估已分配的高密度点）

    Args:
        cluster_labels: 高密度点的聚类标签 (-1表示未分配)
        targets: 真实标签
        known_mask: 已知类掩码
        eval_version: 评估版本 ('v1' 或 'v2')
        X: 特征矩阵 (已废弃，为保持兼容性保留)
        silent: 静默模式（默认False）

    Returns:
        all_acc: 所有样本准确率
        old_acc: 已知类准确率
        new_acc: 未知类准确率
        n_clusters: 聚类数量
        dbcv_score: DBCV分数 (已移除，始终返回None)
    """

    # 只评估已分配的高密度点
    assigned_mask = cluster_labels != -1
    n_total = len(cluster_labels)
    n_assigned = np.sum(assigned_mask)

    if not silent:
        print(f"\n📊 高密度点聚类评估:")
        print(f"   总样本数: {n_total}")
        print(f"   已分配高密度点: {n_assigned} ({n_assigned/n_total*100:.1f}%)")
        print(f"   未分配低密度点: {n_total - n_assigned} ({(n_total - n_assigned)/n_total*100:.1f}%)")

    if n_assigned == 0:
        if not silent:
            print("   ⚠️  没有已分配的高密度点，无法评估")
        return 0.0, 0.0, 0.0, 0, None

    # 提取已分配的高密度点
    assigned_predictions = cluster_labels[assigned_mask]
    assigned_targets = targets[assigned_mask]
    assigned_known_mask = known_mask[assigned_mask]

    # 使用指定版本的ACC计算方法
    if eval_version == 'v1':
        from project_utils.cluster_and_log_utils import split_cluster_acc_v1
        all_acc, old_acc, new_acc = split_cluster_acc_v1(assigned_targets, assigned_predictions, assigned_known_mask)
    else:  # v2
        from project_utils.cluster_and_log_utils import split_cluster_acc_v2
        all_acc, old_acc, new_acc = split_cluster_acc_v2(assigned_targets, assigned_predictions, assigned_known_mask)

    n_clusters = len(np.unique(assigned_predictions))

    if not silent:
        print(f"\n📈 高密度点聚类结果:")
        print(f"   聚类数量: {n_clusters}")
        print(f"   All ACC: {all_acc:.4f}")
        print(f"   Old ACC: {old_acc:.4f}")
        print(f"   New ACC: {new_acc:.4f}")

    # DBCV已移除
    return all_acc, old_acc, new_acc, n_clusters, None
