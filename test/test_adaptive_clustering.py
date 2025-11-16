#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自适应密度聚类算法实现
基于验证的数据加载方案，实现5步自适应密度聚类算法：
1. 密度计算: 使用自适应带宽的高斯核估计每个样本的局部密度
2. 高密度点识别: 选择75分位数以上的点作为聚类种子
3. 聚类构建: 从高密度点开始，通过k近邻扩展形成聚类
4. 冲突处理: 根据已知标签信息解决聚类边界冲突
5. 稀疏点分配: 将剩余点分配给最近聚类或形成单点聚类
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse
from copy import deepcopy

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.get_datasets import get_datasets, get_class_splits
from data.augmentations import get_transform
from data.cifar100_superclass import CIFAR100_SUPERCLASSES, get_single_superclass_datasets
from models import vision_transformer as vits
from config import dino_pretrain_path
from project_utils.general_utils import str2bool
from data.data_utils import MergedDataset
from project_utils.cluster_utils import cluster_acc


def load_model(args, device):
    """
    加载训练好的模型（模仿eval_original_gcd.py）
    """
    print(f"🔄 加载模型...")
    print(f"   模型文件: {args.model_path}")
    print(f"   设备: {device}")

    # 构建base model
    if args.base_model == 'vit_dino':
        model = vits.__dict__['vit_base']()

        # 加载DINO预训练权重
        if os.path.exists(dino_pretrain_path):
            print(f"   加载DINO预训练权重...")
            dino_state_dict = torch.load(dino_pretrain_path, map_location='cpu')
            model.load_state_dict(dino_state_dict, strict=False)

        # 加载训练权重
        print(f"   加载训练权重...")
        gcd_state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(gcd_state_dict)

        model.to(device)

        # 关闭梯度计算
        for param in model.parameters():
            param.requires_grad = False

        print(f"✅ 模型加载成功 (特征维度: {args.feat_dim})")
        return model
    else:
        raise NotImplementedError(f"不支持的模型类型: {args.base_model}")


def extract_features(data_loader, model, device, known_classes=None, use_train_and_test=True):
    """
    提取特征（模仿kmeans的特征提取方式）

    Args:
        data_loader: 数据加载器
        model: 特征提取模型
        device: 设备
        use_train_and_test: 是否使用训练集+测试集，False则只用测试集

    Returns:
        features: 特征矩阵 (n_samples, feat_dim)
        targets: 真实标签 (仅用于评估)
        mask: 已知类别掩码 (True=已知类, False=未知类)
        labeled_mask: 有标签掩码 (True=有标签, False=无标签)
    """
    print(f"🔄 提取特征...")

    model.eval()
    all_feats = []
    targets = np.array([])
    mask = np.array([])  # 已知类别掩码
    labeled_mask = np.array([])  # 有标签掩码

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="提取特征")):
            # 解包数据
            if len(batch_data) == 4:
                images, labels, indices, labeled_or_not = batch_data
                labeled_batch = labeled_or_not.numpy().flatten().astype(bool)
            elif len(batch_data) == 3:
                images, labels, indices = batch_data
                # 测试集全部标记为无标签
                labeled_batch = np.zeros(len(labels), dtype=bool)
            else:
                continue

            # 提取特征
            images = images.to(device)
            feats = model(images)
            feats = torch.nn.functional.normalize(feats, dim=-1)

            # 收集数据
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, labels.cpu().numpy())
            labeled_mask = np.append(labeled_mask, labeled_batch)

            # 创建已知类别掩码（根据known_classes列表）
            if known_classes is not None:
                batch_mask = np.array([True if x.item() in known_classes else False for x in labels])
            else:
                # 默认：前80个类别是已知类
                batch_mask = np.array([True if x.item() < 80 else False for x in labels])
            mask = np.append(mask, batch_mask)

            # 清理GPU内存
            del images, feats
            torch.cuda.empty_cache()

    # 拼接所有特征
    all_feats = np.concatenate(all_feats, axis=0)
    print(f"✅ 特征提取完成: {all_feats.shape}")

    return all_feats, targets.astype(int), mask.astype(bool), labeled_mask.astype(bool)


def compute_simple_density(X, k=10):
    """
    步骤1: 使用简单k近邻平均距离倒数计算密度

    SS-DDBC只需要能够比较密度大小，使用简单方法即可：
    密度 = 1 / k近邻平均距离

    Args:
        X: 特征矩阵 (n_samples, feat_dim)
        k: k近邻数量

    Returns:
        densities: 每个样本的密度值
        knn_distances: k近邻距离矩阵
        neighbors: k近邻索引矩阵
    """
    print(f"📊 计算简化密度 (k={k})...")

    # 计算k近邻
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
    knn_distances, neighbors = nbrs.kneighbors(X)

    # 去除自己（第一个邻居是自己）
    knn_distances = knn_distances[:, 1:]
    neighbors = neighbors[:, 1:]

    # 简单密度计算：k近邻平均距离的倒数
    avg_distances = np.mean(knn_distances, axis=1)
    densities = 1.0 / (avg_distances + 1e-8)  # 避免除零

    print(f"   密度统计: min={densities.min():.3f}, max={densities.max():.3f}, mean={densities.mean():.3f}")

    return densities, knn_distances, neighbors


def identify_high_density_points(densities, percentile=75):
    """
    步骤2: 选择75分位数以上的点作为聚类种子

    Args:
        densities: 密度值数组
        percentile: 百分位数阈值

    Returns:
        high_density_mask: 高密度点掩码
    """
    density_threshold = np.percentile(densities, percentile)
    high_density_mask = densities >= density_threshold

    print(f"📍 识别高密度点:")
    print(f"   密度阈值: {density_threshold:.3f} (第{percentile}百分位数)")
    print(f"   高密度点数量: {np.sum(high_density_mask)} / {len(densities)}")

    return high_density_mask


def ssddbc_conflict_resolution(xi_idx, xj_idx, xi_cluster_center, xj_cluster_center, X, densities):
    """
    SS-DDBC冲突解决算法

    冲突解决流程:
    1. 比较xi和xj的密度
    2. 计算冲突点与xi和xj的距离
    3. 冲突点距离xi较近且xi密度较大，则把冲突点重新分配给xi，否则不做重新分配

    Args:
        xi_idx: 参考点索引
        xj_idx: 冲突点索引
        xi_cluster_center: xi所在聚类的中心
        xj_cluster_center: xj所在聚类的中心
        X: 特征矩阵
        densities: 密度数组

    Returns:
        should_reassign: 是否应该重新分配冲突点
    """
    # 1. 比较xi和xj的密度
    xi_density = densities[xi_idx]
    xj_density = densities[xj_idx]

    # 2. 计算冲突点xj到两个聚类中心的距离
    xj_pos = X[xj_idx]
    distance_to_xi_cluster = np.linalg.norm(xj_pos - xi_cluster_center)
    distance_to_xj_cluster = np.linalg.norm(xj_pos - xj_cluster_center)

    # 3. 判断条件：冲突点距离xi较近且xi密度较大
    closer_to_xi = distance_to_xi_cluster < distance_to_xj_cluster
    xi_density_higher = xi_density > xj_density

    if closer_to_xi and xi_density_higher:
        return True  # 两个条件都满足，重新分配xj给xi
    else:
        return False  # 不重新分配



def build_clusters_ssddbc(X, high_density_mask, neighbors, labeled_mask, targets, densities, known_mask, k=10):
    """
    完全按照SS-DDBC算法构建聚类
    算法流程:
    For each high-density point xi:
        If xi is not assigned to any cluster, create a new cluster pi
        For each neighbor xj in k-neighbors of xi:
            If xj is not assigned to any cluster, add xj to pi
            Else If xj ∈ pj (si ≠ sj) and xj ∈ C (有标签冲突)
                perform Algorithm 3 for conflict resolution
            Else (si = sj 或者有的簇无标签)
                merge pi and pj

    Args:
        X: 特征矩阵
        high_density_mask: 高密度点掩码
        neighbors: k近邻索引矩阵
        labeled_mask: 有标签掩码
        targets: 真实标签
        densities: 密度数组
        known_mask: 已知类掩码
        k: 扩展时的近邻数

    Returns:
        clusters: 聚类列表，每个聚类是一个样本索引集合
        cluster_labels: 每个样本的聚类标签 (-1表示未分配)
    """
    print(f"SS-DDBC聚类构建...")

    n_samples = X.shape[0]
    cluster_labels = np.full(n_samples, -1, dtype=int)
    clusters = []
    current_cluster_id = 0

    # 从每个高密度点开始构建聚类
    high_density_indices = np.where(high_density_mask)[0]

    # 🔍 调试：分析高密度点的类别分布
    print(f"\n🔍 高密度点类别分布分析:")
    known_high_density = 0
    unknown_high_density = 0
    for idx in high_density_indices:
        if known_mask[idx]:
            known_high_density += 1
        else:
            unknown_high_density += 1
    print(f"   已知类高密度点: {known_high_density}个")
    print(f"   未知类高密度点: {unknown_high_density}个")

    # 按类别统计
    class_high_density_count = {}
    for idx in high_density_indices:
        true_label = targets[idx]
        if true_label not in class_high_density_count:
            class_high_density_count[true_label] = 0
        class_high_density_count[true_label] += 1
    print(f"   各类别高密度点数: {class_high_density_count}")

    for xi_idx in high_density_indices:
        if cluster_labels[xi_idx] != -1:
            # 🔍 调试：跟踪被跳过的高密度点
            xi_true_label = targets[xi_idx]
            xi_is_known = known_mask[xi_idx]
            assigned_cluster_id = cluster_labels[xi_idx]
            print(f"   ⏭️ 跳过高密度点{xi_idx} (真实标签={xi_true_label}, 已知类={xi_is_known}) - 已分配给聚类{assigned_cluster_id}")
            continue  # 已经被分配到其他聚类

        # If xi is not assigned to any cluster, create a new cluster pi
        cluster_pi = set([xi_idx])
        cluster_labels[xi_idx] = current_cluster_id
        queue = [xi_idx]

        # 🔍 调试：跟踪聚类创建
        xi_true_label = targets[xi_idx]
        xi_is_known = known_mask[xi_idx]
        print(f"\n🔹 创建聚类{current_cluster_id}: 种子点{xi_idx} (真实标签={xi_true_label}, 已知类={xi_is_known})")

        # BFS扩展聚类
        while queue:
            current_idx = queue.pop(0)

            # 只有高密度点才能扩展其他点
            if high_density_mask[current_idx]:
                # For each neighbor xj in k-neighbors of xi
                for xj_idx in neighbors[current_idx]:
                    xj_cluster_id = cluster_labels[xj_idx]

                    if xj_cluster_id == -1:
                        # If xj is not assigned to any cluster, add xj to pi
                        cluster_pi.add(xj_idx)
                        cluster_labels[xj_idx] = current_cluster_id
                        queue.append(xj_idx)

                    elif xj_cluster_id != current_cluster_id:
                        # xj ∈ pj，需要判断情况
                        xi_has_label = labeled_mask[current_idx]
                        xj_has_label = labeled_mask[xj_idx]

                        # 检查是否有标签冲突 (si ≠ sj) and xj ∈ C
                        has_conflict = False
                        if xi_has_label and xj_has_label:
                            xi_label = targets[current_idx]
                            xj_label = targets[xj_idx]
                            if xi_label != xj_label:
                                has_conflict = True

                        if has_conflict:
                            # Else If xj ∈ pj (si ≠ sj) and xj ∈ C
                            # perform Algorithm 3 for conflict resolution

                            # 计算聚类中心用于冲突解决
                            xi_cluster_center = np.mean(X[list(cluster_pi)], axis=0)
                            if xj_cluster_id < len(clusters) and len(clusters[xj_cluster_id]) > 0:
                                xj_cluster_center = np.mean(X[list(clusters[xj_cluster_id])], axis=0)

                                should_reassign = ssddbc_conflict_resolution(
                                    current_idx, xj_idx, xi_cluster_center, xj_cluster_center, X, densities
                                )

                                if should_reassign:
                                    # 重新分配xj到当前聚类pi
                                    clusters[xj_cluster_id].discard(xj_idx)
                                    cluster_pi.add(xj_idx)
                                    cluster_labels[xj_idx] = current_cluster_id
                                    queue.append(xj_idx)
                                    print(f"   冲突解决: 点{xj_idx}从聚类{xj_cluster_id}重分配到聚类{current_cluster_id}")
                            # 否则不做重新分配

                        else:
                            # Else (si = sj 或者有的簇无标签) merge pi and pj
                            # 无冲突，可以合并聚类
                            if xj_cluster_id < len(clusters) and len(clusters[xj_cluster_id]) > 0:
                                # 🔍 调试：跟踪聚类合并
                                cluster_pj = clusters[xj_cluster_id]

                                # 分析被合并聚类的组成
                                merged_known_count = sum(1 for idx in cluster_pj if known_mask[idx])
                                merged_unknown_count = len(cluster_pj) - merged_known_count

                                # 分析当前聚类的组成
                                current_known_count = sum(1 for idx in cluster_pi if known_mask[idx])
                                current_unknown_count = len(cluster_pi) - current_known_count

                                print(f"   🔄 合并聚类: 聚类{xj_cluster_id}(已知:{merged_known_count},未知:{merged_unknown_count}) → 聚类{current_cluster_id}(已知:{current_known_count},未知:{current_unknown_count})")

                                # 合并聚类pj到pi
                                for idx in cluster_pj:
                                    cluster_pi.add(idx)
                                    cluster_labels[idx] = current_cluster_id
                                    if high_density_mask[idx]:  # 如果是高密度点，加入扩展队列
                                        queue.append(idx)
                                clusters[xj_cluster_id] = set()  # 清空被合并的聚类

        clusters.append(cluster_pi)
        current_cluster_id += 1

    # 移除空聚类并重新编号
    non_empty_clusters = [c for c in clusters if len(c) > 0]
    cluster_labels_new = np.full(n_samples, -1, dtype=int)

    for new_id, cluster in enumerate(non_empty_clusters):
        for idx in cluster:
            cluster_labels_new[idx] = new_id

    print(f"   SS-DDBC聚类数量: {len(non_empty_clusters)}")
    print(f"   已分配样本: {np.sum(cluster_labels_new != -1)} / {n_samples}")

    return non_empty_clusters, cluster_labels_new


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


def identify_unknown_clusters(clusters, labeled_mask):
    """
    识别潜在未知类聚类

    根据SS-DDBC: "簇的类别取决于其有标签样本，不含有标签样本的可能是未知类"

    Args:
        clusters: 聚类列表
        labeled_mask: 有标签掩码

    Returns:
        unknown_clusters: 潜在未知类聚类索引列表
    """
    unknown_clusters = []

    for cluster_id, cluster in enumerate(clusters):
        cluster_indices = list(cluster)
        cluster_labeled_mask = labeled_mask[cluster_indices]

        # 如果聚类中没有有标签样本，标记为潜在未知类
        if not np.any(cluster_labeled_mask):
            unknown_clusters.append(cluster_id)

    return unknown_clusters


def test_kmeans_baseline(test_features, test_targets, test_known_mask, n_clusters, random_state=0):
    """
    K-means基线对比方案

    完全参考eval_original_gcd.py的实现，保持一致性

    Args:
        test_features: 测试集特征 (n_test_samples, feat_dim) - 应该是L2归一化的
        test_targets: 测试集真实标签
        test_known_mask: 测试集已知类掩码
        n_clusters: 聚类数量
        random_state: 随机种子

    Returns:
        kmeans_results: 包含各种评估指标的字典
    """
    print(f"\n🔄 运行K-means基线对比 (与eval_original_gcd保持一致)...")
    print(f"   测试集样本数: {len(test_features)}")
    print(f"   聚类数量: {n_clusters}")
    print(f"   随机种子: {random_state}")

    # 运行K-means聚类 (与原版完全一致)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_predictions = kmeans.fit_predict(test_features)

    # 计算各种ACC指标 (只在测试集上)
    # 使用与eval_original_gcd.py相同的ACC计算方法 (split_cluster_acc_v1)
    from project_utils.cluster_and_log_utils import split_cluster_acc_v1
    all_acc, old_acc, new_acc = split_cluster_acc_v1(test_targets, kmeans_predictions, test_known_mask)


    # 计算其他指标
    nmi = normalized_mutual_info_score(test_targets, kmeans_predictions)
    ari = adjusted_rand_score(test_targets, kmeans_predictions)

    print(f"✅ K-means结果:")
    print(f"   All ACC: {all_acc:.4f}")
    print(f"   Old ACC: {old_acc:.4f}")
    print(f"   New ACC: {new_acc:.4f}")
    print(f"   NMI: {nmi:.4f}")
    print(f"   ARI: {ari:.4f}")

    return {
        'method': 'K-means',
        'n_clusters': n_clusters,
        'all_acc': all_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'nmi': nmi,
        'ari': ari
    }


def build_prototypes(X, clusters, labeled_mask, targets):
    """
    基于partial-clustering结果建立原型

    SS-DDBC步骤: (2) 基于partial-clustering的结果建立原型

    Args:
        X: 特征矩阵
        clusters: 聚类列表
        labeled_mask: 有标签掩码
        targets: 真实标签

    Returns:
        prototypes: 每个聚类的原型 (聚类中心)
        prototype_labels: 每个聚类的主导标签
    """
    print(f"   建立聚类原型...")

    prototypes = []
    prototype_labels = []

    for cluster_id, cluster in enumerate(clusters):
        cluster_indices = list(cluster)

        # 计算聚类中心作为原型
        prototype = np.mean(X[cluster_indices], axis=0)
        prototypes.append(prototype)

        # 确定聚类的主导标签
        cluster_labeled_mask = labeled_mask[cluster_indices]
        if np.any(cluster_labeled_mask):
            # 如果有有标签样本，使用主导标签
            labeled_targets = targets[cluster_indices][cluster_labeled_mask]
            unique_labels, counts = np.unique(labeled_targets, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]
            prototype_labels.append(dominant_label)
        else:
            # 无有标签样本，标记为未知类 (-1)
            prototype_labels.append(-1)

    print(f"   建立原型完成: {len(prototypes)}个原型")
    return np.array(prototypes), np.array(prototype_labels)


def assign_sparse_points_ssddbc(X, clusters, cluster_labels, prototypes, neighbors, labeled_mask, targets, lambda_weight=0.7):
    """
    SS-DDBC稀疏点分配算法

    SS-DDBC步骤: (3) 结合原型距离和近邻情况，标注剩余的稀疏点

    Args:
        X: 特征矩阵
        clusters: 聚类列表
        cluster_labels: 聚类标签
        prototypes: 聚类原型
        neighbors: k近邻索引矩阵
        labeled_mask: 有标签掩码
        targets: 真实标签
        lambda_weight: 原型距离权重

    Returns:
        final_labels: 最终聚类标签
    """
    print(f"稀疏点分配 (结合原型距离和近邻情况)...")

    final_labels = cluster_labels.copy()
    unassigned_indices = np.where(cluster_labels == -1)[0]

    if len(unassigned_indices) == 0:
        print(f"   无需分配稀疏点")
        return final_labels

    print(f"   分配{len(unassigned_indices)}个稀疏点")

    # 分配未分配的点
    for point_idx in unassigned_indices:
        point_features = X[point_idx]

        # 1. 计算到各个原型的距离
        distances_to_prototypes = []
        for prototype in prototypes:
            distance = np.linalg.norm(point_features - prototype)
            distances_to_prototypes.append(distance)

        # 2. 分析k近邻的聚类分布
        point_neighbors = neighbors[point_idx]
        neighbor_clusters = cluster_labels[point_neighbors]
        neighbor_clusters = neighbor_clusters[neighbor_clusters != -1]  # 排除未分配的邻居

        neighbor_confidence = np.zeros(len(prototypes))
        if len(neighbor_clusters) > 0:
            unique_clusters, counts = np.unique(neighbor_clusters, return_counts=True)
            for cluster_id, count in zip(unique_clusters, counts):
                if cluster_id < len(prototypes):
                    neighbor_confidence[cluster_id] = count / len(neighbor_clusters)

        # 3. 结合原型距离和近邻情况
        if len(neighbor_clusters) > 0:
            # 距离置信度 (距离越小，置信度越高)
            max_distance = np.max(distances_to_prototypes)
            if max_distance > 0:
                distance_confidence = 1 - (np.array(distances_to_prototypes) / max_distance)
            else:
                distance_confidence = np.ones(len(prototypes))

            # 综合置信度: lambda * 距离置信度 + (1-lambda) * 邻居置信度
            combined_confidence = lambda_weight * distance_confidence + (1 - lambda_weight) * neighbor_confidence
            best_cluster = np.argmax(combined_confidence)
        else:
            # 没有已分配的邻居，仅基于距离
            best_cluster = np.argmin(distances_to_prototypes)

        # 分配到最佳聚类
        final_labels[point_idx] = best_cluster

    print(f"   稀疏点分配完成")
    return final_labels


def resolve_conflicts_ssddbc(X, clusters, cluster_labels, labeled_mask, targets, known_mask, densities):
    """
    步骤4: SS-DDBC风格的冲突处理算法

    冲突解决流程:
    1. 检测聚类间的冲突点
    2. 比较涉及的核心点密度
    3. 根据密度+距离重新分配冲突点
    4. 标记无标签样本的簇为潜在未知类

    Args:
        X: 特征矩阵
        clusters: 聚类列表
        cluster_labels: 聚类标签
        labeled_mask: 有标签掩码
        targets: 真实标签
        known_mask: 已知类别掩码
        densities: 每个点的密度值

    Returns:
        refined_clusters: 优化后的聚类
        refined_labels: 优化后的聚类标签
        unknown_clusters: 潜在未知类聚类索引
    """
    print(f"⚖️ SS-DDBC冲突处理...")

    refined_clusters = [cluster.copy() for cluster in clusters]
    refined_labels = cluster_labels.copy()
    conflict_count = 0

    # 1. 检测并解决冲突
    for i, cluster_i in enumerate(refined_clusters):
        if len(cluster_i) == 0:
            continue

        cluster_i_indices = list(cluster_i)

        # 获取聚类i的有标签样本标签分布
        cluster_i_labeled = labeled_mask[cluster_i_indices]
        cluster_i_targets = targets[cluster_i_indices]
        cluster_i_labels = cluster_i_targets[cluster_i_labeled]

        if len(cluster_i_labels) == 0:
            continue  # 无标签聚类，稍后处理

        # 检查聚类内是否有标签冲突
        unique_labels_i, counts_i = np.unique(cluster_i_labels, return_counts=True)

        if len(unique_labels_i) > 1:
            # 聚类内部标签冲突，需要解决
            main_label = unique_labels_i[np.argmax(counts_i)]
            conflict_indices = []

            for idx, point_idx in enumerate(cluster_i_indices):
                if cluster_i_labeled[idx] and cluster_i_targets[idx] != main_label:
                    conflict_indices.append(point_idx)

            # 解决冲突点
            for conflict_idx in conflict_indices:
                conflict_count += 1

                # 找到冲突点应该属于的正确聚类
                conflict_label = targets[conflict_idx]
                best_cluster = None
                min_distance = float('inf')

                # 寻找包含相同标签的其他聚类
                for j, cluster_j in enumerate(refined_clusters):
                    if j == i or len(cluster_j) == 0:
                        continue

                    cluster_j_indices = list(cluster_j)
                    cluster_j_labeled = labeled_mask[cluster_j_indices]
                    cluster_j_targets = targets[cluster_j_indices]
                    cluster_j_labels = cluster_j_targets[cluster_j_labeled]

                    if conflict_label in cluster_j_labels:
                        # 计算冲突点到聚类j中心的距离
                        cluster_center = np.mean(X[cluster_j_indices], axis=0)
                        distance = np.linalg.norm(X[conflict_idx] - cluster_center)

                        if distance < min_distance:
                            min_distance = distance
                            best_cluster = j

                # 重新分配冲突点
                if best_cluster is not None:
                    # 比较密度决定是否重新分配
                    conflict_density = densities[conflict_idx]

                    # 找到目标聚类中密度最高的点
                    target_cluster_indices = list(refined_clusters[best_cluster])
                    target_densities = densities[target_cluster_indices]
                    max_target_density = np.max(target_densities)

                    # 如果目标聚类有更高密度的点，则重新分配
                    if max_target_density >= conflict_density:
                        refined_clusters[i].discard(conflict_idx)
                        refined_clusters[best_cluster].add(conflict_idx)
                        refined_labels[conflict_idx] = best_cluster
                        print(f"   冲突点{conflict_idx}从聚类{i}重新分配到聚类{best_cluster}")

    # 2. 识别潜在未知类聚类
    unknown_clusters = []
    for i, cluster in enumerate(refined_clusters):
        if len(cluster) == 0:
            continue

        cluster_indices = list(cluster)
        cluster_labeled = labeled_mask[cluster_indices]

        # 如果聚类中没有有标签样本，标记为潜在未知类
        if np.sum(cluster_labeled) == 0:
            unknown_clusters.append(i)
            print(f"   聚类{i}无有标签样本，标记为潜在未知类")

    print(f"   冲突解决完成: 处理{conflict_count}个冲突点")
    print(f"   识别{len(unknown_clusters)}个潜在未知类聚类")

    return refined_clusters, refined_labels, unknown_clusters


def assign_sparse_points(X, clusters, cluster_labels, lambda_weight=0.7):
    """
    步骤5: 将剩余点分配给最近聚类或形成单点聚类

    Args:
        X: 特征矩阵
        clusters: 聚类列表
        cluster_labels: 聚类标签
        lambda_weight: 距离权重

    Returns:
        final_labels: 最终聚类标签
    """
    print(f"🎯 分配稀疏点...")

    final_labels = cluster_labels.copy()
    unassigned_mask = cluster_labels == -1
    unassigned_indices = np.where(unassigned_mask)[0]

    if len(unassigned_indices) == 0:
        print(f"   无需分配稀疏点")
        return final_labels

    # 计算聚类中心
    cluster_centers = []
    for cluster in clusters:
        if len(cluster) > 0:
            cluster_indices = list(cluster)
            center = np.mean(X[cluster_indices], axis=0)
            cluster_centers.append(center)
        else:
            cluster_centers.append(None)

    # 分配未分配的点
    for idx in unassigned_indices:
        point = X[idx]
        min_distance = float('inf')
        best_cluster = -1

        # 计算到各个聚类中心的距离
        for cluster_id, center in enumerate(cluster_centers):
            if center is not None:
                distance = np.linalg.norm(point - center)
                if distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster_id

        # 分配到最近的聚类
        final_labels[idx] = best_cluster

    print(f"   稀疏点分配完成: {len(unassigned_indices)}个点")

    return final_labels


def adaptive_density_clustering(X, targets, known_mask, labeled_mask,
                               k=10, density_percentile=75, lambda_weight=0.7):
    """
    SS-DDBC风格的自适应密度聚类算法

    Args:
        X: 特征矩阵
        targets: 真实标签 (仅用于评估)
        known_mask: 已知类别掩码
        labeled_mask: 有标签掩码
        k: k近邻参数
        density_percentile: 密度百分位数阈值
        lambda_weight: 冲突解决权重

    Returns:
        predictions: 聚类预测结果
        n_clusters: 聚类数量
        unknown_clusters: 潜在未知类聚类索引
    """
    print("🚀 开始SS-DDBC自适应密度聚类...")

    # 步骤1: 简化密度计算
    densities, knn_distances, neighbors = compute_simple_density(X, k)

    # 步骤2: 高密度点识别
    high_density_mask = identify_high_density_points(densities, density_percentile)

    # 步骤3: SS-DDBC聚类构建 (集成冲突处理)
    clusters, cluster_labels = build_clusters_ssddbc(
        X, high_density_mask, neighbors, labeled_mask, targets, densities, known_mask, k
    )

    # 分析SS-DDBC聚类构建结果
    analyze_ssddbc_clustering_result(clusters, cluster_labels, labeled_mask, targets, known_mask)

    # 步骤4: 基于partial-clustering结果建立原型
    prototypes, prototype_labels = build_prototypes(X, clusters, labeled_mask, targets)

    # 步骤5: 结合原型距离和近邻情况，标注剩余的稀疏点
    final_labels = assign_sparse_points_ssddbc(X, clusters, cluster_labels, prototypes, neighbors, labeled_mask, targets, lambda_weight)

    # 步骤6: 识别潜在未知类聚类
    unknown_clusters = identify_unknown_clusters(clusters, labeled_mask)

    n_clusters = len(clusters)
    print(f"✅ 聚类完成: {n_clusters}个聚类")

    if len(unknown_clusters) > 0:
        print(f"🔍 发现{len(unknown_clusters)}个潜在未知类聚类: {unknown_clusters}")
    else:
        print("🔍 未发现潜在未知类聚类")

    return final_labels, n_clusters, unknown_clusters


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


def test_adaptive_clustering_on_superclass(superclass_name, model_path,
                                         use_train_and_test=True, k=10,
                                         density_percentile=75, lambda_weight=0.7):
    """
    在指定超类上测试自适应聚类算法
    """
    print(f"🧪 测试自适应聚类 - 超类: {superclass_name}")
    print("="*80)

    # 设置参数
    class Args:
        def __init__(self):
            self.dataset_name = 'cifar100_superclass'
            self.superclass_name = superclass_name
            self.prop_train_labels = 0.8
            self.image_size = 224
            self.num_workers = 4
            self.batch_size = 64
            self.base_model = 'vit_dino'
            self.feat_dim = 768
            self.model_path = model_path
            self.interpolation = 3
            self.crop_pct = 0.875
            self.seed = 0

    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = load_model(args, device)

    # 获取超类信息
    superclass_classes = set(CIFAR100_SUPERCLASSES[superclass_name])
    superclass_known_classes_orig = set([cls for cls in superclass_classes if cls < 80])
    superclass_unknown_classes_orig = set([cls for cls in superclass_classes if cls >= 80])

    # 创建标签映射（与超类数据集保持一致）
    all_classes_sorted = sorted(list(superclass_classes))
    label_mapping = {orig_cls: new_cls for new_cls, orig_cls in enumerate(all_classes_sorted)}

    # 映射后的已知/未知类别ID
    known_classes_mapped = set([label_mapping[cls] for cls in superclass_known_classes_orig])
    unknown_classes_mapped = set([label_mapping[cls] for cls in superclass_unknown_classes_orig])

    print(f"📊 超类信息:")
    print(f"   原始已知类: {sorted(list(superclass_known_classes_orig))} -> 映射后: {sorted(list(known_classes_mapped))}")
    print(f"   原始未知类: {sorted(list(superclass_unknown_classes_orig))} -> 映射后: {sorted(list(unknown_classes_mapped))}")

    # 获取数据
    train_transform, test_transform = get_transform('imagenet', image_size=args.image_size, args=args)
    datasets = get_single_superclass_datasets(
        superclass_name=superclass_name,
        train_transform=train_transform,
        test_transform=test_transform,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
        seed=args.seed
    )

    if use_train_and_test:
        # 使用训练集+测试集
        train_dataset = datasets['train_labelled']
        unlabelled_train_dataset = datasets['train_unlabelled']
        test_dataset = datasets['test']

        # 创建MergedDataset
        merged_train_dataset = MergedDataset(
            labelled_dataset=deepcopy(train_dataset),
            unlabelled_dataset=deepcopy(unlabelled_train_dataset)
        )

        # 创建数据加载器
        train_loader = DataLoader(merged_train_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # 提取特征
        print("📊 提取训练集特征...")
        train_feats, train_targets, train_known_mask, train_labeled_mask = extract_features(
            train_loader, model, device, known_classes_mapped
        )

        print("📊 提取测试集特征...")
        test_feats, test_targets, test_known_mask, test_labeled_mask = extract_features(
            test_loader, model, device, known_classes_mapped
        )

        # 合并训练集和测试集
        all_feats = np.concatenate([train_feats, test_feats], axis=0)
        all_targets = np.concatenate([train_targets, test_targets], axis=0)
        all_known_mask = np.concatenate([train_known_mask, test_known_mask], axis=0)
        all_labeled_mask = np.concatenate([train_labeled_mask, test_labeled_mask], axis=0)

    else:
        # 只使用测试集
        test_dataset = datasets['test']
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print("📊 提取测试集特征...")
        all_feats, all_targets, all_known_mask, all_labeled_mask = extract_features(
            test_loader, model, device, known_classes_mapped
        )

    print(f"📊 数据统计:")
    print(f"   总样本数: {len(all_feats)}")
    print(f"   已知类样本: {np.sum(all_known_mask)}")
    print(f"   未知类样本: {np.sum(~all_known_mask)}")
    print(f"   有标签样本: {np.sum(all_labeled_mask)}")
    print(f"   无标签样本: {np.sum(~all_labeled_mask)}")

    # 特征已经在extract_features中进行了L2归一化，与eval_original_gcd保持一致
    # 不再使用StandardScaler，直接使用L2归一化的特征

    # 运行SS-DDBC自适应聚类
    predictions, n_clusters, unknown_clusters = adaptive_density_clustering(
        all_feats, all_targets, all_known_mask, all_labeled_mask,
        k=k, density_percentile=density_percentile, lambda_weight=lambda_weight
    )

    # 确定测试集范围用于ACC计算
    if use_train_and_test:
        # 测试集是后面的部分
        test_start_idx = len(train_feats)
        test_predictions = predictions[test_start_idx:]
        test_targets = all_targets[test_start_idx:]
        test_known_mask = all_known_mask[test_start_idx:]
        print(f"📊 ACC计算范围: 测试集 ({len(test_targets)}个样本, 训练集不参与评估)")
    else:
        # 全部都是测试集
        test_predictions = predictions
        test_targets = all_targets
        test_known_mask = all_known_mask
        print(f"📊 ACC计算范围: 仅测试集 ({len(test_targets)}个样本)")

    # 使用与eval_original_gcd.py相同的ACC计算方法 (split_cluster_acc_v1)
    from project_utils.cluster_and_log_utils import split_cluster_acc_v1
    all_acc, old_acc, new_acc = split_cluster_acc_v1(test_targets, test_predictions, test_known_mask)

    # 计算其他指标（也只在测试集上）
    nmi = normalized_mutual_info_score(test_targets, test_predictions)
    ari = adjusted_rand_score(test_targets, test_predictions)

    print(f"📈 聚类结果:")
    print(f"   聚类数量: {n_clusters}")
    print(f"   潜在未知类: {len(unknown_clusters)}个")
    print(f"   All ACC: {all_acc:.4f}")
    print(f"   Old ACC: {old_acc:.4f}")
    print(f"   New ACC: {new_acc:.4f}")
    print(f"   NMI: {nmi:.4f}")
    print(f"   ARI: {ari:.4f}")

    # 显示每个聚类的内部情况
    analyze_cluster_composition(predictions, all_targets, all_known_mask, all_labeled_mask, unknown_clusters)

    # 提取测试集特征用于K-means对比 (现在使用相同的L2归一化特征)
    if use_train_and_test:
        # 测试集是后面的部分
        test_start_idx = len(train_feats)
        test_features_for_kmeans = all_feats[test_start_idx:]
    else:
        # 全部都是测试集
        test_features_for_kmeans = all_feats

    # 返回结果，包含测试集数据用于K-means对比
    return {
        'method': 'SS-DDBC',
        'all_acc': all_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'nmi': nmi,
        'ari': ari,
        'n_clusters': n_clusters,
        'unknown_clusters': unknown_clusters,
        'test_features': test_features_for_kmeans,
        'test_targets': test_targets,
        'test_known_mask': test_known_mask
    }


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='自适应密度聚类算法测试')

    # 必要参数
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='训练好的模型路径（未提供时需确保缓存特征可用）')
    parser.add_argument('--superclass_name', type=str, default='trees',
                        help='测试的超类名称')

    # 算法参数
    parser.add_argument('--use_train_and_test', type=str2bool, default=True,
                        help='是否合并训练集和测试集')
    parser.add_argument('--k', type=int, default=10,
                        help='k近邻参数')
    parser.add_argument('--density_percentile', type=int, default=75,
                        help='密度阈值百分位数')
    parser.add_argument('--lambda_weight', type=float, default=0.7,
                        help='冲突解决权重')
    parser.add_argument('--run_kmeans_baseline', type=str2bool, default=False,
                        help='是否运行K-means基线对比')

    args = parser.parse_args()

    print("自适应密度聚类算法测试")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"超类名称: {args.superclass_name}")
    print(f"使用训练+测试: {args.use_train_and_test}")
    print(f"算法参数: k={args.k}, density_percentile={args.density_percentile}, lambda={args.lambda_weight}")
    print("="*80)

    try:
        # 运行SS-DDBC算法
        ssddbc_results = test_adaptive_clustering_on_superclass(
            superclass_name=args.superclass_name,
            model_path=args.model_path,
            use_train_and_test=args.use_train_and_test,
            k=args.k,
            density_percentile=args.density_percentile,
            lambda_weight=args.lambda_weight
        )

        # 如果开启K-means基线对比
        if args.run_kmeans_baseline:
            print("\n" + "="*80)
            print("🔄 运行K-means基线对比...")
            print("✅ 现在使用与eval_original_gcd完全相同的L2归一化特征")

            # 使用SS-DDBC测试中已提取的测试集特征 (相同的L2归一化)
            test_features = ssddbc_results['test_features']
            test_targets = ssddbc_results['test_targets']
            test_known_mask = ssddbc_results['test_known_mask']

            # 使用真实的类别数作为K-means聚类数（与eval_original_gcd.py保持一致）
            n_true_classes = len(np.unique(test_targets))
            print(f"🎯 K-means聚类数量: {n_true_classes} (真实类别数)")

            # 运行K-means (使用相同的L2归一化特征)
            kmeans_results = test_kmeans_baseline(
                test_features,  # 相同的L2归一化特征
                test_targets,
                test_known_mask,
                n_clusters=n_true_classes,  # 使用真实类别数
                random_state=0  # 与原版一致的随机种子
            )

            # 对比结果
            print(f"\n📊 算法对比结果:")
            print("="*80)
            print(f"{'指标':<15} {'SS-DDBC':<12} {'K-means':<12} {'差异':<12}")
            print("-"*80)
            print(f"{'All ACC':<15} {ssddbc_results['all_acc']:<12.4f} {kmeans_results['all_acc']:<12.4f} {ssddbc_results['all_acc']-kmeans_results['all_acc']:<+12.4f}")
            print(f"{'Old ACC':<15} {ssddbc_results['old_acc']:<12.4f} {kmeans_results['old_acc']:<12.4f} {ssddbc_results['old_acc']-kmeans_results['old_acc']:<+12.4f}")
            print(f"{'New ACC':<15} {ssddbc_results['new_acc']:<12.4f} {kmeans_results['new_acc']:<12.4f} {ssddbc_results['new_acc']-kmeans_results['new_acc']:<+12.4f}")
            print(f"{'NMI':<15} {ssddbc_results['nmi']:<12.4f} {kmeans_results['nmi']:<12.4f} {ssddbc_results['nmi']-kmeans_results['nmi']:<+12.4f}")
            print(f"{'ARI':<15} {ssddbc_results['ari']:<12.4f} {kmeans_results['ari']:<12.4f} {ssddbc_results['ari']-kmeans_results['ari']:<+12.4f}")
            print(f"{'聚类数':<15} {ssddbc_results['n_clusters']:<12} {kmeans_results['n_clusters']:<12} {'=':<12}")
            print("="*80)

        print("\n测试完成!")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
