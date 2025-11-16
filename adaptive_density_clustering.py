#!/usr/bin/env python3
"""
自适应密度聚类算法
基于提供的原始聚类逻辑重新实现，完全符合start_new函数的版本
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math


# 辅助函数实现
def euclidean_distance(point1, point2):
    """计算欧几里得距离"""
    return np.linalg.norm(point1 - point2)

def compute_and_identify_points(X, k, p):
    """
    计算距离、邻居索引、点类别、核心点标识和密度

    Args:
        X: 特征矩阵
        k: k近邻数量
        p: 密度阈值百分位数

    Returns:
        distances: 距离矩阵
        index_matrix: k近邻索引矩阵
        point_categories: 点类别 (1=核心点, 0=边界点)
        is_core: 核心点标识
        delts: 密度值
    """
    n_samples = len(X)

    # 计算所有点之间的距离
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i][j] = euclidean_distance(X[i], X[j])

    # 计算k近邻索引
    index_matrix = np.zeros((n_samples, k), dtype=int)
    for i in range(n_samples):
        # 获取最近的k个邻居（不包括自身）
        nearest_indices = np.argsort(distances[i])[1:k+1]
        index_matrix[i] = nearest_indices

    # 计算密度（基于k近邻距离的倒数）
    delts = np.zeros(n_samples)
    for i in range(n_samples):
        knn_distances = distances[i][index_matrix[i]]
        avg_knn_dist = np.mean(knn_distances)
        delts[i] = 1.0 / (avg_knn_dist + 1e-10)

    # 确定密度阈值并识别核心点
    density_threshold = np.percentile(delts, p)
    is_core = delts > density_threshold
    point_categories = is_core.astype(int)

    return distances, index_matrix, point_categories, is_core, delts

def compute_knn_distances(index_matrix, distances):
    """计算k近邻距离"""
    n_samples = len(index_matrix)
    knn_distances = []
    for i in range(n_samples):
        row_distances = [distances[i][j] for j in index_matrix[i]]
        knn_distances.append(row_distances)
    return knn_distances

def compute_proto(cluster, X):
    """计算聚类原型（质心）"""
    if len(cluster) == 0:
        return np.zeros(X.shape[1])
    cluster_points = list(cluster)
    return np.mean(X[cluster_points], axis=0)

def compute_confidence(point, clusters_x, clusters_label, total_clusters):
    """计算基于原型的置信度"""
    confidences = []
    for cluster_id in total_clusters:
        if cluster_id < len(clusters_x):
            distance = euclidean_distance(point, clusters_x[cluster_id])
            confidence = 1.0 / (distance + 1e-10)
            confidences.append(confidence)
        else:
            confidences.append(0.0)

    # 归一化
    total = sum(confidences)
    if total > 0:
        confidences = [c / total for c in confidences]

    return confidences

def compute_knn_confidence(point, X, pred, density, k=2):
    """计算基于k近邻的置信度"""
    # 计算到所有点的距离
    distances = [euclidean_distance(point, X[j]) for j in range(len(X))]

    # 找到k个最近邻
    knn_indices = np.argsort(distances)[:k]

    # 统计邻居的聚类分布
    neighbor_clusters = [pred[idx] for idx in knn_indices if pred[idx] != -1]

    if not neighbor_clusters:
        # 如果没有已分配的邻居，返回均匀分布
        n_clusters = len(set(pred[pred != -1])) if len(set(pred[pred != -1])) > 0 else 1
        return [1.0 / n_clusters] * n_clusters

    # 计算每个聚类的置信度
    cluster_counts = Counter(neighbor_clusters)
    max_cluster = max(cluster_counts.keys()) if cluster_counts else 0
    confidences = []

    for cluster_id in range(max_cluster + 1):
        count = cluster_counts.get(cluster_id, 0)
        confidence = count / len(neighbor_clusters)
        confidences.append(confidence)

    return confidences

def calculate_accuracy(pred, true_labels):
    """计算聚类准确率"""
    from project_utils.cluster_utils import cluster_acc
    return cluster_acc(true_labels, pred)

def silhouette_score(X, clusters, x_cluster, clusters_num):
    """计算轮廓系数"""
    if clusters_num <= 1:
        return 0.0

    total_score = 0.0
    n_points = 0

    for i in range(len(X)):
        cluster_id = x_cluster[i]
        if cluster_id == -1 or len(clusters[cluster_id]) <= 1:
            continue

        # 计算簇内平均距离
        a_i = 0.0
        cluster_points = list(clusters[cluster_id])
        for j in cluster_points:
            if i != j:
                a_i += euclidean_distance(X[i], X[j])
        a_i /= (len(cluster_points) - 1)

        # 计算到最近簇的平均距离
        b_i = float('inf')
        for other_cluster_id in range(clusters_num):
            if other_cluster_id != cluster_id and len(clusters[other_cluster_id]) > 0:
                avg_dist = 0.0
                other_points = list(clusters[other_cluster_id])
                for j in other_points:
                    avg_dist += euclidean_distance(X[i], X[j])
                avg_dist /= len(other_points)
                b_i = min(b_i, avg_dist)

        # 计算轮廓系数
        if b_i != float('inf'):
            s_i = (b_i - a_i) / max(a_i, b_i)
            total_score += s_i
            n_points += 1

    return total_score / n_points if n_points > 0 else 0.0

class AdaptiveDensityClustering:
    """
    自适应密度聚类器 - 版本A：融合start_new精细算法 + 保持GCD未知类别发现
    """

    def __init__(self, k_neighbors=3, density_percentile=70, lambda_weight=0.7,
                 min_cluster_size=3, standardize=True, unknown_threshold=0.3):
        """
        初始化聚类参数

        Args:
            k_neighbors: k近邻数量
            density_percentile: 密度阈值百分位数 (对应原始代码中的p参数)
            lambda_weight: 原型置信度与knn置信度的权重 (对应原始代码中的lamda参数)
            min_cluster_size: 最小聚类大小
            standardize: 是否标准化特征
            unknown_threshold: 未知类别检测阈值
        """
        self.k = k_neighbors
        self.density_percentile = density_percentile
        self.lambda_weight = lambda_weight
        self.min_cluster_size = min_cluster_size
        self.standardize = standardize
        self.unknown_threshold = unknown_threshold

        # 聚类结果
        self.clusters = []
        self.cluster_assignments = None
        self.cluster_prototypes = []
        self.densities = None
        self.train_size = 0  # 记录训练集大小，用于分离训练和测试结果

    def enhanced_fit_predict(self, train_x, train_y, train_label_masks, test_x, test_y, test_label_masks, train_classes):
        """
        改进版本A：合并训练集和测试集进行聚类，保持GCD未知类别发现能力

        Args:
            train_x: 训练特征
            train_y: 训练标签
            train_label_masks: 训练集标签掩码（1=有标签，0=无标签）
            test_x: 测试特征
            test_y: 测试标签
            test_label_masks: 测试集标签掩码（1=有标签，0=无标签）
            train_classes: 训练时的已知类别集合

        Returns:
            test_predictions: 测试集预测结果
            test_acc, test_nmi, test_ari: 测试集评估指标
        """
        print("Starting enhanced adaptive density clustering...")

        # 1. 合并训练和测试数据
        X = np.concatenate((train_x, test_x), axis=0)
        Y = np.concatenate((train_y, test_y), axis=0)

        # 确保label_masks是1D数组
        train_label_masks = np.array(train_label_masks).flatten()
        test_label_masks = np.array(test_label_masks).flatten()
        label_masks = np.concatenate((train_label_masks, test_label_masks), axis=0)
        self.train_size = len(train_x)

        print(f"Data info:")
        print(f"   Train samples: {len(train_x)}")
        print(f"   Test samples: {len(test_x)}")
        print(f"   Total samples: {len(X)}")
        print(f"   Feature dimensions: {X.shape[1]}")

        # 2. 创建已知/未知标签掩码（关键改进）
        known_labels = self._create_enhanced_known_labels(Y, train_classes, label_masks)

        # 3. 特征标准化
        if self.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            print("Features standardized")

        # 4. 使用改进的start_new风格聚类算法
        cluster_predictions = self._enhanced_clustering_algorithm(X, known_labels)

        # 5. 只评估测试集部分
        test_predictions = cluster_predictions[self.train_size:]
        test_true_labels = test_y

        # 6. 计算测试集评估指标
        test_acc = calculate_accuracy(test_predictions, test_true_labels)
        test_nmi = normalized_mutual_info_score(test_true_labels, test_predictions)
        test_ari = adjusted_rand_score(test_true_labels, test_predictions)

        print(f"Clustering completed!")
        print(f"Test set results:")
        print(f"   Clusters found: {len(self.clusters)}")
        print(f"   Test accuracy: {test_acc:.4f}")
        print(f"   Test NMI: {test_nmi:.4f}")
        print(f"   Test ARI: {test_ari:.4f}")

        return test_predictions, test_acc, test_nmi, test_ari

    def _create_enhanced_known_labels(self, Y, train_classes, label_masks):
        """
        创建增强的已知标签数组

        关键点：
        1. 只有当样本既属于已知类别又在label_mask中标记为有标签时，才保持原标签
        2. 其他所有情况（未知类别或无标签）都标记为-1

        Args:
            Y: 所有标签
            train_classes: 已知类别集合
            label_masks: 标签掩码（1=有标签，0=无标签）
        """
        known_labels = np.full(len(Y), -1, dtype=int)

        train_labeled_known_count = 0
        train_unlabeled_known_count = 0
        train_labeled_unknown_count = 0
        train_unlabeled_unknown_count = 0
        test_labeled_known_count = 0
        test_unlabeled_known_count = 0
        test_labeled_unknown_count = 0
        test_unlabeled_unknown_count = 0

        # 处理所有样本
        for i in range(len(Y)):
            is_known_class = Y[i] in train_classes
            is_labeled = label_masks[i] == 1
            is_train = i < self.train_size

            # GCD设置：只有训练集中有标签的已知类别样本才保持原标签
            # 测试集中的所有样本都被视为"未知"，用于聚类发现
            if is_train and is_known_class and is_labeled:
                known_labels[i] = Y[i]
                train_labeled_known_count += 1
            else:
                known_labels[i] = -1
                if is_train:
                    if is_known_class:
                        train_unlabeled_known_count += 1
                    else:
                        if is_labeled:
                            train_labeled_unknown_count += 1
                        else:
                            train_unlabeled_unknown_count += 1
                else:
                    # 测试集中的所有样本都被视为"未知"
                    if is_known_class:
                        test_unlabeled_known_count += 1
                    else:
                        test_unlabeled_unknown_count += 1

        print(f"Label mask analysis:")
        print(f"   Train - Known classes with labels: {train_labeled_known_count}")
        print(f"   Train - Known classes without labels: {train_unlabeled_known_count}")
        print(f"   Train - Unknown classes with labels: {train_labeled_unknown_count}")
        print(f"   Train - Unknown classes without labels: {train_unlabeled_unknown_count}")
        print(f"   Test - Known classes with labels: {test_labeled_known_count}")
        print(f"   Test - Known classes without labels: {test_unlabeled_known_count}")
        print(f"   Test - Unknown classes with labels: {test_labeled_unknown_count}")
        print(f"   Test - Unknown classes without labels: {test_unlabeled_unknown_count}")

        return known_labels

    def _enhanced_clustering_algorithm(self, X, known_labels):
        """
        增强版聚类算法：融合start_new的精细算法 + GCD未知类别发现
        """
        print("Executing enhanced clustering algorithm...")

        # 1. 使用start_new的密度计算和核心点识别
        distances, index_matrix, point_categories, is_core, delts = compute_and_identify_points(
            X, self.k, self.density_percentile
        )
        self.densities = delts

        print(f"   核心点数: {np.sum(point_categories == 1)}/{len(X)}")

        # 2. 使用start_new的两阶段聚类构建
        self.clusters, cluster_assignments = self._build_clusters_with_start_new_logic(
            X, distances, index_matrix, point_categories, is_core, known_labels
        )

        # 3. 使用增强的混合置信度分配边界点
        cluster_assignments = self._enhanced_assign_boundary_points(
            X, cluster_assignments, known_labels, delts
        )

        # 4. 后处理：合并小聚类和质量优化
        cluster_assignments = self._post_process_clusters(X, cluster_assignments, known_labels)

        self.cluster_assignments = cluster_assignments
        return cluster_assignments

    def _build_clusters_with_start_new_logic(self, X, distances, index_matrix, point_categories, is_core, known_labels):
        """
        使用start_new的两阶段聚类构建逻辑，但增加标签兼容性检查
        """
        print("Building initial clusters (start_new style)...")

        # 初始化聚类相关变量
        clusters = [set() for _ in range(1000)]
        clusters_num = 0
        is_cluster = np.zeros(len(X), dtype=bool)
        x_cluster = np.full(len(X), -1, dtype=int)
        x_far = np.full(len(X), -1, dtype=int)
        total_clusters = set()

        # 1. 构建初始簇（从核心点开始）
        for i in range(len(X)):
            if point_categories[i] == 1 and is_cluster[i] == False:
                clusters[clusters_num].add(i)
                x_cluster[i] = clusters_num
                total_clusters.add(clusters_num)
                clusters_num = clusters_num + 1
                is_cluster[i] = True
                x_far[i] = i

                flag = True
                new_num = x_cluster[i]
                clusters_copy = list(clusters[new_num])

                while flag:
                    for j in clusters_copy:
                        if point_categories[j] == 1:
                            for a in index_matrix[j]:
                                if a == j:
                                    continue

                                # 增强的标签兼容性检查
                                if not self._enhanced_label_compatible(j, a, known_labels):
                                    continue

                                if is_cluster[a] == False:
                                    clusters[new_num].add(a)
                                    x_cluster[a] = new_num
                                    is_cluster[a] = True
                                    x_far[a] = j
                                else:
                                    if point_categories[a] == 1:
                                        if x_cluster[a] != x_cluster[j]:
                                            # 聚类合并前检查兼容性
                                            if self._clusters_compatible(clusters[x_cluster[a]], clusters[x_cluster[j]], known_labels):
                                                total_clusters.discard(x_cluster[a])
                                                for x in clusters[x_cluster[a]]:
                                                    clusters[x_cluster[j]].add(x)
                                                    x_cluster[x] = x_cluster[j]
                                                x_far[a] = j

                    if clusters_copy == list(clusters[new_num]):
                        flag = False
                    else:
                        clusters_copy = list(clusters[new_num])

        # 2. 重新整理聚类
        clusters_num = len(total_clusters)
        clusters_new = [set() for _ in range(1000)]
        a = 0
        for i in total_clusters:
            clusters_new[a] = clusters[i]
            a += 1

        # 3. 重新分配聚类ID
        cluster_assignments = np.full(len(X), -1, dtype=int)
        for cluster_id, cluster in enumerate(clusters_new[:clusters_num]):
            for point in cluster:
                cluster_assignments[point] = cluster_id

        print(f"   初始聚类数: {clusters_num}")
        return clusters_new[:clusters_num], cluster_assignments

    def _enhanced_label_compatible(self, point1, point2, known_labels):
        """
        增强的标签兼容性检查
        """
        if known_labels is None:
            return True

        label1 = known_labels[point1]
        label2 = known_labels[point2]

        # 如果都是已知标签，必须相同
        if label1 != -1 and label2 != -1:
            return label1 == label2

        # 已知和未知可以在同一聚类（这是关键改进）
        # 但优先保持已知类别的纯度
        return True

    def _clusters_compatible(self, cluster1, cluster2, known_labels):
        """
        检查两个聚类是否可以合并
        """
        if known_labels is None:
            return True

        # 获取两个聚类的已知标签分布
        labels1 = set()
        labels2 = set()

        for point in cluster1:
            if known_labels[point] != -1:
                labels1.add(known_labels[point])

        for point in cluster2:
            if known_labels[point] != -1:
                labels2.add(known_labels[point])

        # 如果两个聚类都有已知标签，它们必须相同
        if labels1 and labels2:
            return labels1 == labels2

        # 如果其中一个或两个都是纯未知聚类，可以合并
        return True

    def _enhanced_assign_boundary_points(self, X, cluster_assignments, known_labels, delts):
        """
        使用增强的混合置信度分配边界点
        """
        print("Assigning boundary points (enhanced mixed confidence)...")

        # 计算聚类原型
        clusters_x = []
        valid_clusters = []
        for cluster_id, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                clusters_x.append(compute_proto(cluster, X))
                valid_clusters.append(cluster_id)

        unassigned_count = 0
        new_cluster_count = 0

        for i in range(len(X)):
            if cluster_assignments[i] == -1:
                # 计算原型置信度
                confidences_p = self._compute_enhanced_prototype_confidence(X[i], clusters_x, valid_clusters)

                # 计算k近邻置信度
                knn_confidences = self._compute_enhanced_knn_confidence(X[i], X, cluster_assignments, delts[i])

                # 混合置信度
                if len(confidences_p) > 0 and len(knn_confidences) > 0:
                    # 确保两个置信度数组长度相同
                    min_len = min(len(confidences_p), len(knn_confidences))
                    confidences_p = confidences_p[:min_len]
                    knn_confidences = knn_confidences[:min_len]

                    combined_confidences = [
                        self.lambda_weight * cp + (1 - self.lambda_weight) * kc
                        for cp, kc in zip(confidences_p, knn_confidences)
                    ]

                    max_confidence = max(combined_confidences) if combined_confidences else 0.0

                    # 未知检测机制
                    if max_confidence > self.unknown_threshold:
                        # 置信度足够高，分配到最佳聚类
                        best_cluster = valid_clusters[np.argmax(combined_confidences)]
                        cluster_assignments[i] = best_cluster
                        self.clusters[best_cluster].add(i)
                        unassigned_count += 1
                    else:
                        # 置信度太低，创建新聚类（潜在未知类别）
                        new_cluster_id = len(self.clusters)
                        self.clusters.append(set([i]))
                        cluster_assignments[i] = new_cluster_id
                        new_cluster_count += 1
                else:
                    # 如果没有有效的聚类，创建新聚类
                    new_cluster_id = len(self.clusters)
                    self.clusters.append(set([i]))
                    cluster_assignments[i] = new_cluster_id
                    new_cluster_count += 1

        print(f"   已分配边界点: {unassigned_count}")
        print(f"   新建聚类数: {new_cluster_count}")

        return cluster_assignments

    def _compute_enhanced_prototype_confidence(self, point, prototypes, valid_clusters):
        """
        计算增强的原型置信度
        """
        if not prototypes:
            return []

        confidences = []
        for prototype in prototypes:
            distance = euclidean_distance(point, prototype)
            confidence = 1.0 / (distance + 1e-10)
            confidences.append(confidence)

        # 归一化
        total = sum(confidences)
        if total > 0:
            confidences = [c / total for c in confidences]

        return confidences

    def _compute_enhanced_knn_confidence(self, point, X, cluster_assignments, density, k=2):
        """
        计算增强的k近邻置信度
        """
        # 计算到所有点的距离
        distances = [euclidean_distance(point, X[j]) for j in range(len(X))]

        # 找到k个最近邻
        knn_indices = np.argsort(distances)[:k]

        # 统计邻居的聚类分布
        neighbor_clusters = [cluster_assignments[idx] for idx in knn_indices
                           if cluster_assignments[idx] != -1]

        if not neighbor_clusters:
            n_clusters = len(self.clusters)
            return [1.0 / n_clusters] * n_clusters if n_clusters > 0 else []

        # 计算每个聚类的置信度
        cluster_counts = Counter(neighbor_clusters)
        n_clusters = len(self.clusters)
        confidences = []

        for cluster_id in range(n_clusters):
            count = cluster_counts.get(cluster_id, 0)
            confidence = count / len(neighbor_clusters)
            confidences.append(confidence)

        return confidences

    def _post_process_clusters(self, X, cluster_assignments, known_labels):
        """
        后处理：合并小聚类和质量优化
        """
        print("Post-processing clusters...")

        # 1. 移除过小的聚类
        cluster_sizes = [len(cluster) for cluster in self.clusters]
        small_clusters = [i for i, size in enumerate(cluster_sizes) if size < self.min_cluster_size]

        for small_cluster_id in small_clusters:
            if len(self.clusters[small_cluster_id]) > 0:
                # 将小聚类的点重新分配到最近的大聚类
                for point in list(self.clusters[small_cluster_id]):
                    best_cluster = self._find_best_cluster_for_point(point, X, known_labels)
                    if best_cluster != -1 and best_cluster != small_cluster_id:
                        cluster_assignments[point] = best_cluster
                        self.clusters[best_cluster].add(point)

                self.clusters[small_cluster_id] = set()

        # 2. 移除空聚类并重新编号
        non_empty_clusters = [cluster for cluster in self.clusters if len(cluster) > 0]
        self.clusters = non_empty_clusters

        # 重新分配聚类ID
        new_assignments = np.full(len(cluster_assignments), -1)
        for new_id, cluster in enumerate(self.clusters):
            for point in cluster:
                new_assignments[point] = new_id

        print(f"   最终聚类数: {len(self.clusters)}")
        return new_assignments

    def _find_best_cluster_for_point(self, point, X, known_labels):
        """
        为单个点找到最佳聚类
        """
        if not self.clusters:
            return -1

        best_cluster = -1
        min_distance = float('inf')

        for cluster_id, cluster in enumerate(self.clusters):
            if len(cluster) >= self.min_cluster_size:
                # 计算到聚类质心的距离
                cluster_points = list(cluster)
                centroid = np.mean(X[cluster_points], axis=0)
                distance = euclidean_distance(X[point], centroid)

                # 考虑标签兼容性
                if self._point_cluster_compatible(point, cluster, known_labels):
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster_id

        return best_cluster

    def _point_cluster_compatible(self, point, cluster, known_labels):
        """
        检查点与聚类的兼容性
        """
        if known_labels is None:
            return True

        point_label = known_labels[point]

        # 如果点是未知标签，可以加入任何聚类
        if point_label == -1:
            return True

        # 如果点是已知标签，检查聚类中已知标签的一致性
        cluster_known_labels = set()
        for cluster_point in cluster:
            if known_labels[cluster_point] != -1:
                cluster_known_labels.add(known_labels[cluster_point])

        # 如果聚类没有已知标签，可以加入
        if not cluster_known_labels:
            return True

        # 如果聚类有已知标签，必须与点的标签相同
        return point_label in cluster_known_labels

    def start_new_clustering(self, train_x, train_y, query_x, query_y):
        """
        基于提供的start_new函数的完整聚类流程

        Args:
            train_x: 训练特征
            train_y: 训练标签
            query_x: 查询特征
            query_y: 查询标签

        Returns:
            acc, nmi, ari, sh: 评估指标
        """
        # 合并数据
        X = np.concatenate((train_x, query_x), axis=0)
        Y = np.concatenate((train_y, query_y), axis=0)

        # 标准化特征
        if self.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # 计算距离、邻居、密度等
        distances, index_matrix, point_categories, is_core, delts = compute_and_identify_points(
            X, self.k, self.density_percentile
        )

        knn_distances = compute_knn_distances(index_matrix, distances)

        # 初始化聚类相关变量
        clusters = [set() for _ in range(1000)]
        clusters_num = 0
        clusters_label = np.full(1000, -1, dtype=int)
        is_cluster = np.zeros(len(X), dtype=bool)
        x_cluster = np.full(len(X), -1, dtype=int)
        x_far = np.full(len(X), -1, dtype=int)
        is_c = set()
        total_clusters = set()

        # 1. 构建初始簇
        for i in range(len(X)):
            if point_categories[i] == 1 and is_cluster[i] == False:
                clusters[clusters_num].add(i)
                x_cluster[i] = clusters_num
                total_clusters.add(clusters_num)
                clusters_num = clusters_num + 1
                is_cluster[i] = True
                x_far[i] = i

                flag = True
                new_num = x_cluster[i]
                clusters_copy = list(clusters[new_num])
                while flag:
                    for j in clusters_copy:
                        if point_categories[j] == 1:
                            for a in index_matrix[j]:
                                if a == j:
                                    continue
                                if is_cluster[a] == False:
                                    clusters[new_num].add(a)
                                    x_cluster[a] = new_num
                                    is_cluster[a] = True
                                    x_far[a] = j
                                else:
                                    if point_categories[a] == 1:
                                        if x_cluster[a] != x_cluster[j]:
                                            total_clusters.discard((x_cluster[a]))
                                            for x in clusters[x_cluster[a]]:
                                                clusters[x_cluster[j]].add(x)
                                                x_cluster[x] = x_cluster[j]
                                            x_far[a] = j
                    if clusters_copy == list(clusters[new_num]):
                        flag = False
                    else:
                        clusters_copy = list(clusters[new_num])

        # 重新整理聚类
        clusters_num = len(total_clusters)
        clusters_new = [set() for _ in range(1000)]
        a = 0
        for i in total_clusters:
            clusters_new[a] = clusters[i]
            a += 1
        total_clusters = set()
        clusters_label = np.full(1000, -1, dtype=int)
        for i in range(clusters_num):
            total_clusters.add(i)
        x_cluster = np.full(len(X), -1, dtype=int)
        l = 0
        pred = np.full(len(X), -1, dtype=int)
        for i in total_clusters:
            for j in clusters_new[i]:
                pred[j] = l
                x_cluster[j] = i
            clusters_label[i] = l
            l += 1

        # 计算聚类原型
        clusters_x = []
        for i in range(clusters_num):
            clusters_x.append(compute_proto(clusters_new[i], X))

        # 分配未分配的点
        a = 0
        for i in range(len(X)):
            if pred[i] == -1:
                confidences_p = compute_confidence(X[i], clusters_x, clusters_label, total_clusters)
                knn_confidences = compute_knn_confidence(X[i], X, pred, delts[i], k=2)
                confidences = [self.lambda_weight * cp + (1 - self.lambda_weight) * kn
                             for cp, kn in zip(confidences_p, knn_confidences)]
                pred[i] = np.argmax(confidences)
                a += 1

        # 保存聚类结果
        self.clusters = clusters_new[:clusters_num]
        self.cluster_assignments = pred.copy()

        # 评估
        pred_sorted = np.sort(pred)
        Y_sorted = np.sort(Y)

        acc = calculate_accuracy(pred_sorted, Y_sorted)
        nmi = normalized_mutual_info_score(pred_sorted, Y_sorted)
        ari = adjusted_rand_score(pred_sorted, Y_sorted)
        sh = silhouette_score(X, clusters_new, x_cluster, clusters_num) / clusters_num if clusters_num > 0 else 0

        return acc, nmi, ari, sh

    def fit_predict(self, X, known_labels=None):
        """
        执行聚类并返回预测标签 - 适配新版本

        Args:
            X: 特征矩阵 [n_samples, n_features]
            known_labels: 已知标签数组，-1表示未知

        Returns:
            predictions: 聚类标签预测
        """
        print("🚀 开始自适应密度聚类...")

        # 为了保持接口兼容性，将单个特征矩阵分割为训练和查询部分
        # 这里假设前一半是训练数据，后一半是查询数据
        mid_point = len(X) // 2
        train_x = X[:mid_point]
        query_x = X[mid_point:]

        # 创建虚拟标签
        if known_labels is not None:
            train_y = known_labels[:mid_point]
            query_y = known_labels[mid_point:]
        else:
            train_y = np.arange(mid_point)
            query_y = np.arange(len(query_x)) + mid_point

        # 调用新的聚类方法
        acc, nmi, ari, sh = self.start_new_clustering(train_x, train_y, query_x, query_y)

        print(f"🎉 聚类完成! 发现 {len(self.clusters)} 个聚类")
        print(f"📊 评估结果: ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}, SH={sh:.4f}")

        return self.cluster_assignments

    def get_unknown_clusters(self, known_labels):
        """
        识别未知类别聚类

        Args:
            known_labels: 已知标签数组

        Returns:
            unknown_cluster_ids: 未知聚类的ID列表
        """
        if known_labels is None:
            return list(range(len(self.clusters)))

        unknown_clusters = []

        for cluster_id, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                continue

            # 检查聚类中是否包含已知标签
            has_known_labels = False

            for point in cluster:
                if point < len(known_labels) and known_labels[point] != -1:
                    has_known_labels = True
                    break

            if not has_known_labels:
                unknown_clusters.append(cluster_id)

        return unknown_clusters


def evaluate_clustering_results(predictions, true_labels):
    """
    评估聚类结果

    Args:
        predictions: 聚类预测
        true_labels: 真实标签

    Returns:
        metrics: 评估指标字典
    """
    from project_utils.cluster_utils import cluster_acc

    # 计算各种指标
    acc = cluster_acc(true_labels, predictions)
    nmi = normalized_mutual_info_score(true_labels, predictions)
    ari = adjusted_rand_score(true_labels, predictions)

    n_clusters_pred = len(set(predictions))
    n_clusters_true = len(set(true_labels))

    metrics = {
        'accuracy': acc,
        'nmi': nmi,
        'ari': ari,
        'n_clusters_predicted': n_clusters_pred,
        'n_clusters_true': n_clusters_true
    }

    print(f"📊 聚类评估结果:")
    print(f"   准确率: {acc:.4f}")
    print(f"   NMI: {nmi:.4f}")
    print(f"   ARI: {ari:.4f}")
    print(f"   预测聚类数: {n_clusters_pred}")
    print(f"   真实聚类数: {n_clusters_true}")

    return metrics


# 使用示例
if __name__ == "__main__":
    # 示例用法
    print("🧪 自适应密度聚类测试")

    # 生成测试数据
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=4, n_features=10,
                      random_state=42, cluster_std=1.5)

    # 模拟已知/未知标签
    known_mask = np.random.random(len(y)) < 0.5
    known_labels = np.where(known_mask, y, -1)

    # 执行聚类
    clusterer = AdaptiveDensityClustering(
        k_neighbors=5,
        density_percentile=75,
        lambda_weight=0.7,
        min_cluster_size=3
    )

    predictions = clusterer.fit_predict(X, known_labels)

    # 评估结果
    metrics = evaluate_clustering_results(predictions, y)

    # 识别未知聚类
    unknown_clusters = clusterer.get_unknown_clusters(known_labels)
    print(f"🔍 发现 {len(unknown_clusters)} 个潜在未知类别聚类: {unknown_clusters}")