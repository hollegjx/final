#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SS-DDBC聚类构建模块
实现基于密度的半监督聚类算法
"""

import numpy as np
from .conflict import ssddbc_conflict_resolution


def build_clusters_ssddbc(X, high_density_mask, neighbors, labeled_mask, targets, densities, known_mask, k=10, co=None, silent=False, logger=None, train_size=None):
    """
    完全按照SS-DDBC算法构建聚类

    算法流程:
    For each high-density point xi:
        If xi is not assigned to any cluster, create a new cluster pi
        For each neighbor xj in k-neighbors of xi:
            If xj is not assigned to any cluster, add xj to pi
            Else If xj ∈ pj (xj已在簇pj中):
                检查簇级别标签冲突:
                - 如果双方都有标签且不同 (si ≠ sj) → 使用密度判断（只有密度高才移动）
                - 否则（标签相同或至少一方无标签）→ 直接合并簇

    核心逻辑:
    1. 样本级标签冲突（有标签样本直接相连且标签不同）= 硬约束，直接拒绝
    2. 簇级标签冲突（两个簇都有标签且不同）= 软约束，通过密度判断
    3. 无簇级冲突（标签相同或至少一方无标签）= 直接合并
    4. 簇的类别取决于其有标签样本（规模≥5，占比>25%，纯度≥80%），否则为未知类
    5. 延迟标签判断机制：簇规模<5时暂不分配标签，避免早期小簇因异类样本失去标签导致簇连接

    Args:
        X: 特征矩阵
        high_density_mask: 高密度点掩码
        neighbors: k近邻索引矩阵（基于所有点计算）
        labeled_mask: 有标签掩码
        targets: 真实标签
        densities: 密度数组
        known_mask: 已知类掩码
        k: 扩展时的近邻数
        co: 截止距离
        silent: 静默模式（默认False）
        logger: 详细日志记录器（可选，用于记录骨干网络聚类详情）
        train_size: 原始训练集大小（用于判断样本来源）

    Returns:
        clusters: 聚类列表，每个聚类是一个样本索引集合
        cluster_labels: 每个样本的聚类标签 (-1表示未分配)
        high_density_neighbors_map: 高密度点的邻居映射字典
        cluster_category_labels: 簇ID到类别标签的映射字典（只包含有类别标签的簇）
    """
    n_samples = X.shape[0]
    cluster_labels = np.full(n_samples, -1, dtype=int)
    clusters = []
    current_cluster_id = 0

    # 调试信息收集器
    debug_info_collector = {}

    # 簇标签字典：记录每个簇的类别标签（满足条件才有标签）
    cluster_category_labels = {}  # cluster_id -> category_label (or None)

    def update_cluster_label(cluster_id, cluster):
        """
        更新簇的类别标签

        规则：
        1. 簇规模 ≥ MIN_CLUSTER_SIZE_FOR_LABEL（延迟标签判断，避免早期小簇因1-2个异类样本失去标签）
        2. 已知标签样本占比 > 25% 且数量 ≠ 0
        3. 已知样本中，单种标签纯度 ≥ 80%

        Returns:
            category_label: 簇的类别标签（如果满足条件），否则返回None
        """
        cluster_indices = list(cluster)
        if len(cluster_indices) == 0:
            return None

        # 【新增】延迟标签判断：只在簇达到一定规模后才判断标签
        # 这样可以避免早期小簇因为1-2个异类样本就失去标签，导致成为"粘合剂"
        MIN_CLUSTER_SIZE_FOR_LABEL = 5
        if len(cluster_indices) < MIN_CLUSTER_SIZE_FOR_LABEL:
            return None  # 簇太小，暂时无标签

        # 统计簇中的已知标签样本
        labeled_in_cluster = [idx for idx in cluster_indices if labeled_mask[idx]]

        if len(labeled_in_cluster) == 0:
            return None  # 没有已知标签样本

        # 检查已知标签样本占比
        labeled_ratio = len(labeled_in_cluster) / len(cluster_indices)
        if labeled_ratio <= 0.25:
            return None  # 已知标签样本占比 ≤ 25%

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
            return dominant_label  # 纯度 ≥ 80%，簇有类别标签
        else:
            return None  # 纯度 < 80%，簇无类别标签

    def log_clustering_action(point_idx, density, co_value, action, cluster_id_info, current_cluster_pi=None):
        """
        记录聚类动作到日志（如果logger存在）

        Args:
            point_idx: 点的索引
            density: 密度值
            co_value: co值
            action: 动作类型 ('create', 'expand', 'merge', 'move', 'reject', 'skip')
            cluster_id_info: 簇ID信息（可能是单个值或列表）
            current_cluster_pi: 当前正在构建的簇（仅用于create动作）
        """
        if logger is None:
            return

        # 准备邻居信息
        neighbors_list = high_density_neighbors_map.get(point_idx, [])
        neighbor_clusters = [cluster_labels[nb] for nb in neighbors_list]
        neighbor_densities = [densities[nb] for nb in neighbors_list]
        neighbor_true_labels = [int(targets[nb]) for nb in neighbors_list]

        # 计算邻居距离：需要找到当前点在高密度索引中的位置
        neighbor_distances = []
        if point_idx in high_density_idx_to_local:
            local_idx = high_density_idx_to_local[point_idx]
            for nb in neighbors_list:
                if nb in high_density_idx_to_local:
                    nb_local_idx = high_density_idx_to_local[nb]
                    # 从knn_distances_hd中查找距离
                    # neighbors_high_density[local_idx]包含邻居的局部索引
                    nb_pos = np.where(neighbors_high_density[local_idx] == nb_local_idx)[0]
                    if len(nb_pos) > 0:
                        dist = knn_distances_hd[local_idx, nb_pos[0]]
                        neighbor_distances.append(float(dist))
                    else:
                        # 如果不在knn列表中，计算欧氏距离
                        dist = np.linalg.norm(X[point_idx] - X[nb])
                        neighbor_distances.append(float(dist))
                else:
                    neighbor_distances.append(0.0)
        else:
            neighbor_distances = [0.0] * len(neighbors_list)

        neighbors_info = {
            'n_neighbors': len(neighbors_list),
            'neighbor_indices': neighbors_list.tolist() if hasattr(neighbors_list, 'tolist') else list(neighbors_list),
            'neighbor_clusters': neighbor_clusters,
            'neighbor_densities': neighbor_densities,
            'neighbor_true_labels': neighbor_true_labels,
            'neighbor_distances': neighbor_distances
        }

        # 确定是否来自训练集
        is_from_train = (point_idx < train_size) if train_size is not None else True

        # 收集当前簇状态信息
        cluster_status = {}
        for cid, cluster in enumerate(clusters):
            if len(cluster) > 0:  # 只记录非空簇
                cluster_label = cluster_category_labels.get(cid, None)
                cluster_status[cid] = {
                    'size': len(cluster),
                    'label': cluster_label
                }

        # 特殊处理：如果是create动作，需要手动添加正在创建的簇（因为还没append到clusters列表）
        if action == 'create' and current_cluster_pi is not None:
            cluster_label = cluster_category_labels.get(cluster_id_info, None)
            cluster_status[cluster_id_info] = {
                'size': len(current_cluster_pi),
                'label': cluster_label
            }

        # 记录日志
        logger.log_point(
            point_idx=point_idx,
            density=density,
            co_value=co_value,
            is_known=bool(known_mask[point_idx]),
            has_label=bool(labeled_mask[point_idx]),
            is_from_train=is_from_train,
            true_label=int(targets[point_idx]),
            action=action,
            cluster_id=cluster_id_info,
            neighbors_info=neighbors_info,
            cluster_status=cluster_status
        )

    # 关键修改：在高密度点子空间中重新计算k近邻
    # 这样可以确保每个高密度点的k个邻居都是高密度点
    high_density_indices_all = np.where(high_density_mask)[0]
    X_high_density = X[high_density_indices_all]

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(high_density_indices_all)), metric='euclidean').fit(X_high_density)
    knn_distances_hd, neighbors_high_density = nbrs.kneighbors(X_high_density)

    # 去除自己（第一个邻居是自己）
    neighbors_high_density = neighbors_high_density[:, 1:]
    knn_distances_hd = knn_distances_hd[:, 1:]

    # 将局部索引映射回全局索引
    neighbors_high_density_global = high_density_indices_all[neighbors_high_density]

    # 创建全局索引到局部索引的映射（用于日志记录时查找距离）
    high_density_idx_to_local = {global_idx: i for i, global_idx in enumerate(high_density_indices_all)}

    # 创建一个字典：高密度点索引 -> 其在高密度子空间中的k近邻
    # 应用co截止距离过滤（KNN_co）
    high_density_neighbors_map = {}

    # 判断co是标量还是数组
    is_scalar_co = isinstance(co, (int, float, np.number))

    for i, global_idx in enumerate(high_density_indices_all):
        # 应用截止距离过滤（KNN_co）
        distances = knn_distances_hd[i]

        # 获取该点的co值
        co_value = co if is_scalar_co else co[global_idx]

        valid_mask = distances <= co_value
        filtered_neighbors = neighbors_high_density_global[i][valid_mask]
        high_density_neighbors_map[global_idx] = filtered_neighbors

    # 从每个高密度点开始构建聚类
    high_density_indices = np.where(high_density_mask)[0]

    # 为确保确定性，按密度从高到低排序高密度点
    # 密度相同时按索引排序
    high_density_points_with_density = [(idx, densities[idx]) for idx in high_density_indices]
    high_density_points_with_density.sort(key=lambda x: (-x[1], x[0]))  # 密度降序，索引升序
    high_density_indices = [idx for idx, _ in high_density_points_with_density]

    for xi_idx in high_density_indices:
        if cluster_labels[xi_idx] != -1:
            continue  # 已经被分配到其他聚类

        # If xi is not assigned to any cluster, create a new cluster pi
        cluster_pi = set([xi_idx])
        cluster_labels[xi_idx] = current_cluster_id
        queue = [xi_idx]

        # 初始化当前簇的类别标签
        cluster_category_labels[current_cluster_id] = update_cluster_label(current_cluster_id, cluster_pi)

        # 记录创建簇的动作（传递cluster_pi以便记录当前簇状态）
        xi_co_value = co if isinstance(co, (int, float, np.number)) else co[xi_idx]
        log_clustering_action(xi_idx, densities[xi_idx], xi_co_value, 'create', current_cluster_id, current_cluster_pi=cluster_pi)

        # BFS扩展聚类
        while queue:
            current_idx = queue.pop(0)

            # For each newly added dense point, check its KNN_co neighbors
            # 使用高密度子空间中的k近邻（确保所有邻居都是高密度点）
            if current_idx not in high_density_neighbors_map:
                continue  # 当前点不是高密度点，跳过

            high_density_neighbors = high_density_neighbors_map[current_idx]

            for xj_idx in high_density_neighbors:
                xj_cluster_id = cluster_labels[xj_idx]

                # 第一层检查：样本级别的标签冲突（硬约束）
                xi_has_label = labeled_mask[current_idx]
                xj_has_label = labeled_mask[xj_idx]

                sample_level_conflict = False
                if xi_has_label and xj_has_label:
                    xi_sample_label = targets[current_idx]
                    xj_sample_label = targets[xj_idx]
                    if xi_sample_label != xj_sample_label:
                        sample_level_conflict = True

                if sample_level_conflict:
                    # 样本级别标签冲突：直接跳过
                    continue

                # 无样本级别冲突，继续处理
                if xj_cluster_id == -1:
                    # xj未分配，直接加入当前簇
                    cluster_pi.add(xj_idx)
                    cluster_labels[xj_idx] = current_cluster_id
                    queue.append(xj_idx)

                    # 更新当前簇的类别标签
                    cluster_category_labels[current_cluster_id] = update_cluster_label(current_cluster_id, cluster_pi)

                    # 记录扩展动作
                    xj_co_value = co if isinstance(co, (int, float, np.number)) else co[xj_idx]
                    log_clustering_action(xj_idx, densities[xj_idx], xj_co_value, 'expand', current_cluster_id)

                elif xj_cluster_id != current_cluster_id:
                    # xj已在簇pj中
                    # 第二层检查：簇级别的类别标签
                    si = cluster_category_labels.get(current_cluster_id, None)  # 当前簇pi的类别标签
                    sj = cluster_category_labels.get(xj_cluster_id, None)       # xj所在簇pj的类别标签

                    # 判断是否可以直接合并的条件：
                    # 1. 两个簇标签相同 (si == sj 且都不为None)
                    # 2. 至少一方无标签 (si is None 或 sj is None 或都是None)
                    # 只有两个簇都有标签且标签不同时才需要密度判断
                    can_merge_directly = False
                    has_label_conflict = (si is not None and sj is not None and si != sj)

                    if not has_label_conflict:
                        # 无标签冲突：两个簇可以直接合并
                        # 情况1: si == sj (标签相同)
                        # 情况2: si is None (当前簇无标签)
                        # 情况3: sj is None (邻居簇无标签)
                        # 情况4: 都是None (都无标签)
                        can_merge_directly = True

                    if can_merge_directly:
                        # 无标签冲突：直接合并簇
                        old_cluster_id = xj_cluster_id
                        cluster_pj = clusters[old_cluster_id]

                        # 将pj的所有点合并到pi
                        for idx in cluster_pj:
                            cluster_pi.add(idx)
                            cluster_labels[idx] = current_cluster_id
                            if high_density_mask[idx]:
                                queue.append(idx)

                        # 清空pj
                        clusters[old_cluster_id] = set()

                        # 更新当前簇的类别标签
                        cluster_category_labels[current_cluster_id] = update_cluster_label(current_cluster_id, cluster_pi)
                        cluster_category_labels[old_cluster_id] = None

                        # 生成详细的合并原因
                        si_str = f"类别{si}" if si is not None else "无标签"
                        sj_str = f"类别{sj}" if sj is not None else "无标签"
                        if si is not None and sj is not None and si == sj:
                            merge_reason = f"标签相同({si_str})"
                        elif si is None and sj is None:
                            merge_reason = "双方都无标签"
                        elif si is None:
                            merge_reason = f"当前簇无标签，邻居簇{sj_str}"
                        else:  # sj is None
                            merge_reason = f"当前簇{si_str}，邻居簇无标签"

                        # 记录合并动作（记录xj，因为它是触发合并的邻居点）
                        xj_co_value = co if isinstance(co, (int, float, np.number)) else co[xj_idx]
                        log_clustering_action(xj_idx, densities[xj_idx], xj_co_value, 'merge', [current_cluster_id, old_cluster_id])

                    else:
                        # 有标签冲突（si ≠ sj 且都不为None）：使用密度判断冲突解决
                        # 条件：当前点密度 > 邻居密度时才移动
                        xi_density = densities[current_idx]
                        xj_density = densities[xj_idx]

                        if xi_density > xj_density:
                            # 当前点密度更高，将xj从原簇移到当前簇
                            old_cluster_id = xj_cluster_id
                            clusters[old_cluster_id].remove(xj_idx)
                            cluster_pi.add(xj_idx)
                            cluster_labels[xj_idx] = current_cluster_id
                            queue.append(xj_idx)

                            # 更新两个簇的类别标签
                            cluster_category_labels[current_cluster_id] = update_cluster_label(current_cluster_id, cluster_pi)
                            cluster_category_labels[old_cluster_id] = update_cluster_label(old_cluster_id, clusters[old_cluster_id])

                            # 记录移动动作
                            xj_co_value = co if isinstance(co, (int, float, np.number)) else co[xj_idx]
                            log_clustering_action(xj_idx, densities[xj_idx], xj_co_value, 'move', [current_cluster_id, old_cluster_id])

                        else:
                            # 当前点密度更低，不移动xj
                            # 记录拒绝动作
                            xj_co_value = co if isinstance(co, (int, float, np.number)) else co[xj_idx]
                            log_clustering_action(xj_idx, densities[xj_idx], xj_co_value, 'reject', None)

        clusters.append(cluster_pi)
        current_cluster_id += 1

    # 移除空聚类并重新编号
    # 同时更新簇类别标签映射（旧ID -> 新ID）
    new_cluster_category_labels = {}
    new_id = 0
    for old_id, cluster in enumerate(clusters):
        if len(cluster) > 0:
            # 如果旧簇有类别标签，则复制到新字典
            if old_id in cluster_category_labels:
                new_cluster_category_labels[new_id] = cluster_category_labels[old_id]
            new_id += 1

    non_empty_clusters = [c for c in clusters if len(c) > 0]
    cluster_labels_new = np.full(n_samples, -1, dtype=int)

    for new_id, cluster in enumerate(non_empty_clusters):
        for idx in cluster:
            cluster_labels_new[idx] = new_id

    return non_empty_clusters, cluster_labels_new, high_density_neighbors_map, new_cluster_category_labels
