#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
骨干网络聚类详细日志记录器
用于记录高密度点聚类过程的详细信息
"""

import os
from datetime import datetime
import numpy as np

from config import clustering_log_dir


class DenseNetworkLogger:
    """
    骨干网络聚类详细日志记录器

    记录内容包括：
    - 点序号、密度值、相对co值
    - 已知/未知状态、是否有标签
    - 来自训练集/测试集
    - 被分配到哪个簇/创建了哪个簇
    - 邻居信息
    - 聚类动作（扩展/合并/移动/拒绝/跳过）
    """

    def __init__(self, log_dir=None, enabled=True):
        """
        初始化日志记录器

        Args:
            log_dir: 日志文件保存目录
            enabled: 是否启用日志记录（detail_dense参数）
        """
        self.enabled = enabled
        self.log_dir = log_dir or clustering_log_dir
        self.records = []

        if self.enabled:
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)

    def set_metadata(self, dataset_name, total_points, n_high_density,
                     train_size, test_size, co_mode, co_value):
        """
        设置聚类元数据

        Args:
            dataset_name: 数据集名称
            total_points: 总样本数
            n_high_density: 高密度点数量
            train_size: 训练集样本数（原始划分）
            test_size: 测试集样本数（原始划分）
            co_mode: co计算模式
            co_value: co值（可能是标量或数组）
        """
        if not self.enabled:
            return

        self.metadata = {
            'dataset_name': dataset_name,
            'total_points': total_points,
            'n_high_density': n_high_density,
            'train_size': train_size,
            'test_size': test_size,
            'co_mode': co_mode,
            'co_value': co_value,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def update_dataset_name(self, dataset_name):
        """
        更新数据集名称（在已设置元数据后）

        Args:
            dataset_name: 新的数据集名称
        """
        if not self.enabled:
            return

        if hasattr(self, 'metadata') and self.metadata is not None:
            self.metadata['dataset_name'] = dataset_name

    def log_point(self, point_idx, density, co_value,
                  is_known, has_label, is_from_train, true_label,
                  action, cluster_id, neighbors_info, cluster_status=None):
        """
        记录单个高密度点的聚类信息

        Args:
            point_idx: 点在全局数据集中的索引
            density: 密度值
            co_value: 该点的相对co值（如果是数组co，取该点的值）
            is_known: 是否为已知类
            has_label: 是否有标签
            is_from_train: 是否来自原始训练集
            true_label: 当前点的真实标签（用于上帝视角分析）
            action: 聚类动作（'create', 'expand', 'merge', 'move', 'reject', 'skip'）
            cluster_id: 所属/创建的簇ID（可能是单个ID或列表）
            neighbors_info: 邻居信息字典
                {
                    'n_neighbors': int,
                    'neighbor_indices': list,
                    'neighbor_clusters': list,  # 邻居所属的簇ID
                    'neighbor_densities': list,
                    'neighbor_true_labels': list,  # 邻居的真实标签
                    'neighbor_distances': list  # 邻居与当前点的距离
                }
            cluster_status: 当前簇状态信息（可选）
                {
                    cluster_id: {'size': int, 'label': int or None}
                }
        """
        if not self.enabled:
            return

        record = {
            'point_idx': point_idx,
            'density': density,
            'co_value': co_value,
            'is_known': is_known,
            'has_label': has_label,
            'is_from_train': is_from_train,
            'true_label': true_label,
            'action': action,
            'cluster_id': cluster_id,
            'neighbors_info': neighbors_info,
            'cluster_status': cluster_status
        }

        self.records.append(record)

    def _format_cluster_id(self, cluster_id):
        """格式化簇ID（可能是单个值或列表）"""
        if isinstance(cluster_id, (list, tuple)):
            return f"[{', '.join(map(str, cluster_id))}]"
        elif cluster_id is None:
            return "None"
        else:
            return str(cluster_id)

    def _format_action_description(self, action, cluster_id):
        """格式化动作描述"""
        action_map = {
            'create': f"创建新簇 {self._format_cluster_id(cluster_id)}",
            'expand': f"加入簇 {self._format_cluster_id(cluster_id)}（扩展）",
            'merge': f"合并簇 {self._format_cluster_id(cluster_id)}",
            'move': f"移动到簇 {self._format_cluster_id(cluster_id)}",
            'reject': "被拒绝（冲突未解决）",
            'skip': "跳过（已分配）"
        }
        return action_map.get(action, f"未知动作: {action}")

    def write_log(self, filename=None):
        """
        将日志写入文件

        Args:
            filename: 日志文件名（不含路径），如果为None则自动生成
        """
        if not self.enabled:
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset = self.metadata.get('dataset_name', 'unknown')
            filename = f"dense_network_{dataset}_{timestamp}.txt"

        log_path = os.path.join(self.log_dir, filename)

        with open(log_path, 'w', encoding='utf-8') as f:
            # 写入头部信息
            f.write("=" * 100 + "\n")
            f.write("骨干网络聚类详细日志\n")
            f.write("=" * 100 + "\n\n")

            # 写入元数据
            f.write("【聚类元数据】\n")
            f.write(f"数据集名称: {self.metadata.get('dataset_name', 'N/A')}\n")
            f.write(f"总样本数: {self.metadata.get('total_points', 'N/A')}\n")
            f.write(f"高密度点数量: {self.metadata.get('n_high_density', 'N/A')}\n")
            f.write(f"原始训练集样本数: {self.metadata.get('train_size', 'N/A')}\n")
            f.write(f"原始测试集样本数: {self.metadata.get('test_size', 'N/A')}\n")
            f.write(f"Co模式: {self.metadata.get('co_mode', 'N/A')}\n")

            co_val = self.metadata.get('co_value', 'N/A')
            if isinstance(co_val, np.ndarray):
                f.write(f"Co值: 数组（每个点不同）\n")
            else:
                f.write(f"Co值: {co_val}\n")

            f.write(f"记录时间: {self.metadata.get('timestamp', 'N/A')}\n")
            f.write("\n" + "=" * 100 + "\n\n")

            # 写入每个点的记录
            f.write("【高密度点聚类过程】\n\n")

            for i, record in enumerate(self.records):
                f.write(f"{'─' * 100}\n")
                f.write(f"处理顺序: #{i+1}\n")
                f.write(f"点序号: {record['point_idx']}\n")
                f.write(f"密度值: {record['density']:.6f}\n")
                f.write(f"相对Co值: {record['co_value']:.6f}\n")

                # 样本状态
                f.write(f"已知/未知: {'已知类' if record['is_known'] else '未知类'}\n")
                f.write(f"是否有标签: {'是' if record['has_label'] else '否'}\n")
                f.write(f"来源: {'原始训练集' if record['is_from_train'] else '原始测试集'}\n")
                f.write(f"真实标签: {record['true_label']}\n")

                # 聚类动作
                f.write(f"聚类动作: {self._format_action_description(record['action'], record['cluster_id'])}\n")

                # 当前簇状态
                cluster_status = record.get('cluster_status')
                if cluster_status is not None and len(cluster_status) > 0:
                    f.write(f"\n当前簇状态:\n")
                    # 按簇ID排序
                    sorted_clusters = sorted(cluster_status.items(), key=lambda x: x[0])
                    status_parts = []
                    for cid, info in sorted_clusters:
                        size = info.get('size', 0)
                        label = info.get('label')
                        if label is not None:
                            status_parts.append(f"簇{cid}:{size}(标签:{label})")
                        else:
                            status_parts.append(f"簇{cid}:{size}(无标签)")
                    f.write(f"  {' '.join(status_parts)}\n")

                # 邻居信息
                neighbors = record['neighbors_info']
                f.write(f"\n邻居信息:\n")
                f.write(f"  邻居数量: {neighbors.get('n_neighbors', 0)}\n")

                if neighbors.get('n_neighbors', 0) > 0:
                    neighbor_indices = neighbors.get('neighbor_indices', [])
                    neighbor_clusters = neighbors.get('neighbor_clusters', [])
                    neighbor_densities = neighbors.get('neighbor_densities', [])
                    neighbor_true_labels = neighbors.get('neighbor_true_labels', [])
                    neighbor_distances = neighbors.get('neighbor_distances', [])

                    f.write(f"  邻居详情:\n")
                    for j, (nb_idx, nb_cluster, nb_density, nb_label, nb_dist) in enumerate(
                        zip(neighbor_indices, neighbor_clusters, neighbor_densities,
                            neighbor_true_labels, neighbor_distances), 1):
                        cluster_str = self._format_cluster_id(nb_cluster)
                        f.write(f"    #{j}: 点{nb_idx}, 簇{cluster_str}, 密度{nb_density:.6f}, 真实标签{nb_label}, 距离{nb_dist:.6f}\n")

                f.write("\n")

            # 写入尾部
            f.write("=" * 100 + "\n")
            f.write(f"日志记录完成，共记录 {len(self.records)} 个高密度点的聚类过程\n")
            f.write("=" * 100 + "\n")

        print(f"\n📝 骨干网络聚类日志已保存至: {log_path}")
        return log_path

    def clear(self):
        """清空记录"""
        self.records = []


# 全局日志实例（用于简化调用）
_global_logger = None


def get_logger():
    """获取全局日志实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = DenseNetworkLogger(enabled=False)
    return _global_logger


def init_logger(log_dir=None, enabled=True):
    """
    初始化全局日志实例

    Args:
        log_dir: 日志保存目录
        enabled: 是否启用（由detail_dense参数控制）

    Returns:
        logger: DenseNetworkLogger实例
    """
    global _global_logger
    _global_logger = DenseNetworkLogger(log_dir=log_dir, enabled=enabled)
    return _global_logger


def reset_logger():
    """重置全局日志实例"""
    global _global_logger
    _global_logger = None
