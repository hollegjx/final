#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
聚类损失函数计算模块
用于计算监督损失L1和无监督损失L2，以及综合损失L
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score

from .l2_strategies import L2_REGISTRY, available_l2_components


def compute_supervised_loss_l1(predictions, targets, labeled_mask, cluster_category_labels=None, loss_type='accuracy'):
    """
    计算监督损失L1

    Args:
        predictions: 聚类预测标签 (n_samples,)
        targets: 真实标签 (n_samples,)
        labeled_mask: 有标签样本的掩码 (n_samples,)
        cluster_category_labels: 簇ID到类别标签的映射字典 {cluster_id: category_label}（可选）
                                如果提供，则将簇ID转换为簇类别标签后再计算ACC
        loss_type: 损失类型，可选'accuracy'(准确率)或'cross_entropy'(交叉熵)

    Returns:
        l1: 监督损失，范围[0, 1]
        metrics: 详细指标字典
    """
    # 直接使用簇ID进行计算（不转换为类别标签）
    predictions_for_acc = predictions

    # 只考虑有标签的样本
    labeled_predictions = predictions_for_acc[labeled_mask]
    labeled_targets = targets[labeled_mask]

    n_labeled = len(labeled_predictions)

    if n_labeled == 0:
        print("⚠️  警告: 没有有标签样本，无法计算监督损失")
        return 0.0, {'n_labeled': 0, 'accuracy': 0.0}

    if loss_type == 'accuracy':
        # 使用准确率作为监督损失
        # 需要将聚类标签映射到真实标签
        from scipy.optimize import linear_sum_assignment

        # 获取聚类标签和真实标签的unique值
        unique_clusters = np.unique(labeled_predictions)
        unique_targets = np.unique(labeled_targets)

        # 构建混淆矩阵
        n_clusters = len(unique_clusters)
        n_classes = len(unique_targets)
        confusion_matrix = np.zeros((n_clusters, n_classes))

        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = (labeled_predictions == cluster_id)
            for j, true_label in enumerate(unique_targets):
                confusion_matrix[i, j] = np.sum(labeled_targets[cluster_mask] == true_label)

        # 使用匈牙利算法找到最优匹配
        row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)

        # 计算最优匹配下的准确样本数
        correct_samples = confusion_matrix[row_ind, col_ind].sum()
        accuracy = correct_samples / n_labeled

        # 损失定义为 1 - accuracy（损失越小越好）
        l1 = 1.0 - accuracy

        metrics = {
            'n_labeled': n_labeled,
            'accuracy': accuracy,
            'correct_samples': int(correct_samples)
        }

    elif loss_type == 'cross_entropy':
        # 方案2: 基于簇内类别分布的交叉熵损失
        # 只使用有标签样本，遵照accuracy方法的簇ID处理方式

        # 获取所有簇ID和类别
        unique_clusters = np.unique(labeled_predictions)
        unique_targets = np.unique(labeled_targets)
        n_classes = len(unique_targets)

        # 为了处理类别标签不连续的情况（如0,2,5），创建映射
        target_to_idx = {label: idx for idx, label in enumerate(unique_targets)}

        # 步骤1: 计算每个簇的类别分布概率
        cluster_class_probs = {}

        for cluster_id in unique_clusters:
            # 找到当前簇的所有有标签样本
            cluster_mask = (labeled_predictions == cluster_id)
            cluster_targets = labeled_targets[cluster_mask]

            if len(cluster_targets) > 0:
                # 计算类别计数
                class_counts = np.zeros(n_classes)
                for target in cluster_targets:
                    class_counts[target_to_idx[target]] += 1

                # 归一化为概率分布
                cluster_class_probs[cluster_id] = class_counts / len(cluster_targets)
            else:
                # 如果簇内没有有标签样本，使用均匀分布
                cluster_class_probs[cluster_id] = np.ones(n_classes) / n_classes

        # 步骤2: 计算交叉熵损失
        epsilon = 1e-10  # 避免log(0)
        cross_entropy_sum = 0.0

        for pred_cluster, true_label in zip(labeled_predictions, labeled_targets):
            # 获取该样本所在簇的类别概率分布
            prob_dist = cluster_class_probs[pred_cluster]

            # 真实标签对应的概率
            true_label_idx = target_to_idx[true_label]
            true_label_prob = prob_dist[true_label_idx]

            # 累加交叉熵：-log(p(true_label))
            cross_entropy_sum += -np.log(true_label_prob + epsilon)

        # 平均交叉熵作为损失
        l1 = cross_entropy_sum / n_labeled

        # 为了便于比较，额外计算一个等效的准确率（非必需，仅供参考）
        # 基于簇类别分布的预测：选择概率最大的类别
        correct_samples = 0
        for pred_cluster, true_label in zip(labeled_predictions, labeled_targets):
            prob_dist = cluster_class_probs[pred_cluster]
            predicted_label_idx = np.argmax(prob_dist)
            predicted_label = unique_targets[predicted_label_idx]
            if predicted_label == true_label:
                correct_samples += 1

        equiv_accuracy = correct_samples / n_labeled

        metrics = {
            'n_labeled': n_labeled,
            'cross_entropy': l1,
            'equiv_accuracy': equiv_accuracy,  # 基于簇分布预测的等效准确率
            'correct_samples': int(correct_samples)
        }

    else:
        raise ValueError(f"未知的loss_type: {loss_type}，支持'accuracy'或'cross_entropy'")

    return l1, metrics


def compute_l2_loss(*,
                    X,
                    predictions,
                    targets,
                    labeled_mask,
                    clusters,
                    k: int,
                    cluster_distance_method: str = 'prototype',
                    neighbors=None,
                    l2_components: Optional[List[str]] = None,
                    l2_component_weights: Optional[Dict[str, float]] = None,
                    l2_component_params: Optional[Dict[str, dict]] = None):
    """
    按照组件配置计算 L2（无监督损失）
    """
    if not l2_components:
        return 0.0, {
            'total_l2': 0.0,
            'components': {},
            'component_order': [],
            'component_weights': {}
        }

    component_weights = dict(l2_component_weights or {})
    component_params = dict(l2_component_params or {})

    components_summary: Dict[str, Dict[str, object]] = {}
    total_l2 = 0.0

    for name in l2_components:
        entry = L2_REGISTRY.get(name)
        if entry is None:
            available = ", ".join(sorted(available_l2_components().keys()))
            raise ValueError(f"未知的 L2 组件 '{name}'，可选组件: {available}")

        fn = entry['fn']
        orientation = entry.get('orientation', 'minimize')
        params = component_params.get(name, {})

        value, metrics = fn(
            clusters=clusters,
            X=X,
            predictions=predictions,
            targets=targets,
            labeled_mask=labeled_mask,
            k=k,
            cluster_distance_method=cluster_distance_method,
            neighbors=neighbors,
            **params
        )

        weight = float(component_weights.get(name, 1.0))

        if orientation == 'maximize':
            contribution = weight * value
        elif orientation == 'minimize':
            contribution = -weight * value
        else:
            contribution = weight * value

        components_summary[name] = {
            'value': value,
            'weight': weight,
            'orientation': orientation,
            'contribution': contribution,
            'metrics': metrics
        }
        total_l2 += contribution

    summary = {
        'total_l2': total_l2,
        'components': components_summary,
        'component_order': list(l2_components),
        'component_weights': {name: components_summary[name]['weight'] for name in l2_components}
    }

    unused_weights = {k: v for k, v in component_weights.items() if k not in components_summary}
    if unused_weights:
        summary['unused_weights'] = unused_weights

    return total_l2, summary


def compute_total_loss(X, predictions, targets, labeled_mask,
                       cluster_category_labels=None,
                       l1_weight=1.0, l2_weight=1.0,
                       l1_type='accuracy', l2_type=None,
                       use_cluster_quality=False,
                       clusters=None,
                       k=10,
                       cluster_distance_method='prototype',
                       neighbors=None,
                       separation_weight=1.0,
                       penalty_weight=1.0,
                       silent=False,
                       l2_components: Optional[List[str]] = None,
                       l2_component_weights: Optional[Dict[str, float]] = None,
                       l2_component_params: Optional[Dict[str, dict]] = None):
    """
    计算综合损失L = l1_weight * L1 + l2_weight * L2

    Args:
        X: 特征矩阵 (n_samples, n_features)
        predictions: 聚类预测标签 (n_samples,)
        targets: 真实标签 (n_samples,)
        labeled_mask: 有标签样本的掩码 (n_samples,)
        cluster_category_labels: 簇ID到类别标签的映射字典（可选）
        l1_weight: L1权重，默认1.0
        l2_weight: L2权重，默认1.0
        l1_type: L1损失类型，默认'accuracy'
        l2_type: L2损失类型 [已废弃]
        use_cluster_quality: 是否启用旧版聚类质量评估（已废弃）
        clusters: 核心簇列表
        k: 近邻数量
        cluster_distance_method: 簇距离计算方法
        neighbors: 预计算的近邻索引
        separation_weight: 默认分离度权重
        penalty_weight: 默认密度惩罚权重
        silent: 是否静默模式
        l2_components: L2 组件列表
        l2_component_weights: 组件权重映射
        l2_component_params: 组件附加参数映射

    Returns:
        loss_dict: 包含所有损失信息的字典
    """
    if l2_type:
        warnings.warn("参数 l2_type 已废弃，请改用 l2_components", DeprecationWarning)

    l1, l1_metrics = compute_supervised_loss_l1(
        predictions, targets, labeled_mask,
        cluster_category_labels=cluster_category_labels,
        loss_type=l1_type
    )

    explicit_components = l2_components is not None
    resolved_components = l2_components

    if resolved_components is None:
        if use_cluster_quality:
            warnings.warn("use_cluster_quality 将在未来移除，请显式设置 l2_components=['separation', 'penalty']",
                          DeprecationWarning)
            resolved_components = ['separation', 'penalty']
        else:
            resolved_components = []
    else:
        if isinstance(resolved_components, str):
            resolved_components = [comp.strip() for comp in resolved_components.split(',') if comp.strip()]
        else:
            resolved_components = list(resolved_components)
        if use_cluster_quality:
            warnings.warn("检测到 use_cluster_quality 与 l2_components 同时设置，将优先使用 l2_components",
                          RuntimeWarning)

    component_weights = {}
    if l2_component_weights:
        component_weights = {str(k): float(v) for k, v in l2_component_weights.items()}

    if 'separation' in resolved_components and 'separation' not in component_weights:
        component_weights['separation'] = separation_weight
    if 'penalty' in resolved_components and 'penalty' not in component_weights:
        component_weights['penalty'] = penalty_weight
    for name in resolved_components:
        component_weights.setdefault(name, 1.0)

    component_params = {str(k): dict(v) for k, v in (l2_component_params or {}).items()}

    if 'separation' in resolved_components and clusters is None:
        raise ValueError("启用 separation 组件时必须提供 clusters 参数")

    l2 = None
    l2_metrics: Dict[str, object] = {}

    if resolved_components:
        l2_value, l2_summary = compute_l2_loss(
            X=X,
            predictions=predictions,
            targets=targets,
            labeled_mask=labeled_mask,
            clusters=clusters,
            k=k,
            cluster_distance_method=cluster_distance_method,
            neighbors=neighbors,
            l2_components=resolved_components,
            l2_component_weights=component_weights,
            l2_component_params=component_params
        )

        l2 = l2_value
        l2_metrics = {
            'method': '+'.join(resolved_components),
            'components': l2_summary.get('components', {}),
            'component_order': l2_summary.get('component_order', []),
            'component_weights': l2_summary.get('component_weights', {}),
            'quality_score': l2_value,
            'total_l2': l2_value
        }
        if 'unused_weights' in l2_summary:
            l2_metrics['unused_weights'] = l2_summary['unused_weights']

        if {'separation', 'penalty'}.issubset(set(resolved_components)):
            sep_entry = l2_summary['components'].get('separation', {})
            pen_entry = l2_summary['components'].get('penalty', {})
            cluster_quality_metrics = {
                'quality_score': l2_value,
                'separation_score': sep_entry.get('value'),
                'penalty_score': pen_entry.get('value'),
                'separation_weight': sep_entry.get('weight'),
                'penalty_weight': pen_entry.get('weight'),
                'weighted_separation': (
                    sep_entry.get('weight', 0.0) * sep_entry.get('value', 0.0) if sep_entry else None
                ),
                'weighted_penalty': (
                    pen_entry.get('weight', 0.0) * pen_entry.get('value', 0.0) if pen_entry else None
                ),
                'separation_metrics': sep_entry.get('metrics'),
                'penalty_metrics': pen_entry.get('metrics')
            }
            l2_metrics['cluster_quality'] = cluster_quality_metrics
    elif explicit_components:
        l2 = 0.0
        l2_metrics = {
            'method': 'none',
            'components': {},
            'component_order': [],
            'component_weights': {},
            'quality_score': 0.0,
            'total_l2': 0.0
        }

    if l2 is not None:
        total_loss = l1_weight * l1 + l2_weight * l2
    else:
        total_loss = l1

    loss_dict = {
        'total_loss': total_loss,
        'l1': l1,
        'l2': l2,
        'l1_weight': l1_weight,
        'l2_weight': l2_weight,
        'l1_metrics': l1_metrics,
        'l2_metrics': l2_metrics,
        'l2_components': resolved_components,
        'l2_component_weights': component_weights,
        'l2_component_params': component_params
    }

    if not silent:
        print(f"\n{'=' * 80}")
        print("📉 损失函数计算")
        print(f"{'=' * 80}")
        print("L1 (监督损失):")
        print(f"   类型: {l1_type}")
        print(f"   有标签样本数: {l1_metrics.get('n_labeled', 0)}")
        if 'accuracy' in l1_metrics:
            print(f"   标签准确率: {l1_metrics['accuracy']:.4f}")
        elif 'cross_entropy' in l1_metrics:
            print(f"   交叉熵: {l1_metrics['cross_entropy']:.4f}")
            print(f"   等效准确率: {l1_metrics['equiv_accuracy']:.4f}")
        print(f"   L1损失值: {l1:.4f}")

        if l2 is not None:
            print("\nL2 (无监督损失):")
            for comp in l2_metrics.get('component_order', []):
                entry = l2_metrics['components'].get(comp, {})
                orientation = entry.get('orientation')
                value = entry.get('value')
                weight = entry.get('weight')
                contribution = entry.get('contribution')
                if value is not None:
                    print(f"   组件[{comp}] (方向: {orientation}) -> 值={value:.4f}, 权重={weight:.3f}, 贡献={contribution:.4f}")
            if 'cluster_quality' in l2_metrics:
                cq = l2_metrics['cluster_quality']
                print(f"   簇间分离度: {cq.get('separation_score', 0.0):.4f}")
                print(f"   密度惩罚: {cq.get('penalty_score', 0.0):.4f}")
            print(f"   L2损失值: {l2:.4f}")

            print("\n综合损失:")
            print(f"   L = {l1_weight:.2f} × L1 + {l2_weight:.2f} × L2")
            print(f"   L = {l1_weight:.2f} × {l1:.4f} + {l2_weight:.2f} × {l2:.4f}")
            print(f"   L = {total_loss:.4f}")
        else:
            print(f"\n总损失: {total_loss:.4f}")

        print(f"{'=' * 80}")

    return loss_dict
