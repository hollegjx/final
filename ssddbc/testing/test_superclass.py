#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
超类测试模块
在CIFAR-100超类数据集上测试SS-DDBC聚类算法
使用ssddbc/data增强型数据提供器，高效读取数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np

# 使用增强型数据提供器（高效、无重复逻辑）
from ssddbc.data import (
    EnhancedDataProvider,
    get_superclass_info
)

# 聚类相关模块
from ..ssddbc.adaptive_clustering import adaptive_density_clustering
from ..ssddbc.analysis import analyze_cluster_composition
from ..baseline.kmeans import test_kmeans_baseline
from ..evaluation.loss_function import compute_total_loss


def test_adaptive_clustering_on_superclass(superclass_name, model_path,
                                         use_train_and_test=True, k=10,
                                         density_percentile=75, random_state=0,
                                         eval_version='v1',
                                         run_kmeans_baseline=False,
                                         use_l2=True,
                                         feature_cache_dir=None,
                                         eval_dense=False, fast_mode=False,
                                         dense_method=0, assign_model=2, voting_k=5,
                                         co_mode=2, co_manual=None, detail_dense=False,
                                         label_guide=False,
                                         use_cluster_quality=False,
                                         cluster_distance_method=1,
                                         l1_type='cross_entropy',
                                         separation_weight=1.0,
                                         penalty_weight=1.0,
                                         l2_components=None,
                                         l2_component_weights=None,
                                         l2_component_params=None):
    """
    在指定超类上测试自适应聚类算法

    Args:
        superclass_name: 超类名称
        model_path: 模型路径
        use_train_and_test: 是否使用训练+测试集
        k: K近邻数量
        density_percentile: 高密度点百分位阈值
        random_state: 随机种子
        eval_version: 评估版本 ('v1' or 'v2')
        run_kmeans_baseline: 是否运行K-means基线
        use_l2: 是否使用L2归一化特征
        feature_cache_dir: 特征缓存根目录（None 时使用配置默认路径）
        eval_dense: 是否仅评估高密度点
        fast_mode: 快速模式，跳过不必要的计算（未知簇识别、簇类别标签），默认False
        dense_method: 密度计算方法
        assign_model: 稀疏点分配策略
        voting_k: KNN投票邻居数量
        co_mode: co计算模式
        co_manual: 手动指定的co值
        detail_dense: 是否记录骨干网络聚类详细日志
        label_guide: 是否启用标签引导模式
        l1_type: L1损失类型 ('accuracy'或'cross_entropy'，默认'cross_entropy')
        separation_weight: L2损失中簇间分离度的权重（默认1.0）
        penalty_weight: L2损失中局部密度惩罚的权重（默认1.0）

    Returns:
        results: 结果字典
    """
    if not fast_mode:
        print(f"🧪 测试自适应聚类 - 超类: {superclass_name}")
        print("="*80)

    # ========== 步骤1: 获取超类配置信息 ==========
    superclass_info = get_superclass_info(superclass_name)
    known_classes_mapped = superclass_info['known_classes_mapped']

    if not fast_mode:
        print(f"📊 超类信息:")
        print(f"   原始已知类: {superclass_info['known_classes']} -> 映射后: {sorted(list(known_classes_mapped))}")
        print(f"   原始未知类: {superclass_info['unknown_classes']} -> 映射后: {sorted(list(superclass_info['unknown_classes_mapped']))}")

    # ========== 步骤2: 加载数据集（使用增强型数据提供器，一次调用获取所有数据） ==========
    provider = EnhancedDataProvider(cache_base_dir=feature_cache_dir)
    dataset = provider.load_dataset(
        dataset_name=superclass_name,
        model_path=model_path,
        use_l2=use_l2,
        use_train_and_test=use_train_and_test,
        silent=fast_mode
    )

    # 打印数据集摘要（替代原来的8行手动打印）
    dataset.print_summary(silent=fast_mode)

    # ========== 步骤3: 获取聚类输入（一行代码） ==========
    X, targets, known_mask, labeled_mask, train_size = dataset.get_clustering_input()

    # ========== 步骤4: 运行SS-DDBC自适应聚类 ==========
    clustering_result = adaptive_density_clustering(
        X, targets, known_mask, labeled_mask,
        k=k, density_percentile=density_percentile, random_state=random_state,
        train_size=train_size,
        eval_dense=eval_dense, eval_version=eval_version,
        fast_mode=fast_mode,  # 传递 fast_mode 参数
        dense_method=dense_method,
        assign_model=assign_model,
        voting_k=voting_k,
        co_mode=co_mode,
        co_manual=co_manual,
        detail_dense=False if fast_mode else detail_dense,
        label_guide=label_guide,
    )

    # 轻量精简：不再更新日志中的数据集名称（detail_dense 功能保留在核心算法中）

    # ========== 步骤5: 处理聚类结果和计算ACC ==========
    if eval_dense:
        (predictions, n_clusters, unknown_clusters, all_acc, old_acc, new_acc, _,
         neighbors, core_clusters, densities) = clustering_result
        clusters = None  # updated_clusters
        cluster_category_labels = {}  # eval_dense模式不返回cluster_category_labels
        if not fast_mode:
            print(f"\n📊 eval_dense模式: 使用高密度点评估结果")
    else:
        (predictions, n_clusters, unknown_clusters, clusters,
         cluster_category_labels, neighbors, core_clusters, densities) = clustering_result

        # 获取测试集数据（使用dataset便捷方法，一行代码替代10行）
        test_data = dataset.get_test_subset(predictions)

        if not fast_mode:
            print(f"📊 ACC计算范围: {'测试集' if dataset.has_train_test_split else '全部数据'} ({test_data['n_samples']}个样本)")

        # 使用簇ID直接计算ACC（不转换为类别标签）
        # 使用指定版本的ACC计算方法
        if eval_version == 'v1':
            from project_utils.cluster_and_log_utils import split_cluster_acc_v1
            all_acc, old_acc, new_acc = split_cluster_acc_v1(
                test_data['targets'], test_data['predictions'], test_data['known_mask']
            )
        else:  # v2
            from project_utils.cluster_and_log_utils import split_cluster_acc_v2
            all_acc, old_acc, new_acc = split_cluster_acc_v2(
                test_data['targets'], test_data['predictions'], test_data['known_mask']
            )

        # 计算错误样本数量（调试版本）
        from project_utils.cluster_utils import linear_assignment
        import numpy as np

        # 使用测试集数据计算线性分配（与split_cluster_acc_v2一致）
        # 使用簇ID直接计算
        test_targets_int = test_data['targets'].astype(int)
        test_predictions_int = test_data['predictions'].astype(int)

        D = max(test_predictions_int.max(), test_targets_int.max()) + 1
        w = np.zeros((D, D), dtype=int)
        for i in range(len(test_predictions_int)):
            w[test_predictions_int[i], test_targets_int[i]] += 1

        ind = linear_assignment(w.max() - w)
        ind = np.vstack(ind).T

        # ind格式: [[预测簇, 真实标签], ...]
        # 创建反向映射: 预测簇 -> 真实标签
        cluster_to_label = {i: j for i, j in ind}

        # 计算ACC（与split_cluster_acc_v2第57行一致）
        acc_from_w = sum([w[i, j] for i, j in ind]) * 1.0 / len(test_predictions_int)

        # 统计错误样本
        error_count = 0
        for idx in range(test_data['n_samples']):
            pred_cluster = test_predictions_int[idx]
            true_label = test_targets_int[idx]
            # 预测簇应该对应的真实标签
            assigned_label = cluster_to_label.get(pred_cluster, -1)
            if assigned_label != true_label:
                error_count += 1

        correct_count = test_data['n_samples'] - error_count

        if not fast_mode:
            print(f"\n[DEBUG] 混淆矩阵前3行:")
            for i in range(min(3, len(w))):
                print(f"  簇{i}: {w[i][:10]} (总和={w[i].sum()})")
            print(f"[DEBUG] 线性分配 (预测簇->真实标签): {dict(list(cluster_to_label.items())[:5])}")
            print(f"[DEBUG] 从混淆矩阵计算的ACC: {acc_from_w:.4f}")
            print(f"[DEBUG] 官方split_cluster_acc_v2的ACC: {all_acc:.4f}")
            print(f"[DEBUG] 差异: {abs(acc_from_w - all_acc):.6f}")
            print(f"\n   正确样本: {correct_count}/{test_data['n_samples']}")
            print(f"   错误样本: {error_count}/{test_data['n_samples']} ({error_count/test_data['n_samples']:.2%})")

    # ========== 步骤6: 显示聚类组成分析 ==========
    if not eval_dense and not fast_mode:
        analyze_cluster_composition(predictions, targets, known_mask, labeled_mask, unknown_clusters)
    elif not fast_mode:
        print(f"\n💡 eval_dense模式: 跳过聚类组成分析（仅评估高密度骨干网络）")

    # ========== 步骤7: 计算综合损失L ==========
    # 将cluster_distance_method数字转换为方法名
    method_map = {1: 'nearest_k', 2: 'all_pairs', 3: 'prototype'}
    cluster_distance_method_str = method_map.get(cluster_distance_method, 'nearest_k')

    # 解析 L2 组件配置
    resolved_l2_components = l2_components
    if resolved_l2_components is None:
        resolved_l2_components = ['separation', 'penalty'] if use_cluster_quality else []
    elif isinstance(resolved_l2_components, str):
        resolved_l2_components = [comp.strip() for comp in resolved_l2_components.split(',') if comp.strip()]
    else:
        resolved_l2_components = list(resolved_l2_components)

    resolved_l2_weights = {}
    if l2_component_weights:
        resolved_l2_weights = {str(k): float(v) for k, v in l2_component_weights.items()}

    if 'separation' in resolved_l2_components and 'separation' not in resolved_l2_weights:
        resolved_l2_weights['separation'] = separation_weight
    if 'penalty' in resolved_l2_components and 'penalty' not in resolved_l2_weights:
        resolved_l2_weights['penalty'] = penalty_weight

    resolved_l2_params = {str(k): dict(v) for k, v in (l2_component_params or {}).items()}

    # 若显式配置组件，则不再使用 use_cluster_quality 旧开关
    use_cluster_quality_flag = use_cluster_quality and l2_components is None

    clusters_for_l2 = None
    if use_cluster_quality_flag:
        clusters_for_l2 = core_clusters
    elif resolved_l2_components and 'separation' in resolved_l2_components:
        if core_clusters is None:
            raise ValueError("启用 separation 组件时需要核心簇信息，当前配置未提供 core_clusters")
        clusters_for_l2 = core_clusters

    loss_dict = compute_total_loss(
        X=X,
        predictions=predictions,
        targets=targets,
        labeled_mask=labeled_mask,
        cluster_category_labels=cluster_category_labels,
        l1_weight=1.0,
        l2_weight=1.0,
        l1_type=l1_type,  # 传递L1损失类型
        use_cluster_quality=use_cluster_quality_flag,
        clusters=clusters_for_l2,
        k=k,
        cluster_distance_method=cluster_distance_method_str,
        neighbors=neighbors,  # 传递预计算的neighbors，避免重复计算KNN
        separation_weight=separation_weight,  # L2中簇间分离度权重
        penalty_weight=penalty_weight,  # L2中局部密度惩罚权重
        silent=fast_mode,
        l2_components=resolved_l2_components,
        l2_component_weights=resolved_l2_weights,
        l2_component_params=resolved_l2_params
    )

    # ========== 步骤7.5: 计算有标签样本ACC（考虑unknown_clusters惩罚） ==========
    if not eval_dense:
        from ..information.labeled_acc_calculation import compute_labeled_acc_with_unknown_penalty
        labeled_acc_new, labeled_acc_metrics = compute_labeled_acc_with_unknown_penalty(
            predictions=predictions,
            targets=targets,
            labeled_mask=labeled_mask,
            unknown_clusters=unknown_clusters,
            silent=fast_mode
        )
        # 用新的labeled_acc覆盖原来的
        loss_dict['l1_metrics']['accuracy'] = labeled_acc_new
        loss_dict['l1_metrics'].update(labeled_acc_metrics)

    if not fast_mode:
        print(f"📈 聚类结果:")
        print(f"   聚类数量: {n_clusters}")
        print(f"   潜在未知类: {len(unknown_clusters)}个")
        print(f"   All ACC: {all_acc:.4f}")
        print(f"   Old ACC: {old_acc:.4f}")
        print(f"   New ACC: {new_acc:.4f}")

    # ========== 步骤8: K-means基线对比（可选，使用test_data） ==========
    kmeans_results = {}
    if run_kmeans_baseline:
        # 获取测试集数据（如果之前没有获取）
        if eval_dense:
            test_data = dataset.get_test_subset()
        # else: test_data 已在步骤5中获取

        n_true_classes = len(np.unique(test_data['targets']))

        kmeans_baseline = test_kmeans_baseline(
            test_data['features'],
            test_data['targets'],
            test_data['known_mask'],
            n_clusters=n_true_classes,
            random_state=0,
            eval_version=eval_version
        )

        kmeans_results = {
            'kmeans_all_acc': kmeans_baseline['all_acc'],
            'kmeans_old_acc': kmeans_baseline['old_acc'],
            'kmeans_new_acc': kmeans_baseline['new_acc'],
            'kmeans_n_clusters': kmeans_baseline['n_clusters']
        }

    # ========== 步骤9: 返回结果 ==========
    results = {
        'method': 'SS-DDBC',
        'all_acc': all_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'n_clusters': n_clusters,
        'unknown_clusters': unknown_clusters,
        'dbcv': None,  # DBCV已移除
        'labeled_acc': loss_dict['l1_metrics'].get('accuracy'),  # 有标签样本的分配准确率
        'loss': loss_dict['total_loss'],
        'l1': loss_dict['l1'],
        'l2': loss_dict['l2'],
        'loss_dict': loss_dict,
        'l2_components': resolved_l2_components,
        'l2_component_weights': resolved_l2_weights,
        'l2_component_params': resolved_l2_params,
        'dataset': dataset,  # 包含所有数据和元信息
        'test_features': dataset.test_features,
        'test_targets': dataset.test_targets,
        'test_known_mask': dataset.test_known_mask,
        'train_features': dataset.train_features,
        'train_targets': dataset.train_targets,
        'train_known_mask': dataset.train_known_mask,
        'train_labeled_mask': dataset.train_labeled_mask,

        # ==== 为训练/上层调用准备的关键信息 ====
        # 完整聚类标签（含核心点+稀疏点）
        'labels': predictions,
        # 简单核心点索引：当前实现中，final_labels != -1 即视为已分配点
        # 后续如需更精细定义，可改为基于 core_clusters 或高密度掩码。
        'core_points': np.where(np.asarray(predictions) != -1)[0].tolist(),
        # 当前这次调用使用的超参数（单点调用时即“最佳参数”）
        'best_params': {
            'k': k,
            'density_percentile': density_percentile,
        },
    }

    # 合并K-means结果
    results.update(kmeans_results)

    return results
