#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存级 SS-DDBC 网格搜索 API

用途：
- 供训练代码直接在内存特征上执行聚类搜索，获取伪标签和核心点
- 不做任何文件读写，也不依赖超类名称或特征缓存路径
- 支持多进程并行搜索（使用 ProcessPoolExecutor）

当前实现：
- 固定使用已裁剪配置：
  dense_method=1, co_mode=2, assign_model=3,
  cluster_distance_method='prototype'
- 使用损失函数（L1 + L2）作为评分标准，权重：
  * l1_weight=1.0
  * separation_weight=0.0
  * silhouette_weight=3.0
- 并行模式：max_workers 控制进程数，1 为单进程，None 为自动检测 CPU 核心数
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from ssddbc.ssddbc.adaptive_clustering import adaptive_density_clustering
from ssddbc.evaluation.loss_function import compute_total_loss
from project_utils.cluster_utils import cluster_acc


@dataclass
class ClusteringSearchResult:
    """内存级聚类搜索结果"""

    labels: np.ndarray
    core_points: np.ndarray
    best_params: Dict[str, int]
    loss: float  # 实际是评分函数值，越大越好
    n_clusters: int
    densities: Optional[np.ndarray] = None
    results_grid: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None  # 调试用：完整网格评分/ACC


def _to_numpy_1d(x: Union[Sequence[int], np.ndarray]) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"期望一维数组，收到形状: {arr.shape}")
    return arr


def _worker_evaluate_config(
    k: int,
    dp: int,
    X: np.ndarray,
    targets: np.ndarray,
    known_mask: np.ndarray,
    labeled_mask: np.ndarray,
    random_state: int,
    silent: bool,
    return_metrics: bool = False,
) -> Union[
    Tuple[float, int, int, np.ndarray, int, Sequence[int], Sequence[set[int]]],
    Tuple[float, int, int, np.ndarray, int, Sequence[int], Sequence[set[int]], Dict[str, float]],
]:
    """
    顶层 worker 函数：评估单个 (k, density_percentile) 配置

    必须在模块级别定义以支持 pickle 序列化

    Returns:
        (score, k, dp, predictions, n_clusters, unknown_clusters, core_clusters)
        注：score 越大越好
    """
    predictions, n_clusters, unknown_clusters, core_clusters = _run_single_configuration(
        X=X,
        targets=targets,
        known_mask=known_mask,
        labeled_mask=labeled_mask,
        k=k,
        density_percentile=dp,
        random_state=random_state,
        silent=silent,
    )

    # 计算损失函数（L1 + L2）
    loss_dict = compute_total_loss(
        X=X,
        predictions=predictions,
        targets=targets,
        labeled_mask=labeled_mask,
        l1_weight=1.0,
        l2_weight=1.0,
        l1_type='cross_entropy',
        l2_components=['separation', 'silhouette'],
        l2_component_weights={'silhouette': 3.0},
        separation_weight=0.0,
        clusters=core_clusters,
        k=k,
        cluster_distance_method='prototype',
        silent=True,
    )

    total_loss = loss_dict['total_loss']

    if return_metrics:
        all_acc, old_acc, new_acc = _compute_accs(predictions, targets, known_mask)
        metrics = {
            "score": float(total_loss),
            "all_acc": all_acc,
            "old_acc": old_acc,
            "new_acc": new_acc,
            "n_clusters": int(n_clusters),
            "l1_loss": float(loss_dict.get("l1_loss", np.nan)),
            "separation_score": float(loss_dict.get("separation_score", np.nan)),
            "silhouette": float(loss_dict.get("silhouette", np.nan)),
            "penalty_score": float(loss_dict.get("penalty_score", np.nan)),
        }
        return (total_loss, k, dp, predictions, n_clusters, unknown_clusters, core_clusters, metrics)

    return (total_loss, k, dp, predictions, n_clusters, unknown_clusters, core_clusters)


def run_clustering_search_on_features(
    features: np.ndarray,
    targets: Sequence[int],
    known_mask: Sequence[bool],
    labeled_mask: Optional[Sequence[bool]] = None,
    *,
    k_range: Iterable[int],
    density_range: Iterable[int],
    random_state: int = 0,
    silent: bool = True,
    max_workers: Optional[int] = None,
    return_all: bool = False,
) -> ClusteringSearchResult:
    """
    在内存特征上执行 SS-DDBC 网格搜索（不涉及磁盘 I/O）

    使用损失函数（L1 + L2）评价参数组合，选择最大分数的配置。
    注：由于L2包含maximize组件，实际是找最大值而非最小损失。
    权重配置：l1_weight=1.0, separation_weight=0.0, silhouette_weight=3.0

    Args:
        features: 特征矩阵 (n_samples, feat_dim)
        targets: 真实标签 (n_samples,)
        known_mask: 已知类掩码 (n_samples,) - True 表示旧类 / 有标签类别
        labeled_mask: 有标注样本掩码 (n_samples,)。如为 None，则默认等于 known_mask
        k_range: KNN k 搜索范围，例如 range(3, 21)
        density_range: 密度百分位搜索范围，例如 range(40, 100, 5)
        random_state: 随机种子
        silent: 是否静默模式
        max_workers: 最大并行工作进程数。None 表示使用 CPU 核心数，1 表示单进程模式
        return_all: 调试开关，True 时返回完整网格评分和 ACC

    Returns:
        ClusteringSearchResult
    """
    X = np.asarray(features, dtype=np.float32)
    targets_np = _to_numpy_1d(targets).astype(int)
    known_mask_np = _to_numpy_1d(known_mask).astype(bool)
    if labeled_mask is None:
        labeled_mask_np = known_mask_np.copy()
    else:
        labeled_mask_np = _to_numpy_1d(labeled_mask).astype(bool)

    n_samples = X.shape[0]
    if (
        targets_np.shape[0] != n_samples
        or known_mask_np.shape[0] != n_samples
        or labeled_mask_np.shape[0] != n_samples
    ):
        raise ValueError(
            f"features/targets/known_mask/labeled_mask 长度不一致："
            f"{n_samples}, {targets_np.shape[0]}, "
            f"{known_mask_np.shape[0]}, {labeled_mask_np.shape[0]}"
        )

    # 决定工作进程数
    if max_workers is None:
        max_workers = mp.cpu_count()

    # 生成所有配置组合
    k_list = list(k_range)
    dp_list = list(density_range)
    configs = [(k, dp) for k in k_list for dp in dp_list]

    if not configs:
        raise ValueError("k_range 和 density_range 不能为空")

    best_score = float('-inf')  # 找最大分数（因为L2组件是maximize）
    best_result: Optional[Tuple[np.ndarray, int, Sequence[int], Sequence[set[int]]]] = None
    best_params: Dict[str, int] = {}
    results_dict: Dict[Tuple[int, int], Dict[str, float]] = {} if return_all else None

    # 创建进度条（始终显示，即使在 silent 模式下）
    pbar = tqdm(
        total=len(configs),
        desc="Grid search",
        unit="config",
        ncols=100,
        leave=True,
        disable=False,  # 始终显示进度条
    )

    # 单进程模式：保持原有串行逻辑
    if max_workers == 1:
        for k, dp in configs:
            predictions, n_clusters, unknown_clusters, core_clusters = _run_single_configuration(
                X=X,
                targets=targets_np,
                known_mask=known_mask_np,
                labeled_mask=labeled_mask_np,
                k=k,
                density_percentile=dp,
                random_state=random_state,
                silent=silent,
            )

            # 计算损失函数
            loss_dict = compute_total_loss(
                X=X,
                predictions=predictions,
                targets=targets_np,
                labeled_mask=labeled_mask_np,
                l1_weight=1.0,
                l2_weight=1.0,
                l1_type='cross_entropy',
                l2_components=['separation', 'silhouette'],
                l2_component_weights={'silhouette': 3.0},
                separation_weight=0.0,
                clusters=core_clusters,
                k=k,
                cluster_distance_method='prototype',
                silent=True,
            )

            total_loss = loss_dict['total_loss']

            if return_all and results_dict is not None:
                all_acc, old_acc, new_acc = _compute_accs(predictions, targets_np, known_mask_np)
                results_dict[(k, dp)] = {
                    "score": float(total_loss),
                    "all_acc": all_acc,
                    "old_acc": old_acc,
                    "new_acc": new_acc,
                    "n_clusters": int(n_clusters),
                    "l1_loss": float(loss_dict.get("l1_loss", np.nan)),
                    "separation_score": float(loss_dict.get("separation_score", np.nan)),
                    "silhouette": float(loss_dict.get("silhouette", np.nan)),
                    "penalty_score": float(loss_dict.get("penalty_score", np.nan)),
                }

            if total_loss > best_score:
                best_score = total_loss
                best_result = (predictions, n_clusters, unknown_clusters, core_clusters)
                best_params = {"k": k, "density_percentile": dp}

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({"k": k, "dp": dp, "best": f"{best_score:.3f}"})

    # 多进程模式：并行评估所有配置
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_params = {
                executor.submit(
                    _worker_evaluate_config,
                    k, dp, X, targets_np, known_mask_np, labeled_mask_np, random_state, silent, return_all,
                ): (k, dp)
                for k, dp in configs
            }

            # 收集结果并找到最佳配置（最大分数）
            for future in as_completed(future_to_params):
                result = future.result()
                if return_all:
                    (total_loss, k, dp, predictions, n_clusters,
                     unknown_clusters, core_clusters, metrics) = result
                else:
                    (total_loss, k, dp, predictions, n_clusters,
                     unknown_clusters, core_clusters) = result

                if total_loss > best_score:
                    best_score = total_loss
                    best_result = (predictions, n_clusters, unknown_clusters, core_clusters)
                    best_params = {"k": k, "density_percentile": dp}

                if return_all and results_dict is not None:
                    results_dict[(k, dp)] = metrics

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"k": k, "dp": dp, "best": f"{best_score:.3f}"})

    # 关闭进度条
    pbar.close()

    if best_result is None:
        raise RuntimeError("聚类搜索未产生任何结果，请检查 k_range / density_range 是否为空。")

    predictions, n_clusters, unknown_clusters, core_clusters = best_result

    # 为最佳配置重新运行一次聚类以获取密度（减少并行模式下的跨进程序列化压力）
    densities = None
    if best_params:
        predictions, n_clusters, unknown_clusters, core_clusters, densities = _run_single_configuration(
            X=X,
            targets=targets_np,
            known_mask=known_mask_np,
            labeled_mask=labeled_mask_np,
            k=best_params["k"],
            density_percentile=best_params["density_percentile"],
            random_state=random_state,
            silent=silent,
            return_densities=True,
        )

    # 核心点定义：使用 core_clusters 中的点集合
    core_indices: set[int] = set()
    for cluster in core_clusters or []:
        core_indices.update(cluster)
    core_points = np.array(sorted(core_indices), dtype=int)

    return ClusteringSearchResult(
        labels=predictions,
        core_points=core_points,
        best_params=best_params,
        loss=float(best_score),
        n_clusters=int(n_clusters),
        densities=densities,
        results_grid=results_dict,
    )


def _run_single_configuration(
    *,
    X: np.ndarray,
    targets: np.ndarray,
    known_mask: np.ndarray,
    labeled_mask: np.ndarray,
    k: int,
    density_percentile: int,
    random_state: int,
    silent: bool,
    return_densities: bool = False,
) -> Union[
    Tuple[np.ndarray, int, Sequence[int], Sequence[set[int]]],
    Tuple[np.ndarray, int, Sequence[int], Sequence[set[int]], np.ndarray],
]:
    """
    使用单组 (k, density_percentile) 配置运行一次 SS-DDBC 聚类。

    返回:
        predictions, n_clusters, unknown_clusters, core_clusters
    """
    # train_size: 这里视为所有样本都属于“训练集”（无显式train/test划分）
    train_size = X.shape[0]

    (predictions, n_clusters, unknown_clusters, _, _, _, core_clusters,
     densities) = adaptive_density_clustering(
        X,
        targets,
        known_mask,
        labeled_mask,
        k=k,
        density_percentile=density_percentile,
        random_state=random_state,
        train_size=train_size,
        co_mode=2,
        co_manual=None,
        eval_dense=False,
        eval_version="v2",
        fast_mode=True,  # 网格搜索使用快速模式，跳过不必要计算
        dense_method=1,
        assign_model=3,
        voting_k=5,
        detail_dense=False,
        label_guide=False,
    )

    if return_densities:
        return predictions, n_clusters, unknown_clusters, core_clusters, densities

    return predictions, n_clusters, unknown_clusters, core_clusters


def _compute_accs(
    predictions: np.ndarray,
    targets: np.ndarray,
    known_mask: np.ndarray
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """计算 all/old/new ACC（仅调试用）。"""
    all_acc = float(cluster_acc(targets, predictions)) if targets.size > 0 else None
    old_acc = None
    new_acc = None
    if known_mask.any():
        old_acc = float(cluster_acc(targets[known_mask], predictions[known_mask]))
    unknown_mask = ~known_mask
    if unknown_mask.any():
        new_acc = float(cluster_acc(targets[unknown_mask], predictions[unknown_mask]))
    return all_acc, old_acc, new_acc
