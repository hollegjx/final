#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存级 SS-DDBC 网格搜索 API

用途：
- 供训练代码直接在内存特征上执行聚类搜索，获取伪标签和核心点
- 不做任何文件读写，也不依赖超类名称或特征缓存路径

当前实现：
- 固定使用已裁剪配置：
  dense_method=1, co_mode=2, assign_model=3,
  use_cluster_quality=False（仅用 ACC 作为评分），
  cluster_distance_method='prototype'
- 使用 split_cluster_acc_v2(all/old/new) 作为评分标准
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ssddbc.ssddbc.adaptive_clustering import adaptive_density_clustering
from project_utils.cluster_and_log_utils import split_cluster_acc_v2


@dataclass
class ClusteringSearchResult:
    """内存级聚类搜索结果"""

    labels: np.ndarray
    core_points: np.ndarray
    best_params: Dict[str, int]
    all_acc: float
    old_acc: float
    new_acc: float
    n_clusters: int


def _to_numpy_1d(x: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"期望一维数组，收到形状: {arr.shape}")
    return arr


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
) -> ClusteringSearchResult:
    """
    在内存特征上执行 SS-DDBC 网格搜索（不涉及磁盘 I/O）

    Args:
        features: 特征矩阵 (n_samples, feat_dim)
        targets: 真实标签 (n_samples,)
        known_mask: 已知类掩码 (n_samples,) - True 表示旧类 / 有标签类别
        labeled_mask: 有标注样本掩码 (n_samples,)。如为 None，则默认等于 known_mask
        k_range: KNN k 搜索范围，例如 range(3, 21)
        density_range: 密度百分位搜索范围，例如 range(40, 100, 5)
        random_state: 随机种子
        silent: 是否静默模式

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

    best_score = -1.0
    best_result: Optional[Tuple[np.ndarray, int, Sequence[int], Sequence[set[int]]]] = None
    best_params: Dict[str, int] = {}

    for k in k_range:
        for dp in density_range:
            # 运行一次 SS-DDBC 聚类（使用裁剪后的固定配置）
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

            # 基于 v2 ACC 评分（与 test_superclass 保持一致）
            all_acc, old_acc, new_acc = split_cluster_acc_v2(
                targets_np,
                predictions,
                known_mask_np,
            )

            if all_acc > best_score:
                best_score = all_acc
                best_result = (predictions, n_clusters, unknown_clusters, core_clusters)
                best_params = {"k": k, "density_percentile": dp}

    if best_result is None:
        raise RuntimeError("聚类搜索未产生任何结果，请检查 k_range / density_range 是否为空。")

    predictions, n_clusters, unknown_clusters, core_clusters = best_result

    # 核心点定义：使用 core_clusters 中的点集合
    core_indices: set[int] = set()
    for cluster in core_clusters or []:
        core_indices.update(cluster)
    core_points = np.array(sorted(core_indices), dtype=int)

    # 使用最佳配置重新计算 ACC（避免浮点误差累积）
    all_acc, old_acc, new_acc = split_cluster_acc_v2(
        targets_np,
        predictions,
        known_mask_np,
    )

    return ClusteringSearchResult(
        labels=predictions,
        core_points=core_points,
        best_params=best_params,
        all_acc=float(all_acc),
        old_acc=float(old_acc),
        new_acc=float(new_acc),
        n_clusters=int(n_clusters),
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
) -> Tuple[np.ndarray, int, Sequence[int], Sequence[set[int]]]:
    """
    使用单组 (k, density_percentile) 配置运行一次 SS-DDBC 聚类。

    返回:
        predictions, n_clusters, unknown_clusters, core_clusters
    """
    # train_size: 这里视为所有样本都属于“训练集”（无显式train/test划分）
    train_size = X.shape[0]

    predictions, n_clusters, unknown_clusters, _, _, _, _, core_clusters = adaptive_density_clustering(
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
        silent=silent,
        dense_method=1,
        assign_model=3,
        voting_k=5,
        detail_dense=False,
        label_guide=False,
    )

    return predictions, n_clusters, unknown_clusters, core_clusters
