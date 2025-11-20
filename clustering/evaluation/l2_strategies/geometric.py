#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何类 L2 组件实现
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any

import numpy as np

from ..cluster_quality import (
    compute_separation_component as base_separation_component,
    compute_penalty_component as base_penalty_component,
)


def _ensure_array(array_like) -> np.ndarray:
    """将输入转换为 numpy 数组，便于统一处理"""
    if isinstance(array_like, np.ndarray):
        return array_like
    return np.asarray(array_like)


def compute_separation_component(*, clusters, X, k: int,
                                 cluster_distance_method: str = 'prototype',
                                 **_) -> Tuple[float, Dict[str, Any]]:
    """
    计算簇间分离度组件（值越大越好）
    """
    if clusters is None:
        raise ValueError("计算 separation 组件时需要提供 clusters")

    separation_score, separation_metrics = base_separation_component(
        clusters=clusters,
        X=_ensure_array(X),
        k=k,
        method=cluster_distance_method
    )

    metrics = {
        'separation_score': separation_score,
        'orientation': 'maximize',
        'details': separation_metrics.get('details', separation_metrics)
    }
    return separation_score, metrics


def compute_penalty_component(*, X, predictions, k: int,
                              neighbors: Optional[np.ndarray] = None,
                              **_) -> Tuple[float, Dict[str, Any]]:
    """
    计算局部密度惩罚组件（值越小越好）
    """
    if predictions is None:
        raise ValueError("计算 penalty 组件时需要提供 predictions 作为聚类标签")

    penalty_score, penalty_metrics = base_penalty_component(
        X=_ensure_array(X),
        labels=_ensure_array(predictions),
        k=k,
        neighbors=neighbors
    )

    metrics = {
        'penalty_score': penalty_score,
        'orientation': 'minimize',
        'details': penalty_metrics.get('details', penalty_metrics)
    }
    return penalty_score, metrics
