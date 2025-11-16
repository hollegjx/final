#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何类 L2 组件实现（精简版）

仅保留 separation 组件，用于计算核心簇间分离度。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..cluster_quality import compute_separation_component as base_separation_component


def _ensure_array(array_like) -> np.ndarray:
    """将输入转换为 numpy 数组，便于统一处理"""
    if isinstance(array_like, np.ndarray):
        return array_like
    return np.asarray(array_like)


def compute_separation_component(
    *,
    clusters,
    X,
    k: int,
    cluster_distance_method: str = 'prototype',
    **_: Any,
) -> Tuple[float, Dict[str, Any]]:
    """
    计算簇间分离度组件（值越大越好）

    这是对 evaluation.cluster_quality.compute_separation_component 的轻量包装，
    以适配 L2 组件统一接口。
    """
    if clusters is None:
        raise ValueError("计算 separation 组件时需要提供 clusters")

    separation_score, separation_metrics = base_separation_component(
        clusters=clusters,
        X=_ensure_array(X),
        k=k,
        method=cluster_distance_method,
    )

    metrics: Dict[str, Any] = {
        'separation_score': separation_score,
        'orientation': 'maximize',
        'details': separation_metrics.get('details', separation_metrics),
    }
    return separation_score, metrics

