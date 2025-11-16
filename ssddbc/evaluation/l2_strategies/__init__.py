#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2 策略注册表
负责管理所有可用的 L2 组件，并提供统一的查找接口
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from .geometric import compute_separation_component
from .silhouette import compute_silhouette_component

L2ComponentFn = Callable[..., Tuple[float, dict]]


L2_REGISTRY: Dict[str, Dict[str, object]] = {
    'separation': {
        'fn': compute_separation_component,
        'orientation': 'maximize',
        'description': '核心簇间分离度，值越大越好',
    },
    'silhouette': {
        'fn': compute_silhouette_component,
        'orientation': 'maximize',
        'description': '无标签样本轮廓系数，值越大越好',
    },
}


def register_l2_component(name: str, fn: L2ComponentFn, orientation: str = 'minimize', description: str = '') -> None:
    """
    注册新的 L2 组件

    Args:
        name: 组件名称
        fn: 计算函数，返回 (value, metrics)
        orientation: 'maximize' 或 'minimize'，描述指标方向
        description: 组件说明
    """
    if name in L2_REGISTRY:
        raise ValueError(f"L2 组件 '{name}' 已存在，禁止重复注册")
    L2_REGISTRY[name] = {
        'fn': fn,
        'orientation': orientation,
        'description': description
    }


def available_l2_components() -> Dict[str, Dict[str, object]]:
    """返回注册表副本，避免外部修改"""
    return dict(L2_REGISTRY)
