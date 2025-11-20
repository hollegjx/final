#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
互信息类 L2 组件占位实现
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple, Any


def compute_mutual_info_component(**_) -> Tuple[float, Dict[str, Any]]:
    """
    互信息组件占位，暂未实现。
    返回 0 并给出警告，避免实验脚本崩溃。
    """
    warnings.warn("L2 组件 'mutual_info' 尚未实现，返回 0 作为占位值", RuntimeWarning)
    metrics = {
        'mutual_info': 0.0,
        'orientation': 'maximize',
        'implemented': False
    }
    return 0.0, metrics
