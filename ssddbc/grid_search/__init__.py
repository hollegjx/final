#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网格搜索模块
包含批量网格搜索功能和独立热力图可视化工具
"""

# 延迟导入，避免在包加载时触发模块执行
def __getattr__(name):
    if name == 'run_grid_search':
        from .batch_runner import run_grid_search
        return run_grid_search
    elif name == 'parse_superclasses':
        from .batch_runner import parse_superclasses
        return parse_superclasses
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # 批量网格搜索
    'run_grid_search',
    'parse_superclasses',
]

# 注意：heatmap.py 是独立的可视化工具，直接运行即可
# python -m ssddbc.grid_search.heatmap [options]
