#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试模块
包含超类测试函数和主程序入口
"""

from .test_superclass import test_adaptive_clustering_on_superclass
from .main import main

__all__ = [
    # 测试函数
    'test_adaptive_clustering_on_superclass',
    'main',
]
