#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
聚类算法模块
包含密度聚类、SS-DDBC算法和相关工具
"""

# 数据相关功能已迁移到clustering.data模块
from .data import (
    set_deterministic_behavior,
    FeatureLoader,
    FeatureExtractor,
    DataProvider,
    ModelLoader,
    DatasetLoader,
    get_superclass_info,
    EnhancedDataProvider,
    EnhancedDataset
)

from .density.density_estimation import compute_simple_density, identify_high_density_points
from .ssddbc.clustering import build_clusters_ssddbc
from .ssddbc.analysis import analyze_ssddbc_clustering_result, analyze_cluster_composition
from .ssddbc.conflict import ssddbc_conflict_resolution, compute_cluster_cohesion
from .ssddbc.assignment import assign_sparse_points_density_based
from .ssddbc.merging import (
    merge_clusters_by_label_consistency,
    merge_clusters_by_category_label,
    get_cluster_category_labels
)
from .ssddbc.adaptive_clustering import adaptive_density_clustering
from .unknown.detection import identify_unknown_clusters, identify_unknown_clusters_from_predictions
from .baseline.kmeans import test_kmeans_baseline
from .prototypes.prototype_builder import build_prototypes

# 评估模块
from .evaluation.loss_function import (
    compute_total_loss,
    compute_supervised_loss_l1,
    compute_l2_loss
)
from .evaluation.cluster_quality import (
    compute_cluster_quality_score,
    compute_cluster_separation,
    compute_local_density_penalty,
    compute_separation_component,
    compute_penalty_component
)

# 工具模块
from .utils.co_calculation import (
    compute_co_value,
    get_co_mode_description,
    apply_co_filter
)

# 信息分析模块
from .information import (
    DenseNetworkLogger,
    get_logger,
    init_logger,
    reset_logger
)

# 测试模块
from .testing.test_superclass import test_adaptive_clustering_on_superclass
from .testing.main import main

# 网格搜索模块 - 延迟导入避免运行时警告
import importlib
_grid_search_spec = importlib.util.find_spec(f"{__name__}.grid_search")
if _grid_search_spec is not None:
    def run_grid_search(*args, **kwargs):
        from .grid_search.batch_runner import run_grid_search as _run
        return _run(*args, **kwargs)

    def parse_superclasses(*args, **kwargs):
        from .grid_search.batch_runner import parse_superclasses as _parse
        return _parse(*args, **kwargs)

    _grid_search_available = True
else:
    print("Warning: Grid search module not available")
    _grid_search_available = False

__all__ = [
    # 数据相关（从clustering.data导入）
    'set_deterministic_behavior',
    'FeatureLoader',
    'FeatureExtractor',
    'DataProvider',
    'EnhancedDataProvider',
    'EnhancedDataset',
    'ModelLoader',
    'DatasetLoader',
    'get_superclass_info',

    # 密度计算
    'compute_simple_density',
    'identify_high_density_points',

    # SS-DDBC聚类
    'build_clusters_ssddbc',
    'analyze_ssddbc_clustering_result',
    'analyze_cluster_composition',
    'ssddbc_conflict_resolution',
    'compute_cluster_cohesion',
    'assign_sparse_points_density_based',
    'merge_clusters_by_label_consistency',
    'merge_clusters_by_category_label',
    'get_cluster_category_labels',
    'adaptive_density_clustering',

    # 未知类识别
    'identify_unknown_clusters',
    'identify_unknown_clusters_from_predictions',

    # 基线方法
    'test_kmeans_baseline',

    # 原型构建
    'build_prototypes',

    # 评估模块
    'compute_total_loss',
    'compute_supervised_loss_l1',
    'compute_l2_loss',
    'compute_cluster_quality_score',
    'compute_cluster_separation',
    'compute_local_density_penalty',
    'compute_separation_component',
    'compute_penalty_component',

    # 工具模块 - Co计算
    'compute_co_value',
    'get_co_mode_description',
    'apply_co_filter',

    # 信息分析模块
    'DenseNetworkLogger',
    'get_logger',
    'init_logger',
    'reset_logger',

    # 测试函数
    'test_adaptive_clustering_on_superclass',
    'main',

]

# 如果网格搜索模块可用，添加到__all__
if _grid_search_available:
    __all__.extend([
        'run_grid_search',
        'parse_superclasses',
    ])
