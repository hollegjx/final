#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 silent 参数和进度条功能
"""

import numpy as np

# 测试1: 验证 density_estimation.py 的 silent 参数
print("=" * 50)
print("测试1: density_estimation.py silent 参数")
print("=" * 50)

from ssddbc.density.density_estimation import (
    compute_median_density,
    compute_relative_density,
    identify_high_density_points
)

# 创建测试数据
np.random.seed(42)
X = np.random.randn(100, 10).astype(np.float32)

print("\n--- silent=False (应该有输出) ---")
densities, knn_distances, neighbors = compute_median_density(X, k=5, silent=False)
relative_densities = compute_relative_density(densities, neighbors, k=5, silent=False)
high_density_mask = identify_high_density_points(relative_densities, percentile=75, use_relative=True, silent=False)

print("\n--- silent=True (应该无输出) ---")
densities, knn_distances, neighbors = compute_median_density(X, k=5, silent=True)
relative_densities = compute_relative_density(densities, neighbors, k=5, silent=True)
high_density_mask = identify_high_density_points(relative_densities, percentile=75, use_relative=True, silent=True)
print("(如果这里没有任何输出，说明 silent=True 生效)")

# 测试2: 验证 api.py 的进度条
print("\n" + "=" * 50)
print("测试2: api.py 进度条")
print("=" * 50)

from ssddbc.grid_search.api import run_clustering_search_on_features

# 创建测试数据
n_samples = 200
X = np.random.randn(n_samples, 64).astype(np.float32)
targets = np.random.randint(0, 5, n_samples)
known_mask = np.array([i < 3 for i in targets])
labeled_mask = np.random.rand(n_samples) < 0.5

print("\n--- 进度条测试 (silent=False) ---")
result = run_clustering_search_on_features(
    features=X,
    targets=targets,
    known_mask=known_mask,
    labeled_mask=labeled_mask,
    k_range=range(3, 7),  # 4个k值
    density_range=range(50, 80, 10),  # 3个dp值 = 12个配置
    random_state=42,
    silent=False,
    max_workers=1,  # 单进程模式
)

print(f"\n最佳参数: {result.best_params}")
print(f"最佳分数: {result.loss:.3f}")
print(f"聚类数: {result.n_clusters}")

print("\n--- 静默模式测试 (silent=True) ---")
result = run_clustering_search_on_features(
    features=X,
    targets=targets,
    known_mask=known_mask,
    labeled_mask=labeled_mask,
    k_range=range(3, 5),
    density_range=range(50, 70, 10),
    random_state=42,
    silent=True,
    max_workers=1,
)
print(f"静默模式完成，最佳参数: {result.best_params}")

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)
