#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silhouette-based L2 component for unlabeled samples.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import silhouette_samples


def compute_silhouette_component(*,
                                 X,
                                 predictions,
                                 labeled_mask,
                                 **_) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the Silhouette Score for unlabeled samples only.

    Args:
        X: feature matrix (n_samples, n_features)
        predictions: cluster labels (n_samples,)
        labeled_mask: boolean mask of labeled samples (n_samples,)

    Returns:
        tuple: (score, metrics) where score is the mean silhouette of unlabeled samples.
    """
    features = np.asarray(X)
    labels = np.asarray(predictions)
    labeled_mask = np.asarray(labeled_mask, dtype=bool)

    n_samples = features.shape[0]
    if features.ndim != 2:
        raise ValueError("silhouette 组件要求特征矩阵为二维 (n_samples, feat_dim)")
    if labels.shape[0] != n_samples or labeled_mask.shape[0] != n_samples:
        raise ValueError("silhouette 组件要求 X、predictions、labeled_mask 长度一致")

    unique_clusters = np.unique(labels)
    if unique_clusters.size < 2:
        metrics = {
            'silhouette_score': 0.0,
            'orientation': 'maximize',
            'implemented': True,
            'reason': '簇数量不足，无法计算Silhouette'
        }
        return 0.0, metrics

    unlabeled_mask = ~labeled_mask
    n_unlabeled = int(np.sum(unlabeled_mask))
    if n_unlabeled == 0:
        metrics = {
            'silhouette_score': 0.0,
            'orientation': 'maximize',
            'implemented': True,
            'reason': '无无标签样本'
        }
        return 0.0, metrics

    all_silhouettes = silhouette_samples(features, labels, metric='euclidean')
    unlabeled_silhouettes = all_silhouettes[unlabeled_mask]

    score = float(np.mean(unlabeled_silhouettes))

    metrics = {
        'silhouette_score': score,
        'orientation': 'maximize',
        'implemented': True,
        'n_unlabeled': n_unlabeled,
        'n_labeled': int(np.sum(labeled_mask)),
        'n_clusters': int(unique_clusters.size),
        'unlabeled_silhouette_mean': score,
        'unlabeled_silhouette_median': float(np.median(unlabeled_silhouettes)),
        'unlabeled_silhouette_std': float(np.std(unlabeled_silhouettes)),
        'global_silhouette_mean': float(np.mean(all_silhouettes))
    }
    return score, metrics
