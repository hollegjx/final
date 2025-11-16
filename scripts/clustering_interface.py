#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练流程与聚类算法之间的接口层
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # 在纯numpy环境可用

from clustering.config import DEFAULT_SEARCH_SPACE, DEFAULT_EVAL_WEIGHTS
from clustering.search.grid_search import grid_search_ssddbc
from clustering.core.ssddbc import SSDDBC
from clustering.evaluation.metrics import compute_all_metrics


def _to_numpy(features) -> np.ndarray:
    if isinstance(features, np.ndarray):
        return features.astype(np.float32)
    if torch is not None and isinstance(features, torch.Tensor):
        return features.detach().cpu().float().numpy()
    raise TypeError(f"不支持的特征类型: {type(features)}")


@dataclass
class ClusteringResult:
    labels: np.ndarray
    core_sample_indices: np.ndarray
    best_params: Dict
    metrics: Dict[str, float]
    num_clusters: int
    noise_ratio: float

    def summary(self) -> str:
        core_ratio = len(self.core_sample_indices) / len(self.labels) if len(self.labels) else 0.0
        return (
            f"Clusters: {self.num_clusters} | "
            f"Core: {len(self.core_sample_indices)} ({core_ratio*100:.1f}%) | "
            f"Noise: {self.noise_ratio*100:.1f}%"
        )


def run_clustering_search(
    features,
    search_space: Dict[str, Iterable] | None = None,
    eval_weights: Dict[str, float] | None = None,
) -> ClusteringResult:
    X = _to_numpy(features)

    best_params, _, metrics = grid_search_ssddbc(
        X,
        param_grid=search_space or DEFAULT_SEARCH_SPACE,
        eval_weights=eval_weights or DEFAULT_EVAL_WEIGHTS,
    )

    clusterer = SSDDBC(**best_params)
    labels = clusterer.fit_predict(X)
    core_indices = (
        clusterer.core_sample_indices_
        if clusterer.core_sample_indices_ is not None
        else np.array([], dtype=int)
    )
    metrics = compute_all_metrics(X, labels)

    return ClusteringResult(
        labels=labels,
        core_sample_indices=core_indices,
        best_params=best_params,
        metrics=metrics,
        num_clusters=metrics.get("num_clusters", 0),
        noise_ratio=metrics.get("noise_ratio", 0.0),
    )


def extract_and_cluster(
    model,
    dataloader: DataLoader,
    device: Optional["torch.device"] = None,
    normalize: bool = True,
    search_space: Dict[str, Iterable] | None = None,
    eval_weights: Dict[str, float] | None = None,
) -> ClusteringResult:
    if torch is None:
        raise ImportError("需要 PyTorch 环境以执行特征提取")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    feats = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            outputs = model(images)
            if normalize:
                outputs = torch.nn.functional.normalize(outputs, dim=-1)
            feats.append(outputs.cpu().float())
            del images, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    features = torch.cat(feats, dim=0)
    return run_clustering_search(
        features,
        search_space=search_space,
        eval_weights=eval_weights,
    )
