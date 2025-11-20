#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
伪标签缓存读写工具

职责：
- 规范化 offline SSDDBC 输出的 indices / labels / core_mask / best_params / metadata
- 提供 save/load API 供训练脚本或离线脚本复用，避免重复实现 npz 读写逻辑
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "PseudoLabelPacket",
    "PseudoLabelSchemaError",
    "save_pseudo_labels",
    "load_pseudo_labels",
    "PseudoLabelCache",
    "load_pseudo_label_cache",
]


class PseudoLabelSchemaError(ValueError):
    """在解析伪标签文件时触发的格式错误。"""


@dataclass(frozen=True)
class PseudoLabelPacket:
    """
    统一的伪标签载体，方便训练/调度脚本使用。

    Attributes:
        indices: 样本全局索引 (N,)
        labels: 伪标签 (N,)
        core_mask: 核心点布尔数组 (N,)
        best_params: SSDDBC 搜索的最佳参数
        metadata: 其他统计信息（ACC、来源 ckpt、生成时间等）
        densities: 样本密度值（可选，用于伪标签加权）
    """

    indices: np.ndarray
    labels: np.ndarray
    core_mask: np.ndarray
    best_params: Dict[str, Any]
    metadata: Dict[str, Any]
    densities: Optional[np.ndarray] = None

    def core_indices(self) -> np.ndarray:
        """返回核心点对应的原始索引。"""
        return self.indices[self.core_mask]

    def num_core(self) -> int:
        """核心点数量。"""
        return int(self.core_mask.sum())

    def validate(self) -> None:
        """执行基本的尺寸校验。"""
        n = self.indices.shape[0]
        if self.labels.shape[0] != n or self.core_mask.shape[0] != n:
            raise PseudoLabelSchemaError(
                "indices/labels/core_mask 长度不一致: "
                f"{n}, {self.labels.shape[0]}, {self.core_mask.shape[0]}"
            )
        if self.densities is not None and self.densities.shape[0] != n:
            raise PseudoLabelSchemaError(
                "densities 长度与 indices 不一致: "
                f"{n}, {self.densities.shape[0]}"
            )


@dataclass
class PseudoLabelCache:
    """
    面向训练阶段的伪标签缓存。

    将 PseudoLabelPacket 展开为稠密数组，便于按 `uq_idx` 进行 O(1) 查询。
    """

    packet: PseudoLabelPacket
    dense_labels: np.ndarray
    dense_core_mask: np.ndarray
    present_mask: np.ndarray
    dense_weights: Optional[np.ndarray] = None
    path: Optional[str] = None

    @classmethod
    def from_packet(cls, packet: PseudoLabelPacket, path: Optional[str] = None) -> "PseudoLabelCache":
        if packet.indices.size == 0:
            raise PseudoLabelSchemaError("伪标签文件中没有任何样本")

        max_index = int(packet.indices.max())
        if max_index < 0:
            raise PseudoLabelSchemaError("伪标签索引必须为非负整数")

        dense_labels = np.full(max_index + 1, -1, dtype=packet.labels.dtype)
        dense_core_mask = np.zeros(max_index + 1, dtype=bool)
        present_mask = np.zeros(max_index + 1, dtype=bool)
        dense_weights = np.zeros(max_index + 1, dtype=np.float32)

        indices = packet.indices.astype(np.int64)
        dense_labels[indices] = packet.labels
        dense_core_mask[indices] = packet.core_mask
        present_mask[indices] = True
        if packet.densities is not None:
            weights = _compute_density_weights(packet.densities)
            dense_weights[indices] = weights.astype(np.float32)
        else:
            dense_weights[indices] = 1.0

        return cls(
            packet=packet,
            dense_labels=dense_labels,
            dense_core_mask=dense_core_mask,
            present_mask=present_mask,
            dense_weights=dense_weights,
            path=path,
        )

    @property
    def num_total(self) -> int:
        return int(self.packet.indices.shape[0])

    @property
    def num_core(self) -> int:
        return int(self.packet.core_mask.astype(bool).sum())

    @property
    def core_ratio(self) -> float:
        if self.num_total == 0:
            return 0.0
        return float(self.num_core) / float(self.num_total)

    @property
    def has_density_weights(self) -> bool:
        return self.packet.densities is not None

    def missing_mask(self, indices: np.ndarray) -> np.ndarray:
        """返回给定索引是否缺少伪标签的布尔掩码。"""
        idx = np.asarray(indices, dtype=np.int64)
        missing = (idx < 0) | (idx >= self.dense_labels.shape[0])
        valid = ~missing
        missing_valid = ~self.present_mask[idx[valid]]
        missing[valid] = missing_valid
        return missing

    def subset_stats(self, indices: np.ndarray) -> Dict[str, Any]:
        """
        针对一组样本索引计算伪标签覆盖与核心点比例。

        Returns:
            dict(total, covered, missing, core_count, core_ratio)
        """
        idx = np.asarray(indices, dtype=np.int64)
        missing_mask = self.missing_mask(idx)
        missing_count = int(missing_mask.sum())
        covered = idx.size - missing_count

        if covered > 0:
            core_count = int(self.dense_core_mask[idx[~missing_mask]].sum())
            core_ratio = core_count / covered
        else:
            core_count = 0
            core_ratio = 0.0

        return {
            "total": int(idx.size),
            "covered": int(covered),
            "missing": missing_count,
            "core_count": core_count,
            "core_ratio": core_ratio,
        }

    def lookup_labels(self, indices: np.ndarray) -> np.ndarray:
        """返回给定索引的伪标签（缺失为 -1）。"""
        idx = np.asarray(indices, dtype=np.int64)
        result = np.full(idx.shape, -1, dtype=self.dense_labels.dtype)
        valid = ~self.missing_mask(idx)
        result[valid] = self.dense_labels[idx[valid]]
        return result

    def lookup_core_mask(self, indices: np.ndarray) -> np.ndarray:
        """返回给定索引是否为核心点（缺失为 False）。"""
        idx = np.asarray(indices, dtype=np.int64)
        result = np.zeros(idx.shape, dtype=bool)
        valid = ~self.missing_mask(idx)
        result[valid] = self.dense_core_mask[idx[valid]]
        return result

    def lookup_weights(self, indices: np.ndarray) -> np.ndarray:
        """返回给定索引的伪标签权重（缺失为0）。"""
        idx = np.asarray(indices, dtype=np.int64)
        result = np.zeros(idx.shape, dtype=np.float32)
        valid = ~self.missing_mask(idx)
        if not np.any(valid):
            return result
        if self.dense_weights is None:
            result[valid] = 1.0
        else:
            result[valid] = self.dense_weights[idx[valid]]
        return result


def _ensure_numpy_1d(value: Any, *, name: str, dtype: Optional[np.dtype] = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise PseudoLabelSchemaError(f"{name} 需要是一维数组，实际 ndim={array.ndim}")
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _pack_object(obj: Optional[Dict[str, Any]]) -> np.ndarray:
    payload = obj if obj is not None else {}
    return np.array(payload, dtype=object)


def _restore_object(arr: np.ndarray, *, name: str) -> Dict[str, Any]:
    if arr.size == 0:
        return {}
    try:
        value = arr.item()
    except ValueError as exc:
        raise PseudoLabelSchemaError(f"{name} 不是标量对象，无法恢复为字典") from exc
    if not isinstance(value, dict):
        raise PseudoLabelSchemaError(f"{name} 需要是字典，实际类型: {type(value)}")
    return value


def _compute_density_weights(densities: np.ndarray) -> np.ndarray:
    """
    根据密度值生成 [0, 1] 之间的权重，遵循 sigmoid(12 * (rank - 0.5)) 公式。

    Args:
        densities: 一维密度数组

    Returns:
        与输入等长的权重数组
    """
    values = np.asarray(densities, dtype=np.float64)
    if values.ndim != 1:
        raise PseudoLabelSchemaError(f"densities 需要是一维数组，实际 ndim={values.ndim}")
    n = values.size
    if n == 0:
        return np.array([], dtype=np.float32)

    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float64)

    start = 0
    current_rank = 1.0
    while start < n:
        end = start + 1
        while end < n and np.isclose(sorted_values[end], sorted_values[start]):
            end += 1
        count = end - start
        first_rank = current_rank
        last_rank = current_rank + count - 1
        avg_rank = (first_rank + last_rank) / 2.0
        ranks[order[start:end]] = avg_rank
        current_rank += count
        start = end

    normalized = ranks / n
    weights = 1.0 / (1.0 + np.exp(-12.0 * (normalized - 0.5)))
    return weights.astype(np.float32)


def save_pseudo_labels(
    path: str,
    *,
    indices: Any,
    labels: Any,
    core_mask: Any,
    densities: Any = None,
    best_params: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_compression: bool = True,
) -> None:
    """
    将离线 SSDDBC 产出的伪标签信息写入 npz 文件。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    indices_np = _ensure_numpy_1d(indices, name="indices", dtype=np.int64)
    labels_np = _ensure_numpy_1d(labels, name="labels")
    core_mask_np = _ensure_numpy_1d(core_mask, name="core_mask").astype(bool)
    densities_np = None
    if densities is not None:
        densities_np = _ensure_numpy_1d(densities, name="densities").astype(np.float32)

    if len(indices_np) != len(labels_np) or len(indices_np) != len(core_mask_np):
        raise PseudoLabelSchemaError(
            f"indices/labels/core_mask 长度必须一致，当前分别为 "
            f"{len(indices_np)}, {len(labels_np)}, {len(core_mask_np)}"
        )
    if densities_np is not None and len(indices_np) != len(densities_np):
        raise PseudoLabelSchemaError(
            f"densities 长度必须与 indices 一致，当前为 "
            f"{len(indices_np)} vs {len(densities_np)}"
        )

    saver = np.savez_compressed if use_compression else np.savez
    payload = {
        "indices": indices_np,
        "labels": labels_np,
        "core_mask": core_mask_np,
        "best_params": _pack_object(best_params),
        "metadata": _pack_object(metadata),
    }
    if densities_np is not None:
        payload["densities"] = densities_np
    saver(path, **payload)


def load_pseudo_labels(path: str) -> PseudoLabelPacket:
    """
    从 npz 文件加载伪标签数据，并完成基本校验。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"伪标签文件不存在: {path}")

    with np.load(path, allow_pickle=True) as data:
        required = {"indices", "labels", "core_mask"}
        missing = required.difference(data.files)
        if missing:
            raise PseudoLabelSchemaError(f"伪标签文件缺少必要字段: {', '.join(sorted(missing))}")

        indices = data["indices"].astype(np.int64)
        labels = data["labels"]
        core_mask = data["core_mask"].astype(bool)

        best_params = (
            _restore_object(data["best_params"], name="best_params")
            if "best_params" in data
            else {}
        )
        metadata = (
            _restore_object(data["metadata"], name="metadata") if "metadata" in data else {}
        )
        densities = data["densities"].astype(np.float32) if "densities" in data else None

    packet = PseudoLabelPacket(
        indices=indices,
        labels=labels,
        core_mask=core_mask,
        best_params=best_params,
        metadata=metadata,
        densities=densities,
    )
    packet.validate()
    return packet


def load_pseudo_label_cache(path: str) -> PseudoLabelCache:
    """加载伪标签文件并返回稠密缓存结构。"""
    packet = load_pseudo_labels(path)
    return PseudoLabelCache.from_packet(packet, path=path)
