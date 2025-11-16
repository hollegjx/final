#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练检查点工具

用途：
- 统一保存 / 恢复训练过程中涉及的关键状态，支持尽量等价的断点续训：
  - 模型与投影头参数
  - 优化器与学习率调度器状态
  - 训练会话（早停监控、最佳性能统计等）
  - 随机数生成器状态（torch / cuda / numpy / python）

注意：
- 本模块只负责“状态的打包与还原”，不改变原有训练逻辑；
- 调用方需要在合适的时机主动调用 save_training_state / load_training_state。
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from utils.training_utils import TrainingSession


def _get_rng_state() -> Dict[str, Any]:
    """获取当前随机数状态（CPU / CUDA / NumPy / Python）。"""
    rng_state: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "cuda": None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            # 在未初始化 CUDA 或多 GPU 情况下保持 None
            rng_state["cuda"] = None
    return rng_state


def _set_rng_state(rng_state: Dict[str, Any]) -> None:
    """恢复随机数状态。缺失的部分将被忽略。"""
    if not rng_state:
        return

    if "torch" in rng_state and rng_state["torch"] is not None:
        torch.set_rng_state(rng_state["torch"])

    if "cuda" in rng_state and rng_state["cuda"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        except RuntimeError:
            # 如果当前环境无法设置 CUDA RNG（例如 GPU 数量不同），则忽略
            pass

    if "numpy" in rng_state and rng_state["numpy"] is not None:
        np.random.set_state(rng_state["numpy"])

    if "python" in rng_state and rng_state["python"] is not None:
        random.setstate(rng_state["python"])


def save_training_state(
    path: str,
    *,
    epoch: int,
    model: torch.nn.Module,
    projection_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    training_session: TrainingSession,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    保存当前训练状态到指定路径。

    Args:
        path: 目标文件路径（.pt 或 .pth）
        epoch: 当前轮次编号（0-based）
        model: 主模型
        projection_head: 投影头
        optimizer: 优化器
        scheduler: 学习率调度器（可为 None）
        training_session: 训练会话管理器
        extra_state: 额外自定义状态（例如伪标签版本号）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state: Dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "proj": projection_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "training_session": training_session.state_dict(),
        "rng_state": _get_rng_state(),
        "extra": extra_state or {},
    }

    torch.save(state, path)


def load_training_state(
    path: str,
    *,
    model: torch.nn.Module,
    projection_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    training_session: TrainingSession,
) -> Tuple[int, Dict[str, Any]]:
    """
    从指定路径加载训练状态，并就地恢复各组件。

    Args:
        path: 检查点文件路径
        model: 已构建好的主模型实例
        projection_head: 已构建好的投影头实例
        optimizer: 已构建好的优化器实例
        scheduler: 已构建好的学习率调度器实例（可为 None）
        training_session: 已构建好的 TrainingSession 实例

    Returns:
        tuple: (epoch, extra_state)
            epoch: 上一次保存的 epoch（0-based）
            extra_state: 保存时附带的额外状态字典
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到检查点文件: {path}")

    # 始终以 CPU 方式加载，再由各模块负责将权重移动到目标设备
    state: Dict[str, Any] = torch.load(path, map_location="cpu")

    model.load_state_dict(state["model"])
    projection_head.load_state_dict(state["proj"])
    optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    ts_state = state.get("training_session")
    if ts_state is not None:
        training_session.load_state_dict(ts_state)

    rng_state = state.get("rng_state")
    if rng_state is not None:
        _set_rng_state(rng_state)

    epoch = int(state.get("epoch", 0))
    extra_state = state.get("extra", {}) or {}

    return epoch, extra_state

