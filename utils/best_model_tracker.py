#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最佳模型追踪器 - 使用 JSON 文件记录全局最佳模型信息

用途：
- 在 pipeline 的多个阶段之间持久化最佳模型信息
- 避免阶段切换时丢失历史最佳记录
- 提供人类可读的最佳模型元信息

设计：
- 使用独立的 JSON 文件存储（<run_dir>/best_model_info.json）
- 原子性写入（先写临时文件，再重命名）
- 轻量级查询（无需加载大的 checkpoint）
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


class BestModelTracker:
    """最佳模型追踪器"""

    def __init__(self, run_dir: str):
        """
        初始化追踪器

        Args:
            run_dir: 训练运行目录（包含 checkpoints/log/pseudo_labels 等）
        """
        self.run_dir = run_dir
        self.json_path = os.path.join(run_dir, "best_model_info.json")

    def load(self) -> Dict[str, Any]:
        """
        加载最佳模型信息

        Returns:
            包含最佳模型信息的字典，如果文件不存在则返回默认值
        """
        if not os.path.exists(self.json_path):
            return self._get_default_info()

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            return info
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  警告: 无法读取 {self.json_path}: {e}")
            print(f"   将使用默认值初始化")
            return self._get_default_info()

    def save(self, info: Dict[str, Any]) -> bool:
        """
        保存最佳模型信息（原子性写入）

        Args:
            info: 最佳模型信息字典

        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

            # 添加更新时间戳
            info["last_updated"] = datetime.now().isoformat()

            # 原子性写入：先写临时文件，再重命名
            tmp_path = self.json_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            # 重命名是原子操作（在同一文件系统上）
            os.replace(tmp_path, self.json_path)

            return True
        except (IOError, OSError) as e:
            print(f"❌ 错误: 无法保存 {self.json_path}: {e}")
            return False

    def update_if_better(
        self,
        new_acc: float,
        epoch: int,
        model_path: str,
        proj_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        stage: str = "unknown"
    ) -> bool:
        """
        如果新的 ACC 更好，则更新最佳模型信息

        Args:
            new_acc: 新的准确率
            epoch: 对应的 epoch
            model_path: 模型文件路径（相对于 run_dir）
            proj_path: 投影头文件路径（相对于 run_dir）
            metadata: 额外的元信息（如 old_acc, new_acc, train_loss 等）
            hyperparameters: 训练超参数
            stage: 当前训练阶段（stage1, stage3 等）

        Returns:
            是否更新了最佳模型
        """
        current_info = self.load()
        current_best = current_info.get("best_acc", 0.0)

        if new_acc > current_best:
            new_info = {
                "best_acc": float(new_acc),
                "best_epoch": int(epoch),
                "best_model_path": model_path,
                "best_proj_path": proj_path,
                "stage": stage,
            }

            if metadata:
                new_info["metadata"] = metadata

            if hyperparameters:
                new_info["hyperparameters"] = hyperparameters

            success = self.save(new_info)

            if success:
                print(f"🏆 更新全局最佳模型: ACC {current_best:.4f} → {new_acc:.4f} (epoch {epoch})")

            return success

        return False

    def get_best_acc(self) -> float:
        """获取当前最佳 ACC"""
        info = self.load()
        return info.get("best_acc", 0.0)

    def get_best_epoch(self) -> int:
        """获取达到最佳的 epoch"""
        info = self.load()
        return info.get("best_epoch", -1)

    def print_summary(self):
        """打印最佳模型摘要信息"""
        info = self.load()

        if info.get("best_epoch", -1) < 0:
            print("📊 尚未记录最佳模型")
            return

        print(f"\n{'='*60}")
        print(f"🏆 全局最佳模型信息")
        print(f"{'='*60}")
        print(f"最佳 ACC:    {info['best_acc']:.4f}")
        print(f"最佳 Epoch:  {info['best_epoch']}")
        print(f"训练阶段:    {info.get('stage', 'unknown')}")

        if "metadata" in info:
            meta = info["metadata"]
            if "old_acc" in meta and "new_acc" in meta:
                print(f"  - Old ACC: {meta['old_acc']:.4f}")
                print(f"  - New ACC: {meta['new_acc']:.4f}")
            if "train_loss" in meta:
                print(f"  - Train Loss: {meta['train_loss']:.4f}")

        print(f"模型路径:    {os.path.join(self.run_dir, info['best_model_path'])}")
        print(f"投影头路径:  {os.path.join(self.run_dir, info['best_proj_path'])}")

        if "last_updated" in info:
            print(f"更新时间:    {info['last_updated']}")

        print(f"{'='*60}\n")

    def _get_default_info(self) -> Dict[str, Any]:
        """获取默认的最佳模型信息"""
        return {
            "best_acc": 0.0,
            "best_epoch": -1,
            "best_model_path": "",
            "best_proj_path": "",
            "stage": "none",
            "last_updated": datetime.now().isoformat()
        }
