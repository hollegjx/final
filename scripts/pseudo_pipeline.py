#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三阶段串行 orchestrator：
    1) 预热训练到 stop_at_epoch；
    2) 离线 SSDDBC 生成伪标签；
    3) 携带伪标签继续训练。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXEC = sys.executable


def _run(cmd, cwd=None):
    print("🚀 运行命令:")
    print("    ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=os.environ.copy())


def run_stage1(args, run_dir: Path):
    log_dir = run_dir / "log"
    ckpt_dir = run_dir / "checkpoints" / args.superclass_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_EXEC,
        "scripts/train_superclass.py",
        "--superclass_name", args.superclass_name,
        "--epochs", str(args.stage1_epochs),
        "--stop_at_epoch", str(args.stage1_epochs),
        "--save_ckpt_every", str(args.stage1_epochs),
        "--save_features_and_exit",
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--gpu", str(args.gpu),
        "--prop_train_labels", str(args.prop_train_labels),
        "--seed", str(args.seed),
        "--reuse_log_dir", str(log_dir),
        "--exp_root", str(run_dir),
    ]
    _run(cmd)
    latest_ckpt = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))[-1]
    return latest_ckpt, log_dir


def run_stage2(args, ckpt_path: Path, run_dir: Path):
    pseudo_dir = run_dir / "pseudo_labels"
    pseudo_dir.mkdir(exist_ok=True)
    cmd = [
        PYTHON_EXEC,
        "scripts/offline_ssddbc_superclass.py",
        "--superclass_name", args.superclass_name,
        "--ckpt_path", str(ckpt_path),
        "--feature_cache_dir", args.feature_cache_dir,
        "--pseudo_output_dir", str(pseudo_dir),
    ]
    _run(cmd)
    newest = sorted(pseudo_dir.glob("*.npz"))[-1]
    return newest


def run_feature_cache_for_ckpt(args, ckpt_path: Path):
    cmd = [
        PYTHON_EXEC,
        "scripts/cache_features.py",
        "--superclass_name", args.superclass_name,
        "--model_path", str(ckpt_path),
        "--auto_find_best", "False",
        "--cache_dir", args.feature_cache_dir,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--gpu", str(args.gpu),
        "--prop_train_labels", str(args.prop_train_labels),
        "--seed", str(args.seed),
        "--overwrite",
    ]
    _run(cmd)


def run_stage3(args, ckpt_path: Path, pseudo_path: Path, log_dir: Path,
              run_dir: Path, start_epoch: int, end_epoch: int):
    epochs_to_train = max(end_epoch - start_epoch, 0)
    if epochs_to_train <= 0:
        return ckpt_path
    target_epoch = end_epoch
    save_every = max(1, args.update_interval)
    cmd = [
        PYTHON_EXEC,
        "scripts/train_superclass.py",
        "--superclass_name", args.superclass_name,
        "--resume_from_ckpt", str(ckpt_path),
        "--pseudo_labels_path", str(pseudo_path),
        "--reuse_log_dir", str(log_dir),
        "--epochs", str(target_epoch),
        "--save_ckpt_every", str(save_every),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--gpu", str(args.gpu),
        "--prop_train_labels", str(args.prop_train_labels),
        "--seed", str(args.seed),
        "--exp_root", str(run_dir),
    ]
    _run(cmd)
    ckpt_dir = run_dir / "checkpoints" / args.superclass_name
    latest_ckpt = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))[-1]
    return latest_ckpt


def main():
    parser = argparse.ArgumentParser(description="三阶段伪标签训练管线")
    parser.add_argument("--superclass_name", required=True)
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--update_interval", type=int, default=5)
    parser.add_argument("--total_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prop_train_labels", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--feature_cache_dir", type=str, default="/data/gjx/checkpoints/features1")
    parser.add_argument("--runs_root", type=str, default="runs_pipeline")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_root) / args.superclass_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Stage1: 预热到 epoch {args.stage1_epochs} ===")
    ckpt_path, log_dir = run_stage1(args, run_dir)
    feature_cache_ready = True  # Stage1 已导出特征
    current_epoch = args.stage1_epochs

    while current_epoch < args.total_epochs:
        if not feature_cache_ready:
            run_feature_cache_for_ckpt(args, ckpt_path)
            feature_cache_ready = True
        print(f"=== Stage2: 离线聚类 (当前 epoch {current_epoch}) ===")
        pseudo_path = run_stage2(args, ckpt_path, run_dir)
        next_epoch = min(current_epoch + args.update_interval, args.total_epochs)
        print(f"=== Stage3: 伪标签续训 {current_epoch} -> {next_epoch} ===")
        ckpt_path = run_stage3(
            args,
            ckpt_path,
            pseudo_path,
            log_dir,
            run_dir,
            start_epoch=current_epoch,
            end_epoch=next_epoch,
        )
        current_epoch = next_epoch
        feature_cache_ready = False
        print(f"✅ 已完成到 epoch {current_epoch}")

    print(f"✅ pipeline 完成，运行目录: {run_dir}")


if __name__ == "__main__":
    main()
