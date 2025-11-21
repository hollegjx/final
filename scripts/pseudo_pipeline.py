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

# 添加项目根目录到路径以导入工具模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.best_model_tracker import BestModelTracker
from config import feature_cache_dir as DEFAULT_FEATURE_CACHE_DIR


def _run(cmd, cwd=None):
    print("🚀 运行命令:")
    print("    ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=os.environ.copy())


def run_stage1(args, run_dir: Path):
    log_dir = run_dir / "log"
    ckpt_dir = run_dir / "checkpoints" / args.superclass_name
    features_dir = run_dir / "features"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_EXEC,
        "scripts/train_superclass.py",
        "--superclass_name", args.superclass_name,
        "--epochs", str(args.total_epochs),
        "--stop_at_epoch", str(args.stage1_epochs),
        "--save_ckpt_every", str(args.stage1_epochs),
        "--save_features_and_exit",
        "--feature_cache_dir", str(features_dir),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--gpu", str(args.gpu),
        "--prop_train_labels", str(args.prop_train_labels),
        "--seed", str(args.seed),
        "--reuse_log_dir", str(log_dir),
        "--exp_root", str(run_dir),
        # 🆕 超参数配置
        "--lr", str(args.lr),
        "--grad_from_block", str(args.grad_from_block),
        "--sup_con_weight", str(args.sup_con_weight),
        "--momentum", str(args.momentum),
        "--weight_decay", str(args.weight_decay),
        "--contrast_unlabel_only", args.contrast_unlabel_only,
        "--temperature", str(args.temperature),
    ]
    _run(cmd)
    ckpts = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))
    if not ckpts:
        raise RuntimeError(f"Stage1 训练失败：未在 {ckpt_dir} 生成 checkpoint 文件")
    latest_ckpt = ckpts[-1]
    return latest_ckpt, log_dir


def run_stage2(args, ckpt_path: Path, run_dir: Path):
    pseudo_dir = run_dir / "pseudo_labels"
    features_dir = run_dir / "features"
    pseudo_dir.mkdir(exist_ok=True)
    cmd = [
        PYTHON_EXEC,
        "scripts/offline_ssddbc_superclass.py",
        "--superclass_name", args.superclass_name,
        "--ckpt_path", str(ckpt_path),
        "--feature_cache_dir", str(features_dir),
        "--pseudo_output_dir", str(pseudo_dir),
        "--skip_feature_extraction",  # 🆕 跳过特征提取，直接使用缓存
    ]
    _run(cmd)
    npz_files = sorted(pseudo_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"Stage2 聚类失败：未在 {pseudo_dir} 生成伪标签文件")
    newest = npz_files[-1]
    return newest


def run_feature_cache_for_ckpt(args, ckpt_path: Path, run_dir: Path):
    features_dir = run_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_EXEC,
        "scripts/cache_features.py",
        "--superclass_name", args.superclass_name,
        "--model_path", str(ckpt_path),
        "--auto_find_best", "False",
        "--cache_dir", str(features_dir),
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
    save_every = max(1, args.update_interval)
    cmd = [
        PYTHON_EXEC,
        "scripts/train_superclass.py",
        "--superclass_name", args.superclass_name,
        "--resume_from_ckpt", str(ckpt_path),
        "--pseudo_labels_path", str(pseudo_path),
        "--pseudo_weight_mode", args.pseudo_weight_mode,
        "--pseudo_loss_weight", str(args.pseudo_loss_weight),
        "--pseudo_for_labeled_mode", args.pseudo_for_labeled_mode,
        "--warmup_epochs", str(args.stage1_epochs),  # 🆕 使用 stage1_epochs 作为 warmup_epochs
        "--reuse_log_dir", str(log_dir),
        "--epochs", str(args.total_epochs),
        "--stop_at_epoch", str(end_epoch),  # 训练到 end_epoch-1，保持区间长度一致
        "--save_ckpt_every", str(save_every),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--gpu", str(args.gpu),
        "--prop_train_labels", str(args.prop_train_labels),
        "--seed", str(args.seed),
        "--exp_root", str(run_dir),
        # 🆕 超参数配置
        "--lr", str(args.lr),
        "--grad_from_block", str(args.grad_from_block),
        "--sup_con_weight", str(args.sup_con_weight),
        "--momentum", str(args.momentum),
        "--weight_decay", str(args.weight_decay),
        "--contrast_unlabel_only", args.contrast_unlabel_only,
        "--temperature", str(args.temperature),
    ]
    _run(cmd)
    ckpt_dir = run_dir / "checkpoints" / args.superclass_name
    ckpts = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))
    if not ckpts:
        raise RuntimeError(f"Stage3 训练失败：未在 {ckpt_dir} 生成 checkpoint 文件")
    latest_ckpt = ckpts[-1]
    return latest_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="三阶段伪标签训练管线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 基础配置
    parser.add_argument("--superclass_name", required=True,
                        help="要训练的超类名称（如 trees）")
    parser.add_argument("--stage1_epochs", type=int, default=50,
                        help="Stage1 预热训练轮数（无伪标签）")
    parser.add_argument("--update_interval", type=int, default=5,
                        help="伪标签更新间隔，每 N 轮重新聚类")
    parser.add_argument("--total_epochs", type=int, default=200,
                        help="总训练轮数（包含预热和续训）")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="训练批次大小")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader 工作线程数")
    parser.add_argument("--gpu", type=int, default=0,
                        help="使用的 GPU 设备编号")
    parser.add_argument("--prop_train_labels", type=float, default=0.8,
                        help="已知类样本中用于训练的比例")
    parser.add_argument("--seed", type=int, default=1,
                        help="随机种子，用于保证实验可复现")
    parser.add_argument("--feature_cache_dir", type=str, default=DEFAULT_FEATURE_CACHE_DIR,
                        help="特征缓存根目录")
    parser.add_argument("--runs_root", type=str, default="runs_pipeline",
                        help="Pipeline 运行输出根目录")
    parser.add_argument("--resume_run_dir", type=str, default=None,
                        help="从已有任务目录恢复（支持断点续训）")
    parser.add_argument("--pseudo_weight_mode", type=str, default="none",
                        choices=["none", "density"],
                        help="阶段3训练使用的伪标签加权模式")
    parser.add_argument("--pseudo_loss_weight", type=float, default=1.0,
                        help="伪标签损失的整体权重系数 λ，最终权重 = γ × λ（默认: 1.0）")
    parser.add_argument("--pseudo_for_labeled_mode", type=str, default="off",
                        choices=["off", "all"],
                        help="伪标签损失的样本范围：off=仅未标注样本（默认），all=已标注与未标注一起参与")

    # 🆕 训练超参数配置
    parser.add_argument("--lr", type=float, default=0.1,
                        help="初始学习率 (默认: 0.1)")
    parser.add_argument("--grad_from_block", type=int, default=11,
                        help="ViT 解冻起始 block，范围 0-11 (默认: 11，仅解冻最后一层)")
    parser.add_argument("--sup_con_weight", type=float, default=0.5,
                        help="监督对比损失权重，范围 0-1 (默认: 0.5)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD 动量系数 (默认: 0.9)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减系数（L2正则化） (默认: 1e-4)")
    parser.add_argument("--contrast_unlabel_only", type=str, default="False",
                        help="是否仅对无标签样本计算对比损失 (默认: False)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="对比学习温度系数 (默认: 1.0)")

    args = parser.parse_args()

    # 初始化/恢复运行目录
    # 恢复或初始化运行目录
    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir)
        log_dir = run_dir / "log"
        ckpt_dir = run_dir / "checkpoints" / args.superclass_name
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"恢复失败：未找到 ckpt 目录 {ckpt_dir}")
        ckpts = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"恢复失败：{ckpt_dir} 下没有 ckpt_epoch_*.pt")
        ckpt_path = ckpts[-1]
        current_epoch = int(ckpt_path.stem.split("_")[-1])
        feature_cache_ready = False  # 恢复时需要重新生成特征缓存
        print(f"🔁 从 {run_dir} 恢复: ckpt={ckpt_path.name}, current_epoch={current_epoch}")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.runs_root) / args.superclass_name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Stage1: 预热到 epoch {args.stage1_epochs} ===")
        ckpt_path, log_dir = run_stage1(args, run_dir)
        feature_cache_ready = True  # Stage1 已导出特征
        current_epoch = args.stage1_epochs

    # 定义目录路径（确保在循环中可用）
    ckpt_dir = run_dir / "checkpoints" / args.superclass_name
    pseudo_dir = run_dir / "pseudo_labels"

    while current_epoch < args.total_epochs:
        if not feature_cache_ready:
            run_feature_cache_for_ckpt(args, ckpt_path, run_dir)
            feature_cache_ready = True

        # 因为 checkpoint 只在更新点保存，current_epoch 一定是更新点
        # 所以 pseudo_base_epoch = current_epoch，next_epoch = current_epoch + interval
        print(f"=== Stage2: 离线聚类 (epoch {current_epoch}) ===")
        pseudo_dir.mkdir(exist_ok=True)

        # 查找对应 epoch 的伪标签
        existing_pseudo = sorted(pseudo_dir.glob(f"*epoch_{current_epoch:03d}*.npz"))
        if existing_pseudo:
            pseudo_path = existing_pseudo[-1]
            print(f"   ↪ 复用已有伪标签: {pseudo_path.name}")
        else:
            # 缺少伪标签，使用当前 checkpoint 重新聚类
            print(f"   ↪ 未找到该 epoch 的伪标签，使用 {ckpt_path.name} 生成新的伪标签")
            pseudo_path = run_stage2(args, ckpt_path, run_dir)

        # 计算下一个训练终点
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

    # 显示全局最佳模型信息
    tracker = BestModelTracker(str(run_dir))
    tracker.print_summary()


if __name__ == "__main__":
    main()
