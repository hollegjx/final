#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线 SSDDBC 管线封装脚本（单超类）

设计目标：
- 在「保存好 ckpt」之后，离线执行的流程应当与原先手动运行的
  `scripts/cache_features.py` + `python -m ssddbc.grid_search.batch_runner` 完全一致：
    1) 使用指定 ckpt 提取并缓存特征；
    2) 调用 batch_runner 在特征缓存上做网格搜索。

特点：
- 本脚本本身不做特征计算和聚类实现，只是顺序调用已有脚本；
- 提特征阶段在一个进程里完成，结束后进程退出；
- 聚类阶段由 batch_runner 独立进程完成，行为与过去使用方式一致。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import feature_cache_dir as DEFAULT_FEATURE_CACHE_DIR
from utils.data.feature_loader import FeatureLoader
from utils.pseudo_labels import save_pseudo_labels
from ssddbc.grid_search.api import run_clustering_search_on_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线 SSDDBC：ckpt -> 特征缓存 -> batch_runner（单超类）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必要参数
    parser.add_argument(
        "--superclass_name",
        type=str,
        required=True,
        help="要处理的超类名称，如 'trees'。",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="训练阶段保存的完整 ckpt 路径（模型+投影头），将作为特征提取用的模型。",
    )

    # 特征缓存参数（保持与原有 scripts/cache_features.py 一致）
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default=DEFAULT_FEATURE_CACHE_DIR,
        help="特征缓存根目录（与之前的 FEATURE_CACHE_DIR 一致）。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="特征提取批大小（沿用 cache_features 的默认）。",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="特征提取 DataLoader 的工作线程数。",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="特征提取所用 GPU 编号。",
    )
    parser.add_argument(
        "--prop_train_labels",
        type=float,
        default=0.8,
        help="训练集中有标签样本占比（需与训练时一致）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="特征提取阶段的数据划分随机种子（与 cache_features/test_feature 保持一致）。",
    )

    # SSDDBC 网格搜索参数（保持原来 batch_runner 的使用方式）
    parser.add_argument("--k_min", type=int, default=3, help="KNN k 最小值（包含）。")
    parser.add_argument("--k_max", type=int, default=21, help="KNN k 最大值上界（不含）。")
    parser.add_argument(
        "--density_min",
        type=int,
        default=40,
        help="密度百分位最小值（包含）。",
    )
    parser.add_argument(
        "--density_max",
        type=int,
        default=100,
        help="密度百分位最大值上界（不含）。",
    )
    parser.add_argument(
        "--density_step",
        type=int,
        default=5,
        help="密度百分位步长。",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="batch_runner 并行进程数上限（None 表示由脚本自行决定，使用 CPU 一半核心）。",
    )
    parser.add_argument(
        "--pseudo_output_dir",
        type=str,
        default=None,
        help="伪标签输出目录（默认写入 feature_cache_dir/<superclass_name>/pseudo_labels）。",
    )
    parser.add_argument(
        "--skip_feature_extraction",
        action="store_true",
        help="跳过特征提取阶段，直接使用已有缓存（适用于 pipeline 场景）。",
    )

    return parser


def run_cache_features(args: argparse.Namespace) -> None:
    """
    调用 scripts/cache_features.py，使用指定 ckpt 提取并缓存特征。
    """
    cmd = [
        sys.executable,
        "scripts/cache_features.py",
        "--superclass_name",
        args.superclass_name,
        "--model_path",
        args.ckpt_path,
        "--auto_find_best",
        "False",
        "--cache_dir",
        args.feature_cache_dir,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--gpu",
        str(args.gpu),
        "--prop_train_labels",
        str(args.prop_train_labels),
        "--seed",
        str(args.seed),
    ]

    print("🚀 [Stage 1/2] 使用 ckpt 提取并缓存特征：")
    print("    ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _build_pseudo_metadata(args: argparse.Namespace, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ckpt_path": args.ckpt_path,
        "superclass": args.superclass_name,
        "feature_cache_dir": args.feature_cache_dir,
        "generated_at": datetime.now().isoformat(),
        **result,
    }


def _resolve_pseudo_output_path(args: argparse.Namespace, filename: str) -> str:
    if args.pseudo_output_dir:
        base_dir = args.pseudo_output_dir
    else:
        base_dir = os.path.join(args.feature_cache_dir, args.superclass_name, "pseudo_labels")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)


def _load_cached_features(args: argparse.Namespace) -> Dict[str, Any]:
    loader = FeatureLoader(cache_base_dir=args.feature_cache_dir)
    feature_dict = loader.load(args.superclass_name, use_l2=True, silent=True)
    if feature_dict is None:
        cache_path = loader.get_cache_path(args.superclass_name, use_l2=True)
        raise FileNotFoundError(f"找不到特征缓存，请确认 Stage1 成功完成: {cache_path}")
    return feature_dict


def run_offline_clustering(args: argparse.Namespace) -> str:
    """
    读取缓存特征并执行内存级 SSDDBC 搜索，生成伪标签文件。
    """
    feature_dict = _load_cached_features(args)
    features = feature_dict["all_features"]
    targets = feature_dict["all_targets"]
    known_mask = feature_dict["all_known_mask"]
    labeled_mask = feature_dict["all_labeled_mask"]
    indices = feature_dict.get("all_indices")
    if indices is None:
        print("⚠️  缓存缺少 all_indices 字段，将默认使用顺序索引。建议重新生成缓存。")
        indices = np.arange(features.shape[0], dtype=np.int64)

    # 显示搜索配置
    k_range = range(args.k_min, args.k_max)
    density_range = range(args.density_min, args.density_max, args.density_step)
    n_configs = len(list(k_range)) * len(list(density_range))

    mode_str = f"并行模式 (max_workers={args.max_workers})" if args.max_workers != 1 else "单进程模式"
    print(f"🚀 [Stage 2/2] 在缓存特征上执行 SSDDBC 网格搜索 ({mode_str})...")
    print(f"   搜索空间: k={list(k_range)}, density={list(density_range)} (共 {n_configs} 个配置)")

    search_result = run_clustering_search_on_features(
        features=features,
        targets=targets,
        known_mask=known_mask,
        labeled_mask=labeled_mask,
        k_range=k_range,
        density_range=density_range,
        random_state=0,
        silent=True,
        max_workers=args.max_workers,
    )

    core_mask = np.zeros_like(indices, dtype=bool)
    if len(search_result.core_points) > 0:
        core_mask[search_result.core_points] = True

    ckpt_base = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    filename = (
        f"{args.superclass_name}_{ckpt_base}"
        f"_k{search_result.best_params['k']}_dp{search_result.best_params['density_percentile']}.npz"
    )
    pseudo_path = _resolve_pseudo_output_path(args, filename)

    metadata = _build_pseudo_metadata(
        args,
        {
            "score": search_result.loss,  # 字段名保持loss，但实际是score（越大越好）
            "n_clusters": search_result.n_clusters,
            "num_core_points": int(core_mask.sum()),
        },
    )

    save_pseudo_labels(
        pseudo_path,
        indices=indices,
        labels=search_result.labels,
        core_mask=core_mask,
        best_params=search_result.best_params,
        metadata=metadata,
        densities=search_result.densities,
    )

    print(
        f"✅ 伪标签已保存: {pseudo_path}\n"
        f"   Score = {search_result.loss:.4f} (越大越好)\n"
        f"   核心点: {core_mask.sum()} / {len(core_mask)} | 最佳参数: {search_result.best_params}"
    )
    return pseudo_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 规范化路径
    args.feature_cache_dir = os.path.abspath(args.feature_cache_dir)
    args.ckpt_path = os.path.abspath(args.ckpt_path)

    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"找不到指定 ckpt 文件: {args.ckpt_path}")

    # 条件性执行特征提取
    if args.skip_feature_extraction:
        print("⏭️  跳过特征提取阶段（使用已有缓存）")
        # 验证缓存是否存在
        from utils.data.feature_loader import FeatureLoader
        loader = FeatureLoader(cache_base_dir=args.feature_cache_dir)
        cache_path = loader.get_cache_path(args.superclass_name, use_l2=True)
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(
                f"❌ 缓存文件不存在: {cache_path}\n"
                f"提示: 请先运行特征提取，或移除 --skip_feature_extraction 参数"
            )
        print(f"✅ 找到特征缓存: {cache_path}")
    else:
        run_cache_features(args)

    pseudo_path = run_offline_clustering(args)

    print("\n✅ 离线 SSDDBC 管线执行完成（提特征 + SSDDBC 网格搜索）。")
    print(f"📁 伪标签文件: {pseudo_path}")


if __name__ == "__main__":
    main()
