#!/usr/bin/env python3
"""
调试工具：读取 run_dir/debug/epoch_xxx/results_grid.json，穷举 L1/L2 权重并筛选 ACC。

用途：
- 基于调试模式生成的 results_grid.json（包含 score、各 ACC、L1/L2 分量、簇数）
- 穷举给定总权重下的 (w_l1, w_sep, w_sil, w_pen) 组合（非负整数，和为 weight_sum）
- 计算新的加权分数（-w_l1*l1_loss + w_sep*separation + w_sil*silhouette - w_pen*penalty）
- 根据 all_acc 阈值筛选；对每组权重只保留加权分最高的 (k, dp) 结果，输出到同目录下的 txt 报告（默认：results_grid_weight_scan.txt）

注意：
- 仅用于调试，不用于正式评估；数据来源于调试开关 --debug_cluster_heatmap 生成的 JSON。
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from typing import Dict, Tuple, Any, List


def load_results_grid(json_path: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    parsed: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for key, metrics in raw.items():
        # key 格式 "k3_dp40" 或 "k=3,dp=40" 均容忍
        if key.startswith("k") and "_dp" in key:
            parts = key.replace("k", "").split("_dp")
            k_val = int(parts[0].replace("=", ""))
            dp_val = int(parts[1].replace("=", ""))
        elif "dp" in metrics and "k" in metrics:
            k_val = int(metrics["k"])
            dp_val = int(metrics["dp"])
        else:
            raise ValueError(f"无法解析键 {key}，期望形如 k3_dp40")
        parsed[(k_val, dp_val)] = metrics
    return parsed


def enumerate_weights(weight_sum: int):
    # 穷举非负整数解 w_l1 + w_sep + w_sil + w_pen = weight_sum
    for w_l1 in range(weight_sum + 1):
        for w_sep in range(weight_sum - w_l1 + 1):
            for w_sil in range(weight_sum - w_l1 - w_sep + 1):
                w_pen = weight_sum - w_l1 - w_sep - w_sil
                yield (w_l1, w_sep, w_sil, w_pen)


def score_with_weights(metrics: Dict[str, Any], weights: Tuple[int, int, int, int]) -> float:
    w_l1, w_sep, w_sil, w_pen = weights
    return (
        -w_l1 * metrics.get("l1_loss", 0.0)  # L1: 最小化，取负号
        + w_sep * metrics.get("separation_score", 0.0)  # separation: maximize
        + w_sil * metrics.get("silhouette", 0.0)  # silhouette: maximize
        - w_pen * metrics.get("penalty_score", 0.0)  # penalty: minimize，取负
    )


def score_check(metrics: Dict[str, Any]) -> float:
    """使用硬编码公式 -l1 + 3*silhouette 计算分数，用于对照 JSON 内的 score。"""
    return -1.0 * metrics.get("l1_loss", 0.0) + 3.0 * metrics.get("silhouette", 0.0)


def main():
    parser = argparse.ArgumentParser(description="从 results_grid.json 穷举权重筛选 ACC（调试用）")
    parser.add_argument("--grid_dir", required=True, help="调试目录，例如 run_dir/debug/epoch_010")
    parser.add_argument("--weight_sum", type=int, default=10, help="权重总和（非负整数，默认10）")
    parser.add_argument("--all_acc_thresh", type=float, default=0.0, help="all_acc 阈值，低于阈值的组合不输出")
    parser.add_argument("--top_k", type=int, default=50, help="输出前多少条（按 all_acc 降序，再按 score 降序）")
    args = parser.parse_args()

    json_path = os.path.join(args.grid_dir, "results_grid.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"找不到 results_grid.json: {json_path}")

    results = load_results_grid(json_path)

    # 评估所有权重组合：对每一组权重只保留 score_weighted 最高的 (k, dp)
    candidates: List[Tuple[Tuple[int, int, int, int], Tuple[int, int], Dict[str, Any], float]] = []
    for weights in enumerate_weights(args.weight_sum):
        best_for_weight: Tuple[Tuple[int, int, int, int], Tuple[int, int], Dict[str, Any], float] | None = None
        for (k, dp), metrics in results.items():
            all_acc = metrics.get("all_acc")
            if all_acc is None or all_acc < args.all_acc_thresh:
                continue
            new_score = score_with_weights(metrics, weights)
            if (
                best_for_weight is None
                or new_score > best_for_weight[3]
                or (new_score == best_for_weight[3] and (all_acc or 0.0) > (best_for_weight[2].get("all_acc") or 0.0))
            ):
                best_for_weight = (weights, (k, dp), metrics, new_score)
        if best_for_weight is not None:
            candidates.append(best_for_weight)

    # 排序：all_acc 降序，其次 new_score 降序
    candidates.sort(key=lambda x: (x[2].get("all_acc", 0.0), x[3]), reverse=True)
    if args.top_k > 0:
        candidates = candidates[:args.top_k]

    out_path = os.path.join(args.grid_dir, "results_grid_weight_scan.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 调试权重扫描（weight_sum={args.weight_sum}, all_acc_thresh={args.all_acc_thresh})\n")
        f.write("# 格式: weights(w_l1,w_sep,w_sil,w_pen) | k,dp | all/old/new_acc | score(orig) | score_weighted | score_check(-l1+3*sil) | n_clusters | l1/separation/silhouette/penalty\n\n")
        for weights, (k, dp), m, new_score in candidates:
            sc_check = score_check(m)
            sc_orig = m.get("score")
            if sc_orig is not None and abs(sc_check - sc_orig) > 1e-6:
                print(f"⚠️  score 不一致: k={k}, dp={dp}, score(orig)={sc_orig:.6f}, score_check={sc_check:.6f}")
            line = (
                f"w={weights} | k={k},dp={dp} | "
                f"all={m.get('all_acc'):.4f}, old={m.get('old_acc')}, new={m.get('new_acc')} | "
                f"score_orig={sc_orig:.4f} | score_weighted={new_score:.4f} | score_check={sc_check:.4f} | "
                f"n_clusters={m.get('n_clusters')} | "
                f"l1={m.get('l1_loss')}, sep={m.get('separation_score')}, "
                f"sil={m.get('silhouette')}, pen={m.get('penalty_score')}"
            )
            f.write(line + "\n")

    print(f"✅ 扫描完成，输出: {out_path}")


if __name__ == "__main__":
    main()
