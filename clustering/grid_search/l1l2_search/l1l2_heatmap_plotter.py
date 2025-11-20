#!/usr/bin/env python3
"""L1+L2权重热力图绘制模块

职责：
- 对标 L2：按权重配置 (w_l1, w_sep, w_sil) 生成热力图，
  背景按 ACC 指标（all/new/old）着色，注释显示 combined_score；
- 生成单组件（l1_loss、separation_score、silhouette）热力图保存在 single_metrics/。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple
import re
from config import grid_search_output_dir, l1l2_search_output_dir

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from multiprocessing import Pool

try:  # 与 L2 保持一致的进度条依赖
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from .l1l2_weight_calculator import (
    WeightTriplet,
    compute_weighted_l1l2,
    load_raw_scores,
    validate_l1l2_requirements,
)

# 旧流程（切片模式）已移除以减少冗余


# ---------------------------------
# 新流程：对标 L2 的权重配置热力图
# ---------------------------------

def plot_weighted_l1l2_heatmap(
    weighted_results: Dict[Tuple[int, int], Dict],
    color_metric: str,
    superclass_name: str,
    output_dir: str,
    weights: WeightTriplet,
) -> Dict[str, float]:
    """绘制单个权重配置下的加权热力图（对标 L2）。

    - 背景：color_metric（all_acc/new_acc/old_acc），越高越好（配色方向使用 'viridis'）。
    - 注释：combined_score（越小越好）。
    - 保存路径：{output_dir}/{superclass}/{all|new|old}/color_metric_*.png
    """

    if not weighted_results:
        return {}

    # 目录映射
    metric_dir_map = {'all_acc': 'all', 'new_acc': 'new', 'old_acc': 'old'}
    metric_short = metric_dir_map.get(color_metric, color_metric)
    superclass_output_dir = os.path.join(output_dir, superclass_name, metric_short)
    os.makedirs(superclass_output_dir, exist_ok=True)

    # 网格维度
    k_values = sorted({k for k, _ in weighted_results.keys()})
    dp_values = sorted({dp for _, dp in weighted_results.keys()})
    if not k_values or not dp_values:
        return {}

    color_data = np.full((len(dp_values), len(k_values)), np.nan)
    display_data = np.full((len(dp_values), len(k_values)), np.nan)

    # 选择最优点（combined_score 越小越好）
    best: Optional[Tuple[float, int, int, float]] = None  # (combined, i, j, color_value)
    valid_entries = []

    for i, dp in enumerate(dp_values):
        for j, k in enumerate(k_values):
            entry = weighted_results.get((k, dp)) or {}
            color_val = entry.get(color_metric)
            comb = entry.get('combined_score')
            if color_val is not None:
                color_data[i, j] = float(color_val)
            if comb is not None:
                display_data[i, j] = float(comb)
                valid_entries.append((comb, i, j))
                if best is None or comb < best[0]:
                    best = (comb, i, j, float(color_val) if color_val is not None else np.nan)

    if not valid_entries:
        return {}

    fig, ax = plt.subplots(figsize=(12, 8))
    # color_data 背景，display_data 注释
    cbar_label = f'{color_metric} (background)'
    sns.heatmap(
        color_data,
        xticklabels=k_values,
        yticklabels=dp_values,
        annot=display_data,
        fmt='.4f',
        cmap='viridis',
        cbar_kws={'label': cbar_label},
        ax=ax
    )

    best_k = best_dp = None
    best_combined = None
    best_color_metric_value = None
    if best:
        best_combined, i, j, cval = best
        best_k = k_values[j]
        best_dp = dp_values[i]
        ax.plot(j + 0.5, i + 0.5, marker='*', markersize=16, color='red',
                markeredgecolor='white', markeredgewidth=1.2)
        best_color_metric_value = cval
    if best_color_metric_value is None:
        best_color_metric_value = 0.0

    weight_str = f"{int(weights.w_l1)}_{int(weights.w_sep)}_{int(weights.w_sil)}"
    title_detail = (
        f"Best: k={best_k}, dp={best_dp}, {color_metric}={best_color_metric_value:.2f}, combined={best_combined:.4f}"
        if best_k is not None else "Best: N/A"
    )
    ax.set_title(
        f"Weighted L1+L2 (w_l1={int(weights.w_l1)}, w_sep={int(weights.w_sep)}, w_sil={int(weights.w_sil)}) - {superclass_name}\n"
        f"{title_detail}",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('density_percentile', fontsize=12)

    plt.tight_layout()
    filename = f"{color_metric}_{best_color_metric_value:.4f}_{weight_str}.png"
    output_path = os.path.join(superclass_output_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    summary = {
        'file': output_path,
        'entries': len(valid_entries),
        'min_combined': float(np.nanmin(display_data)),
        'max_combined': float(np.nanmax(display_data))
    }
    if best:
        score, i, j, _ = best
        summary.update({
            'best_combined': score,
            'best_k': k_values[j],
            'best_density_percentile': dp_values[i],
            f'best_{color_metric}': best_color_metric_value
        })
    return summary


def _plot_single_weight_task(
    raw_results: Dict[Tuple[int, int], Dict],
    weights: WeightTriplet,
    superclass_name: str,
    output_dir: str,
    color_metrics: List[str],
    available_components: List[str],
) -> Dict[str, object]:
    try:
        weighted = compute_weighted_l1l2(raw_results, weights, available_components)
        if not weighted:
            return {'error': 'weighted_empty', 'weights': weights}
        summaries = []
        for metric in color_metrics:
            s = plot_weighted_l1l2_heatmap(
                weighted_results=weighted,
                color_metric=metric,
                superclass_name=superclass_name,
                output_dir=output_dir,
                weights=weights,
            )
            if s:
                summaries.append(s)
        return {
            'weights': (int(weights.w_l1), int(weights.w_sep), int(weights.w_sil)),
            'summaries': summaries,
        }
    except Exception as exc:  # pragma: no cover
        return {'error': str(exc), 'weights': weights}


def create_weighted_l1l2_heatmaps(
    raw_results: Dict[Tuple[int, int], Dict],
    weight_sets: List[WeightTriplet],
    superclass_name: str,
    output_dir: str,
    color_metrics: List[str],
    available_components: List[str],
    num_workers: Optional[int] = None,
) -> List[Dict[str, object]]:
    """批量生成每个权重配置在各个 ACC 背景下的热力图。"""
    if not weight_sets:
        return []
    cpu_count = os.cpu_count() or 2
    max_default = max(1, cpu_count - 1)
    if num_workers is None:
        num_workers = max_default
    num_workers = max(1, min(num_workers, cpu_count))

    results: List[Dict[str, object]] = []
    if num_workers <= 1:
        for w in weight_sets:
            out = _plot_single_weight_task(raw_results, w, superclass_name, output_dir, color_metrics, available_components)
            if out and not out.get('error'):
                results.append(out)
        return results

    task_args = [
        (raw_results, w, superclass_name, output_dir, color_metrics, available_components) for w in weight_sets
    ]
    with Pool(processes=num_workers) as pool:
        iterator = pool.starmap(_plot_single_weight_task, task_args)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(task_args), desc=f"Heatmaps {superclass_name}")
        for out in iterator:
            if isinstance(out, dict) and out.get('error'):
                print(f"⚠️ 权重 {out.get('weights')} 生成失败: {out['error']}")
                continue
            if out:
                results.append(out)
    return results


def create_l1l2_component_heatmap(
    results_dict: Dict[Tuple[int, int], Dict],
    component_name: str,
    color_metric: str,
    superclass_name: str,
    output_dir: str,
) -> Dict:
    """绘制单组件热力图到 single_metrics/，对标 L2。"""
    component_name = component_name.strip()
    if not component_name:
        return {}

    k_values = sorted({k for k, _ in results_dict.keys()})
    dp_values = sorted({dp for _, dp in results_dict.keys()})
    if not k_values or not dp_values:
        return {}

    color_data = np.full((len(dp_values), len(k_values)), np.nan)
    display_data = np.full((len(dp_values), len(k_values)), np.nan)

    # 组件取值及方向
    def get_component_value(metrics: Dict) -> Optional[float]:
        if component_name == 'l1_loss':
            val = metrics.get('l1_loss')
            return float(val) if val is not None else None
        if component_name == 'separation_score':
            from .l1l2_weight_calculator import _extract_separation  # 兼容新旧格式
            val = _extract_separation(metrics)
            return float(val) if val is not None else None
        if component_name == 'silhouette':
            from .l1l2_weight_calculator import _extract_silhouette  # 局部导入避免循环
            val = _extract_silhouette(metrics)
            return float(val) if val is not None else None
        return None

    for i, dp in enumerate(dp_values):
        for j, k in enumerate(k_values):
            m = results_dict.get((k, dp))
            if not m:
                continue
            cval = get_component_value(m)
            if color_metric in m and m[color_metric] is not None:
                color_data[i, j] = float(m[color_metric])
            if cval is not None:
                display_data[i, j] = float(cval)

    if np.all(np.isnan(display_data)):
        return {}

    # 组件方向：l1_loss 越小越好，其余越大越好，仅影响标题描述
    orientation = 'minimize' if component_name == 'l1_loss' else 'maximize'

    fig, ax = plt.subplots(figsize=(12, 8))
    cbar_label = f'{color_metric} (background)'
    sns.heatmap(
        color_data,
        xticklabels=k_values,
        yticklabels=dp_values,
        annot=display_data,
        fmt='.4f',
        cmap='viridis',
        cbar_kws={'label': cbar_label},
        ax=ax
    )
    ax.set_title(
        f"Component: {component_name} (bg={color_metric}) - {superclass_name}\n"
        f"Annotation: {component_name} value; orientation: {orientation}",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('density_percentile', fontsize=12)

    plt.tight_layout()
    superclass_dir = os.path.join(output_dir, superclass_name, 'single_metrics')
    os.makedirs(superclass_dir, exist_ok=True)
    filename = f"component_{component_name}_colored_by_{color_metric}.png"
    path = os.path.join(superclass_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"📁 组件热力图已保存: {path}")
    return {'file': path}


def create_l1l2_single_metric_heatmap(
    results_dict: Dict[Tuple[int, int], Dict],
    metric: str,
    superclass_name: str,
    output_dir: str,
    save_plots: bool = True,
    higher_is_better: bool = True,
) -> Dict:
    """绘制 ACC 单指标热力图（背景与注释均为同一指标）。"""

    if not results_dict:
        return {}

    k_values = sorted({k for k, _ in results_dict.keys()})
    dp_values = sorted({dp for _, dp in results_dict.keys()})
    if not k_values or not dp_values:
        return {}

    metric_data = np.full((len(dp_values), len(k_values)), np.nan)
    for i, dp in enumerate(dp_values):
        for j, k in enumerate(k_values):
            entry = results_dict.get((k, dp))
            if entry is None:
                continue
            val = entry.get(metric)
            if val is not None:
                metric_data[i, j] = float(val)

    if np.all(np.isnan(metric_data)):
        print(f"⚠️ 指标 {metric} 完全缺失，跳过单指标热力图")
        return {}

    # 选出Top-3
    valid = []
    for i in range(metric_data.shape[0]):
        for j in range(metric_data.shape[1]):
            v = metric_data[i, j]
            if not np.isnan(v):
                valid.append((v, i, j))
    if not valid:
        return {}
    valid.sort(key=lambda x: x[0], reverse=higher_is_better)
    top3 = valid[:3]

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = 'viridis' if higher_is_better else 'viridis_r'
    cbar_label = f"{metric} ({'higher better' if higher_is_better else 'lower better'})"
    sns.heatmap(
        metric_data,
        xticklabels=k_values,
        yticklabels=dp_values,
        annot=True,
        fmt='.4f',
        cmap=cmap,
        cbar_kws={'label': cbar_label},
        ax=ax,
    )
    for rank, (val, i, j) in enumerate(top3, 1):
        ax.plot(j + 0.5, i + 0.5, marker='*', markersize=18,
                color='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                markeredgecolor='white', markeredgewidth=1.0)

    ax.set_title(f'{metric} Single Metric Heatmap - {superclass_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('density_percentile', fontsize=12)
    plt.tight_layout()

    if save_plots:
        superclass_dir = os.path.join(output_dir, superclass_name, 'single_metrics')
        os.makedirs(superclass_dir, exist_ok=True)
        filename = f"single_metric_{metric}.png"
        out_path = os.path.join(superclass_dir, filename)
        plt.savefig(out_path, dpi=300)
        print(f"📁 单指标热力图已保存: {out_path}")
    plt.close(fig)

    return {
        'metric': metric,
        'top3': [(k_values[j], dp_values[i], float(val)) for val, i, j in top3],
        'min': float(np.nanmin(metric_data)),
        'max': float(np.nanmax(metric_data)),
        'mean': float(np.nanmean(metric_data)),
        'std': float(np.nanstd(metric_data)),
    }


def generate_l1l2_report(
    weighted_results: Dict[Tuple[int, int], Dict],
    weights: WeightTriplet,
    superclass_name: str,
    output_path: str,
) -> Tuple[List[Tuple[Tuple[int, int], Dict]], Dict]:
    """生成单个权重配置的文本报告（combined_score 越小越好）。"""
    if not weighted_results:
        print("⚠️ 无有效数据，跳过报告生成")
        return [], {}

    sorted_results = sorted(weighted_results.items(), key=lambda x: x[1]['combined_score'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    values = [d['combined_score'] for _, d in sorted_results]
    import numpy as _np
    stats = {
        'count': len(sorted_results),
        'min_combined': float(min(values)),
        'max_combined': float(max(values)),
        'mean_combined': float(_np.mean(values)),
        'std_combined': float(_np.std(values)),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"L1+L2 权重探索报告 - {superclass_name}\n")
        f.write(
            f"权重配置: w_l1={int(weights.w_l1)}, w_sep={int(weights.w_sep)}, w_sil={int(weights.w_sil)} (sum={int(weights.total)})\n"
        )
        f.write("=" * 60 + "\n\n")
        f.write("统计摘要:\n")
        f.write(f"  参数组合总数: {stats['count']}\n")
        f.write(f"  Combined范围: [{stats['min_combined']:.4f}, {stats['max_combined']:.4f}]\n")
        f.write(f"  Combined均值: {stats['mean_combined']:.4f} ± {stats['std_combined']:.4f}\n\n")
        f.write("参数组合排序（按 Combined 从小到大）:\n\n")
        for rank, (key, data) in enumerate(sorted_results, 1):
            k, dp = key
            f.write(f"#{rank:2d} k={k}, dp={dp}, combined={data['combined_score']:.4f}, all_acc={data.get('all_acc')}, new_acc={data.get('new_acc')}, old_acc={data.get('old_acc')}\n")
    print(f"📄 L1+L2报告已保存: {output_path}")
    return sorted_results[:3], stats


def generate_l1l2_summary_report(
    all_stats: Dict[Tuple[int, int, int], Dict],
    superclass_name: str,
    output_path: str,
) -> None:
    """生成跨权重配置的汇总报告（对标 L2 的 summary）。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"L1+L2 权重探索汇总报告 - {superclass_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write("权重配置统计:\n")
        f.write("-" * 40 + "\n")
        for (wl1, wsep, wsil), stats in all_stats.items():
            f.write(f"配置 w_l1={wl1}, w_sep={wsep}, w_sil={wsil}:\n")
            f.write(f"  参数组合数: {stats.get('count', 0)}\n")
            f.write(f"  Combined范围: [{stats.get('min_combined', float('nan')):.4f}, {stats.get('max_combined', float('nan')):.4f}]\n")
            f.write(f"  Combined均值: {stats.get('mean_combined', float('nan')):.4f} ± {stats.get('std_combined', float('nan')):.4f}\n\n")
    print(f"📄 汇总报告已保存: {output_path}")


def plot_all_l1l2_configurations(
    superclass_name: str,
    search_dir: str = grid_search_output_dir,
    output_dir: str = l1l2_search_output_dir,
    weight_sets: Optional[List[WeightTriplet]] = None,
    color_metrics: Optional[List[str]] = None,
    weight_sum: int = 10,
    cleanup_reports: bool = True,
) -> Dict:
    """统一调度：对标 L2，批量绘制所有权重配置。"""
    if not color_metrics:
        color_metrics = ['new_acc', 'all_acc', 'old_acc']

    print("=" * 60)
    print(f"🎨 L1+L2权重探索 - {superclass_name} (metrics: {','.join(color_metrics)}, weight_sum={weight_sum})")
    print("=" * 60)

    print("\n📂 Step 1: 加载原始搜索数据")
    raw_loaded = load_raw_scores(superclass_name, search_dir)
    if not raw_loaded:
        print(f"❌ 无法加载 {superclass_name} 的搜索结果")
        return {}
    raw_data, available_components = raw_loaded

    valid, info = validate_l1l2_requirements(raw_data, available_components)
    if not valid:
        print(f"❌ 数据验证失败: {info}")
        return {}
    print(f"✅ 数据验证通过: {info}")

    if not weight_sets:
        from .l1l2_weight_calculator import get_weight_configurations
        weight_sets = get_weight_configurations(weight_sum)
    print(f"🔧 Step 2: 处理 {len(weight_sets)} 种权重配置")

    # 每个权重配置生成三张热力图
    _ = create_weighted_l1l2_heatmaps(
        raw_data,
        weight_sets,
        superclass_name,
        output_dir,
        color_metrics=color_metrics,
        available_components=available_components,
        num_workers=None,
    )

    # 单组件热力图（以 new_acc 作为背景）
    try:
        create_l1l2_component_heatmap(raw_data, 'l1_loss', 'new_acc', superclass_name, output_dir)
        create_l1l2_component_heatmap(raw_data, 'separation_score', 'new_acc', superclass_name, output_dir)
        create_l1l2_component_heatmap(raw_data, 'silhouette', 'new_acc', superclass_name, output_dir)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ 单组件热力图生成失败: {exc}")

    # 单ACC指标热力图（背景与注释均同一指标）
    try:
        create_l1l2_single_metric_heatmap(raw_data, 'all_acc', superclass_name, output_dir)
        create_l1l2_single_metric_heatmap(raw_data, 'new_acc', superclass_name, output_dir)
        create_l1l2_single_metric_heatmap(raw_data, 'old_acc', superclass_name, output_dir)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ 单指标热力图生成失败: {exc}")

    # 简要汇总报告（跨配置）
    all_stats: Dict[Tuple[int, int, int], Dict] = {}
    for w in weight_sets:
        weighted = compute_weighted_l1l2(raw_data, w, available_components)
        if not weighted:
            continue
        top3, stats = generate_l1l2_report(
            weighted,
            w,
            superclass_name,
            os.path.join(output_dir, superclass_name, f"l1l2_report_wl1{int(w.w_l1)}_sep{int(w.w_sep)}_sil{int(w.w_sil)}.txt"),
        )
        all_stats[(int(w.w_l1), int(w.w_sep), int(w.w_sil))] = {'count': stats.get('count', 0),
                                                                 'min_combined': stats.get('min_combined', float('nan')),
                                                                 'max_combined': stats.get('max_combined', float('nan')),
                                                                 'mean_combined': stats.get('mean_combined', float('nan')),
                                                                 'std_combined': stats.get('std_combined', float('nan'))}

    generate_l1l2_summary_report(all_stats, superclass_name, os.path.join(output_dir, superclass_name, 'l1l2_weights_summary.txt'))

    if cleanup_reports:
        try:
            result = cleanup_intermediate_reports(superclass_name, output_dir, dry_run=False)
            if result.get('deleted_count', 0) > 0:
                freed_kb = result.get('freed_bytes', 0) / 1024.0
                print(f"🧹 已清理 {result['deleted_count']} 个中间报告，释放空间 {freed_kb:.2f} KB")
        except Exception as exc:  # pragma: no cover
            print(f"⚠️ 清理中间报告失败: {exc}")

    return {
        'superclass': superclass_name,
        'output_dir': os.path.join(output_dir, superclass_name),
        'color_metrics': color_metrics,
        'weight_sum': weight_sum,
    }


INTERMEDIATE_REPORT_PATTERN = re.compile(r'^l1l2_report_wl1\d+_sep\d+_sil\d+\.txt$')


def cleanup_intermediate_reports(superclass_name: str,
                                 output_dir: str,
                                 dry_run: bool = False) -> Dict[str, int]:
    """清理单个超类目录下的中间txt报告文件，保留汇总报告与PNG。

    返回 {'deleted_count': int, 'freed_bytes': int}
    """
    base_dir = os.path.join(output_dir, superclass_name)
    if not os.path.isdir(base_dir):
        return {'deleted_count': 0, 'freed_bytes': 0}
    deleted = 0
    freed = 0
    for name in os.listdir(base_dir):
        if not name.endswith('.txt'):
            continue
        if name == 'l1l2_weights_summary.txt':
            continue
        if not INTERMEDIATE_REPORT_PATTERN.match(name):
            continue
        path = os.path.join(base_dir, name)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if not dry_run:
            try:
                os.remove(path)
                deleted += 1
                freed += int(size)
            except OSError:
                # 忽略删除失败
                pass
    return {'deleted_count': deleted, 'freed_bytes': freed}
