#!/usr/bin/env python3
"""
L2权重热力图绘制模块

作用：
- 提供 L2 组件与 ACC 指标的可视化（单组件视角）
- 提供加权 L2 的可视化（离线权重探索）

说明：
- 保持与 heatmap.py 的风格一致，尽量复用既有绘图逻辑
- 组件热力图支持两种背景：ACC 指标或组件自身（用于簇数量注释）
- 加权 L2 热力图按照背景指标分目录保存：
  {output_dir}/{superclass}/{all|new|old|labeled}/
- 性能优化：当网格规模较大时（k×dp 单元数超过阈值）自动关闭格内注释并降低 PNG DPI，避免保存图片时过慢或被中断。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from multiprocessing import Pool
from config import grid_search_output_dir, l2_search_output_dir

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入L2权重计算器
from .l2_weight_calculator import (
    load_raw_scores, 
    compute_weighted_l2, 
    generate_l2_report, 
    get_weight_configurations,
    validate_l2_requirements,
    analyze_weight_stability
)
from clustering.evaluation.l2_strategies import L2_REGISTRY


def create_l2_component_heatmap(
    results_dict: Dict[Tuple[int, int], Dict],
    component_name: str,
    color_metric: str,
    superclass_name: str,
    output_dir: str,
    save_plots: bool = True,
    annotation_data: Optional[Dict[Tuple[int, int], float]] = None,
    color_by_component: bool = False,
    higher_is_better: Optional[bool] = None,
    filename: Optional[str] = None,
) -> Dict:
    """
    绘制单个 L2 组件的热力图。

    默认：背景为 `color_metric`（如 new_acc/all_acc），格子内显示该组件的数值。
    当 `color_by_component=True` 时：背景为组件数值，格子内显示 `annotation_data`（如簇数量）。

    Args:
        results_dict: {(k, dp): metrics}
        component_name: 组件名称，如 'silhouette'、'separation'、'penalty'
        color_metric: 背景指标（在 color_by_component=False 时使用）
        superclass_name: 超类名
        output_dir: 输出目录根路径
        save_plots: 是否保存
        annotation_data: 可选，(k, dp)->数值，用于在格子中显示（如簇数量）
        color_by_component: 若为 True，背景按组件值着色
        higher_is_better: 控制配色方向；None 时自动推断
        filename: 可选，自定义输出文件名（含后缀）

    Returns:
        dict: 摘要信息（包含 Top-3 条目等）
    """
    component_name = component_name.strip()
    if not component_name:
        return {}

    k_values = sorted({k for k, _ in results_dict.keys()})
    dp_values = sorted({dp for _, dp in results_dict.keys()})
    if not k_values or not dp_values:
        print(f"⚠️  {superclass_name} 缺少参数组合，跳过组件 {component_name}")
        return {}

    color_data = np.full((len(dp_values), len(k_values)), np.nan)
    display_data = np.full((len(dp_values), len(k_values)), np.nan)

    for i, dp in enumerate(dp_values):
        for j, k in enumerate(k_values):
            metrics = results_dict.get((k, dp))
            if not metrics:
                continue
            comp_info = metrics.get('l2_components', {}).get(component_name)
            comp_val = None
            if comp_info and comp_info.get('value') is not None:
                comp_val = float(comp_info['value'])

            if color_by_component:
                # 背景按组件值着色
                if comp_val is not None:
                    color_data[i, j] = comp_val
                # 注释显示外部 annotation 数据（如簇数量）
                if annotation_data is not None:
                    ann = annotation_data.get((k, dp))
                    if ann is not None:
                        display_data[i, j] = float(ann)
            else:
                # 背景按指标着色
                if color_metric in metrics and metrics[color_metric] is not None:
                    color_data[i, j] = float(metrics[color_metric])
                # 注释显示组件数值
                if comp_val is not None:
                    display_data[i, j] = comp_val

    if np.all(np.isnan(display_data)):
        print(f"⚠️  {superclass_name} 的组件 {component_name} 没有可用数据，跳过。")
        return {}

    # 确定配色方向
    orientation = L2_REGISTRY.get(component_name, {}).get('orientation', 'maximize')
    if higher_is_better is None:
        if color_by_component:
            higher_is_better = (orientation == 'maximize')
        else:
            # ACC 类指标通常越大越好
            higher_is_better = True

    cmap = 'RdYlGn' if higher_is_better else 'RdYlGn_r'

    fig, ax = plt.subplots(figsize=(12, 8))
    # 性能优化开关：大网格禁用注释并降低DPI
    cell_count = int(len(dp_values) * len(k_values))
    large_grid_threshold = 1200
    annotate_enabled = cell_count <= large_grid_threshold
    # 注释格式：若显式提供 annotation_data 且均为整数，优先用 '.0f'（兼容 float/NaN）
    fmt = '.4f'
    if annotate_enabled and annotation_data is not None:
        ann_vals = [v for v in display_data.flatten() if not np.isnan(v)]
        if ann_vals and all(abs(v - round(v)) < 1e-6 for v in ann_vals):
            fmt = '.0f'

    if color_by_component:
        cbar_label = f"{component_name} value (background)"
    else:
        cbar_label = f"{color_metric} (background)"

    sns.heatmap(
        color_data,
        xticklabels=k_values,
        yticklabels=dp_values,
        annot=(display_data if annotate_enabled else False),
        fmt=fmt,
        cmap=cmap,
        cbar_kws={'label': cbar_label},
        ax=ax
    )

    if color_by_component:
        title = (
            f"Cluster Count vs {component_name} - {superclass_name}\n"
            f"Background: {component_name} ({'maximize' if orientation == 'maximize' else 'minimize'} green)"
        )
    else:
        title = (
            f"Component: {component_name} (bg={color_metric}) - {superclass_name}\n"
            f"Annotation: {component_name} value; orientation: {orientation}"
        )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('density_percentile', fontsize=12)

    # Highlight top-3 entries based on component value
    valid_entries = []
    # 使用组件值进行排名：
    ranking_data = color_data if color_by_component else display_data
    for i in range(ranking_data.shape[0]):
        for j in range(ranking_data.shape[1]):
            val = ranking_data[i, j]
            if not np.isnan(val):
                valid_entries.append((val, i, j))
    reverse = (orientation == 'maximize')
    valid_entries.sort(key=lambda item: item[0], reverse=reverse)
    top3 = valid_entries[:3]
    for rank, (val, i, j) in enumerate(top3, 1):
        ax.plot(j + 0.5, i + 0.5, marker='*',
                markersize=18,
                color='red' if rank == 1 else 'orange',
                markeredgecolor='white', markeredgewidth=1.2)

    plt.tight_layout()

    if save_plots:
        superclass_dir = os.path.join(output_dir, superclass_name, 'single_metrics')
        os.makedirs(superclass_dir, exist_ok=True)
        if not filename:
            filename = f"component_{component_name}_colored_by_{color_metric}.png"
        path = os.path.join(superclass_dir, filename)
        dpi = 300 if annotate_enabled else 200
        bbox = 'tight' if annotate_enabled else None
        if not annotate_enabled:
            print(f"⚡ 大网格({cell_count}格)已禁用注释并降低DPI以加速保存: {path}")
        plt.savefig(path, dpi=dpi, bbox_inches=bbox)
        print(f"📁 组件热力图已保存: {path}")

    plt.close(fig)

    summary = {
        'component': component_name,
        'color_metric': component_name if color_by_component else color_metric,
        'orientation': orientation,
        'top_entries': [
            {
                'value': val,
                'k': k_values[j],
                'density_percentile': dp_values[i]
            }
            for val, i, j in top3
        ]
    }
    return summary


def enumerate_weight_combinations(components: List[str], weight_sum: int, step: int = 1) -> List[Dict[str, int]]:
    """
    Generate all ordered non-negative integer weight assignments for components.

    Args:
        components: ordered component names to assign weights.
        weight_sum: total weight budget.
        step: discrete increment for each weight.

    Returns:
        List of dictionaries mapping component name to assigned weight.
    """
    if not components:
        raise ValueError("components不能为空")
    if weight_sum < 0:
        raise ValueError("weight_sum必须≥0")
    if step <= 0:
        raise ValueError("step必须>0")
    if weight_sum % step != 0:
        raise ValueError("weight_sum必须能被step整除")

    scaled_total = weight_sum // step
    n = len(components)
    current = [0] * n
    results: List[Dict[str, int]] = []

    def backtrack(idx: int, remaining: int) -> None:
        if idx == n - 1:
            current[idx] = remaining
            results.append({comp: current[i] * step for i, comp in enumerate(components)})
            return
        for value in range(remaining + 1):
            current[idx] = value
            backtrack(idx + 1, remaining - value)

    backtrack(0, scaled_total)
    return results


def calculate_weighted_l2_scores(
    results_dict: Dict[Tuple[int, int], Dict],
    components: List[str],
    weight_map: Dict[str, int],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Combine component values into a weighted L2 score for each parameter pair.

    Args:
        results_dict: parsed grid-search results keyed by (k, density_percentile).
        components: ordered component names to consider.
        weight_map: mapping component name to weight.

    Returns:
        Dictionary mapping (k, dp) to metrics including weighted_l2.
    """
    if not components:
        raise ValueError("components不能为空")
    weighted_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}

    for key, metrics in results_dict.items():
        component_values = metrics.get('l2_components', {})
        total = 0.0
        # 仅对权重大于0的组件做存在性检查与累加；权重为0的组件忽略缺失
        active_components = [c for c in components if float(weight_map.get(c, 0)) > 0]
        # 若所有权重皆为0，则保留 weighted_l2=0（但这类配置通常不会被生成）
        missing = False
        for comp in active_components:
            comp_info = component_values.get(comp)
            if comp_info is None or comp_info.get('value') is None:
                missing = True
                break
            orientation = L2_REGISTRY.get(comp, {}).get('orientation', 'minimize')
            value = float(comp_info['value'])
            signed = value if orientation == 'maximize' else -value
            total += float(weight_map.get(comp, 0)) * signed
        if missing:
            continue
        weighted_metrics[key] = {
            'weighted_l2': total,
            'new_acc': metrics.get('new_acc'),
            'all_acc': metrics.get('all_acc'),
            'old_acc': metrics.get('old_acc')
        }

    return weighted_metrics


def plot_weighted_l2_heatmap(
    weighted_results: Dict[Tuple[int, int], Dict[str, float]],
    color_metric: str,
    superclass_name: str,
    output_dir: str,
    weight_map: Dict[str, int],
    components: List[str],
    higher_is_better: bool = False
) -> Dict[str, float]:
    """
    Render a heatmap showing weighted L2 values with accuracy as background color.

    保存路径：{output_dir}/{superclass}/{metric_short}/，其中 metric_short 由
    {'all_acc':'all','new_acc':'new','old_acc':'old','labeled_acc':'labeled'} 映射得到。

    Args:
        weighted_results: metrics per (k, density_percentile).
        color_metric: accuracy metric used for coloring.
        superclass_name: current superclass identifier.
        output_dir: base directory for outputs.
        weight_map: mapping components to weights for naming.
        higher_is_better: whether weighted L2 larger value is preferred.

    Returns:
        Summary statistics with best parameter and metric values.
    """
    if not weighted_results:
        return {}

    # 统一在 {output_dir}/{superclass_name}/{metric_short}/ 下保存权重热力图
    superclass_output_dir = os.path.join(output_dir, superclass_name)
    metric_dir_map = {'all_acc': 'all', 'new_acc': 'new', 'old_acc': 'old', 'labeled_acc': 'labeled'}
    metric_short = metric_dir_map.get(color_metric)
    if metric_short:
        superclass_output_dir = os.path.join(superclass_output_dir, metric_short)
    os.makedirs(superclass_output_dir, exist_ok=True)

    k_values = sorted({k for k, _ in weighted_results.keys()})
    density_values = sorted({dp for _, dp in weighted_results.keys()})

    color_data = np.full((len(density_values), len(k_values)), np.nan)
    display_data = np.full((len(density_values), len(k_values)), np.nan)

    for i, dp in enumerate(density_values):
        for j, k in enumerate(k_values):
            metrics = weighted_results.get((k, dp))
            if not metrics:
                continue
            color_value = metrics.get(color_metric)
            if color_value is not None:
                color_data[i, j] = color_value
            display_value = metrics.get('weighted_l2')
            if display_value is not None:
                display_data[i, j] = display_value

    if np.all(np.isnan(display_data)):
        return {}

    valid_entries = [
        (display_data[i, j], i, j, color_data[i, j])
        for i in range(display_data.shape[0])
        for j in range(display_data.shape[1])
        if not np.isnan(display_data[i, j]) and not np.isnan(color_data[i, j])
    ]

    valid_entries.sort(key=lambda item: item[0], reverse=higher_is_better)
    best = valid_entries[0] if valid_entries else None

    fig, ax = plt.subplots(figsize=(14, 9))
    cell_count = int(len(density_values) * len(k_values))
    large_grid_threshold = 1200
    annotate_enabled = cell_count <= large_grid_threshold
    cbar_label = f"{color_metric} (background color)"
    sns.heatmap(
        color_data,
        xticklabels=k_values,
        yticklabels=density_values,
        annot=(display_data if annotate_enabled else False),
        fmt='.4f',
        cmap='viridis',
        cbar_kws={'label': cbar_label},
        ax=ax
    )

    best_k = best_dp = None
    best_weighted = None
    # 使用更准确的命名：记录最优点对应的配色指标值（可能是 all_acc/new_acc/old_acc）
    best_color_metric_value = None
    if best:
        best_weighted, i, j, _ = best
        best_k = k_values[j]
        best_dp = density_values[i]
        ax.plot(j + 0.5, i + 0.5, marker='*', markersize=18,
                color='red', markeredgecolor='white', markeredgewidth=1.2)
        best_entry = weighted_results.get((best_k, best_dp), {})
        best_color_metric_value = best_entry.get(color_metric) or best_entry.get('new_acc')
    if best_color_metric_value is None:
        best_color_metric_value = 0.0

    weight_signature = "_".join(f"{comp}-{weight_map[comp]}" for comp in weight_map)
    weight_desc = ", ".join(f"w_{comp}={weight_map.get(comp, 0)}" for comp in weight_map)
    if best_k is not None:
        title_detail = f"Best: k={best_k}, dp={best_dp}, {color_metric}={best_color_metric_value:.2f}, weighted_l2={best_weighted:.4f}"
    else:
        title_detail = "Best: N/A"
    ax.set_title(
        f"Weighted L2 ({weight_desc}) - {superclass_name}\n"
        f"{title_detail}",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('density_percentile', fontsize=12)

    plt.tight_layout()
    weight_str = "_".join(str(weight_map.get(comp, 0)) for comp in components)
    filename = f"{color_metric}_{best_color_metric_value:.4f}_{weight_str}.png"
    output_path = os.path.join(superclass_output_dir, filename)
    dpi = 300 if annotate_enabled else 200
    bbox = 'tight' if annotate_enabled else None
    if not annotate_enabled:
        print(f"⚡ 大网格({cell_count}格)已禁用注释并降低DPI以加速保存: {output_path}")
    plt.savefig(output_path, dpi=dpi, bbox_inches=bbox)

    summary = {
        'file': output_path,
        'weight_signature': weight_signature,
        'entries': len(valid_entries),
        'min_weighted_l2': float(np.nanmin(display_data)),
        'max_weighted_l2': float(np.nanmax(display_data))
    }
    if best:
        score, i, j, color_val = best
        summary.update({
            'best_weighted_l2': score,
            'best_k': k_values[j],
            'best_density_percentile': density_values[i],
            f'best_{color_metric}': best_color_metric_value
        })
    return summary


def _plot_single_heatmap_task(
    results_dict: Dict[Tuple[int, int], Dict],
    components: List[str],
    weight_map: Dict[str, int],
    superclass_name: str,
    output_dir: str,
    color_metric: str,
    higher_is_better: bool
) -> Dict[str, float]:
    try:
        weighted_results = calculate_weighted_l2_scores(results_dict, components, weight_map)
        if not weighted_results:
            return {'error': 'weighted_results_empty', 'weight_map': weight_map}
        summary = plot_weighted_l2_heatmap(
            weighted_results,
            color_metric=color_metric,
            superclass_name=superclass_name,
            output_dir=output_dir,
            weight_map=weight_map,
            components=components,
            higher_is_better=higher_is_better
        )
        summary['weight_signature'] = "_".join(f"{comp}-{weight_map[comp]}" for comp in weight_map)
        return summary
    except Exception as exc:  # pragma: no cover - defensive
        return {'error': str(exc), 'weight_map': weight_map}


def create_weighted_l2_heatmaps(
    results_dict: Dict[Tuple[int, int], Dict],
    components: List[str],
    weight_sets: List[Dict[str, int]],
    superclass_name: str,
    output_dir: str,
    color_metric: str,
    num_workers: Optional[int] = None
) -> List[Dict[str, float]]:
    """
    Generate heatmaps for each weight combination and return summary stats.

    Args:
        results_dict: parsed grid-search results.
        components: ordered component names to explore.
        weight_sets: list of weight dictionaries for each combination.
        superclass_name: identifier for output grouping.
        output_dir: base directory for outputs.
        color_metric: accuracy metric used for coloring.

    Returns:
        List of summary dictionaries per weight configuration.
    """
    summaries: List[Dict[str, float]] = []
    if not weight_sets:
        return summaries

    cpu_count = os.cpu_count() or 2
    max_default = max(1, cpu_count - 1)
    if num_workers is None:
        num_workers = max_default
    num_workers = max(1, min(num_workers, cpu_count))

    if num_workers <= 1:
        for weight_map in weight_sets:
            weighted_results = calculate_weighted_l2_scores(results_dict, components, weight_map)
            if not weighted_results:
                continue
            summary = plot_weighted_l2_heatmap(
                weighted_results,
                color_metric=color_metric,
                superclass_name=superclass_name,
                output_dir=output_dir,
                weight_map=weight_map,
                components=components,
                # 注意：weighted_l2 已按组件方向（maximize 加，minimize 减）合成，值越大越好
                higher_is_better=True
            )
            if summary:
                summaries.append(summary)
        return summaries

    # 注意：weighted_l2 已按组件方向（maximize 加，minimize 减）合成，值越大越好
    higher_is_better = True
    task_args = [
        (results_dict, components, weight_map, superclass_name, output_dir, color_metric, higher_is_better)
        for weight_map in weight_sets
    ]

    with Pool(processes=num_workers) as pool:
        iterator = pool.starmap(_plot_single_heatmap_task, task_args)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(weight_sets), desc=f"Heatmaps {superclass_name}")
        for result in iterator:
            if isinstance(result, dict) and result.get('error'):
                print(f"⚠️  权重 {result.get('weight_map')} 生成失败: {result['error']}")
                continue
            if result:
                summaries.append(result)

    return summaries


def create_l2_weighted_heatmap(results_dict: Dict, w_sep: float, w_pen: float,
                               superclass_name: str, output_dir: str, save_plots: bool = True,
                               color_metric: str = 'new_acc') -> Tuple[List, Dict]:
    """
    绘制单张L2权重热力图
    基于heatmap.py的create_mixed_heatmap函数逻辑
    
    Args:
        results_dict: 加权结果字典 (来自compute_weighted_l2)
        w_sep: separation权重
        w_pen: penalty权重
        superclass_name: 超类名称
        output_dir: 输出目录
        save_plots: 是否保存图片
        color_metric: 背景着色所使用的指标字段
    
    Returns:
        tuple: (top3_params, stats)
    """
    print(f"🎨 绘制L2权重热力图 - 权重: sep={w_sep}, pen={w_pen}，配色指标: {color_metric}")
    
    if not results_dict:
        print(f"⚠️  无有效数据，跳过热力图绘制")
        return [], {}
    
    # 创建输出目录
    if save_plots:
        superclass_output_dir = os.path.join(output_dir, superclass_name)
        os.makedirs(superclass_output_dir, exist_ok=True)
    
    # 提取参数和结果（复用heatmap.py的逻辑）
    k_values = sorted(list(set([k for k, _ in results_dict.keys()])))
    density_percentile_values = sorted(list(set([dp for _, dp in results_dict.keys()])))
    
    # 创建两个数据矩阵
    color_data = np.zeros((len(density_percentile_values), len(k_values)))  # color_metric用于着色
    display_data = np.zeros((len(density_percentile_values), len(k_values)))  # L2值用于显示
    
    for i, dp in enumerate(density_percentile_values):
        for j, k in enumerate(k_values):
            if (k, dp) in results_dict:
                # 着色数据：color_metric（背景色）
                color_value = results_dict[(k, dp)].get(color_metric)
                color_data[i, j] = color_value if color_value is not None else np.nan
                
                # 显示数据：L2权重损失值
                display_value = results_dict[(k, dp)].get('l2_weighted')
                display_data[i, j] = display_value if display_value is not None else np.nan
            else:
                color_data[i, j] = np.nan
                display_data[i, j] = np.nan
    
    if np.all(np.isnan(color_data)):
        print(f"⚠️  配色指标 {color_metric} 缺失，将以 NaN 背景展示")
    
    # 找到基于L2损失的前3名位置（L2越小越好）
    valid_data = []
    for i in range(display_data.shape[0]):
        for j in range(display_data.shape[1]):
            if not np.isnan(display_data[i, j]):
                valid_data.append((display_data[i, j], i, j))
    
    # 按L2损失从小到大排序（损失越小越好）
    valid_data.sort(key=lambda x: x[0])
    top3 = valid_data[:3] if len(valid_data) >= 3 else valid_data
    
    # 创建热力图（复用heatmap.py的样式）
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 用color_metric着色，显示L2值
    cbar_label = f'{color_metric.upper()} (background color)'
    sns.heatmap(color_data,
                xticklabels=k_values,
                yticklabels=density_percentile_values,
                annot=display_data,  # 显示L2权重损失值
                fmt='.4f',
                cmap='viridis',  # 指标越高颜色越深
                cbar_kws={'label': cbar_label},
                ax=ax)
    
    # 标注前3名（基于L2损失，最小的为最佳）
    for rank, (l2_value, i, j) in enumerate(top3, 1):
        ax.plot(j + 0.5, i + 0.5, marker='*', markersize=20,
                color='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                markeredgecolor='white', markeredgewidth=1.5)
        ax.text(j + 0.5, i + 0.2, f'#{rank}',
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                         edgecolor='white', linewidth=1.5, alpha=0.8))
    
    # 设置标题
    ax.set_title(f'L2 Weighted Loss (colored by {color_metric.upper()}) - {superclass_name}\n'
                 f'Weights: separation={w_sep}, penalty={w_pen} (sum={w_sep + w_pen})\n'
                 f'Parameters: k vs density_percentile (Top 3 by L2 marked)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Density Percentile', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图片
    if save_plots:
        current_time = datetime.now()
        filename = f"l2_weighted_{color_metric}_sep{w_sep}_pen{w_pen}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}.png"
        output_path = os.path.join(superclass_output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📁 L2权重热力图已保存: {output_path}")
    
    # 输出前3名参数组合
    print(f"\n🏆 Top 3 参数组合 (权重 sep={w_sep}, pen={w_pen}):")
    print("-" * 60)
    top3_params = []
    for rank, (l2_value, i, j) in enumerate(top3, 1):
        k_val = k_values[j]
        dp_val = density_percentile_values[i]
        metric_val = color_data[i, j]
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        
        param_info = (k_val, dp_val, l2_value)
        top3_params.append(param_info)
        
        if not np.isnan(metric_val):
            print(f"{emoji} #{rank}: k={k_val:<3}, dp={dp_val:<3}, L2={l2_value:.4f}, {color_metric}={metric_val:.4f}")
        else:
            print(f"{emoji} #{rank}: k={k_val:<3}, dp={dp_val:<3}, L2={l2_value:.4f}, {color_metric}=N/A")
    
    # 统计信息
    l2_values = [data['l2_weighted'] for data in results_dict.values()]
    stats = {
        'count': len(l2_values),
        'min_l2': min(l2_values) if l2_values else 0,
        'max_l2': max(l2_values) if l2_values else 0,
        'mean_l2': np.mean(l2_values) if l2_values else 0,
        'std_l2': np.std(l2_values) if l2_values else 0,
        'w_sep': w_sep,
        'w_pen': w_pen
    }
    
    return top3_params, stats


def create_single_metric_heatmap(results_dict: Dict, metric: str,
                                 superclass_name: str, output_dir: str,
                                 save_plots: bool = True, higher_is_better: bool = True) -> Dict:
    """
    绘制单个指标的热力图（着色与显示使用相同指标）
    
    Args:
        results_dict: 原始结果字典（通常来自 load_raw_scores）
        metric: 指标名称，如 'separation_score' 或 'penalty_score'
        superclass_name: 超类名称
        output_dir: 输出目录
        save_plots: 是否保存图片
        higher_is_better: 指标越大是否越好，用于排序方向
    
    Returns:
        dict: {'top3': [...], 'stats': {...}}
    """
    direction = "↑ 优先" if higher_is_better else "↓ 优先"
    print(f"🎨 绘制单指标热力图 - 指标: {metric} ({direction})")
    
    if not results_dict:
        print("⚠️  无有效数据，跳过单指标热力图绘制")
        return {}
    
    if save_plots:
        superclass_output_dir = os.path.join(output_dir, superclass_name)
        os.makedirs(superclass_output_dir, exist_ok=True)
    
    k_values = sorted(list(set([k for k, _ in results_dict.keys()])))
    density_percentile_values = sorted(list(set([dp for _, dp in results_dict.keys()])))
    
    metric_data = np.full((len(density_percentile_values), len(k_values)), np.nan)
    
    for i, dp in enumerate(density_percentile_values):
        for j, k in enumerate(k_values):
            if (k, dp) in results_dict:
                value = results_dict[(k, dp)].get(metric)
                metric_data[i, j] = value if value is not None else np.nan
    
    if np.all(np.isnan(metric_data)):
        print(f"⚠️  指标 {metric} 完全缺失，无法生成热力图")
        return {}
    
    valid_points = []
    for i in range(metric_data.shape[0]):
        for j in range(metric_data.shape[1]):
            if not np.isnan(metric_data[i, j]):
                valid_points.append((metric_data[i, j], i, j))
    
    reverse = higher_is_better
    valid_points.sort(key=lambda x: x[0], reverse=reverse)
    top3 = valid_points[:3]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = 'viridis' if higher_is_better else 'viridis_r'
    cbar_label = f'{metric} ({"higher better" if higher_is_better else "lower better"})'

    sns.heatmap(metric_data,
                xticklabels=k_values,
                yticklabels=density_percentile_values,
                annot=True,
                fmt='.4f',
                cmap=cmap,
                cbar_kws={'label': cbar_label},
                ax=ax)

    for rank, (value, i, j) in enumerate(top3, 1):
        ax.plot(j + 0.5, i + 0.5, marker='*', markersize=20,
                color='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                markeredgecolor='white', markeredgewidth=1.5)
        ax.text(j + 0.5, i + 0.2, f'#{rank}',
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                          edgecolor='white', linewidth=1.5, alpha=0.8))

    ax.set_title(f'{metric} Single Metric Heatmap ({direction}) - {superclass_name}\nParameters: k vs density_percentile (Top 3 highlighted)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Density Percentile', fontsize=12)
    
    plt.tight_layout()
    
    if save_plots:
        current_time = datetime.now()
        filename = f"single_metric_{metric}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}.png"
        output_path = os.path.join(superclass_output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📁 单指标热力图已保存: {output_path}")
    
    print(f"\n🏆 {metric} Top 3 参数组合 ({'高' if higher_is_better else '低'}值优先):")
    print("-" * 60)
    top3_params = []
    for rank, (value, i, j) in enumerate(top3, 1):
        k_val = k_values[j]
        dp_val = density_percentile_values[i]
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print(f"{emoji} #{rank}: k={k_val:<3}, dp={dp_val:<3}, {metric}={value:.4f}")
        top3_params.append((k_val, dp_val, value))
    
    stats = {
        'count': len(valid_points),
        'min': float(np.nanmin(metric_data)),
        'max': float(np.nanmax(metric_data)),
        'mean': float(np.nanmean(metric_data)),
        'std': float(np.nanstd(metric_data)),
        'higher_is_better': higher_is_better
    }
    
    return {'top3': top3_params, 'stats': stats}


def plot_all_l2_configurations(superclass_name: str,
                               search_dir: str = grid_search_output_dir,
                               output_dir: str = l2_search_output_dir,
                               color_metric: str = 'new_acc',
                               weight_sum: int = 10) -> Dict:
    """
    批量绘制所有9种权重配置的热力图
    
    Args:
        superclass_name: 超类名称
        search_dir: 网格搜索结果目录
        output_dir: L2探索结果输出目录
        color_metric: 背景着色所使用的指标字段
        weight_sum: 分离度与惩罚项权重之和
    
    Returns:
        dict: 所有配置的结果汇总
    """
    print("=" * 60)
    print(f"🎨 L2权重探索 - {superclass_name} (配色指标: {color_metric}, 权重总和: {weight_sum})")
    print("=" * 60)
    
    # 1. 加载原始数据
    print("\n📂 Step 1: 加载原始搜索数据")
    raw_data = load_raw_scores(superclass_name, search_dir)
    
    if raw_data is None:
        print(f"❌ 无法加载 {superclass_name} 的搜索结果")
        return {}
    
    # 2. 验证数据要求
    is_valid, validation_info = validate_l2_requirements(raw_data)
    if not is_valid:
        print(f"❌ 数据验证失败: {validation_info}")
        return {}
    
    print(f"✅ 数据验证通过: {validation_info}")
    
    # 3. 获取权重配置
    weight_configs = get_weight_configurations(weight_sum)
    total_configs = len(weight_configs)
    
    print(f"\n🔧 Step 2: 处理 {total_configs} 种权重配置")
    
    # 存储所有结果
    all_results = {}  # {(w_sep, w_pen): {(k, dp): result_data}}
    all_top3 = {}     # {(w_sep, w_pen): [(k, dp, l2_value), ...]}
    all_stats = {}    # {(w_sep, w_pen): stats_dict}
    
    # 4. 逐个处理权重配置
    for config_idx, (w_sep, w_pen) in enumerate(weight_configs, 1):
        print(f"\n配置 {config_idx}/{total_configs}: sep={w_sep}, pen={w_pen}")
        print("-" * 30)
        
        # 4.1 计算加权L2
        weighted_results = compute_weighted_l2(raw_data, w_sep, w_pen)
        all_results[(w_sep, w_pen)] = weighted_results
        
        if not weighted_results:
            print(f"⚠️  权重配置 (sep={w_sep}, pen={w_pen}) 无有效结果")
            continue
        
        # 4.2 生成文本报告
        report_path = os.path.join(output_dir, superclass_name, f"l2_report_sep{w_sep}_pen{w_pen}.txt")
        top3, stats = generate_l2_report(weighted_results, w_sep, w_pen, superclass_name, report_path)
        all_top3[(w_sep, w_pen)] = top3
        all_stats[(w_sep, w_pen)] = stats
        
        if not any(data.get(color_metric) is not None for data in weighted_results.values()):
            print(f"⚠️  警告: 指标 {color_metric} 在此配置下不存在，将以 NaN 显示背景颜色")
        
        # 4.3 绘制热力图
        top3_params, heatmap_stats = create_l2_weighted_heatmap(
            weighted_results, w_sep, w_pen, superclass_name, output_dir, save_plots=True,
            color_metric=color_metric
        )
    
    # 5. 跨配置稳定性分析
    print(f"\n📊 Step 3: 跨配置稳定性分析")
    print("=" * 60)
    
    stability_analysis = analyze_weight_stability(all_results)
    stable_params = stability_analysis['stability_ranking']
    
    print(f"📊 稳定性分析结果:")
    print(f"   总权重配置: {stability_analysis['total_configs']}")
    print(f"   稳定参数组合数: {stability_analysis['analysis_summary']['stable_count']}")
    
    if stable_params:
        print(f"\n🏆 跨配置稳定性排名 (Top 5):")
        for rank, (param_key, stability_data) in enumerate(stable_params[:5], 1):
            k, dp = param_key
            freq = stability_data['top3_frequency']
            stability_score = stability_data['stability_score']
            avg_rank = stability_data['avg_rank']
            
            stars = "⭐" * min(3, freq)  # 最多3颗星
            print(f"   #{rank}: k={k}, dp={dp} - {freq}/{total_configs}配置进入Top3 {stars}")
            print(f"        稳定性得分: {stability_score:.1%}, 平均排名: {avg_rank:.1f}")
    
    # 6. 生成汇总报告
    summary_path = os.path.join(output_dir, superclass_name, "l2_weights_summary.txt")
    generate_summary_report(all_stats, stability_analysis, superclass_name, summary_path)
    
    print(f"\n✅ L2权重探索完成！")
    print(f"📁 所有结果保存在: {os.path.join(output_dir, superclass_name)}")
    print("=" * 60)
    
    return {
        'superclass': superclass_name,
        'all_results': all_results,
        'all_top3': all_top3,
        'all_stats': all_stats,
        'stability_analysis': stability_analysis,
        'output_dir': os.path.join(output_dir, superclass_name),
        'color_metric': color_metric,
        'raw_data': raw_data,
        'weight_sum': weight_sum
    }


def generate_summary_report(all_stats: Dict, stability_analysis: Dict, 
                          superclass_name: str, output_path: str) -> None:
    """
    生成跨权重配置的汇总报告
    
    Args:
        all_stats: 所有权重配置的统计信息
        stability_analysis: 稳定性分析结果
        superclass_name: 超类名称
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"L2权重探索汇总报告 - {superclass_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("权重配置统计:\n")
        f.write("-" * 40 + "\n")
        
        # 按权重配置列出统计信息
        for (w_sep, w_pen), stats in all_stats.items():
            f.write(f"配置 sep={w_sep}, pen={w_pen}:\n")
            f.write(f"  参数组合数: {stats['count']}\n")
            f.write(f"  L2损失范围: [{stats['min_l2']:.4f}, {stats['max_l2']:.4f}]\n")
            f.write(f"  L2损失均值: {stats['mean_l2']:.4f} ± {stats['std_l2']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("跨配置稳定性分析:\n")
        f.write("-" * 40 + "\n")
        
        stable_params = stability_analysis['stability_ranking']
        total_configs = stability_analysis['total_configs']
        
        f.write(f"总权重配置数: {total_configs}\n")
        f.write(f"稳定参数组合数: {stability_analysis['analysis_summary']['stable_count']}\n\n")
        
        if stable_params:
            f.write("稳定性排名 (按稳定性得分降序):\n\n")
            for rank, (param_key, data) in enumerate(stable_params, 1):
                k, dp = param_key
                freq = data['top3_frequency']
                stability_score = data['stability_score']
                avg_rank = data['avg_rank']
                
                f.write(f"#{rank:2d}: k={k}, dp={dp}\n")
                f.write(f"     Top3频次: {freq}/{total_configs} ({stability_score:.1%})\n")
                f.write(f"     平均排名: {avg_rank:.2f}\n")
                f.write(f"     权重配置: {data['configs']}\n\n")
        
        # 最稳定参数推荐
        if stable_params:
            best_param = stable_params[0]
            f.write("=" * 80 + "\n")
            f.write("🏆 推荐参数组合:\n")
            f.write("-" * 40 + "\n")
            param_key, param_data = best_param
            k, dp = param_key
            f.write(f"k = {k}\n")
            f.write(f"density_percentile = {dp}\n")
            f.write(f"稳定性得分: {param_data['stability_score']:.1%}\n")
            f.write(f"Top3频次: {param_data['top3_frequency']}/{total_configs}\n")
            f.write(f"平均排名: {param_data['avg_rank']:.2f}\n")
            f.write(f"推荐理由: 在多种权重配置下均表现优异，具有良好的鲁棒性\n")
    
    print(f"📄 汇总报告已保存: {output_path}")


def create_l2_comparison_subplot(superclass_name: str, search_dir: str, output_dir: str,
                                 color_metric: str = 'new_acc', weight_sum: int = 10) -> None:
    """
    可选：绘制3x3子图，一次性展示所有9种配置
    
    Args:
        superclass_name: 超类名称
        search_dir: 搜索结果目录
        output_dir: 输出目录
        color_metric: 背景着色所使用的指标字段
        weight_sum: 分离度与惩罚项权重之和
    """
    print(f"🎨 创建L2权重对比子图 - {superclass_name} (配色指标: {color_metric}, 权重总和: {weight_sum})")
    
    # 加载数据
    raw_data = load_raw_scores(superclass_name, search_dir)
    if raw_data is None:
        return
    
    weight_configs = get_weight_configurations(weight_sum)
    
    # 创建3x3子图
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'L2 Weight Exploration Comparison ({color_metric}) - {superclass_name}', fontsize=16, fontweight='bold')
    
    for idx, (w_sep, w_pen) in enumerate(weight_configs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # 计算加权L2
        weighted_results = compute_weighted_l2(raw_data, w_sep, w_pen)
        
        if not weighted_results:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'sep={w_sep}, pen={w_pen}')
            continue
        
        # 准备数据矩阵
        k_values = sorted(list(set([k for k, _ in weighted_results.keys()])))
        dp_values = sorted(list(set([dp for _, dp in weighted_results.keys()])))
        
        color_data = np.zeros((len(dp_values), len(k_values)))
        display_data = np.zeros((len(dp_values), len(k_values)))
        
        for i, dp in enumerate(dp_values):
            for j, k in enumerate(k_values):
                if (k, dp) in weighted_results:
                    color_data[i, j] = weighted_results[(k, dp)].get(color_metric, np.nan)
                    display_data[i, j] = weighted_results[(k, dp)].get('l2_weighted', np.nan)
                else:
                    color_data[i, j] = np.nan
                    display_data[i, j] = np.nan
        
        # 绘制子图
        sns.heatmap(color_data, 
                    xticklabels=k_values[::2],  # 减少标签密度
                    yticklabels=dp_values[::2],
                    annot=False,  # 子图太小，不显示数值
                    cmap='viridis',
                    ax=ax,
                    cbar=False)  # 不显示颜色条
        
        if np.all(np.isnan(color_data)):
            note = '无配色数据'
        else:
            note = ''
        ax.set_title(f'sep={w_sep}, pen={w_pen} {note}', fontsize=10)
        ax.set_xlabel('k' if row == 2 else '', fontsize=8)
        ax.set_ylabel('dp' if col == 0 else '', fontsize=8)
    
    plt.tight_layout()
    
    # 保存子图
    current_time = datetime.now()
    filename = f"l2_comparison_subplot_{color_metric}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}.png"
    output_path = os.path.join(output_dir, superclass_name, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📁 L2对比子图已保存: {output_path}")
