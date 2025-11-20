#!/usr/bin/env python3
"""
离线L2权重探索工具

职责：
1. 读取 batch_runner 生成的网格搜索 txt 结果
2. 支持单指标热力图可视化（ACC/组件）
3. 枚举 L2 组件权重并生成对比热力图
4. 权重探索模式下，自动为 all_acc/new_acc/old_acc 三个指标各生成一套热力图，
   输出目录：{output_dir}/{superclass}/{all|new|old}/

注意：本工具不再触发网格搜索，请先使用 batch_runner 保存结果；
      参数 --color_metric 在权重探索模式下已弃用，仅非权重探索模式下用于单图背景指标。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure the project root is importable when running as a module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from clustering.grid_search.heatmap import load_existing_results, detect_available_superclasses
from config import grid_search_output_dir, l2_search_output_dir
from .l2_heatmap_plotter import (
    create_single_metric_heatmap,
    enumerate_weight_combinations,
    create_weighted_l2_heatmaps,
    create_l2_component_heatmap,
)


def generate_acc_heatmaps(results_dict, superclass_name, output_dir, metrics=None):
    metrics = metrics or ['all_acc', 'old_acc', 'new_acc']
    single_dir = Path(output_dir) / superclass_name / 'single_metrics'
    single_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        print(f"🎨 生成 {metric} 热力图...")
        create_single_metric_heatmap(
            results_dict,
            metric=metric,
            superclass_name=superclass_name,
            output_dir=str(single_dir),
            save_plots=True,
            higher_is_better=True
        )


def generate_component_heatmaps(results_dict, superclass_name, output_dir, components, color_metrics=None):
    """
    生成 L2 组件的单指标热力图：在不同背景（new_acc 与 all_acc）下观察组件值。

    文件命名：{component}_bg_{metric}.png，输出到 single_metrics 子目录。
    """
    single_dir = Path(output_dir) / superclass_name / 'single_metrics'
    single_dir.mkdir(parents=True, exist_ok=True)

    if color_metrics is None:
        color_metrics = ['new_acc', 'all_acc']
    if isinstance(color_metrics, str):
        color_metrics = [color_metrics]

    for metric in color_metrics:
        for comp in components:
            comp = comp.strip()
            if not comp:
                continue
            print(f"🎨 生成组件 {comp} 热力图（背景: {metric}）...")
            create_l2_component_heatmap(
                results_dict,
                component_name=comp,
                color_metric=metric,
                superclass_name=superclass_name,
                output_dir=str(Path(output_dir)),
                save_plots=True,
                filename=f"{comp}_bg_{metric}.png"
            )


def generate_cluster_count_heatmaps(results_dict, superclass_name, output_dir, components):
    """
    生成“簇数量 vs L2组件”的关联热力图：
    - 背景为组件值（按组件方向选择配色正/反）
    - 注释显示簇数量

    文件命名：cluster_count_vs_{component}.png，输出到 single_metrics 子目录。
    """
    single_dir = Path(output_dir) / superclass_name / 'single_metrics'
    single_dir.mkdir(parents=True, exist_ok=True)

    # 提取簇数量：兼容不同字段
    annotation_data = {}
    for (k, dp), data in results_dict.items():
        val = data.get('n_clusters')
        if val is None:
            val = data.get('clusters')
        if val is None:
            val = data.get('cluster_count')
        if val is not None:
            annotation_data[(k, dp)] = float(val)

    if not annotation_data:
        print("⚠️  结果中未找到簇数量字段（n_clusters/clusters/cluster_count），跳过簇数量热力图。")
        return

    for comp in components:
        comp = comp.strip()
        if not comp:
            continue
        print(f"🎨 生成簇数量关联热力图（组件: {comp}）...")
        create_l2_component_heatmap(
            results_dict,
            component_name=comp,
            color_metric='component',  # 占位，无实际使用
            superclass_name=superclass_name,
            output_dir=str(Path(output_dir)),
            save_plots=True,
            annotation_data=annotation_data,
            color_by_component=True,
            higher_is_better=None,  # 由组件方向自动推断
            filename=f"cluster_count_vs_{comp}.png"
        )


def parse_l2_results(superclass_name: str, search_dir: str) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Load grid-search results for the specified superclass.
    """
    results = load_existing_results(superclass_name, search_dir)
    if not results:
        raise FileNotFoundError(
            f"未在 {search_dir}/{superclass_name} 找到有效的搜索结果，"
            f"请确认已运行网格搜索或指定正确目录。"
        )
    return results


def create_l2_heatmap(
    results_dict: Dict[Tuple[int, int], Dict[str, float]],
    superclass_name: str,
    output_dir: str,
    color_metric: str = 'new_acc',
) -> Dict:
    """
    Render a heatmap using the selected accuracy metric as colour background.
    """
    print("=" * 80)
    print(f"🎨 生成热力图 - 指标: {color_metric}")
    print(f"   输出目录: {output_dir}")
    print("=" * 80)

    higher_is_better = color_metric in {'all_acc', 'old_acc', 'new_acc', 'labeled_acc'}
    return create_single_metric_heatmap(
        results_dict,
        metric=color_metric,
        superclass_name=superclass_name,
        output_dir=output_dir,
        save_plots=True,
        higher_is_better=higher_is_better
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="L2权重探索后处理工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--superclass_name', type=str, default=None,
                        help='待探索的超类名称（留空则处理search_dir下全部超类）')
    parser.add_argument('--search_dir', type=str, default=grid_search_output_dir,
                        help='网格搜索结果目录（需先使用batch_runner生成）')
    parser.add_argument('--output_dir', type=str,
                        default=l2_search_output_dir,
                        help='热力图输出目录')
    # 保留但在权重探索模式下不再生效（向后兼容）
    parser.add_argument('--color_metric', type=str, default='new_acc',
                        choices=['new_acc', 'all_acc', 'old_acc', 'labeled_acc'],
                        help='[deprecated] 仅非权重探索模式使用；权重探索将自动生成 all/new/old 三套热力图')
    parser.add_argument('--skip_search', action='store_true',
                        help='兼容旧参数；当前工具总是使用已有结果进行后处理')
    parser.add_argument('--weight_exploration', action='store_true',
                        help='启用离线权重探索模式')
    parser.add_argument('--explore_components', type=str,
                        default='silhouette,separation,penalty',
                        help='参与权重探索的L2组件列表，逗号分隔')
    parser.add_argument('--weight_sum', type=int, default=10,
                        help='权重探索的总权重')
    parser.add_argument('--weight_step', type=int, default=1,
                        help='权重离散步长')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行绘制热力图的进程数（默认CPU核心数-1，设为1回退串行）')
    parser.add_argument('--skip_single_metrics', action='store_true',
                        help='跳过单指标（ACC/L2组件）热力图生成')

    args = parser.parse_args()

    if not args.skip_search:
        print("ℹ️  当前工具不再触发网格搜索，请先使用 batch_runner 生成txt结果。")

    if args.superclass_name:
        superclass_list = [args.superclass_name]
    else:
        superclass_list = detect_available_superclasses(args.search_dir)
        if not superclass_list:
            print(f"❌ 在 {args.search_dir} 中未发现任何超类结果，请先运行 batch_runner 生成txt。")
            return 1
        print(f"📁 未指定超类，将处理 {len(superclass_list)} 个超类: {superclass_list}")

    for idx, superclass in enumerate(superclass_list, 1):
        print(f"\n{'='*80}")
        print(f"🔄 处理超类 [{idx}/{len(superclass_list)}]: {superclass}")
        print(f"{'='*80}")
        try:
            results = parse_l2_results(superclass, args.search_dir)
        except FileNotFoundError:
            print(f"❌ 未找到 {superclass} 的网格搜索txt，跳过。")
            print("   提示: python -m clustering.grid_search.batch_runner --superclass_name "
                  f"{superclass} --use_l2 --l2_components silhouette separation penalty")
            continue
        if not results:
            print(f"❌ {superclass} 解析结果为空，跳过。")
            continue

        if not args.skip_single_metrics:
            print(f"\n{'-'*80}")
            print("📊 生成单指标热力图...")
            generate_acc_heatmaps(results, superclass, args.output_dir)
            explore_components = [comp.strip() for comp in args.explore_components.split(',') if comp.strip()]
            # 双背景的组件热力图
            generate_component_heatmaps(results, superclass, args.output_dir, explore_components, ['new_acc', 'all_acc'])
            # 新增簇数量关联热力图
            generate_cluster_count_heatmaps(results, superclass, args.output_dir, explore_components)
            print("✅ 单指标热力图生成完成！")

        has_component_data = any(metrics.get('l2_components') for metrics in results.values())

        if args.weight_exploration and not has_component_data:
            print(f"⚠️ {superclass} 的结果不包含组件数据（未找到 component_* 字段），请先用最新代码重新运行 batch_runner。")
            continue

        if args.weight_exploration:
            components = [comp.strip() for comp in args.explore_components.split(',') if comp.strip()]
            if not components:
                print("❌ 权重探索组件列表为空，跳过。")
                continue
            try:
                weight_sets = enumerate_weight_combinations(components, args.weight_sum, args.weight_step)
            except ValueError as exc:
                print(f"❌ 权重参数错误: {exc}，跳过 {superclass}")
                continue

            if len(weight_sets) > 150:
                print(f"⚠️ 权重组合数量为 {len(weight_sets)}，生成热力图较多，可增大weight_step控制。")

            supported_count = sum(
                1 for metrics in results.values()
                if all(comp in metrics.get('l2_components', {}) for comp in components)
            )
            if supported_count == 0:
                print(f"⚠️ {superclass} 的结果缺少权重探索所需的全部组件 {components}，跳过。")
                continue

            # 自动遍历全部 ACC 指标，分别生成权重热力图
            metrics_to_explore = ['all_acc', 'new_acc', 'old_acc']
            metric_dir_map = {'all_acc': 'all', 'new_acc': 'new', 'old_acc': 'old'}
            print("ℹ️  将为 all_acc、new_acc、old_acc 分别生成权重探索热力图（--color_metric 已弃用）")

            for metric in metrics_to_explore:
                metric_short = metric_dir_map.get(metric, metric)
                print(f"\n🎨 生成 {metric} 权重探索热力图 -> 子目录: {metric_short}/")

                summaries = create_weighted_l2_heatmaps(
                    results,
                    components=components,
                    weight_sets=weight_sets,
                    superclass_name=superclass,
                    output_dir=args.output_dir,  # 实际保存子目录在绘图模块内按 metric 划分
                    color_metric=metric,
                    num_workers=args.workers
                )

                superclass_output_dir = Path(args.output_dir) / superclass / metric_short
                print(f"✅ 权重探索完成，输出目录: {superclass_output_dir}")
                if summaries:
                    key_metric = f"best_{metric}"
                    ranked = [s for s in summaries if key_metric in s]
                    ranked.sort(key=lambda item: item[key_metric], reverse=True)
                    if ranked:
                        print("🏆 最佳权重组合:")
                        best = ranked[0]
                        print(f"   文件: {best['file']}")
                        print(f"   权重: {best['weight_signature']}")
                        print(f"   最佳(k, dp): ({best['best_k']}, {best['best_density_percentile']})")
                        print(f"   {metric}: {best[key_metric]:.4f}")
                else:
                    print(f"⚠️ 未生成任何 {metric} 热力图——可能是组件数据缺失或所有组合被过滤。")
        else:
            heatmap_stats = create_l2_heatmap(
                results,
                superclass_name=superclass,
                output_dir=args.output_dir,
                color_metric=args.color_metric,
            )

            superclass_output_dir = Path(args.output_dir) / superclass
            print(f"✅ 热力图已生成，输出目录: {superclass_output_dir}")
            if heatmap_stats:
                top_entries = heatmap_stats.get('top3', [])
                if top_entries:
                    print("🏆 指标Top3参数组合:")
                    for rank, (k_val, dp_val, metric_val) in enumerate(top_entries, 1):
                        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                        print(f"   {emoji} #{rank}: k={k_val}, dp={dp_val}, {args.color_metric}={metric_val:.4f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
