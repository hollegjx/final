#!/usr/bin/env python3
"""L1+L2联合权重搜索命令行入口

- 输出结构与 L2 保持一致：{superclass}/{all,new,old,single_metrics}/
- 默认使用自动权重配置（--weight_sum）逐配置生成热力图
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import os
import sys

from config import grid_search_output_dir, l1l2_search_output_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from clustering.grid_search.heatmap import detect_available_superclasses
from .l1l2_heatmap_plotter import plot_all_l1l2_configurations


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='L1+L2联合权重搜索 (离线)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--superclass_name', type=str, default=None, help='指定单个超类，留空则批量处理')
    # 新：与 batch_runner 对齐的任务目录解析
    parser.add_argument('--output_dir', type=str, default=grid_search_output_dir,
                        help=f'[输入] batch_runner 输出根目录（默认: {grid_search_output_dir}）')
    parser.add_argument('--task_folder', type=str, default=None, help='[输入] 任务文件夹名（可选，格式: <N>class_MM_DD_HH_MM），例如: 4class_11_06_21_09')
    parser.add_argument('--search_dir', type=str, default=None, help='[兼容] 直接传入完整搜索目录；若提供 --task_folder 将忽略此参数')
    parser.add_argument('--output_dir_heatmap', type=str, default=l1l2_search_output_dir,
                        help=f'[输出] 结果输出根目录（热力图等，默认: {l1l2_search_output_dir}）')
    parser.add_argument('--weight_sum', type=int, default=10, help='三组件权重总和 (默认: 10)')
    parser.add_argument('--color_metrics', type=str, default='all_acc,new_acc,old_acc', help='热力图背景指标列表，逗号分隔')
    parser.add_argument('--coverage_threshold', type=float, default=0.5, help='有效样本覆盖率阈值')
    parser.add_argument('--keep_reports', action='store_true', help='保留所有中间txt报告（默认清理以节省空间）')
    args = parser.parse_args()

    print('=' * 80)
    print('🔍 L1+L2联合权重探索')
    print('=' * 80)
    # 解析 search_dir：优先 task_folder，其次兼容 search_dir
    def _validate_task_folder(task_folder: str, output_dir: str) -> Path:
        pattern = r'^\d+class_\d{2}_\d{2}_\d{2}_\d{2}$'
        if not task_folder or not re.match(pattern, task_folder.strip()):
            raise SystemExit("❌ 错误：--task_folder 格式应为 <N>class_MM_DD_HH_MM，例如: 4class_11_06_21_09")
        p = Path(output_dir) / task_folder.strip()
        if not p.exists():
            print(f"❌ 错误：任务目录不存在: {p}")
            raise SystemExit(1)
        return p

    if args.task_folder:
        search_dir_path = _validate_task_folder(args.task_folder, args.output_dir)
        resolved_search_dir: str = str(search_dir_path)
    else:
        if not args.search_dir:
            print('❌ 错误：必须提供 --task_folder 或 --search_dir 参数')
            print('💡 推荐使用: --task_folder 4class_11_06_21_09')
            return 1
        resolved_search_dir = args.search_dir

    # 任务文件夹名：优先使用 --task_folder；兼容 --search_dir 直接带路径的情况
    task_folder_name = args.task_folder.strip() if args.task_folder else Path(resolved_search_dir).name
    task_output_dir = Path(args.output_dir_heatmap) / task_folder_name
    _ensure_dir(str(task_output_dir))

    print(f"搜索目录: {resolved_search_dir}")
    print(f"任务文件夹: {task_folder_name}")
    print(f"输出目录: {task_output_dir}")
    print(f"权重总和: {args.weight_sum}")
    print(f"指标: {args.color_metrics}")

    try:
        color_metrics = [metric.strip() for metric in args.color_metrics.split(',') if metric.strip()]
    except ValueError as exc:
        print(f"❌ 参数解析失败: {exc}")
        return 1

    if not color_metrics:
        color_metrics = ['new_acc', 'all_acc', 'old_acc']
    if not isinstance(args.weight_sum, int) or args.weight_sum < 1:
        print("❌ weight_sum 必须为整数且 >= 1")
        return 1

    if args.superclass_name:
        superclasses = [args.superclass_name]
    else:
        superclasses = detect_available_superclasses(resolved_search_dir)
        if not superclasses:
            print(f"❌ 未在 {resolved_search_dir} 中找到搜索结果")
            return 1

    success = 0
    failures = []

    for idx, superclass in enumerate(superclasses, start=1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(superclasses)}] 处理超类: {superclass}")
        print('=' * 80)

        try:
            result = plot_all_l1l2_configurations(
                superclass_name=superclass,
                search_dir=resolved_search_dir,
                output_dir=str(task_output_dir),
                color_metrics=color_metrics,
                weight_sum=args.weight_sum,
                cleanup_reports=(not args.keep_reports),
            )
            if result:
                success += 1
                print(f"✅ {superclass} 处理完成，结果目录: {result['output_dir']}")
            else:
                failures.append(superclass)
                print("❌ 无有效结果，跳过")
        except Exception as e:
            failures.append(superclass)
            print(f"❌ {superclass} 处理失败: {e}")

    print('\n' + '=' * 80)
    print('🎯 处理完成')
    print('=' * 80)
    print(f"成功: {success}/{len(superclasses)}")
    if failures:
        print(f"失败: {len(failures)}/{len(superclasses)} -> {failures}")
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
