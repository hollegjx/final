#!/usr/bin/env python3
"""L1+L2 权重区域搜索命令行入口

用法示例：
  - 基于 all 模式，针对 trees 和 humans 设定阈值（任务隔离目录）：
    python -m clustering.grid_search.l1l2_search.run_l1l2_region_search \
      --acc_mode all --output_dir /data/gjx/checkpoints/l1l2_search \
      --task_folder 4class_11_06_21_06 \
      --trees 0.5 --humans 0.6

说明：
  - 在 `output_dir/{task_folder}` 下检测已完成的超类目录，并为其动态添加命令行阈值参数 `--<superclass>`
  - 不传任何超类阈值将直接退出并提示
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List
import re

from config import l1l2_search_output_dir, l1l2_region_report_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from .l1l2_weight_region_search import (
    detect_superclasses_with_l1l2_results,
    detect_available_task_folders,
    find_common_weights,
    collect_weight_details,
    generate_region_report,
)


def build_parser_with_dynamic_superclasses(resolved_output_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='L1+L2 权重区域搜索 (离线，基于已生成热力图文件名)')
    parser.add_argument('--acc_mode', choices=['all', 'new', 'old'], required=True,
                        help='选择 ACC 模式，对应子目录 all/new/old')
    parser.add_argument('--output_dir', type=str, default=l1l2_search_output_dir,
                        help='L1L2 热力图输出根目录（任务根）')
    parser.add_argument('--task_folder', type=str, default=None,
                        help='任务文件夹名（例如: 4class_11_06_21_06）')
    parser.add_argument('--report_dir', type=str, default=l1l2_region_report_dir,
                        help='区域搜索报告输出目录')

    # 动态添加超类阈值参数：--<superclass> <threshold>
    superclasses = detect_superclasses_with_l1l2_results(resolved_output_dir)
    if superclasses:
        group = parser.add_argument_group('superclass thresholds', '为下列超类设置 ACC 阈值（可选）')
        for sc in superclasses:
            group.add_argument(f'--{sc}', type=float, default=None, help=f'{sc} 的阈值 (例如 0.6)')
    else:
        # 不添加动态项，但允许继续运行（用户可能传了不同的 output_dir）
        pass
    return parser


def parse_superclass_thresholds(args: argparse.Namespace, output_dir: str) -> Dict[str, float]:
    # 以 output_dir 的实际存在的超类为基准收集；若没有，返回空
    discovered = detect_superclasses_with_l1l2_results(output_dir)
    thresholds: Dict[str, float] = {}
    for sc in discovered:
        if hasattr(args, sc):
            val = getattr(args, sc)
            if val is not None:
                thresholds[sc] = float(val)
    return thresholds


def main() -> int:
    # 预解析：先拿到任务根目录与任务名
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--acc_mode', choices=['all', 'new', 'old'], required=True)
    pre_parser.add_argument('--output_dir', type=str, default=l1l2_search_output_dir)
    pre_parser.add_argument('--task_folder', type=str, default=None)
    pre_parser.add_argument('--report_dir', type=str, default=l1l2_region_report_dir)
    pre_args, _ = pre_parser.parse_known_args()

    # 如果未提供 --task_folder，列出可用任务并退出
    if not pre_args.task_folder:
        available_tasks = detect_available_task_folders(pre_args.output_dir)
        if not available_tasks:
            print(f"❌ 在 {pre_args.output_dir} 下未找到任何 L1L2 任务文件夹")
            print("💡 请先运行 L1L2 探索：")
            print("   python -m clustering.grid_search.l1l2_search.run_l1l2_exploration \\")
            print("       --weight_sum 20 --task_folder <任务名>")
            return 1

        print("📁 检测到以下可用的 L1L2 任务文件夹：")
        print("=" * 60)
        for task in available_tasks:
            m = re.match(r'^(\d+)class_(\d{2})_(\d{2})_(\d{2})_(\d{2})$', task)
            if m:
                n_classes, month, day, hour, minute = m.groups()
                print(f"  • {task}")
                print(f"    ├─ 超类数量: {n_classes}")
                print(f"    └─ 时间戳: {month}月{day}日 {hour}:{minute}")
            else:
                print(f"  • {task}")
        print("=" * 60)
        print("💡 请使用 --task_folder 参数指定任务，例如：")
        print(f"   python -m clustering.grid_search.l1l2_search.run_l1l2_region_search \\")
        print(f"       --acc_mode all --task_folder {available_tasks[0]} \\")
        print(f"       --trees 0.5 --humans 0.6")
        return 0

    # 验证任务文件夹格式
    pattern = r'^\d+class_\d{2}_\d{2}_\d{2}_\d{2}$'
    if not re.match(pattern, pre_args.task_folder.strip()):
        print(f"❌ 错误：任务文件夹格式不正确: {pre_args.task_folder}")
        print("   期望格式: <N>class_MM_DD_HH_MM（例如: 4class_11_06_21_06）")
        return 1

    resolved_output_dir = str(Path(pre_args.output_dir).joinpath(pre_args.task_folder.strip()))

    # 检查任务目录是否存在，提前给出明确提示
    if not Path(resolved_output_dir).exists():
        print(f"❌ 错误：任务目录不存在: {resolved_output_dir}")
        print(f"💡 请确保已运行 L1L2 探索：")
        print(f"   python -m clustering.grid_search.l1l2_search.run_l1l2_exploration \\")
        print(f"       --task_folder {pre_args.task_folder}")
        return 1

    # 基于任务级目录构建包含动态超类阈值参数的最终解析器
    parser = build_parser_with_dynamic_superclasses(resolved_output_dir)
    args = parser.parse_args()

    # 收集阈值配置
    sc_thresholds = parse_superclass_thresholds(args, resolved_output_dir)
    if not sc_thresholds:
        print('❌ 未指定任何超类阈值。请使用 --<superclass> <threshold> 传入至少一个超类阈值。')
        detected = detect_superclasses_with_l1l2_results(resolved_output_dir)
        if detected:
            print(f'ℹ️  已检测到的超类: {detected}')
        return 1

    print('=' * 80)
    print('🔎 L1+L2 权重区域搜索')
    print('=' * 80)
    print(f'任务目录: {resolved_output_dir}')
    print(f'模式: {args.acc_mode}')
    print(f'超类与阈值: {sc_thresholds}')

    # 交集
    common = find_common_weights(sc_thresholds, resolved_output_dir, args.acc_mode)
    if not common:
        print('⚠️ 无重合权重（交集为空）。')
        return 0

    # 详情
    ordered_superclasses: List[str] = sorted(sc_thresholds.keys())
    details = collect_weight_details(common, ordered_superclasses, resolved_output_dir, args.acc_mode)
    # 传递任务文件夹名用于构建与任务同步的报告文件名
    report_path = generate_region_report(
        details,
        sc_thresholds,
        args.acc_mode,
        args.report_dir,
        task_folder=pre_args.task_folder.strip() if pre_args.task_folder else None,
    )
    print(f'✅ 报告已生成: {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
