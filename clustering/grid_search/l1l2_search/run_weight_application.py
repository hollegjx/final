#!/usr/bin/env python3
"""权重应用评估命令行入口

用途：
- 从 findL 区域搜索报告解析重合权重
- 在指定的 search 任务数据上评估这些权重的泛化表现
"""

from __future__ import annotations

import argparse
import os
import sys

from config import grid_search_output_dir, weight_application_report_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from .weight_application_evaluator import (
    parse_findl_weights,
    load_task_data,
    evaluate_all_weights,
    generate_application_report,
    generate_simplified_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='将 findL 报告中的权重应用到指定任务，评估泛化能力',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--findl_report', type=str, required=True, help='findL 报告txt完整路径')
    parser.add_argument('--search_dir_base', type=str, default=grid_search_output_dir, help='search 根目录')
    parser.add_argument('--task_folder', type=str, required=True, help='目标任务文件夹名，例如 15class_11_06_22_30')
    parser.add_argument('--acc_mode', type=str, default='all_acc', choices=['all_acc', 'new_acc', 'old_acc'], help='ACC 指标')
    parser.add_argument('--report_dir', type=str, default=weight_application_report_dir, help='评估报告输出目录')
    parser.add_argument('--sort', action='store_true',
                        help='按平均差值降序排序权重（默认保持findL报告原序）')
    args = parser.parse_args()

    if not os.path.isfile(args.findl_report):
        print(f"❌ 报告文件不存在: {args.findl_report}")
        return 1

    task_path = os.path.join(args.search_dir_base, args.task_folder)
    if not os.path.isdir(task_path):
        print(f"❌ 任务目录不存在: {task_path}")
        print("💡 请确认 search_dir_base 与 task_folder 是否正确")
        return 1

    print('=' * 80)
    print('🧪 权重应用评估')
    print('=' * 80)
    print(f"来源报告: {args.findl_report}")
    print(f"任务目录: {task_path}")
    print(f"ACC指标: {args.acc_mode}")

    try:
        weights = parse_findl_weights(args.findl_report)
    except Exception as exc:
        print(f"❌ 解析报告失败: {exc}")
        return 1
    print(f"✅ 解析到 {len(weights)} 个权重配置")

    try:
        task_data = load_task_data(args.search_dir_base, args.task_folder)
    except Exception as exc:
        print(f"❌ 加载任务数据失败: {exc}")
        return 1
    print(f"✅ 任务加载完成，超类数量: {len(task_data)}")

    results = evaluate_all_weights(weights, task_data, args.acc_mode, sort_by_avg=args.sort)
    out_path = generate_application_report(
        args.findl_report, args.task_folder, args.acc_mode, results, args.report_dir, sorted_by_avg=args.sort
    )
    # 生成简化版报告（仅权重与平均差值）
    simplified_path = generate_simplified_report(
        args.findl_report, args.task_folder, args.acc_mode, results, args.report_dir,
        weights=weights
    )
    print('=' * 80)
    print('🎯 评估完成')
    print('=' * 80)
    print(f"详细报告: {out_path}")
    print(f"简化报告: {simplified_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
