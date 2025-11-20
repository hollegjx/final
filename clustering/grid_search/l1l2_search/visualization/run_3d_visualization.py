#!/usr/bin/env python3
"""从简化版权重评估报告生成 3D 可视化图

用法示例：
  python -m clustering.grid_search.l1l2_search.visualization.run_3d_visualization \
    --simplified_report /data/gjx/checkpoints/weight_application/41_..._simplified.txt \
    --output_path /data/gjx/checkpoints/weight_application/weights_3d.png \
    --cmap RdYlGn --elev 20 --azim 45

说明：
  - 仅依赖简化版报告（生成自 run_weight_application.py）
  - 三轴为 (w_l1, w_sep, w_sil)，颜色映射为平均差值（越接近 0 越好）
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from .weight_3d_visualizer import parse_simplified_report, plot_weight_3d_scatter


def _default_output_path(report_path: str) -> str:
    base = os.path.splitext(os.path.basename(report_path))[0]
    out_dir = os.path.dirname(report_path) or '.'
    return os.path.join(out_dir, f"{base}_3d.png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='将简化版权重评估报告可视化为 3D 散点图',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--simplified_report', type=str, required=True, help='简化报告路径 (*.txt)')
    parser.add_argument('--output_path', type=str, default=None, help='输出图片路径（省略则同目录 *_3d.png）')
    parser.add_argument('--cmap', type=str, default='RdYlGn', help='颜色映射 (matplotlib colormap)')
    parser.add_argument('--elev', type=int, default=20, help='视角仰角')
    parser.add_argument('--azim', type=int, default=45, help='视角方位角')
    parser.add_argument('--top_k', type=int, default=3, help='标注Top-K权重')
    parser.add_argument('--dpi', type=int, default=300, help='输出DPI')
    args = parser.parse_args()

    try:
        data = parse_simplified_report(args.simplified_report)
    except Exception as exc:
        print(f"❌ 解析简化报告失败: {exc}")
        return 1

    out_path = args.output_path or _default_output_path(args.simplified_report)

    try:
        # 局部视图（自动范围，允许 Top-K 文本标注）
        saved_local = plot_weight_3d_scatter(
            data,
            out_path,
            cmap=args.cmap,
            elev=args.elev,
            azim=args.azim,
            top_k=args.top_k,
            dpi=args.dpi,
            fixed_range=None,
        )
        print(f"✅ 局部视图已生成: {saved_local}")
        print(f"  数据点数量: {len(data)}")
        print(f"  颜色映射: {args.cmap} (Green=Better)")

        # 全局视图（固定 [0,20]，不标注）
        global_out_path = out_path.replace('_3d.png', '_3d_global.png')
        if global_out_path == out_path:
            base, ext = os.path.splitext(out_path)
            global_out_path = f"{base}_global{ext}"

        saved_global = plot_weight_3d_scatter(
            data,
            global_out_path,
            cmap=args.cmap,
            elev=args.elev,
            azim=args.azim,
            top_k=0,
            dpi=args.dpi,
            fixed_range=(0, 20),
        )
        print(f"✅ 全局视图已生成: {saved_global}")
        print(f"  坐标范围: [0, 20] × [0, 20] × [0, 20]")
        print(f"  颜色映射: {args.cmap} (Green=Better)")
        return 0
    except Exception as exc:
        print(f"❌ 绘图失败: {exc}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
