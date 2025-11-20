#!/usr/bin/env python3
"""L1+L2 权重三维可视化 - 解析简化报告并绘制 3D 散点图

文件作用：
- 解析简化版评估报告（仅权重与平均差值），提取 (w_l1, w_sep, w_sil, avg_diff)
- 使用 matplotlib 在 3D 坐标系中绘制散点，并以颜色映射 avg_diff（越接近 0 越好）

注意：
- 简化报告行格式由 weight_application_evaluator.generate_simplified_report 生成：
  权重N [w_l1:X, w_sep:Y, w_sil:Z] 平均差值: ±X.XXX
- 解析时跳过 "N/A" 行，避免绘图异常
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # 兼容无显示环境
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (确保3D后端注册)


def parse_simplified_report(report_path: str) -> List[Tuple[int, int, int, float]]:
    """解析简化报告，返回 [(w_l1, w_sep, w_sil, avg_diff), ...]。

    跳过 "N/A" 的条目；若文件不存在或无有效数据，将抛出异常。
    """
    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"简化报告不存在: {report_path}")

    # 匹配示例：权重1 [w_l1:9, w_sep:5, w_sil:6] 平均差值: -0.012
    pattern = re.compile(
        r"\[\s*w_l1\s*:\s*(\d+)\s*,\s*w_sep\s*:\s*(\d+)\s*,\s*w_sil\s*:\s*(\d+)\s*\]\s*"
        r"平均差值\s*:\s*([+\-]?\d+(?:\.\d+)?|N/A)")

    data: List[Tuple[int, int, int, float]] = []
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            wl1, wsep, wsil, val = m.groups()
            if val.strip().upper() == 'N/A':
                continue
            try:
                avg = float(val)
            except ValueError:
                continue
            data.append((int(wl1), int(wsep), int(wsil), avg))

    if not data:
        raise ValueError("简化报告中没有可用的数据条目（可能均为 N/A 或格式不匹配）")
    return data


def plot_weight_3d_scatter(
    data: List[Tuple[int, int, int, float]],
    output_path: str,
    cmap: str = 'RdYlGn',
    elev: int = 20,
    azim: int = 45,
    top_k: int = 3,
    dpi: int = 300,
    fixed_range: Optional[Tuple[int, int]] = None,
) -> str:
    """绘制三维散点图并保存。

    - 三轴：w_l1, w_sep, w_sil
    - 颜色：avg_diff 线性归一化到 [0,1] 后映射到 cmap（更接近 0 → 更偏向绿色）
    - 标注 Top-K（avg_diff 最大者）
    返回输出图片路径
    """
    if not data:
        raise ValueError("空数据，无法绘图")

    xs = [t[0] for t in data]
    ys = [t[1] for t in data]
    zs = [t[2] for t in data]
    vals = [t[3] for t in data]

    vmin = min(vals)
    vmax = max(vals)
    if abs(vmax - vmin) < 1e-12:
        norm_vals = [0.5 for _ in vals]
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        norm_vals = [norm(v) for v in vals]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=norm_vals, cmap=cmap, s=40, depthshade=True, edgecolors='k', linewidths=0.4)

    ax.set_xlabel('w_l1')
    ax.set_ylabel('w_sep')
    ax.set_zlabel('w_sil')
    ax.set_title('L1+L2 Weight Space (Green=Better Performance)')
    ax.view_init(elev=elev, azim=azim)

    # 固定全局坐标范围（如 [0, 20]）
    if fixed_range is not None:
        ax.set_xlim(fixed_range)
        ax.set_ylim(fixed_range)
        ax.set_zlim(fixed_range)

    # 颜色条
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)
    cbar.set_label('Average Difference (Green = Better)')

    # 标注 Top-K（avg_diff 值最大）
    try:
        top_k = max(0, min(top_k, len(data)))
        if top_k > 0 and fixed_range is None:
            # 索引按 vals 降序排列（数值越大越好）
            idx_sorted = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:top_k]
            for i in idx_sorted:
                # 仅文本标注，不再绘制遮挡性星标
                ax.text(
                    xs[i], ys[i], zs[i] + 0.5,  # z 轴上移，避免覆盖散点
                    f"({xs[i]},{ys[i]},{zs[i]})\n{vals[i]:.3f}",
                    fontsize=8,
                    ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.3')
                )
    except Exception:
        pass

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
