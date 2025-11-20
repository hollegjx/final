#!/usr/bin/env python3
"""L1+L2 权重区域搜索（基于已生成热力图文件名的离线分析）

职责：
- 解析 {metric}_{acc:.4f}_{w_l1}_{w_sep}_{w_sil}.png 文件名，提取 ACC 与权重
- 按超类与阈值过滤满足条件的权重集合
- 计算多超类共同满足条件的权重交集
- 收集各超类在这些权重下的 ACC 并生成文本报告

说明：
- 不做任何重新计算，完全基于已有热力图文件名
- acc_mode 与目录映射：all -> all/, new -> new/, old -> old/
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

from config import l1l2_region_report_dir

# WeightTriplet 在本模块内不直接使用，解析结果以整数三元组表示


HEATMAP_NAME_PATTERN = re.compile(r"^(?P<metric>\w+?)_(?P<acc>[-\d.]+)_(?P<wl1>\d+)_(?P<wsep>\d+)_(?P<wsil>\d+)\.png$")


def parse_heatmap_filename(filename: str) -> Optional[Tuple[str, float, int, int, int]]:
    """解析热力图文件名: metric_acc_wl1_wsep_wsil.png。

    返回:
        (metric, acc, w_l1, w_sep, w_sil) 或 None 当不匹配时。
    """
    m = HEATMAP_NAME_PATTERN.match(filename)
    if not m:
        return None
    groups = m.groupdict()
    return (
        groups["metric"],
        float(groups["acc"]),
        int(groups["wl1"]),
        int(groups["wsep"]),
        int(groups["wsil"]),
    )


def detect_superclasses_with_l1l2_results(output_dir: str) -> List[str]:
    """检测在 output_dir 下已完成 L1L2 搜索的超类（含 all/new/old 子目录）。"""
    if not os.path.exists(output_dir):
        return []
    superclasses: List[str] = []
    for item in os.listdir(output_dir):
        spath = os.path.join(output_dir, item)
        if not os.path.isdir(spath):
            continue
        if all(os.path.isdir(os.path.join(spath, sub)) for sub in ("all", "new", "old")):
            superclasses.append(item)
    return sorted(superclasses)


def detect_available_task_folders(output_dir: str) -> List[str]:
    """检测在 output_dir 下所有有效的 L1L2 任务文件夹。

    返回格式：['15class_11_06_22_30', '4class_11_06_21_06', ...]
    按时间戳降序排序（最新的在前）。
    有效性标准：
      1) 子目录名匹配 ^\d+class_\d{2}_\d{2}_\d{2}_\d{2}$
      2) 目录下存在至少一个有效的超类子目录（含 all/new/old）
    """
    if not os.path.exists(output_dir):
        return []

    pattern = re.compile(r'^\d+class_\d{2}_\d{2}_\d{2}_\d{2}$')
    tasks: List[str] = []

    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if not os.path.isdir(item_path):
            continue
        if not pattern.match(item):
            continue
        # 验证该任务目录下存在有效超类
        superclasses = detect_superclasses_with_l1l2_results(item_path)
        if superclasses:
            tasks.append(item)

    def _ts_key(task_name: str) -> tuple:
        # 任务名格式: <N>class_MM_DD_HH_MM
        parts = task_name.split('_')
        if len(parts) >= 5:
            try:
                mm, dd, hh, mi = int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1])
                return (mm, dd, hh, mi)
            except Exception:
                return (0, 0, 0, 0)
        return (0, 0, 0, 0)

    tasks.sort(key=_ts_key, reverse=True)
    return tasks


def _acc_mode_to_folder(acc_mode: str) -> str:
    mode = acc_mode.strip().lower()
    if mode not in {"all", "new", "old"}:
        raise ValueError("acc_mode 必须是 all/new/old 之一")
    return mode


def filter_weights_by_threshold(superclass: str, threshold: float, output_dir: str, acc_mode: str) -> Set[Tuple[int, int, int]]:
    """从指定超类的指定 acc_mode 目录中过滤满足阈值的权重组合集合。

    返回集合中元素为 (w_l1, w_sep, w_sil) 整数三元组。
    """
    folder = os.path.join(output_dir, superclass, _acc_mode_to_folder(acc_mode))
    valid: Set[Tuple[int, int, int]] = set()
    if not os.path.isdir(folder):
        return valid
    for fname in os.listdir(folder):
        if not fname.endswith('.png'):
            continue
        parsed = parse_heatmap_filename(fname)
        if not parsed:
            continue
        metric, acc, wl1, wsep, wsil = parsed
        # 文件所在文件夹即 acc_mode，因此无需用 metric 再过滤
        try:
            if float(acc) >= float(threshold):
                valid.add((wl1, wsep, wsil))
        except Exception:
            continue
    return valid


def find_common_weights(superclass_thresholds: Dict[str, float], output_dir: str, acc_mode: str) -> Set[Tuple[int, int, int]]:
    """对每个超类执行阈值过滤后，求权重组合交集。"""
    sets: List[Set[Tuple[int, int, int]]] = []
    for sc, th in superclass_thresholds.items():
        s = filter_weights_by_threshold(sc, th, output_dir, acc_mode)
        sets.append(s)
    if not sets:
        return set()
    common = sets[0].copy()
    for s in sets[1:]:
        common &= s
    return common


def collect_weight_details(common_weights: Set[Tuple[int, int, int]],
                           superclasses: Sequence[str],
                           output_dir: str,
                           acc_mode: str) -> List[Dict[str, object]]:
    """收集每个权重在各超类下的 ACC 值详情。

    返回每个元素格式：
        {
          'weights': (wl1,wsep,wsil),
          'acc': {superclass: acc_value, ...},
          'avg_acc': 平均值,
          'min_acc': 最小值
        }
    """
    folder_name = _acc_mode_to_folder(acc_mode)
    results: List[Dict[str, object]] = []
    for wl1, wsep, wsil in sorted(common_weights):
        acc_map: Dict[str, float] = {}
        for sc in superclasses:
            dir_path = os.path.join(output_dir, sc, folder_name)
            if not os.path.isdir(dir_path):
                continue
            # 查找对应权重的文件：*_{wl1}_{wsep}_{wsil}.png
            target_suffix = f"_{wl1}_{wsep}_{wsil}.png"
            matched_file = None
            for fname in os.listdir(dir_path):
                if fname.endswith(target_suffix):
                    matched_file = fname
                    break
            if not matched_file:
                continue
            parsed = parse_heatmap_filename(matched_file)
            if not parsed:
                continue
            metric, acc, _wl1, _wsep, _wsil = parsed
            acc_map[sc] = float(acc)
        if not acc_map:
            continue
        vals = list(acc_map.values())
        entry = {
            'weights': (wl1, wsep, wsil),
            'acc': acc_map,
            'avg_acc': sum(vals) / len(vals),
            'min_acc': min(vals),
        }
        results.append(entry)
    # 默认按平均 ACC 降序，再按最小 ACC 降序排序
    results.sort(key=lambda e: (e['avg_acc'], e['min_acc']), reverse=True)
    return results


def generate_region_report(details: List[Dict[str, object]],
                           superclass_thresholds: Dict[str, float],
                           acc_mode: str,
                           report_dir: str = l1l2_region_report_dir,
                           task_folder: Optional[str] = None) -> str:
    """生成区域搜索报告，返回报告路径。

    当提供 task_folder 时，报告文件名采用 "<重合数量>_<task_folder>.txt"，
    否则退回旧命名逻辑（超类列表+当前时间戳）。
    """
    os.makedirs(report_dir, exist_ok=True)

    num_common = len(details)
    if task_folder:
        filename = f"{num_common}_{task_folder}.txt"
    else:
        names = sorted(list(superclass_thresholds.keys()))
        ts = datetime.now().strftime('%m_%d_%H_%M')
        filename = f"{'_'.join(names)}_{ts}.txt" if names else f"region_{ts}.txt"
    path = os.path.join(report_dir, filename)

    lines: List[str] = []
    lines.append('=' * 60)
    lines.append('L1+L2 权重区域搜索报告')
    lines.append('=' * 60)
    lines.append(f"搜索时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"ACC模式: {acc_mode}")
    lines.append('')
    lines.append('超类阈值配置:')
    names = sorted(list(superclass_thresholds.keys()))
    for sc in names:
        lines.append(f"  {sc}: >= {superclass_thresholds[sc]:.4f}")
    lines.append('')
    lines.append('=' * 60)
    lines.append(f"重合权重数量: {len(details)}")
    lines.append('=' * 60)
    lines.append('')

    for idx, item in enumerate(details, 1):
        wl1, wsep, wsil = item['weights']  # type: ignore[assignment]
        lines.append(f"#{idx} 权重组合: (w_l1={wl1}, w_sep={wsep}, w_sil={wsil})")
        acc_map: Dict[str, float] = item['acc']  # type: ignore[assignment]
        for sc in names:
            val = acc_map.get(sc)
            if val is None:
                lines.append(f"   {sc}:  N/A")
            else:
                lines.append(f"   {sc}:  {acc_mode}_acc = {val:.4f}")
        lines.append('')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path
