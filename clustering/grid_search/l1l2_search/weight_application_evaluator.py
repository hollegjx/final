#!/usr/bin/env python3
"""权重应用评估核心逻辑

文件作用：
- 解析 findL 区域搜索报告中的重合权重 (w_l1, w_sep, w_sil)
- 从指定任务(search目录)加载各超类的原始搜索结果与可用组件
- 复用 L1+L2 动态加权逻辑，对每个权重×超类评估最优点ACC与全局最佳ACC的差值
- 生成结构化结果供 CLI 入口写入报告

实现约束：
- 仅依赖原始搜索数据（search_dir_base/{task_folder}/{superclass}/*.txt）
- 组件按“动态检测 + 自适应加权”，与现有 L1+L2 搜索保持一致
"""

from __future__ import annotations

import os
import re
import sys
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from clustering.grid_search.heatmap import detect_available_superclasses
from .l1l2_weight_calculator import (
    WeightTriplet,
    load_raw_scores,
    compute_weighted_l1l2,
)


def parse_findl_weights(report_path: str) -> List[WeightTriplet]:
    """从 findL 区域搜索报告中解析所有权重组合。

    兼容行形如：
      "#1 权重组合: (w_l1=1, w_sep=4, w_sil=8)"

    返回去重后的 WeightTriplet 列表，按 (w_l1, w_sep, w_sil) 升序。
    """
    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"报告不存在: {report_path}")
    pattern = re.compile(r"w_l1\s*=\s*(\d+).*?w_sep\s*=\s*(\d+).*?w_sil\s*=\s*(\d+)")
    seen: set[Tuple[int, int, int]] = set()
    ordered: List[WeightTriplet] = []
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            wl1, wsep, wsil = int(m.group(1)), int(m.group(2)), int(m.group(3))
            key = (wl1, wsep, wsil)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(WeightTriplet(wl1, wsep, wsil))
    if not ordered:
        raise ValueError("未在报告中解析到任何权重配置（期望包含 w_l1=, w_sep=, w_sil=）")
    # 保持报告原始顺序返回
    return ordered


def load_task_data(search_dir_base: str, task_folder: str) -> Dict[str, Tuple[Dict, List[str]]]:
    """加载任务下所有超类的原始搜索数据与可用组件。

    返回 {superclass: (results_dict, available_components)}。
    若个别超类加载失败则自动跳过。
    """
    task_path = os.path.join(search_dir_base, task_folder)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(f"任务目录不存在: {task_path}")

    superclasses = detect_available_superclasses(task_path)
    if not superclasses:
        raise ValueError(f"未在任务目录中发现任何超类结果: {task_path}")

    out: Dict[str, Tuple[Dict, List[str]]] = {}
    for sc in superclasses:
        try:
            loaded = load_raw_scores(sc, search_dir=task_path)
            if not loaded:
                print(f"⚠️ 超类 {sc} 数据为空，跳过")
                continue
            results_dict, available_components = loaded
            out[sc] = (results_dict, available_components)
            print(f"ℹ️ 超类 {sc} 可用组件: {','.join(available_components)}")
        except Exception as exc:
            print(f"⚠️ 超类 {sc} 加载失败: {exc}")
    return out


def _best_acc_over_grid(results_dict: Dict, acc_mode: str) -> Optional[float]:
    best: Optional[float] = None
    for metrics in results_dict.values():
        v = metrics.get(acc_mode)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if best is None or fv > best:
            best = fv
    return best


def evaluate_weight_on_superclass(results_dict: Dict,
                                  available_components: List[str],
                                  weight: WeightTriplet,
                                  acc_mode: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """对单个超类应用单个权重，返回 (diff, optimal_acc, best_acc)。

    diff = optimal_acc - best_acc；若无法计算则返回 (None, None, best_acc)。
    """
    if not results_dict:
        return None, None, None

    best_acc = _best_acc_over_grid(results_dict, acc_mode)
    try:
        weighted = compute_weighted_l1l2(results_dict, weight, available_components)
        if not weighted:
            return None, None, best_acc
        # 选择 combined_score 最小的点
        best_key, best_entry = min(weighted.items(), key=lambda kv: kv[1]['combined_score'])
        optimal_acc = weighted[best_key].get(acc_mode)
        if optimal_acc is None or best_acc is None:
            return None, optimal_acc if optimal_acc is not None else None, best_acc
        return float(optimal_acc) - float(best_acc), float(optimal_acc), float(best_acc)
    except Exception:
        return None, None, best_acc


def evaluate_all_weights(weights: Sequence[WeightTriplet],
                         task_data: Dict[str, Tuple[Dict, List[str]]],
                         acc_mode: str,
                         sort_by_avg: bool = False) -> List[Dict[str, object]]:
    """对所有权重在所有超类上评估，返回结构化结果。

    返回列表中每个元素：
        {
          'weight': (wl1,wsep,wsil),
          'results': {superclass: {'diff': float|None, 'optimal': float|None, 'best': float|None}},
          'stats': {'pos': int, 'neg': int, 'zero': int, 'fail': int, 'avg': float|None, 'median': float|None}
        }
    """
    outputs: List[Dict[str, object]] = []
    for w in weights:
        per_sc: Dict[str, Dict[str, Optional[float]]] = {}
        diffs: List[float] = []
        pos = neg = zero = fail = 0
        for sc, (results_dict, available_components) in task_data.items():
            diff, opt_acc, best_acc = evaluate_weight_on_superclass(results_dict, available_components, w, acc_mode)
            entry = {'diff': diff, 'optimal': opt_acc, 'best': best_acc}
            per_sc[sc] = entry
            if diff is None:
                fail += 1
            else:
                diffs.append(float(diff))
                if diff > 1e-12:
                    pos += 1
                elif diff < -1e-12:
                    neg += 1
                else:
                    zero += 1
        avg_val = float(sum(diffs) / len(diffs)) if diffs else None
        med_val = float(median(diffs)) if diffs else None
        outputs.append({
            'weight': (int(w.w_l1), int(w.w_sep), int(w.w_sil)),
            'results': per_sc,
            'stats': {'pos': pos, 'neg': neg, 'zero': zero, 'fail': fail, 'avg': avg_val, 'median': med_val},
        })

    # 可选排序：按平均差值降序（平均差值越大越好，None排在最后）
    if sort_by_avg:
        outputs.sort(key=lambda x: x['stats']['avg'] if x['stats']['avg'] is not None else float('-inf'), reverse=True)
    return outputs


def generate_application_report(findl_report_path: str,
                                task_folder: str,
                                acc_mode: str,
                                results: List[Dict[str, object]],
                                report_dir: str,
                                sorted_by_avg: bool = False) -> str:
    """将评估结果写入报告文件，并返回路径。"""
    os.makedirs(report_dir, exist_ok=True)
    report_name = os.path.splitext(os.path.basename(findl_report_path))[0]
    out_path = os.path.join(report_dir, f"{report_name}_applied_to_{task_folder}.txt")

    lines: List[str] = []
    lines.append('=' * 40)
    lines.append('权重应用评估报告')
    lines.append('=' * 40)
    lines.append(f"来源报告: {os.path.basename(findl_report_path)}")
    lines.append(f"目标任务: {task_folder}")
    # 超类数量：从一个结果项取长度（若为空则0）
    sc_count = 0
    if results:
        any_item = results[0]
        sc_count = len(any_item.get('results', {}))  # type: ignore[arg-type]
    lines.append(f"超类数量: {sc_count}")
    lines.append(f"ACC指标: {acc_mode}")
    lines.append(f"权重排序: {'按平均差值降序' if sorted_by_avg else '按findL报告原序'}")
    import datetime as _dt
    lines.append(f"评估时间: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('=' * 40)
    lines.append('')

    for idx, item in enumerate(results, 1):
        wl1, wsep, wsil = item['weight']  # type: ignore[assignment]
        stats = item['stats']  # type: ignore[assignment]
        per_sc: Dict[str, Dict[str, Optional[float]]] = item['results']  # type: ignore[assignment]
        lines.append(f"权重{idx} (w_l1:{wl1}, w_sep:{wsep}, w_sil:{wsil}):")
        for sc in sorted(per_sc.keys()):
            r = per_sc[sc]
            d = r.get('diff')
            opt = r.get('optimal')
            bst = r.get('best')
            if d is None:
                lines.append(f"  {sc}: N/A")
            else:
                sign = '+' if d > 0 else ''
                if opt is None or bst is None:
                    lines.append(f"  {sc}: {sign}{d:.3f}")
                else:
                    lines.append(f"  {sc}: {sign}{d:.3f} ({opt:.3f}/{bst:.3f})")
        lines.append(f"  [正向: {stats['pos']}, 负向: {stats['neg']}, 持平: {stats['zero']}, 失败: {stats['fail']}]")
        avg_val = stats.get('avg')
        med_val = stats.get('median')
        if avg_val is not None and med_val is not None:
            lines.append(f"  [平均差值: {avg_val:.3f}, 中位数差值: {med_val:.3f}]")
        lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"📄 评估报告已保存: {out_path}")
    return out_path


def generate_simplified_report(findl_report_path: str,
                               task_folder: str,
                               acc_mode: str,
                               results: List[Dict[str, object]],
                               report_dir: str,
                               weights: Sequence[WeightTriplet]) -> str:
    """生成简化版评估报告，仅显示每个权重的平均差值。

    - 保持传入 results 的顺序（受 --sort 控制）。
    - 文件名：{report_name}_applied_to_{task_folder}_simplified.txt
    """
    os.makedirs(report_dir, exist_ok=True)
    report_name = os.path.splitext(os.path.basename(findl_report_path))[0]
    out_path = os.path.join(report_dir, f"{report_name}_applied_to_{task_folder}_simplified.txt")

    lines: List[str] = []
    lines.append('=' * 60)
    lines.append('权重应用评估报告 (简化版)')
    lines.append('=' * 60)
    lines.append(f"来源报告: {os.path.basename(findl_report_path)}")
    lines.append(f"目标任务: {task_folder}")
    lines.append(f"ACC指标: {acc_mode}")
    lines.append("权重排序: 保持findL报告原序")
    import datetime as _dt
    lines.append(f"评估时间: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('=' * 60)
    lines.append('')

    # 构建 (wl1,wsep,wsil) -> result 映射
    results_map: Dict[Tuple[int, int, int], Dict[str, object]] = {}
    for item in results:
        key = item['weight']  # type: ignore[assignment]
        results_map[key] = item  # type: ignore[assignment]

    # 按原始 weights 顺序输出
    for idx, w in enumerate(weights, 1):
        wl1, wsep, wsil = int(w.w_l1), int(w.w_sep), int(w.w_sil)
        item = results_map.get((wl1, wsep, wsil))
        if not item:
            continue
        stats = item['stats']  # type: ignore[assignment]
        avg_val = stats.get('avg')
        weight_str = f"[w_l1:{wl1}, w_sep:{wsep}, w_sil:{wsil}]"
        if avg_val is not None:
            sign = '+' if avg_val > 0 else ''
            avg_str = f"{sign}{avg_val:.3f}"
        else:
            avg_str = "N/A"
        lines.append(f"权重{idx} {weight_str} 平均差值: {avg_str}")

    lines.append('')
    lines.append('=' * 60)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"📄 简化版报告已保存: {out_path}")
    return out_path
