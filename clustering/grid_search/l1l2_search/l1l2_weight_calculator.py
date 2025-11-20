#!/usr/bin/env python3
"""L1+L2联合权重计算器

文件作用：
- 加载网格搜索结果并进行可用性校验（动态检测可用 L2 组件）
- 基于 (w_l1, w_sep, w_sil) 进行自适应加权：仅对“当前样本中存在”的 L2 组件分配 L2 总权重
- 生成权重组合（自动与手动）供上层探索与绘图

重要变更（动态组件）：
- 不再强制 separation、silhouette 同时存在；只要 L1 存在且“至少一个 L2 组件（separation/silhouette/penalty）”存在即可计算
- 若某些组件缺失，则将 L2 总权重在“可用组件”之间按比例或等分重分配；分母仍为原始三元组之和 ``weights.total`` 以保持向后兼容
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from config import grid_search_output_dir

# 确保可以导入 heatmap.load_existing_results
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from clustering.grid_search.heatmap import load_existing_results


@dataclass(frozen=True)
class WeightTriplet:
    """权重组合的不可变表示，便于作为字典键。"""

    w_l1: float
    w_sep: float
    w_sil: float

    def __post_init__(self) -> None:
        precision = 6
        object.__setattr__(self, 'w_l1', round(float(self.w_l1), precision))
        object.__setattr__(self, 'w_sep', round(float(self.w_sep), precision))
        object.__setattr__(self, 'w_sil', round(float(self.w_sil), precision))

    @property
    def total(self) -> float:
        return float(self.w_l1 + self.w_sep + self.w_sil)


def get_weight_configurations(weight_sum: int = 10) -> List[WeightTriplet]:
    """按总和穷举所有非负整数三元组 (w_l1, w_sep, w_sil)。

    目标：
    - 给定权重总和 `weight_sum`，返回所有满足 `w_l1 + w_sep + w_sil = weight_sum` 的整数组合；
      允许某些分量为 0，以覆盖极端点（如 (weight_sum,0,0)）。

    约束：
    - `weight_sum` 必须为整数且 >= 1。
    - 结果按 (w_l1, w_sep, w_sil) 升序排序。
    - 返回值规模为 C(weight_sum+2, 2)，即 (S+2 取 2)。
    """

    if not isinstance(weight_sum, int):
        raise TypeError("weight_sum 必须为整数")
    if weight_sum < 1:
        raise ValueError("weight_sum 必须大于等于 1")

    configs: List[WeightTriplet] = []
    for wl1 in range(0, weight_sum + 1):
        remaining = weight_sum - wl1
        for wsep in range(0, remaining + 1):
            wsil = remaining - wsep
            configs.append(WeightTriplet(wl1, wsep, wsil))

    return configs


def _extract_silhouette(metrics: Dict) -> Optional[float]:
    """兼容两种来源的轮廓系数字段。"""

    if metrics.get('silhouette') is not None:
        return float(metrics['silhouette'])
    comp = metrics.get('l2_components', {}).get('silhouette')
    if comp and comp.get('value') is not None:
        return float(comp['value'])
    return None


def _extract_separation(metrics: Dict) -> Optional[float]:
    """兼容两种来源的 separation 字段。"""

    if metrics.get('separation_score') is not None:
        return float(metrics['separation_score'])
    comp = metrics.get('l2_components', {}).get('separation')
    if comp and comp.get('value') is not None:
        return float(comp['value'])
    return None


def _extract_penalty(metrics: Dict) -> Optional[float]:
    """兼容两种来源的 penalty 字段。"""

    if metrics.get('penalty_score') is not None:
        return float(metrics['penalty_score'])
    comp = metrics.get('l2_components', {}).get('penalty')
    if comp and comp.get('value') is not None:
        return float(comp['value'])
    return None


def load_raw_scores(
    superclass_name: str,
    search_dir: str = grid_search_output_dir
) -> Optional[Tuple[Dict, List[str]]]:
    """加载 batch_runner 结果并动态检测可用组件。

    返回:
        (results_dict, available_components)
        - available_components ∈ {'separation_score','silhouette','penalty_score'} 的子集
    规则:
        - 必须存在: l1_loss（至少一个样本）
        - 至少一个 L2 组件存在（全局至少一个样本出现过即可）
    """

    print(f"📂 加载 {superclass_name} 的网格搜索结果…")
    results = load_existing_results(superclass_name, search_dir)
    if not results:
        return None

    total = len(results)
    l1_available = sum(1 for v in results.values() if v.get('l1_loss') is not None)
    sep_available = sum(1 for v in results.values() if _extract_separation(v) is not None)
    pen_available = sum(1 for v in results.values() if _extract_penalty(v) is not None)
    sil_available = sum(1 for v in results.values() if _extract_silhouette(v) is not None)

    print(
        f"📊 字段覆盖率: l1={l1_available}/{total}, separation={sep_available}/{total}, "
        f"penalty={pen_available}/{total}, silhouette={sil_available}/{total}"
    )

    available_components: List[str] = []
    if sep_available > 0:
        available_components.append('separation_score')
    if pen_available > 0:
        available_components.append('penalty_score')
    if sil_available > 0:
        available_components.append('silhouette')

    if l1_available == 0:
        print("❌ l1_loss 完全缺失，无法进行 L1+L2 搜索")
        return None
    if not available_components:
        print("❌ L2 组件完全缺失（separation/penalty/silhouette 均为0），无法进行 L1+L2 搜索")
        return None

    skipped_components = [c for c in ['separation_score', 'penalty_score', 'silhouette'] if c not in available_components]
    print(f"ℹ️  检测到可用组件: l1_loss{(',' if available_components else '')} {', '.join(available_components) if available_components else ''}")
    if skipped_components:
        print(f"⚠️  跳过的组件: {', '.join(skipped_components)}（数据缺失）")

    return results, available_components


def validate_l1l2_requirements(
    results_dict: Dict,
    available_components: List[str],
    coverage_threshold: float = 0.5
) -> Tuple[bool, str]:
    """按“至少一个 L2 可用”的标准验证有效覆盖率。

    判定一个样本有效：l1_loss 存在 且 在 available_components 中至少一个组件存在。
    """

    if not results_dict:
        return False, "结果为空"
    total = len(results_dict)

    def has_any_l2(m: Dict) -> bool:
        for name in available_components:
            if name == 'separation_score' and m.get('separation_score') is not None:
                return True
            if name == 'penalty_score' and m.get('penalty_score') is not None:
                return True
            if name == 'silhouette' and _extract_silhouette(m) is not None:
                return True
        return False

    valid = sum(1 for m in results_dict.values() if m.get('l1_loss') is not None and has_any_l2(m))
    coverage = valid / total if total else 0.0
    if coverage < coverage_threshold:
        return False, f"有效数据覆盖率不足: {coverage:.1%} (有效/总: {valid}/{total})"
    used = ','.join(available_components) if available_components else 'N/A'
    return True, f"✅ 使用组件[{used}] 的有效占比 {coverage:.1%} ({valid}/{total})"


def compute_weighted_l1l2(
    results_dict: Dict[Tuple[int, int], Dict],
    weights: WeightTriplet,
    available_components: List[str],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """按指定权重组合计算综合得分（动态自适应 L2 组件）。

    规则：
    - L2 总权重 = weights.w_sep + weights.w_sil（保持接口兼容）
    - 将 L2 总权重在“当前样本中存在”的 L2 组件之间重分配：
        • 若这些组件在原始三元组中有非零基准权重，则按比例分配；
        • 若基准和为 0（如仅 penalty 存在），则在存在的组件间等分。
    - 合成方向：l1_loss（越小越好，正权相加）；separation/silhouette（越大越好，负号相加）；penalty_score（越小越好，正号相加）。
    - 分母始终使用原始 ``weights.total`` 以保持与图名/汇总一致。
    """

    if weights.total <= 0:
        raise ValueError("权重总和必须大于0")

    l2_total = float(weights.w_sep + weights.w_sil)
    base_map = {
        'separation_score': float(weights.w_sep),
        'silhouette': float(weights.w_sil),
        'penalty_score': 0.0,  # 无单独配置，作为动态分配候选
    }

    combined: Dict[Tuple[int, int], Dict[str, float]] = {}
    skipped_l1 = 0
    skipped_no_l2 = 0

    for key, metrics in results_dict.items():
        l1_val = metrics.get('l1_loss')
        if l1_val is None:
            skipped_l1 += 1
            continue

        # 当前样本中可用的 L2 组件及其值
        values: Dict[str, Optional[float]] = {
            'separation_score': _extract_separation(metrics) if 'separation_score' in available_components else None,
            'penalty_score': _extract_penalty(metrics) if 'penalty_score' in available_components else None,
            'silhouette': _extract_silhouette(metrics) if 'silhouette' in available_components else None,
        }
        present = [name for name, val in values.items() if val is not None]
        if not present:
            skipped_no_l2 += 1
            continue

        # 在“当前样本存在的组件”之间重分配 L2 总权重
        base_sum = sum(base_map.get(name, 0.0) for name in present)
        if base_sum > 1e-12:
            weights_l2 = {name: (base_map[name] / base_sum) * l2_total for name in present}
        else:
            share = l2_total / float(len(present)) if len(present) > 0 else 0.0
            weights_l2 = {name: share for name in present}

        # 合成得分（越小越好）
        score_num = float(weights.w_l1) * float(l1_val)
        # 分别累加 L2 贡献（注意方向）
        for name in present:
            val = float(values[name])  # type: ignore[arg-type]
            w = float(weights_l2[name])
            if name in ('separation_score', 'silhouette'):
                score_num += -w * val  # 越大越好 → 取负
            elif name == 'penalty_score':
                score_num += +w * val  # 越小越好 → 取正

        score = score_num / float(weights.total)

        # 输出项：尽可能保留已用到的原始指标，便于绘图/报告
        out_entry = {
            'combined_score': score,
            'all_acc': metrics.get('all_acc'),
            'old_acc': metrics.get('old_acc'),
            'new_acc': metrics.get('new_acc'),
            'l1_loss': float(l1_val),
        }
        if values['separation_score'] is not None:
            out_entry['separation_score'] = float(values['separation_score'])  # type: ignore[index]
        if values['silhouette'] is not None:
            out_entry['silhouette'] = float(values['silhouette'])  # type: ignore[index]
        if values['penalty_score'] is not None:
            out_entry['penalty_score'] = float(values['penalty_score'])  # type: ignore[index]

        combined[key] = out_entry  # type: ignore[assignment]

    if skipped_l1 or skipped_no_l2:
        print(
            f"⚠️ 权重 {weights}: 跳过 {skipped_l1} 个样本(无l1) / {skipped_no_l2} 个样本(无可用L2)"
        )
    return combined


def enumerate_weight_grid(
    w_l1_values: Iterable[float],
    w_sep_values: Iterable[float],
    w_sil_values: Iterable[float],
) -> Iterable[WeightTriplet]:
    """生成权重组合的笛卡尔积。"""

    for wl1 in w_l1_values:
        for wsep in w_sep_values:
            for wsil in w_sil_values:
                yield WeightTriplet(float(wl1), float(wsep), float(wsil))


def summarize_best_entry(
    weighted_results: Dict[Tuple[int, int], Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """从单次权重评估结果中选出综合得分最小的(k, dp)。"""

    if not weighted_results:
        return None
    best_key, best_metrics = min(weighted_results.items(), key=lambda item: item[1]['combined_score'])
    summary = {
        'k': best_key[0],
        'density_percentile': best_key[1],
        'combined_score': best_metrics['combined_score'],
        'all_acc': best_metrics.get('all_acc'),
        'old_acc': best_metrics.get('old_acc'),
        'new_acc': best_metrics.get('new_acc'),
    }
    return summary
