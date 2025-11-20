#!/usr/bin/env python3
"""
L2权重计算器模块
基于现有heatmap.py的load_existing_results函数，精确复用其解析逻辑
用于读取网格搜索结果并计算不同权重组合下的加权L2损失
"""

import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

from config import grid_search_output_dir

# 添加项目路径以便导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 直接复用现有的load_existing_results函数
from clustering.grid_search.heatmap import load_existing_results


def load_raw_scores(superclass_name: str, search_dir: str = grid_search_output_dir) -> Optional[Dict]:
    """
    从batch_runner的输出文件中解析原始指标
    直接复用heatmap.py中已经验证的load_existing_results函数
    
    Args:
        superclass_name: 超类名称
        search_dir: 搜索结果目录
    
    Returns:
        dict: {(k, dp): {'all_acc': float, 'separation_score': float, 'penalty_score': float, ...}} 或 None
    """
    print(f"📂 加载 {superclass_name} 的搜索结果...")
    
    # 直接使用现有的解析函数
    results_dict = load_existing_results(superclass_name, search_dir)
    
    if results_dict is None:
        return None
    
    # 验证L2权重探索所需的关键字段
    total_count = len(results_dict)
    sep_available = sum(1 for v in results_dict.values() if v.get('separation_score') is not None)
    pen_available = sum(1 for v in results_dict.values() if v.get('penalty_score') is not None)
    
    print(f"📊 关键指标统计:")
    print(f"   总参数组合: {total_count}")
    print(f"   separation_score可用: {sep_available}/{total_count}")
    print(f"   penalty_score可用: {pen_available}/{total_count}")
    
    if sep_available == 0:
        print("❌ 错误: separation_score完全缺失，无法进行L2权重探索")
        return None
    
    if pen_available == 0:
        print("❌ 错误: penalty_score完全缺失，无法进行L2权重探索")
        return None
    
    valid_pairs = sum(1 for v in results_dict.values() 
                     if v.get('separation_score') is not None and v.get('penalty_score') is not None)
    
    if valid_pairs < total_count * 0.5:  # 至少50%的数据有效
        print(f"⚠️  警告: 有效数据对不足 ({valid_pairs}/{total_count})")
        print("   建议重新运行网格搜索以获取完整的聚类质量指标")
        return None
    
    print(f"✅ 数据验证通过，有效数据对: {valid_pairs}/{total_count}")
    return results_dict


def compute_weighted_l2(results_dict: Dict, w_sep: float, w_pen: float) -> Dict:
    """
    计算加权L2损失
    
    Args:
        results_dict: 从load_raw_scores获取的原始结果字典
        w_sep: separation权重 (1-9)
        w_pen: penalty权重 (1-9)
    
    Returns:
        dict: {(k, dp): {'all_acc': float, 'l2_weighted': float, 'separation_score': float, 'penalty_score': float, ...}}
    """
    weighted_results = {}
    valid_count = 0
    skipped_count = 0
    
    for key, data in results_dict.items():
        separation_score = data.get('separation_score')
        penalty_score = data.get('penalty_score')
        
        # 检查必要数据是否存在
        if separation_score is None or penalty_score is None:
            skipped_count += 1
            continue
        
        # 计算加权L2损失
        # 公式：L2 = (penalty权重 × penalty分数 - separation权重 × separation分数) / (w_sep + w_pen)
        # 注意：separation_score越大越好，penalty_score越大越差
        weight_total = w_sep + w_pen
        if weight_total <= 0:
            raise ValueError("权重总和必须大于0")
        l2_weighted = (w_pen * penalty_score - w_sep * separation_score) / float(weight_total)
        
        # 保留原始数据并添加L2计算结果
        weighted_results[key] = {
            # 基础指标（来自heatmap.py解析）
            'all_acc': data.get('all_acc', 0.0),
            'old_acc': data.get('old_acc', 0.0),
            'new_acc': data.get('new_acc', 0.0),
            'n_clusters': data.get('n_clusters', 0),
            
            # L2相关指标
            'l2_weighted': l2_weighted,
            'separation_score': separation_score,
            'penalty_score': penalty_score,
            'w_sep': w_sep,
            'w_pen': w_pen,
            
            # 可选指标（如果存在）
            'quality_score': data.get('quality_score'),
            'labeled_acc': data.get('labeled_acc'),
            'l1_loss': data.get('l1_loss'),
            'silhouette': data.get('silhouette'),
            'db_score': data.get('db_score'),
            
            # K-means基线（如果存在）
            'kmeans_all_acc': data.get('kmeans_all_acc', 0.0),
            'kmeans_old_acc': data.get('kmeans_old_acc', 0.0),
            'kmeans_new_acc': data.get('kmeans_new_acc', 0.0)
        }
        valid_count += 1
    
    print(f"📊 L2计算完成: 有效={valid_count}, 跳过={skipped_count}, 权重=(sep={w_sep}, pen={w_pen})")
    
    return weighted_results


def generate_l2_report(results_dict: Dict, w_sep: float, w_pen: float, 
                      superclass_name: str, output_path: str) -> Tuple[list, dict]:
    """
    生成L2权重探索文本报告
    
    Args:
        results_dict: 加权结果字典
        w_sep: separation权重
        w_pen: penalty权重
        superclass_name: 超类名称
        output_path: 输出文件路径
    
    Returns:
        tuple: (top3_list, stats_dict)
    """
    if not results_dict:
        print(f"⚠️  无有效数据，跳过报告生成")
        return [], {}
    
    # 按L2损失从小到大排序（损失越小越好）
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['l2_weighted'])
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 统计信息
    l2_values = [data['l2_weighted'] for _, data in sorted_results]
    stats = {
        'count': len(sorted_results),
        'min_l2': min(l2_values),
        'max_l2': max(l2_values),
        'mean_l2': np.mean(l2_values),
        'std_l2': np.std(l2_values)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"L2权重探索报告 - {superclass_name}\n")
        f.write(f"权重配置: separation={w_sep}, penalty={w_pen} (sum={w_sep + w_pen})\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("统计摘要:\n")
        f.write(f"  参数组合总数: {stats['count']}\n")
        f.write(f"  L2损失范围: [{stats['min_l2']:.4f}, {stats['max_l2']:.4f}]\n")
        f.write(f"  L2损失均值: {stats['mean_l2']:.4f} ± {stats['std_l2']:.4f}\n\n")
        
        f.write("参数组合排序（按L2损失从小到大）:\n\n")
        
        for rank, (key, data) in enumerate(sorted_results, 1):
            k, dp = key
            l2_loss = data['l2_weighted']
            all_acc = data['all_acc']
            separation = data['separation_score']
            penalty = data['penalty_score']
            
            # 选择排名标志
            if rank == 1:
                emoji = "🥇"
            elif rank == 2:
                emoji = "🥈"
            elif rank == 3:
                emoji = "🥉"
            else:
                emoji = f"#{rank:2d}"
            
            f.write(f"{emoji} Rank {rank}: k={k}, dp={dp}\n")
            f.write(f"     L2 Loss: {l2_loss:.4f}\n")
            f.write(f"     all_acc: {all_acc:.4f}\n")
            f.write(f"     separation: {separation:.4f}\n")
            f.write(f"     penalty: {penalty:.4f}\n")
            
            # 添加可选字段
            if data.get('labeled_acc') is not None:
                f.write(f"     labeled_acc: {data['labeled_acc']:.4f}\n")
            if data.get('quality_score') is not None:
                f.write(f"     quality_score: {data['quality_score']:.4f}\n")
            if data.get('n_clusters') is not None:
                f.write(f"     n_clusters: {data['n_clusters']}\n")
            
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write(f"配置详情:\n")
        f.write(f"  权重: separation={w_sep}, penalty={w_pen}\n")
        f.write(f"  公式: L2 = ({w_pen} × penalty - {w_sep} × separation) / 10\n")
        
        if sorted_results:
            best_key, best_data = sorted_results[0]
            f.write(f"  最佳参数: k={best_key[0]}, dp={best_key[1]}\n")
            f.write(f"  最佳L2损失: {best_data['l2_weighted']:.4f}\n")
    
    print(f"📄 L2报告已保存: {output_path}")
    
    # 返回Top 3和统计信息
    top3 = [(key, data) for key, data in sorted_results[:3]]
    return top3, stats


def get_weight_configurations(weight_sum: int = 10) -> list:
    """
    获取所有权重配置
    
    Args:
        weight_sum: 分离度与惩罚项权重之和，决定九组配置的尺度
    
    Returns:
        list: [(w_sep, w_pen), ...] 权重配置列表
    """
    if not isinstance(weight_sum, int):
        raise TypeError("weight_sum 必须为整数")
    if weight_sum < 2:
        raise ValueError("weight_sum 必须大于等于 2")
    
    configs = []
    prev_sep = 0
    max_sep = weight_sum - 1
    
    for idx in range(1, 10):
        # 基于原始1-9比例按总和缩放，并保持单调递增
        scaled = int(round((idx / 10.0) * weight_sum))
        w_sep = max(1, min(max_sep, scaled))
        if w_sep <= prev_sep:
            w_sep = min(max_sep, prev_sep + 1)
        w_pen = weight_sum - w_sep
        if w_pen < 1:
            w_pen = 1
            w_sep = weight_sum - w_pen
        prev_sep = w_sep
        configs.append((w_sep, w_pen))
    
    return configs


def validate_l2_requirements(results_dict: Dict) -> Tuple[bool, str]:
    """
    验证数据是否满足L2权重探索要求
    
    Args:
        results_dict: 从load_raw_scores获取的结果字典
    
    Returns:
        tuple: (是否满足要求, 详细信息)
    """
    if not results_dict:
        return False, "结果字典为空"
    
    total_count = len(results_dict)
    sep_available = sum(1 for v in results_dict.values() if v.get('separation_score') is not None)
    pen_available = sum(1 for v in results_dict.values() if v.get('penalty_score') is not None)
    
    if sep_available == 0:
        return False, f"separation_score完全缺失 (0/{total_count})"
    
    if pen_available == 0:
        return False, f"penalty_score完全缺失 (0/{total_count})"
    
    valid_pairs = sum(1 for v in results_dict.values() 
                     if v.get('separation_score') is not None and v.get('penalty_score') is not None)
    
    coverage = valid_pairs / total_count if total_count > 0 else 0
    
    if coverage < 0.5:  # 至少50%的数据有效
        return False, f"有效数据覆盖率不足: {coverage:.1%} ({valid_pairs}/{total_count})"
    
    info = f"✅ 数据满足要求: separation={sep_available}/{total_count}, penalty={pen_available}/{total_count}, 覆盖率={coverage:.1%}"
    return True, info


def analyze_weight_stability(all_results: Dict[Tuple[float, float], Dict]) -> Dict:
    """
    分析不同权重配置下参数组合的稳定性
    
    Args:
        all_results: {(w_sep, w_pen): {(k, dp): result_data}} 所有权重配置的结果
    
    Returns:
        dict: 稳定性分析结果
    """
    param_stability = {}  # {(k, dp): [出现次数, 平均排名, 权重配置列表]}
    
    # 分析每个权重配置的Top 3
    for (w_sep, w_pen), results in all_results.items():
        if not results:
            continue
            
        # 按L2损失排序，获取Top 3
        sorted_results = sorted(results.items(), key=lambda x: x[1]['l2_weighted'])
        top3_params = [key for key, _ in sorted_results[:3]]
        
        # 统计每个参数组合的出现情况
        for rank, param_key in enumerate(top3_params, 1):
            if param_key not in param_stability:
                param_stability[param_key] = {'count': 0, 'ranks': [], 'configs': []}
            
            param_stability[param_key]['count'] += 1
            param_stability[param_key]['ranks'].append(rank)
            param_stability[param_key]['configs'].append((w_sep, w_pen))
    
    # 计算稳定性指标
    stability_analysis = {}
    for param_key, data in param_stability.items():
        k, dp = param_key
        count = data['count']
        avg_rank = np.mean(data['ranks'])
        
        stability_analysis[param_key] = {
            'k': k,
            'dp': dp, 
            'top3_frequency': count,
            'avg_rank': avg_rank,
            'stability_score': count / len(all_results),  # 稳定性得分
            'configs': data['configs']
        }
    
    # 按稳定性得分排序
    stable_params = sorted(stability_analysis.items(), 
                          key=lambda x: (x[1]['stability_score'], -x[1]['avg_rank']), 
                          reverse=True)
    
    return {
        'stability_ranking': stable_params,
        'total_configs': len(all_results),
        'analysis_summary': {
            'most_stable': stable_params[0] if stable_params else None,
            'stable_count': len([p for p in stable_params if p[1]['stability_score'] >= 0.5])
        }
    }
