#!/usr/bin/env python3
"""
参数热力图绘制工具
绘制k和density_percentile参数与ACC关系的热力图，用于可视化参数调优结果
调用test_adaptive_ssddbc.py中的函数进行网格搜索，避免代码重复
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime
from itertools import product
import warnings
import re
import glob
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入test_adaptive_clustering中的主要函数（修复相对导入问题）
from ssddbc.testing.test_superclass import test_adaptive_clustering_on_superclass
from config import (
    grid_search_output_dir,
    heatmap_output_dir,
    superclass_model_root
)


def load_existing_results(superclass_name, search_dir=grid_search_output_dir):
    """
    从已保存的搜索结果文件中加载数据，避免重复计算

    Args:
        superclass_name: 超类名称
        search_dir: 搜索结果目录

    Returns:
        dict: {(k, density_percentile): {'all_acc': xx, 'old_acc': xx, ...}} 或 None
    """
    superclass_dir = os.path.join(search_dir, superclass_name)

    if not os.path.exists(superclass_dir):
        print(f"⚠️  搜索结果目录不存在: {superclass_dir}")
        return None

    # 查找最新的结果文件
    result_files = glob.glob(os.path.join(superclass_dir, "*.txt"))
    if not result_files:
        print(f"⚠️  没有找到搜索结果文件: {superclass_dir}")
        return None

    # 选择最新的文件
    latest_file = max(result_files, key=os.path.getmtime)
    print(f"📂 加载已有搜索结果: {latest_file}")

    try:
        results_dict = {}
        with open(latest_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用正则表达式解析batch_runner.py生成的文件格式
        # 更新pattern以包含labeled_acc
        pattern = r'k=(\d+), density_percentile=(\d+)\n-+\n.*?all_acc: (\d+\.?\d*)\nold_acc: (\d+\.?\d*)\nnew_acc: (\d+\.?\d*)\nclusters: (\d+)'

        matches = re.findall(pattern, content, re.DOTALL)

        # 解析labeled_acc（可选字段，可能为N/A）
        labeled_acc_pattern = r'k=(\d+), density_percentile=(\d+).*?labeled_acc: (N/A|[-\d]+\.?\d*)'
        labeled_acc_matches = re.findall(labeled_acc_pattern, content, re.DOTALL)

        # 创建字典方便查找
        labeled_acc_dict = {}
        for m in labeled_acc_matches:
            key = (int(m[0]), int(m[1]))
            labeled_acc_dict[key] = None if m[2] == 'N/A' else float(m[2])

        # 解析l1_loss（新增字段）
        l1_loss_pattern = r'k=(\d+), density_percentile=(\d+).*?l1_loss: (N/A|[-\d]+\.?\d*)'
        l1_loss_matches = re.findall(l1_loss_pattern, content, re.DOTALL)

        # 创建字典方便查找
        l1_loss_dict = {}
        for m in l1_loss_matches:
            key = (int(m[0]), int(m[1]))
            l1_loss_dict[key] = None if m[2] == 'N/A' else float(m[2])

        # 同时尝试解析轮廓系数和DB指数（旧版可选字段）
        # 注意：轮廓系数可以是负数（范围-1到1），所以需要匹配负号
        silhouette_pattern = r'k=(\d+), density_percentile=(\d+).*?Silhouette\s+(-?\d+\.?\d*)'
        db_pattern = r'k=(\d+), density_percentile=(\d+).*?DB Score\s+(-?\d+\.?\d*)'

        silhouette_matches = re.findall(silhouette_pattern, content, re.DOTALL)
        db_matches = re.findall(db_pattern, content, re.DOTALL)

        # 创建字典方便查找
        silhouette_dict = {(int(m[0]), int(m[1])): float(m[2]) for m in silhouette_matches}
        db_dict = {(int(m[0]), int(m[1])): float(m[2]) for m in db_matches}

        # 解析聚类质量指标（新增）
        quality_score_pattern = r'k=(\d+), density_percentile=(\d+).*?quality_score: ([-\d]+\.?\d*)'
        separation_score_pattern = r'k=(\d+), density_percentile=(\d+).*?separation_score: ([-\d]+\.?\d*)'
        penalty_score_pattern = r'k=(\d+), density_percentile=(\d+).*?penalty_score: ([-\d]+\.?\d*)'

        quality_score_matches = re.findall(quality_score_pattern, content, re.DOTALL)
        separation_score_matches = re.findall(separation_score_pattern, content, re.DOTALL)
        penalty_score_matches = re.findall(penalty_score_pattern, content, re.DOTALL)

        quality_score_dict = {(int(m[0]), int(m[1])): float(m[2]) for m in quality_score_matches}
        separation_score_dict = {(int(m[0]), int(m[1])): float(m[2]) for m in separation_score_matches}
        penalty_score_dict = {(int(m[0]), int(m[1])): float(m[2]) for m in penalty_score_matches}

        # 解析通用L2组件指标 - 改用分块解析避免跨参数组合匹配
        component_data = {}

        # 按参数组合分块（以 "k=数字, density_percentile=数字" 为分隔符）
        param_blocks = re.split(r'(?=k=\d+, density_percentile=\d+)', content)

        for block in param_blocks:
            if not block.strip():
                continue

            # 提取当前块的 k 和 dp
            header_match = re.match(r'k=(\d+), density_percentile=(\d+)', block)
            if not header_match:
                continue

            k = int(header_match.group(1))
            dp = int(header_match.group(2))
            key = (k, dp)

            # 在当前块内查找所有组件
            comp_value_matches = re.findall(r'component_(\w+)_value: ([-\d]+\.?\d*)', block)
            comp_contrib_matches = re.findall(r'component_(\w+)_contribution: ([-\d]+\.?\d*)', block)
            comp_orient_matches = re.findall(r'component_(\w+)_orientation: ([A-Za-z]+)', block)

            if comp_value_matches:
                comp_entry = component_data.setdefault(key, {})

                for comp_name, value_str in comp_value_matches:
                    entry = comp_entry.setdefault(comp_name, {})
                    entry['value'] = float(value_str)

                for comp_name, contrib_str in comp_contrib_matches:
                    entry = comp_entry.setdefault(comp_name, {})
                    entry['contribution'] = float(contrib_str)

                for comp_name, orient in comp_orient_matches:
                    entry = comp_entry.setdefault(comp_name, {})
                    entry['orientation'] = orient

        # 尝试解析K-means缓存结果（只在第一次运行时记录）
        kmeans_cache = {'all_acc': 0.0, 'old_acc': 0.0, 'new_acc': 0.0}

        for match in matches:
            k = int(match[0])
            density_percentile = int(match[1])
            ss_ddbc_all_acc = float(match[2])
            ss_ddbc_old_acc = float(match[3])
            ss_ddbc_new_acc = float(match[4])
            ss_ddbc_clusters = int(match[5])

            # 获取所有质量指标（如果存在）
            silhouette_score = silhouette_dict.get((k, density_percentile), None)
            db_score = db_dict.get((k, density_percentile), None)
            labeled_acc_score = labeled_acc_dict.get((k, density_percentile), None)

            # 获取聚类质量指标（新增）
            quality_score = quality_score_dict.get((k, density_percentile), None)
            separation_score = separation_score_dict.get((k, density_percentile), None)
            penalty_score = penalty_score_dict.get((k, density_percentile), None)

            results_dict[(k, density_percentile)] = {
                'all_acc': ss_ddbc_all_acc,
                'old_acc': ss_ddbc_old_acc,
                'new_acc': ss_ddbc_new_acc,
                'n_clusters': ss_ddbc_clusters,
                'kmeans_all_acc': kmeans_cache['all_acc'],  # 使用缓存值或默认值
                'kmeans_old_acc': kmeans_cache['old_acc'],
                'kmeans_new_acc': kmeans_cache['new_acc'],
                'silhouette': silhouette_score,
                'db_score': db_score,
                'labeled_acc': labeled_acc_score,
                'l1_loss': l1_loss_dict.get((k, density_percentile), None),  # 新增L1损失字段
                'quality_score': quality_score,
                'separation_score': separation_score,
                'penalty_score': penalty_score,
                'l2_components': component_data.get((k, density_percentile), {})
            }

        print(f"✅ 成功加载 {len(results_dict)} 个参数组合的结果")
        return results_dict

    except Exception as e:
        print(f"❌ 解析结果文件失败: {e}")
        return None


def detect_available_superclasses(search_dir=grid_search_output_dir):
    """
    自动探测搜索结果目录中存在的所有超类

    Args:
        search_dir: 搜索结果目录

    Returns:
        list: 包含结果文件的超类名称列表
    """
    if not os.path.exists(search_dir):
        print(f"⚠️  搜索结果目录不存在: {search_dir}")
        return []

    superclass_list = []

    # 遍历搜索目录下的所有子文件夹
    for item in os.listdir(search_dir):
        item_path = os.path.join(search_dir, item)

        # 只处理文件夹
        if os.path.isdir(item_path):
            # 检查文件夹中是否有.txt结果文件
            txt_files = glob.glob(os.path.join(item_path, "*.txt"))
            if txt_files:
                superclass_list.append(item)

    return sorted(superclass_list)  # 按字母顺序排序


def print_progress_bar(current, total, prefix='Progress', suffix='Complete', length=50):
    """打印进度条"""
    percent = f"{100 * (current / float(total)):.1f}"
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} ({current}/{total})', end='', flush=True)
    if current == total:
        print()


def run_parameter_grid_search(superclass_name, model_path, k_range=(3, 21),
                             density_percentile_range=(20, 100), step=5,
                             eval_version='v2', merge_clusters=True):
    """
    运行参数网格搜索，收集数据用于绘制热力图

    Args:
        superclass_name: 超类名称
        model_path: 模型路径
        k_range: k值范围 (start, end)
        density_percentile_range: 密度百分位范围 (start, end)
        step: 密度百分位步长
        eval_version: 评估版本
        merge_clusters: 是否合并聚类

    Returns:
        results_dict: {(k, density_percentile): {'all_acc': xx, 'old_acc': xx, 'new_acc': xx, ...}}
    """
    print(f"🔍 开始参数网格搜索 - 超类: {superclass_name}")
    print("=" * 80)

    # 参数范围
    k_values = list(range(k_range[0], k_range[1]))
    density_percentile_values = list(range(density_percentile_range[0],
                                         density_percentile_range[1], step))

    print(f"K值范围: {k_values}")
    print(f"密度百分位范围: {density_percentile_values}")
    print(f"评估版本: {eval_version}")
    print(f"聚类合并: {'启用' if merge_clusters else '禁用'}")
    print(f"总参数组合数: {len(k_values) * len(density_percentile_values)}")

    # 存储结果
    results_dict = {}
    total_combinations = len(k_values) * len(density_percentile_values)
    current_combination = 0

    for k, density_percentile in product(k_values, density_percentile_values):
        current_combination += 1

        # 显示进度
        print_progress_bar(current_combination - 1, total_combinations,
                         prefix=f'网格搜索进度', suffix=f'k={k}, density_percentile={density_percentile}')

        try:
            # 调用test_adaptive_ssddbc.py中的函数
            results = test_adaptive_clustering_on_superclass(
                superclass_name=superclass_name,
                model_path=model_path,
                k=k,
                density_percentile=density_percentile,
                eval_version=eval_version,
                fast_mode=True,  # 网格搜索使用快速模式
                run_kmeans_baseline=True
            )

            if results and isinstance(results, dict):
                # 存储结果
                loss_dict = results.get('loss_dict') or {}
                l2_metrics = loss_dict.get('l2_metrics', {})
                cluster_quality = l2_metrics.get('cluster_quality', {})
                results_dict[(k, density_percentile)] = {
                    'all_acc': results.get('all_acc', 0.0),
                    'old_acc': results.get('old_acc', 0.0),
                    'new_acc': results.get('new_acc', 0.0),
                    'n_clusters': results.get('n_clusters', 0),
                    'kmeans_all_acc': results.get('kmeans_all_acc', 0.0),
                    'kmeans_old_acc': results.get('kmeans_old_acc', 0.0),
                    'kmeans_new_acc': results.get('kmeans_new_acc', 0.0),
                    'l1_loss': results.get('l1'),
                    'loss': results.get('loss'),
                    'l1': results.get('l1'),
                    'l2': results.get('l2'),
                    'loss_dict': loss_dict,
                    'labeled_acc': results.get('labeled_acc'),
                    'quality_score': cluster_quality.get('quality_score'),
                    'separation_score': cluster_quality.get('separation_score'),
                    'penalty_score': cluster_quality.get('penalty_score')
                }
            else:
                print(f"\n⚠️  参数组合 k={k}, density_percentile={density_percentile} 失败")
                results_dict[(k, density_percentile)] = {
                    'all_acc': 0.0, 'old_acc': 0.0, 'new_acc': 0.0,
                    'n_clusters': 0,
                    'kmeans_all_acc': 0.0, 'kmeans_old_acc': 0.0, 'kmeans_new_acc': 0.0,
                    'l1_loss': None,
                    'loss': None,
                    'l1': None,
                    'l2': None,
                    'loss_dict': None,
                    'labeled_acc': None,
                    'quality_score': None,
                    'separation_score': None,
                    'penalty_score': None
                }

        except Exception as e:
            print(f"\n❌ 参数组合 k={k}, density_percentile={density_percentile} 出现异常: {str(e)}")
            results_dict[(k, density_percentile)] = {
                'all_acc': 0.0, 'old_acc': 0.0, 'new_acc': 0.0,
                'n_clusters': 0,
                'kmeans_all_acc': 0.0, 'kmeans_old_acc': 0.0, 'kmeans_new_acc': 0.0,
                'l1_loss': None,
                'loss': None,
                'l1': None,
                'l2': None,
                'loss_dict': None,
                'labeled_acc': None,
                'quality_score': None,
                'separation_score': None,
                'penalty_score': None
            }

    # 完成时显示最终进度条
    print_progress_bar(total_combinations, total_combinations,
                     prefix=f'网格搜索进度', suffix='完成!')

    print(f"\n✅ 参数网格搜索完成")
    print(f"📊 成功完成: {len([r for r in results_dict.values() if r['all_acc'] > 0])}/{total_combinations} 个参数组合")

    return results_dict


def create_heatmap(results_dict, metric='all_acc', superclass_name='',
                  output_dir=heatmap_output_dir, save_plots=True,
                  show_clusters=False):
    """
    创建参数热力图（生成两张图：一张显示ACC值，一张显示聚类数）

    Args:
        results_dict: 网格搜索结果字典
        metric: 要可视化的指标 ('all_acc', 'old_acc', 'new_acc', 'l1_loss'等)
        superclass_name: 超类名称
        output_dir: 输出目录
        save_plots: 是否保存图片
        show_clusters: 是否额外生成显示聚类数量的热力图（仅对all_acc启用）
    """
    print(f"🎨 绘制热力图 - 指标: {metric}")

    # 根据指标类型决定排序方向：损失类指标越小越好，准确率类指标越大越好
    loss_metrics = ['l1_loss', 'quality_score', 'penalty_score']  # 损失类指标，越小越好

    # 创建输出目录（包含超类名称子目录）
    if save_plots:
        superclass_output_dir = os.path.join(output_dir, superclass_name)
        os.makedirs(superclass_output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {superclass_output_dir}")

    # 提取参数和结果
    k_values = sorted(list(set([k for k, _ in results_dict.keys()])))
    density_percentile_values = sorted(list(set([dp for _, dp in results_dict.keys()])))

    # 创建热力图数据矩阵（用于着色）
    heatmap_data = np.zeros((len(density_percentile_values), len(k_values)))
    # 创建聚类数矩阵（用于第二张图的标注）
    cluster_data = np.zeros((len(density_percentile_values), len(k_values)))

    for i, dp in enumerate(density_percentile_values):
        for j, k in enumerate(k_values):
            if (k, dp) in results_dict:
                # 检查指标是否存在，如果不存在则使用NaN
                if metric in results_dict[(k, dp)] and results_dict[(k, dp)][metric] is not None:
                    heatmap_data[i, j] = results_dict[(k, dp)][metric]
                else:
                    heatmap_data[i, j] = np.nan
                    print(f"⚠️  警告: (k={k}, dp={dp}) 缺少指标 '{metric}'，使用NaN填充")

                cluster_data[i, j] = results_dict[(k, dp)].get('n_clusters', np.nan)
            else:
                heatmap_data[i, j] = np.nan
                cluster_data[i, j] = np.nan

    # 找到前3名的位置
    valid_data = []
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            if not np.isnan(heatmap_data[i, j]):
                valid_data.append((heatmap_data[i, j], i, j))

    if metric in loss_metrics:
        valid_data.sort(key=lambda x: x[0])  # 升序排列，最小值在前
    else:
        valid_data.sort(key=lambda x: x[0], reverse=True)  # 降序排列，最大值在前

    top3 = valid_data[:3]

    # ==================== 第一张图：显示ACC值 ====================
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # 根据指标类型选择颜色映射：损失类指标用反向映射，准确率类指标用正向映射
    if metric in loss_metrics:
        cmap = 'viridis_r'  # 反向映射，低值（好）用深色，高值（差）用浅色
        cbar_label = f'{metric.upper()} (lower better)'
    else:
        cmap = 'viridis'  # 正向映射,高值（好）用深色，低值（差）用浅色
        cbar_label = f'{metric.upper()} (higher better)'

    sns.heatmap(heatmap_data,
                xticklabels=k_values,
                yticklabels=density_percentile_values,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                cbar_kws={'label': cbar_label},
                ax=ax1)

    # 标注前3名
    for rank, (value, i, j) in enumerate(top3, 1):
        ax1.plot(j + 0.5, i + 0.5, marker='*', markersize=20,
                color='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                markeredgecolor='white', markeredgewidth=1.5)
        ax1.text(j + 0.5, i + 0.2, f'#{rank}',
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                         edgecolor='white', linewidth=1.5, alpha=0.8))

    ax1.set_title(f'{metric.upper()} Heatmap - {superclass_name}\nParameters: k vs density_percentile (Top 3 marked)',
              fontsize=14, fontweight='bold')
    ax1.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax1.set_ylabel('Density Percentile', fontsize=12)

    plt.tight_layout()

    # 保存第一张图
    if save_plots:
        current_time = datetime.now()
        filename1 = f"{metric}_heatmap_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}.png"
        output_path1 = os.path.join(superclass_output_dir, filename1)
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        print(f"📁 热力图1 (ACC值) 已保存: {output_path1}")

    plt.show()

    # ==================== 第二张图：显示聚类数（只对all_acc显示） ====================
    if show_clusters and metric == 'all_acc':
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        # 使用指标值着色，但标注显示聚类数（只对all_acc显示聚类数图）
        sns.heatmap(heatmap_data,
                    xticklabels=k_values,
                    yticklabels=density_percentile_values,
                    annot=cluster_data,  # 显示聚类数
                    fmt='.0f',  # 整数格式
                    cmap=cmap,  # 使用与第一张图相同的颜色映射
                    cbar_kws={'label': cbar_label},
                    ax=ax2)

        # 标注前3名
        for rank, (value, i, j) in enumerate(top3, 1):
            ax2.plot(j + 0.5, i + 0.5, marker='*', markersize=20,
                     color='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                     markeredgecolor='white', markeredgewidth=1.5)
            ax2.text(j + 0.5, i + 0.2, f'#{rank}',
                     ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                               edgecolor='white', linewidth=1.5, alpha=0.8))

        ax2.set_title(f'Number of Clusters (colored by {metric.upper()}) - {superclass_name}\nParameters: k vs density_percentile (Top 3 marked)',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('k (Number of Neighbors)', fontsize=12)
        ax2.set_ylabel('Density Percentile', fontsize=12)

        plt.tight_layout()

        # 保存第二张图
        if save_plots:
            filename2 = f"{metric}_clusters_heatmap_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}.png"
            output_path2 = os.path.join(superclass_output_dir, filename2)
            plt.savefig(output_path2, dpi=300, bbox_inches='tight')
            print(f"📁 热力图2 (聚类数) 已保存: {output_path2}")

        plt.show()

    # 找到最佳参数组合（根据指标类型选择argmax或argmin）
    if metric in loss_metrics:
        best_idx = np.unravel_index(np.nanargmin(heatmap_data), heatmap_data.shape)
    else:
        best_idx = np.unravel_index(np.nanargmax(heatmap_data), heatmap_data.shape)
    best_dp = density_percentile_values[best_idx[0]]
    best_k = k_values[best_idx[1]]
    best_value = heatmap_data[best_idx]

    # 输出前3名参数组合
    print(f"\n🏆 Top 3 参数组合 ({metric.upper()}):")
    print("-" * 60)
    for rank, (value, i, j) in enumerate(top3, 1):
        k_val = k_values[j]
        dp_val = density_percentile_values[i]
        n_clusters_val = int(cluster_data[i, j])
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print(f"{emoji} #{rank}: k={k_val:<3}, density_percentile={dp_val:<3}, {metric}={value:.4f}, n_clusters={n_clusters_val}")

    return best_k, best_dp, best_value


def create_quality_metrics_heatmap(results_dict, acc_metric='all_acc', superclass_name='',
                                   output_dir=heatmap_output_dir, save_plots=True):
    """
    创建新式热力图：背景色使用ACC值，单元格文字显示轮廓系数和DB指数

    Args:
        results_dict: 网格搜索结果字典
        acc_metric: 用于着色的ACC指标 ('all_acc', 'old_acc', 'new_acc')
        superclass_name: 超类名称
        output_dir: 输出目录
        save_plots: 是否保存图片
    """
    print(f"🎨 绘制质量指标热力图 - 背景色指标: {acc_metric}")

    # 创建输出目录
    if save_plots:
        superclass_output_dir = os.path.join(output_dir, superclass_name)
        os.makedirs(superclass_output_dir, exist_ok=True)

    # 提取参数和结果
    k_values = sorted(list(set([k for k, _ in results_dict.keys()])))
    density_percentile_values = sorted(list(set([dp for _, dp in results_dict.keys()])))

    # 创建数据矩阵
    acc_data = np.zeros((len(density_percentile_values), len(k_values)))
    silhouette_data = np.full((len(density_percentile_values), len(k_values)), np.nan)
    db_data = np.full((len(density_percentile_values), len(k_values)), np.nan)

    for i, dp in enumerate(density_percentile_values):
        for j, k in enumerate(k_values):
            if (k, dp) in results_dict:
                acc_data[i, j] = results_dict[(k, dp)][acc_metric]

                # 读取轮廓系数和DB指数（可能为None）
                silhouette_val = results_dict[(k, dp)].get('silhouette', None)
                db_val = results_dict[(k, dp)].get('db_score', None)

                if silhouette_val is not None:
                    silhouette_data[i, j] = silhouette_val
                if db_val is not None:
                    db_data[i, j] = db_val
            else:
                acc_data[i, j] = np.nan

    # 创建自定义标注矩阵：每个单元格显示 "S: xx.xxx\nDB: xx.xxx"
    annot_data = np.empty((len(density_percentile_values), len(k_values)), dtype=object)
    for i in range(len(density_percentile_values)):
        for j in range(len(k_values)):
            s_val = silhouette_data[i, j]
            db_val = db_data[i, j]

            if not np.isnan(s_val) and not np.isnan(db_val):
                annot_data[i, j] = f"S: {s_val:.3f}\nDB: {db_val:.3f}"
            elif not np.isnan(s_val):
                annot_data[i, j] = f"S: {s_val:.3f}\nDB: N/A"
            elif not np.isnan(db_val):
                annot_data[i, j] = f"S: N/A\nDB: {db_val:.3f}"
            else:
                annot_data[i, j] = "N/A"

    # 找到前3名ACC的位置
    valid_data = []
    for i in range(acc_data.shape[0]):
        for j in range(acc_data.shape[1]):
            if not np.isnan(acc_data[i, j]):
                valid_data.append((acc_data[i, j], i, j))

    valid_data.sort(key=lambda x: x[0], reverse=True)
    top3 = valid_data[:3]

    # 创建热力图
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(acc_data,
                xticklabels=k_values,
                yticklabels=density_percentile_values,
                annot=annot_data,
                fmt='',  # 使用自定义格式
                cmap='viridis',
                cbar_kws={'label': f'{acc_metric.upper()} (background color)'},
                ax=ax,
                annot_kws={'fontsize': 7})  # 调整字体大小以容纳两行文字

    # 标注前3名
    for rank, (value, i, j) in enumerate(top3, 1):
        ax.plot(j + 0.5, i + 0.5, marker='*', markersize=20,
                color='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                markeredgecolor='white', markeredgewidth=1.5)
        ax.text(j + 0.5, i + 0.15, f'#{rank}',
                ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                         edgecolor='white', linewidth=1.5, alpha=0.8))

    ax.set_title(f'Quality Metrics (Silhouette & DB) colored by {acc_metric.upper()} - {superclass_name}\n'
                 f'Parameters: k vs density_percentile (Top 3 marked)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Density Percentile', fontsize=12)

    plt.tight_layout()

    # 保存图片
    if save_plots:
        current_time = datetime.now()
        filename = f"{acc_metric}_quality_metrics_heatmap_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}.png"
        output_path = os.path.join(superclass_output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📁 质量指标热力图 已保存: {output_path}")

    plt.show()

    # 输出前3名参数组合及其质量指标
    print(f"\n🏆 Top 3 参数组合 (按 {acc_metric.upper()} 排序，显示质量指标):")
    print("-" * 80)
    for rank, (value, i, j) in enumerate(top3, 1):
        k_val = k_values[j]
        dp_val = density_percentile_values[i]
        s_val = silhouette_data[i, j]
        db_val = db_data[i, j]
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"

        s_str = f"{s_val:.4f}" if not np.isnan(s_val) else "N/A"
        db_str = f"{db_val:.4f}" if not np.isnan(db_val) else "N/A"

        print(f"{emoji} #{rank}: k={k_val:<3}, density_percentile={dp_val:<3}, "
              f"{acc_metric}={value:.4f}, Silhouette={s_str}, DB={db_str}")

    return top3[0][1], top3[0][2]  # 返回最佳参数的索引


def create_multiple_quality_metrics_heatmaps(results_dict, superclass_name='',
                                             output_dir=heatmap_output_dir,
                                             acc_metrics=['all_acc', 'old_acc', 'new_acc'],
                                             save_plots=True):
    """
    创建多个ACC指标的质量指标热力图（每个ACC生成一张热力图）
    """
    print(f"🎨 创建多ACC质量指标热力图...")

    for acc_metric in acc_metrics:
        print(f"\n{'='*60}")
        create_quality_metrics_heatmap(
            results_dict, acc_metric, superclass_name, output_dir, save_plots
        )

    print(f"\n✅ 所有质量指标热力图创建完成!")


def create_multiple_heatmaps(results_dict, superclass_name='', output_dir=heatmap_output_dir,
                           metrics=['all_acc', 'old_acc', 'new_acc'], save_plots=True):
    """
    创建多个指标的热力图
    """
    print(f"🎨 创建多指标热力图...")

    best_params = {}

    for metric in metrics:
        print(f"\n{'='*60}")
        try:
            # 只对all_acc显示聚类数
            show_clusters = (metric == 'all_acc')
            best_k, best_dp, best_value = create_heatmap(
                results_dict, metric, superclass_name, output_dir, save_plots, show_clusters
            )
            best_params[metric] = {
                'k': best_k,
                'density_percentile': best_dp,
                'value': best_value
            }
        except Exception as e:
            print(f"❌ {superclass_name} 热力图绘制失败: {e}")
            best_params[metric] = {
                'k': None,
                'density_percentile': None,
                'value': None,
                'error': str(e)
            }
            continue

    # 输出最佳参数总结
    print(f"\n{'='*80}")
    print(f"📊 {superclass_name} 最佳参数总结:")
    print(f"{'='*80}")
    for metric, params in best_params.items():
        print(f"{metric.upper():<12}: k={params['k']:<3}, density_percentile={params['density_percentile']:<3}, value={params['value']:.4f}")

    return best_params


def create_mixed_heatmap(results_dict, color_metric='all_acc', display_metric='labeled_acc',
                         superclass_name='', output_dir=heatmap_output_dir,
                         save_plots=True):
    """
    创建混合热力图：用一个指标着色，显示另一个指标的数值

    Args:
        results_dict: 网格搜索结果字典
        color_metric: 用于着色的指标 (例如 'all_acc')
        display_metric: 用于显示数值的指标 (例如 'labeled_acc')
        superclass_name: 超类名称
        output_dir: 输出目录
        save_plots: 是否保存图片

    Returns:
        best_k, best_dp, best_value: 基于color_metric的最佳参数
    """
    print(f"🎨 绘制混合热力图 - 着色: {color_metric}, 显示: {display_metric}")

    # 创建输出目录
    if save_plots:
        superclass_output_dir = os.path.join(output_dir, superclass_name)
        os.makedirs(superclass_output_dir, exist_ok=True)

    # 提取参数和结果
    k_values = sorted(list(set([k for k, _ in results_dict.keys()])))
    density_percentile_values = sorted(list(set([dp for _, dp in results_dict.keys()])))

    # 创建两个数据矩阵
    color_data = np.zeros((len(density_percentile_values), len(k_values)))  # 用于着色
    display_data = np.zeros((len(density_percentile_values), len(k_values)))  # 用于显示

    for i, dp in enumerate(density_percentile_values):
        for j, k in enumerate(k_values):
            if (k, dp) in results_dict:
                # 着色数据
                color_value = results_dict[(k, dp)].get(color_metric)
                color_data[i, j] = color_value if color_value is not None else np.nan

                # 显示数据
                display_value = results_dict[(k, dp)].get(display_metric)
                display_data[i, j] = display_value if display_value is not None else np.nan
            else:
                color_data[i, j] = np.nan
                display_data[i, j] = np.nan

    # 找到基于display_metric的前3名位置（按显示的指标排序，而不是背景颜色）
    valid_data = []
    for i in range(display_data.shape[0]):
        for j in range(display_data.shape[1]):
            if not np.isnan(display_data[i, j]):
                valid_data.append((display_data[i, j], i, j))

    valid_data.sort(key=lambda x: x[0], reverse=True)
    top3 = valid_data[:3] if len(valid_data) >= 3 else valid_data

    # 创建热力图
    fig, ax = plt.subplots(figsize=(14, 9))

    # 用color_metric着色，但显示display_metric的数值
    annot_fmt = '.4f'

    sns.heatmap(color_data,
                xticklabels=k_values,
                yticklabels=density_percentile_values,
                annot=display_data,  # 显示display_metric的数值
                fmt=annot_fmt,
                cmap='viridis',
                cbar_kws={'label': f'{color_metric.upper()} (coloring)'},
                ax=ax)

    # 标注前3名（基于color_metric）
    for rank, (value, i, j) in enumerate(top3, 1):
        ax.plot(j + 0.5, i + 0.5, marker='*', markersize=20,
                color='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                markeredgecolor='white', markeredgewidth=1.5)
        ax.text(j + 0.5, i + 0.2, f'#{rank}',
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                         edgecolor='white', linewidth=1.5, alpha=0.8))

    # 设置标题
    metric_names = {
        'all_acc': 'All ACC',
        'old_acc': 'Old ACC',
        'new_acc': 'New ACC',
        'labeled_acc': 'Labeled ACC'
    }
    color_name = metric_names.get(color_metric, color_metric.upper())
    display_name = metric_names.get(display_metric, display_metric.upper())

    ax.set_title(f'{display_name} Values (colored by {color_name}) - {superclass_name}\n'
                 f'Parameters: k vs density_percentile (Top 3 by {display_name} marked)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Density Percentile', fontsize=12)

    plt.tight_layout()

    # 保存图片
    if save_plots:
        current_time = datetime.now()
        filename = f"{display_metric}_colored_by_{color_metric}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}.png"
        output_path = os.path.join(superclass_output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📁 混合热力图已保存: {output_path}")

    plt.show()

    # 输出前3名参数组合（基于display_metric）
    print(f"\n🏆 Top 3 参数组合 (按 {display_name} 排序):")
    print("-" * 80)
    for rank, (display_value, i, j) in enumerate(top3, 1):
        k_val = k_values[j]
        dp_val = density_percentile_values[i]
        color_value = color_data[i, j]
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"

        if not np.isnan(color_value):
            print(f"{emoji} #{rank}: k={k_val:<3}, density_percentile={dp_val:<3}, "
                  f"{display_metric}={display_value:.4f}, {color_metric}={color_value:.4f}")
        else:
            print(f"{emoji} #{rank}: k={k_val:<3}, density_percentile={dp_val:<3}, "
                  f"{display_metric}={display_value:.4f}, {color_metric}=N/A")

    # 返回基于color_metric的最佳参数
    best_idx = np.unravel_index(np.nanargmax(color_data), color_data.shape)
    best_dp = density_percentile_values[best_idx[0]]
    best_k = k_values[best_idx[1]]
    best_value = color_data[best_idx]

    return best_k, best_dp, best_value


def create_cluster_quality_heatmaps(results_dict, color_metric='all_acc',
                                    superclass_name='', output_dir=heatmap_output_dir,
                                    save_plots=True):
    """
    创建三张聚类质量指标热力图，用all_acc着色

    生成三张图：
    1. 质量分数 (quality_score) - 以all_acc着色
    2. 簇间分离度 (separation_score) - 以all_acc着色
    3. 密度惩罚 (penalty_score) - 以all_acc着色

    Args:
        results_dict: 网格搜索结果字典
        color_metric: 用于着色的指标 (默认 'all_acc')
        superclass_name: 超类名称
        output_dir: 输出目录
        save_plots: 是否保存图片
    """
    print(f"🎨 生成聚类质量指标热力图 - 背景色指标: {color_metric}")

    # 创建输出目录
    if save_plots:
        superclass_output_dir = os.path.join(output_dir, superclass_name)
        os.makedirs(superclass_output_dir, exist_ok=True)

    # 定义三个指标（指标名, 标题, 是否越大越好）
    metrics_to_plot = [
        ('quality_score', 'Quality Score (Separation - Penalty)', True),   # 越大越好
        ('separation_score', 'Cluster Separation Score', True),             # 越大越好
        ('penalty_score', 'Local Density Penalty Score', False)             # 越小越好
    ]

    # 提取参数和结果
    k_values = sorted(list(set([k for k, _ in results_dict.keys()])))
    density_percentile_values = sorted(list(set([dp for _, dp in results_dict.keys()])))

    # 为每个指标创建一张热力图
    for metric_key, metric_title, higher_is_better in metrics_to_plot:
        print(f"\n{'='*60}")
        print(f"📊 生成热力图: {metric_title}")
        direction = "higher better" if higher_is_better else "lower better"
        print(f"   排序方向: {direction}")

        # 创建两个数据矩阵
        color_data = np.zeros((len(density_percentile_values), len(k_values)))  # 用于着色
        display_data = np.zeros((len(density_percentile_values), len(k_values)))  # 用于显示

        for i, dp in enumerate(density_percentile_values):
            for j, k in enumerate(k_values):
                if (k, dp) in results_dict:
                    # 着色数据（all_acc）
                    color_value = results_dict[(k, dp)].get(color_metric)
                    color_data[i, j] = color_value if color_value is not None else np.nan

                    # 显示数据（当前指标）
                    display_value = results_dict[(k, dp)].get(metric_key)
                    display_data[i, j] = display_value if display_value is not None else np.nan
                else:
                    color_data[i, j] = np.nan
                    display_data[i, j] = np.nan

        # 检查是否有有效数据
        if np.all(np.isnan(display_data)):
            print(f"⚠️  {metric_key} 无有效数据，跳过此热力图")
            continue

        # 找到基于当前指标（display_data）的前3名位置
        valid_data = []
        for i in range(display_data.shape[0]):
            for j in range(display_data.shape[1]):
                if not np.isnan(display_data[i, j]):
                    valid_data.append((display_data[i, j], i, j))

        # 根据指标特性排序（越大越好 or 越小越好）
        valid_data.sort(key=lambda x: x[0], reverse=higher_is_better)
        top3 = valid_data[:3] if len(valid_data) >= 3 else valid_data

        # 创建热力图
        fig, ax = plt.subplots(figsize=(14, 9))

        # 用color_metric着色，显示当前指标的数值
        sns.heatmap(color_data,
                    xticklabels=k_values,
                    yticklabels=density_percentile_values,
                    annot=display_data,  # 显示当前指标的数值
                    fmt='.4f',
                    cmap='viridis',
                    cbar_kws={'label': f'{color_metric.upper()} (coloring)'},
                    ax=ax)

        # 标注前3名
        for rank, (value, i, j) in enumerate(top3, 1):
            ax.plot(j + 0.5, i + 0.5, marker='*', markersize=20,
                    color='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.text(j + 0.5, i + 0.2, f'#{rank}',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='red' if rank == 1 else 'orange' if rank == 2 else 'yellow',
                             edgecolor='white', linewidth=1.5, alpha=0.8))

        # 设置标题
        direction_label = "highest" if higher_is_better else "lowest"
        ax.set_title(f'{metric_title} (colored by {color_metric.upper()}) - {superclass_name}\n'
                     f'Parameters: k vs density_percentile (Top 3 {direction_label} {metric_key} marked)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
        ax.set_ylabel('Density Percentile', fontsize=12)

        plt.tight_layout()

        # 保存图片
        if save_plots:
            current_time = datetime.now()
            filename = f"{metric_key}_colored_by_{color_metric}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}.png"
            output_path = os.path.join(superclass_output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📁 热力图已保存: {output_path}")

        plt.show()

        # 输出前3名参数组合的指标值
        direction_desc = f"{direction_label} {metric_key}"
        print(f"\n🏆 Top 3 参数组合 (按 {direction_desc} 排序):")
        print("-" * 80)
        for rank, (metric_value, i, j) in enumerate(top3, 1):
            k_val = k_values[j]
            dp_val = density_percentile_values[i]
            color_value = color_data[i, j]
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"

            if not np.isnan(color_value):
                print(f"{emoji} #{rank}: k={k_val:<3}, dp={dp_val:<3}, "
                      f"{metric_key}={metric_value:.4f}, {color_metric}={color_value:.4f}")
            else:
                print(f"{emoji} #{rank}: k={k_val:<3}, dp={dp_val:<3}, "
                      f"{metric_key}={metric_value:.4f}, {color_metric}=N/A")

    print(f"\n✅ 所有聚类质量指标热力图创建完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='参数热力图绘制工具')

    # 必要参数
    parser.add_argument('--superclass_name', type=str, default=None,
                        help='测试的超类名称（如果不指定，则自动探测并处理所有超类）')
    parser.add_argument('--model_path', type=str, default=None,
                        help='训练好的模型路径')

    # 搜索参数
    parser.add_argument('--k_min', type=int, default=3,
                        help='k值最小值')
    parser.add_argument('--k_max', type=int, default=21,
                        help='k值最大值')
    parser.add_argument('--dp_min', type=int, default=20,
                        help='密度百分位最小值')
    parser.add_argument('--dp_max', type=int, default=100,
                        help='密度百分位最大值')
    parser.add_argument('--dp_step', type=int, default=5,
                        help='密度百分位步长')

    # 其他参数
    parser.add_argument('--eval_version', type=str, default='v2', choices=['v1', 'v2'],
                        help='评估版本')
    parser.add_argument('--merge_clusters', type=str, default='true',
                        help='是否启用聚类合并')
    parser.add_argument('--output_dir', type=str, default=heatmap_output_dir,
                        help='输出目录')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['l1_loss'],
                        help='要绘制的指标（默认显示交叉熵损失l1_loss）。可选: all_acc, old_acc, new_acc, l1_loss, labeled_acc, quality_score等')
    parser.add_argument('--use_existing_results', type=str, default='true',
                        help='是否优先使用已有搜索结果')
    parser.add_argument('--quality_metrics', type=str, default='false',
                        help='是否生成质量指标热力图（轮廓系数和DB指数）')
    parser.add_argument('--cluster_quality_heatmaps', type=str, default='false',
                        help='是否生成聚类质量指标热力图（quality_score、separation_score、penalty_score）')

    args = parser.parse_args()
    merge_clusters = args.merge_clusters.lower() == 'true'
    use_existing_results = args.use_existing_results.lower() == 'true'
    quality_metrics = args.quality_metrics.lower() == 'true'
    cluster_quality_heatmaps = args.cluster_quality_heatmaps.lower() == 'true'

    # ========== 自动探测超类（如果未指定） ==========
    if args.superclass_name is None:
        print("🔍 未指定超类名称，自动探测网格搜索结果目录...")
        search_dir = grid_search_output_dir
        superclass_list = detect_available_superclasses(search_dir)

        if not superclass_list:
            print(f"❌ 未在 {search_dir} 中找到任何超类搜索结果")
            print("💡 请先运行网格搜索或手动指定超类名称 (--superclass_name)")
            return

        print(f"✅ 发现 {len(superclass_list)} 个超类: {superclass_list}")
        print("="*80)
    else:
        superclass_list = [args.superclass_name]
        print(f"📊 处理指定超类: {args.superclass_name}")
        print("="*80)

    # ========== 批量处理所有超类 ==========
    for idx, superclass_name in enumerate(superclass_list, 1):
        print(f"\n{'='*80}")
        print(f"🔄 处理超类 [{idx}/{len(superclass_list)}]: {superclass_name}")
        print(f"{'='*80}")

        print("参数热力图绘制工具")
        print("="*80)
        print(f"超类名称: {superclass_name}")
        print(f"模型路径: {args.model_path}")
        print(f"k值范围: {args.k_min}-{args.k_max-1}")
        print(f"密度百分位范围: {args.dp_min}-{args.dp_max-1} (步长{args.dp_step})")
        print(f"评估版本: {args.eval_version}")
        print(f"聚类合并: {merge_clusters}")
        print(f"输出目录: {args.output_dir}")
        print(f"绘制指标: {args.metrics}")
        print(f"质量指标热力图: {'启用' if quality_metrics else '禁用'}")
        print(f"聚类质量热力图: {'启用' if cluster_quality_heatmaps else '禁用'}")
        print("="*80)

        try:
            results = None

            # 根据参数决定是否使用已有结果
            if use_existing_results:
                print("\n🔍 检查是否已有搜索结果...")
                results = load_existing_results(superclass_name)

            if results is None:
                if use_existing_results:
                    print("📊 未找到已有结果，开始新的网格搜索...")
                else:
                    print("📊 强制重新运行网格搜索...")
                # 运行网格搜索
                results = run_parameter_grid_search(
                    superclass_name=superclass_name,
                    model_path=args.model_path,
                    k_range=(args.k_min, args.k_max),
                    density_percentile_range=(args.dp_min, args.dp_max),
                    step=args.dp_step,
                    eval_version=args.eval_version,
                    merge_clusters=merge_clusters
                )
            else:
                print("🎯 使用已有搜索结果生成热力图，避免重复计算")

            # 绘制热力图
            best_params = create_multiple_heatmaps(
                results_dict=results,
                superclass_name=superclass_name,
                output_dir=args.output_dir,
                metrics=args.metrics,
                save_plots=True
            )

            # 如果启用质量指标热力图，则生成轮廓系数和DB指数热力图
            if quality_metrics:
                print(f"\n{'='*80}")
                print("🎯 生成质量指标热力图（轮廓系数和DB指数）...")
                print(f"{'='*80}")
                create_multiple_quality_metrics_heatmaps(
                    results_dict=results,
                    superclass_name=superclass_name,
                    output_dir=args.output_dir,
                    acc_metrics=args.metrics,  # 使用相同的ACC指标列表
                    save_plots=True
                )

            # 生成混合热力图：以all_acc着色，显示labeled_acc（默认运行）
            print(f"\n{'='*80}")
            print("🎨 生成混合热力图（以all_acc着色，显示labeled_acc）...")
            print(f"{'='*80}")

            # 第一张：labeled_acc（以all_acc着色）
            try:
                create_mixed_heatmap(
                    results_dict=results,
                    color_metric='all_acc',
                    display_metric='labeled_acc',
                    superclass_name=superclass_name,
                    output_dir=args.output_dir,
                    save_plots=True
                )
            except Exception as e:
                print(f"⚠️  labeled_acc混合热力图生成失败: {e}")

            # 第二部分：聚类质量指标热力图（quality_score, separation_score, penalty_score）
            if cluster_quality_heatmaps:
                print(f"\n{'='*80}")
                print("🎨 生成聚类质量指标热力图（quality_score、separation_score、penalty_score）...")
                print(f"{'='*80}")

                try:
                    create_cluster_quality_heatmaps(
                        results_dict=results,
                        color_metric='all_acc',
                        superclass_name=superclass_name,
                        output_dir=args.output_dir,
                        save_plots=True
                    )
                except Exception as e:
                    print(f"⚠️  聚类质量指标热力图生成失败: {e}")

            print(f"\n✅ {superclass_name} 热力图绘制完成！")
            final_output_dir = os.path.join(args.output_dir, superclass_name)
            print(f"📁 结果保存在: {final_output_dir}")

        except Exception as e:
            print(f"❌ {superclass_name} 热力图绘制失败: {e}")
            import traceback
            traceback.print_exc()
            continue  # 继续处理下一个超类

    # ========== 全部完成 ==========
    print(f"\n{'='*80}")
    print(f"🎉 全部完成！成功处理 {len(superclass_list)} 个超类")
    print(f"📁 所有结果保存在: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
