#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主测试程序
命令行入口，用于运行SS-DDBC聚类算法测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import numpy as np

from config import clustering_log_dir, error_analysis_log_dir
from project_utils.general_utils import str2bool
from ssddbc.data import set_deterministic_behavior  # 从ssddbc/data导入

# 兼容两种运行方式的导入
try:
    from ..ssddbc.analysis import analyze_cluster_composition
    from ..baseline.kmeans import test_kmeans_baseline
    from .test_superclass import test_adaptive_clustering_on_superclass
except ImportError:
    # 直接运行时的导入
    from ssddbc.ssddbc.analysis import analyze_cluster_composition
    from ssddbc.baseline.kmeans import test_kmeans_baseline
    from ssddbc.testing.test_superclass import test_adaptive_clustering_on_superclass


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='自适应密度聚类算法测试')

    # ========== 1. 全局设置 ==========
    parser.add_argument('--model_path', type=str, default=None,
                        help='训练好的模型路径（未指定时需结合缓存特征使用）')
    parser.add_argument('--superclass_name', type=str, default='trees',
                        help='测试的超类名称')
    parser.add_argument('--use_train_and_test', type=str2bool, default=True,
                        help='是否合并训练集和测试集进行聚类。True=合并(默认)，在更大数据集上构建更鲁棒的聚类；False=仅测试集，更快但可能不稳定')
    parser.add_argument('--l2', type=str2bool, default=True,
                        help='是否使用L2归一化特征，默认True。True=使用L2归一化(推荐，与eval_original_gcd保持一致)，False=使用原始特征(不推荐)')
    parser.add_argument('--fast_mode', type=str2bool, default=False,
                        help='快速模式，默认False。True=跳过不必要的计算（未知簇识别、簇类别标签）和调试输出，仅显示最终结果，用于网格搜索加速或批量实验；False=完整计算和输出')

    # ========== 2. 密度计算相关参数 ==========
    parser.add_argument('--k', type=int, default=10,
                        help='K近邻数量，用于密度计算和邻居发现。范围[3-30]，默认10。较大值(15-20)适合稠密数据，较小值(5-10)适合稀疏数据')
    parser.add_argument('--density_percentile', type=int, default=75,
                        help='高密度点选择的百分位阈值，范围[20-100]，默认75。值越高选择的核心点越少但越可靠(80-90适合噪声数据)，值越低选择的核心点越多但可能包含噪声(60-70适合干净数据)')
    parser.add_argument('--dense_method', type=int, default=0, choices=[0, 1, 2, 3],
                        help='密度计算方法，默认0。0=平均距离倒数(通用默认)，1=中位数距离倒数(抗噪声)，2=归一化倒数(密度值归一化到[0,1])，3=指数密度(强调局部密集度)。详见ssddbc/density/DENSITY_METHODS.md')

    # ========== 3. 骨干网络（高密度点）聚类相关参数 ==========
    parser.add_argument('--co_mode', type=int, default=2, choices=[1, 2, 3],
                        help='co截止距离计算模式，默认2。1=手动指定(需配合--co_manual)，2=K近邻平均距离(通用默认)，3=相对自适应距离(每点自适应，适合不均匀分布)。详见ssddbc/utils/CO_MODES.md')
    parser.add_argument('--co_manual', type=float, default=None,
                        help='手动指定的co截止距离值(仅当co_mode=1时使用)。通过网格搜索或先验知识确定。典型范围[1.0-5.0]取决于特征空间尺度')
    parser.add_argument('--eval_dense', type=str2bool, default=False,
                        help='是否仅评估高密度点(骨干网络)，默认False。True=只构建和评估高密度聚类，跳过低密度点分配，用于评估核心聚类质量')

    # ========== 4. 稀疏点分配相关参数 ==========
    parser.add_argument('--assign_model', type=int, default=2, choices=[1, 2, 3],
                        help='低密度点分配策略，默认2。1=簇原型就近(速度最快)，2=KNN投票加权(推荐默认，考虑邻域信息)，3=簇内K近邻平均距离(最精细但慢)。详见ssddbc/ssddbc/ASSIGNMENT_STRATEGIES.md')
    parser.add_argument('--voting_k', type=int, default=5,
                        help='KNN投票时使用的近邻数量，默认5(仅当assign_model=2时生效)。范围[3-15]，值越大分配越稳定但计算越慢')

    # ========== 5. 评估和对比相关参数 ==========
    parser.add_argument('--eval_version', type=str, default='v2', choices=['v1', 'v2'],
                        help='评估指标计算版本，默认v1。v1=原始版本(匈牙利算法)，v2=新版本(不同的匹配策略)。保持默认即可')
    parser.add_argument('--run_kmeans_baseline', type=str2bool, default=False,
                        help='是否运行K-means基线对比，默认False。True=同时运行K-means并输出对比结果，用于验证SS-DDBC的优势')
    parser.add_argument('--kmeans_merge', type=str2bool, default=False,
                        help='K-means是否合并训练集和测试集，默认False。仅在--run_kmeans_baseline=True时生效。True=合并数据集运行K-means')

    # ========== 6. 调试和分析相关参数 ==========
    parser.add_argument('--detail_dense', type=str2bool, default=False,
                        help='是否记录骨干网络聚类详细日志（当前实现已精简为无效，占位参数）。')
    parser.add_argument('--label_guide', type=str2bool, default=False,
                        help='是否启用标签引导模式，默认False。True=在核心点聚类完成后，将剩余稀疏点中的已知标签样本直接分配到对应标签的核心点簇，然后再分配剩余无标签样本')
    parser.add_argument('--analyze_errors', type=str2bool, default=False,
                        help='是否分析测试集错误样本（当前实现已移除错误样本分析逻辑，占位参数）。')

    # ========== 6.5 聚类质量评估参数 ==========
    parser.add_argument('--use_cluster_quality', type=str2bool, default=False,
                        help='是否使用聚类质量评估指标作为L2损失，默认False。True=使用基于簇间分离度和局部密度的聚类质量评估')
    parser.add_argument('--cluster_distance_method', type=int, default=1, choices=[1, 2, 3],
                        help='簇距离计算方法，默认1。1=最近k对点平均距离（关注簇边界），2=所有点对平均距离（整体分离度），3=原型距离（计算效率高）')
    parser.add_argument('--l1_type', type=str, default='cross_entropy', choices=['accuracy', 'cross_entropy'],
                        help='L1监督损失类型，默认cross_entropy。accuracy=基于匈牙利算法的准确率损失(1-ACC)，cross_entropy=基于簇类别分布的交叉熵损失')

    # L2损失权重参数
    parser.add_argument('--separation_weight', type=float, default=1.0,
                        help='L2损失中簇间分离度的权重（默认1.0）。增大此值会更加重视簇之间的分离程度。')
    parser.add_argument('--penalty_weight', type=float, default=1.0,
                        help='L2损失中局部密度惩罚的权重（默认1.0）。增大此值会更加重视样本的局部密度一致性（同簇样本近，异簇样本远）。')
    parser.add_argument('--l2_components', type=str, default=None,
                        help="指定启用的 L2 组件，使用空格或逗号分隔，例如 'separation penalty'")
    parser.add_argument('--l2_component_weights', type=str, default=None,
                        help="指定 L2 组件权重，支持 JSON 字符串或 key=value 的逗号形式，如 "
                             "'{\"separation\":0.5,\"penalty\":0.5}' 或 'separation=0.5,penalty=0.5'")

    # ========== 7. 网格搜索参数范围 ==========
    parser.add_argument('--grid_search', type=str2bool, default=False,
                        help='是否启用网格搜索优化参数，默认False。True=自动搜索k和density_percentile的最佳组合，结果保存为CSV文件')
    parser.add_argument('--k_min', type=int, default=3,
                        help='网格搜索时k值的最小值，默认3。建议范围[3-10]，太小会导致密度估计不稳定')
    parser.add_argument('--k_max', type=int, default=21,
                        help='网格搜索时k值的最大值，默认21。建议范围[15-30]，太大会过度平滑密度分布。搜索范围为[k_min, k_max)，步长为2')
    parser.add_argument('--dp_min', type=int, default=50,
                        help='网格搜索时密度百分位的最小值，默认20。建议范围[20-50]，太小会选择过多噪声点作为核心')
    parser.add_argument('--dp_max', type=int, default=100,
                        help='网格搜索时密度百分位的最大值，默认100。建议范围[80-100]。搜索范围为[dp_min, dp_max]，步长由--dp_step指定')
    parser.add_argument('--dp_step', type=int, default=5,
                        help='网格搜索时密度百分位的步长，默认5。步长越小搜索越精细但耗时越长。建议范围[5-10]')

    args = parser.parse_args()

    def _parse_l2_components(value: str):
        if value is None:
            return None
        parts = [part.strip() for part in value.replace(',', ' ').split() if part.strip()]
        return parts

    def _parse_l2_component_weights(value: str):
        if value is None:
            return None
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return {str(k): float(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

        weights = {}
        for chunk in value.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            if '=' not in chunk:
                raise ValueError(f"无法解析 l2_component_weights 条目: {chunk}")
            key, val = chunk.split('=', 1)
            weights[key.strip()] = float(val.strip())
        return weights

    parsed_l2_components = _parse_l2_components(args.l2_components) if args.l2_components else None
    parsed_l2_weights = _parse_l2_component_weights(args.l2_component_weights) if args.l2_component_weights else None

    # 设置确定性行为（仅torch相关）
    set_deterministic_behavior()

    print("自适应密度聚类算法测试")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"超类名称: {args.superclass_name}")
    print(f"使用训练+测试: {args.use_train_and_test}")
    print(f"特征归一化: {'L2归一化' if args.l2 else '无L2归一化'}")
    print(f"评估版本: {args.eval_version}")
    print(f"K-means合并: {'启用' if args.kmeans_merge else '禁用'}")
    print(f"网格搜索: {'启用' if args.grid_search else '禁用'}")
    print(f"随机种子: 0 (与K-means保持一致)")
    print(f"算法参数: k={args.k}, density_percentile={args.density_percentile}")
    print(f"稀疏点分配: assign_model={args.assign_model}, voting_k={args.voting_k}")
    if parsed_l2_components is not None:
        print(f"L2 组件: {parsed_l2_components} (自定义)")
        if parsed_l2_weights:
            print(f"L2 组件权重: {parsed_l2_weights}")
    else:
        print(f"L2 组件: {'默认(separation, penalty)' if args.use_cluster_quality else '未启用'}")
    print("="*80)

    try:
        # 网格搜索功能已迁移到 ssddbc.grid_search.batch_runner
        if args.grid_search:
            print("🔍 网格搜索功能已迁移到 ssddbc.grid_search.batch_runner")
            print("请使用: python -m ssddbc.grid_search.batch_runner")
            return
        # 运行SS-DDBC算法（遵守K-means的random_state=0设计）
        ssddbc_results = test_adaptive_clustering_on_superclass(
            superclass_name=args.superclass_name,
            model_path=args.model_path,
            use_train_and_test=args.use_train_and_test,
            k=args.k,
            density_percentile=args.density_percentile,
            random_state=0,  # 与K-means保持一致，固定使用0
            eval_version=args.eval_version,
            run_kmeans_baseline=args.run_kmeans_baseline,  # 传递K-means基线参数
            use_l2=args.l2,
            eval_dense=args.eval_dense,
            fast_mode=args.fast_mode,  # 传递快速模式参数
            dense_method=args.dense_method,  # 传递密度计算方法
            assign_model=args.assign_model,  # 传递稀疏点分配策略
            voting_k=args.voting_k,  # 传递KNN投票邻居数量
            co_mode=args.co_mode,  # 传递co计算模式
            co_manual=args.co_manual,  # 传递手动指定的co值
            detail_dense=args.detail_dense,  # 占位参数（当前实现不再记录详细日志）
            label_guide=args.label_guide,  # 传递标签引导模式参数
            # analyze_errors 参数已移除，这里保持向后兼容占位
            use_cluster_quality=args.use_cluster_quality,  # 传递聚类质量评估参数
            cluster_distance_method=args.cluster_distance_method,  # 传递簇距离计算方法
            l1_type=args.l1_type,  # 传递L1损失类型
            separation_weight=args.separation_weight,  # 传递L2中簇间分离度权重
            penalty_weight=args.penalty_weight,  # 传递L2中局部密度惩罚权重
            l2_components=parsed_l2_components,
            l2_component_weights=parsed_l2_weights
        )

        # 如果开启K-means基线对比
        if args.run_kmeans_baseline:
            print("\n" + "="*80)
            print("🔄 运行K-means基线对比...")
            print("✅ 现在使用与eval_original_gcd完全相同的L2归一化特征")

            # 使用SS-DDBC测试中已提取的测试集特征 (相同的L2归一化)
            test_features = ssddbc_results['test_features']
            test_targets = ssddbc_results['test_targets']
            test_known_mask = ssddbc_results['test_known_mask']

            # 使用真实的类别数作为K-means聚类数（与eval_original_gcd.py保持一致）
            n_true_classes = len(np.unique(test_targets))
            print(f"🎯 K-means聚类数量: {n_true_classes} (真实类别数)")

            # 运行K-means (使用相同的L2归一化特征和评估版本)
            kmeans_results = test_kmeans_baseline(
                test_features,  # 相同的L2归一化特征
                test_targets,
                test_known_mask,
                n_clusters=n_true_classes,  # 使用真实类别数
                random_state=0,  # K-means保持原始的random_state=0
                eval_version=args.eval_version,  # 使用相同的评估版本
                kmeans_merge=args.kmeans_merge,  # 是否合并训练集和测试集
                train_features=ssddbc_results.get('train_features'),  # 训练集特征
                train_targets=ssddbc_results.get('train_targets'),   # 训练集标签
                train_known_mask=ssddbc_results.get('train_known_mask')  # 训练集已知掩码
            )

            # 显示K-means聚类细节分析
            print(f"\n🔍 K-means聚类结果详细分析:")
            kmeans_predictions = kmeans_results['predictions']
            kmeans_n_clusters = kmeans_results['n_clusters']

            # 检查是否使用了kmeans_merge模式
            if args.kmeans_merge and 'all_predictions' in kmeans_results:
                # 合并模式：显示整个合并数据集的聚类分析
                print(f"   模式: 合并训练+测试集分析")
                all_predictions_kmeans = kmeans_results['all_predictions']
                all_targets_kmeans = np.concatenate([ssddbc_results['train_targets'], test_targets], axis=0)
                all_known_mask_kmeans = np.concatenate([ssddbc_results['train_known_mask'], test_known_mask], axis=0)
                all_labeled_mask_kmeans = np.concatenate([ssddbc_results['train_labeled_mask'], np.zeros_like(test_known_mask, dtype=bool)], axis=0)

                analyze_cluster_composition(
                    all_predictions_kmeans,
                    all_targets_kmeans,
                    all_known_mask_kmeans,
                    all_labeled_mask_kmeans,
                    []  # K-means没有潜在未知类
                )
            else:
                # 仅测试集模式：显示测试集聚类分析
                print(f"   模式: 仅测试集分析")
                analyze_cluster_composition(
                    kmeans_predictions,
                    test_targets,
                    test_known_mask,
                    np.zeros_like(test_known_mask, dtype=bool),  # K-means没有labeled_mask概念，设为全False
                    []  # K-means没有潜在未知类
                )

            # 对比结果
            print(f"\n📊 算法对比结果:")
            print("="*80)
            print(f"{'指标':<15} {'SS-DDBC':<12} {'K-means':<12} {'差异':<12}")
            print("-"*80)
            print(f"{'All ACC':<15} {ssddbc_results['all_acc']:<12.4f} {kmeans_results['all_acc']:<12.4f} {ssddbc_results['all_acc']-kmeans_results['all_acc']:<+12.4f}")
            print(f"{'Old ACC':<15} {ssddbc_results['old_acc']:<12.4f} {kmeans_results['old_acc']:<12.4f} {ssddbc_results['old_acc']-kmeans_results['old_acc']:<+12.4f}")
            print(f"{'New ACC':<15} {ssddbc_results['new_acc']:<12.4f} {kmeans_results['new_acc']:<12.4f} {ssddbc_results['new_acc']-kmeans_results['new_acc']:<+12.4f}")
            print(f"{'聚类数':<15} {ssddbc_results['n_clusters']:<12} {kmeans_results['n_clusters']:<12} {'=':<12}")
            print("="*80)

        print("\n测试完成!")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
