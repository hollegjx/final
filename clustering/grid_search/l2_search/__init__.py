"""
L2权重探索模块

用法：
    from clustering.grid_search.l2_search import plot_all_l2_configurations
    
    plot_all_l2_configurations(
        superclass_name='vehicles',
        search_dir='/data/gjx/checkpoints/search',
        output_dir='/data/gjx/checkpoints/l2_search',
        color_metric='new_acc',
        weight_sum=10
    )

模块组件：
    - l2_weight_calculator: L2权重计算和数据加载
    - l2_heatmap_plotter: 热力图绘制和可视化
"""

from .l2_weight_calculator import (
    load_raw_scores,
    compute_weighted_l2,
    generate_l2_report,
    get_weight_configurations,
    validate_l2_requirements,
    analyze_weight_stability
)

from .l2_heatmap_plotter import (
    create_l2_weighted_heatmap,
    plot_all_l2_configurations,
    create_l2_comparison_subplot,
    create_single_metric_heatmap,
    generate_summary_report
)

__all__ = [
    # L2权重计算器
    'load_raw_scores',
    'compute_weighted_l2',
    'generate_l2_report',
    'get_weight_configurations',
    'validate_l2_requirements',
    'analyze_weight_stability',
    
    # 热力图绘制器
    'create_l2_weighted_heatmap',
    'plot_all_l2_configurations',
    'create_l2_comparison_subplot',
    'create_single_metric_heatmap',
    'generate_summary_report'
]

# 版本信息
__version__ = "1.2.0"
__author__ = "L2 Weight Explorer"
__description__ = "L2权重探索工具：针对separation_score和penalty_score的权重优化分析"
