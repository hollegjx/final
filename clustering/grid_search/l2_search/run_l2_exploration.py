#!/usr/bin/env python3
"""
L2权重探索命令行工具

运行L2权重探索，支持单个或多个超类的批量处理

命令行示例：
    # 单个超类
    python -m clustering.grid_search.l2_search.run_l2_exploration \
        --superclass_name vehicles \
        --search_dir /data/gjx/checkpoints/search \
        --output_dir /data/gjx/checkpoints/l2_search
    
    # 自动处理所有超类
    python -m clustering.grid_search.l2_search.run_l2_exploration \
        --search_dir /data/gjx/checkpoints/search \
        --output_dir /data/gjx/checkpoints/l2_search
"""

import argparse
import os
import sys
import glob
import re
from pathlib import Path
from typing import List, Optional

from config import grid_search_output_dir, l2_search_output_dir

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from .l2_heatmap_plotter import (
    plot_all_l2_configurations,
    create_l2_comparison_subplot,
    create_single_metric_heatmap
)


def detect_available_superclasses(search_dir: str) -> List[str]:
    """
    自动探测搜索结果目录中存在的所有超类
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
    
    return sorted(superclass_list)


def _validate_task_folder(task_folder: str, output_dir: str) -> Optional[Path]:
    """验证任务文件夹名并返回完整 search_dir 路径。"""
    if not task_folder or not task_folder.strip():
        print("❌ 错误：--task_folder 参数不能为空")
        return None
    task_folder = task_folder.strip()
    pattern = r'^\d+class_\d{2}_\d{2}_\d{2}_\d{2}$'
    if not re.match(pattern, task_folder):
        print("❌ 错误：任务文件夹格式不正确")
        print(f"   当前输入: {task_folder}")
        print("   期望格式: <N>class_MM_DD_HH_MM (例如: 4class_11_06_21_09)")
        return None
    search_dir = Path(output_dir) / task_folder
    if not search_dir.exists():
        print("❌ 错误：任务目录不存在")
        print(f"   完整路径: {search_dir}")
        root = Path(output_dir)
        if root.exists():
            candidates = [d.name for d in root.iterdir() if d.is_dir() and re.match(pattern, d.name)]
            if candidates:
                print("\n   可用的任务文件夹:")
                for name in sorted(candidates):
                    print(f"   - {name}")
        return None
    return search_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='L2权重探索工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 处理单个超类:
     python -m clustering.grid_search.l2_search.run_l2_exploration \\
         --superclass_name vehicles

  2. 自动处理所有超类:
     python -m clustering.grid_search.l2_search.run_l2_exploration

  3. 自定义目录:
     python -m clustering.grid_search.l2_search.run_l2_exploration \\
         --superclass_name trees \\
         --search_dir /custom/path/search \\
         --output_dir /custom/path/l2_search

说明:
  - 如果不指定superclass_name，将自动处理search_dir中的所有超类
  - 需要先运行网格搜索(batch_runner.py)生成包含separation_score和penalty_score的结果文件
  - 输出包括9张权重热力图、文本报告和稳定性分析
        """
    )
    
    parser.add_argument('--superclass_name', type=str, default=None,
                        help='超类名称 (如不指定则自动处理所有超类)')
    # 搜索结果根目录 + 任务文件夹（推荐新用法）
    parser.add_argument('--output_dir', type=str,
                        default=grid_search_output_dir,
                        help=f'[输入] 网格搜索结果根目录（与 batch_runner.py 一致，默认: {grid_search_output_dir}）')
    parser.add_argument('--task_folder', type=str, required=True,
                        help='[输入] 任务文件夹名（必填，格式: <N>class_MM_DD_HH_MM）。例如: 4class_11_06_21_09')
    # 兼容旧参数：直接传入完整 search_dir
    parser.add_argument('--search_dir', type=str, default=None,
                        help='[兼容] 直接指定完整搜索目录；若提供 --task_folder 将忽略此参数')
    # 输出目录（热力图等）
    parser.add_argument('--output_dir_heatmap', type=str,
                        default=l2_search_output_dir,
                        help=f'[输出] L2 探索结果输出目录（热力图、报告等，默认: {l2_search_output_dir}）')
    parser.add_argument('--create_subplot', action='store_true',
                        help='额外创建3x3对比子图')
    parser.add_argument('--weight_sum', type=int, default=10,
                        help='分离度与惩罚项权重总和 (默认: 10)')
    parser.add_argument('--skip_single_metric', action='store_true',
                        help='跳过 separation/penalty 单指标热力图绘制')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 L2权重探索工具")
    print("=" * 80)
    # 解析 search_dir：优先 task_folder，其次兼容 search_dir
    if args.task_folder:
        resolved_search_dir = _validate_task_folder(args.task_folder, args.output_dir)
        if resolved_search_dir is None:
            return 1
        search_dir: str = str(resolved_search_dir)
    else:
        # 防御性分支（按理不会命中，因为 --task_folder 为必填）
        if not args.search_dir:
            print('❌ 错误：必须提供 --task_folder 或 --search_dir 参数')
            print('💡 推荐使用: --task_folder 4class_11_06_21_09')
            return 1
        search_dir = args.search_dir

    print(f"搜索目录: {search_dir}")
    print(f"输出目录: {args.output_dir_heatmap}")
    print(f"对比子图: {'启用' if args.create_subplot else '禁用'}")
    print(f"权重总和: {args.weight_sum}")
    print(f"单指标热力图: {'跳过' if args.skip_single_metric else '生成'}")
    
    if args.weight_sum < 2 or args.weight_sum > 100:
        print("❌ weight_sum 参数超出允许范围 (2-100)")
        return 1
    
    # 确定要处理的超类列表
    if args.superclass_name:
        superclass_list = [args.superclass_name]
        print(f"处理模式: 单个超类 - {args.superclass_name}")
    else:
        superclass_list = detect_available_superclasses(search_dir)
        if not superclass_list:
            print(f"\n❌ 未在 {search_dir} 中找到任何超类搜索结果")
            print("💡 请先运行batch_runner.py进行网格搜索，或手动指定超类名称")
            return 1
        print(f"处理模式: 批量处理 - 发现 {len(superclass_list)} 个超类")
    
    print(f"超类列表: {superclass_list}")
    print("=" * 80)
    
    # 处理结果统计
    success_count = 0
    failed_superclasses = []
    
    # 逐个处理超类
    for idx, superclass_name in enumerate(superclass_list, 1):
        print(f"\n{'='*80}")
        print(f"🔄 处理超类 [{idx}/{len(superclass_list)}]: {superclass_name}")
        print(f"{'='*80}")
        
        try:
            # 运行L2权重探索
            result = plot_all_l2_configurations(
                superclass_name=superclass_name,
                search_dir=search_dir,
                output_dir=args.output_dir_heatmap,
                weight_sum=args.weight_sum
            )
            
            if result and 'output_dir' in result:
                success_count += 1
                print(f"✅ {superclass_name} 处理完成")
                print(f"📁 结果保存在: {result['output_dir']}")
                
                # 可选：创建对比子图
                if not args.skip_single_metric:
                    raw_results = result.get('raw_data')
                    if raw_results:
                        try:
                            create_single_metric_heatmap(
                                raw_results, 'separation_score', superclass_name,
                                args.output_dir_heatmap, save_plots=True, higher_is_better=True
                            )
                            create_single_metric_heatmap(
                                raw_results, 'penalty_score', superclass_name,
                                args.output_dir_heatmap, save_plots=True, higher_is_better=False
                            )
                        except Exception as metric_exc:
                            print(f"⚠️  单指标热力图生成失败: {metric_exc}")
                    else:
                        print("⚠️  未获取到原始数据，跳过单指标热力图")
                
                if args.create_subplot:
                    try:
                        create_l2_comparison_subplot(
                            superclass_name, search_dir, args.output_dir_heatmap,
                            color_metric=result.get('color_metric', 'new_acc'),
                            weight_sum=args.weight_sum
                        )
                        print("📊 对比子图已创建")
                    except Exception as e:
                        print(f"⚠️  对比子图创建失败: {e}")
                
            else:
                failed_superclasses.append(superclass_name)
                print(f"❌ {superclass_name} 处理失败：无有效结果")
        
        except Exception as e:
            failed_superclasses.append(superclass_name)
            print(f"❌ {superclass_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 最终统计
    print("\n" + '='*80)
    print("🎉 L2权重探索完成！")
    print('='*80)
    print("📊 处理统计:")
    print(f"   成功: {success_count}/{len(superclass_list)}")
    print(f"   失败: {len(failed_superclasses)}/{len(superclass_list)}")
    
    if failed_superclasses:
        print(f"   失败超类: {failed_superclasses}")
    
    print(f"📁 所有结果保存在: {args.output_dir_heatmap}")
    
    # 输出使用建议
    if success_count > 0:
        print("\n💡 后续分析建议:")
        print("   1. 查看各超类的稳定性分析报告: */l2_weights_summary.txt")
        print("   2. 对比不同权重配置的热力图: */l2_weighted_sep*_pen*.png")
        if not args.skip_single_metric:
            print("   3. 查看单指标热力图: */single_metric_separation_score*.png 与 */single_metric_penalty_score*.png")
            print("   4. 参考推荐参数组合进行最终选择")
        else:
            print("   3. 参考推荐参数组合进行最终选择")
    
    print("=" * 80)
    
    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
