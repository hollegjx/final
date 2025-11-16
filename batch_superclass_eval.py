#!/usr/bin/env python3
"""
批量超类评估脚本

先进行全数据集训练，然后在所有超类上进行评估
比较不同超类在GCD任务上的难度
"""

import subprocess
import os
import sys
import time
from data.cifar100_superclass import CIFAR100_SUPERCLASSES

def run_training_and_evaluation(superclass_name, base_args, gpu_id=0):
    """运行训练并在指定超类上评估"""
    print(f"\n{'='*80}")
    print(f"🎯 开始评估超类: {superclass_name}")
    print(f"   包含类别: {CIFAR100_SUPERCLASSES[superclass_name]}")
    print(f"{'='*80}")

    # 构建训练命令
    cmd = [
        'python', 'methods/contrastive_training/contrastive_training.py',
        '--dataset_name', 'cifar100',
        '--eval_superclass', superclass_name,
        '--exp_name', f'gcd_full_train_{superclass_name}_eval',
        '--gpu', str(gpu_id)
    ]

    # 添加基础参数
    for key, value in base_args.items():
        cmd.extend([f'--{key}', str(value)])

    print(f"🚀 执行命令: {' '.join(cmd)}")

    # 执行训练
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ 超类 '{superclass_name}' 评估完成")

        # 提取关键结果
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if f"超类 '{superclass_name}' 评估结果:" in line:
                # 找到评估结果附近的行
                idx = output_lines.index(line)
                for i in range(idx, min(idx + 5, len(output_lines))):
                    if 'ACC:' in output_lines[i]:
                        print(f"   {output_lines[i].strip()}")

        duration = time.time() - start_time
        print(f"   用时: {duration:.1f}秒")

        return True, result.stdout

    except subprocess.CalledProcessError as e:
        print(f"❌ 超类 '{superclass_name}' 评估失败:")
        print(f"   错误: {e}")
        if e.stdout:
            print("   标准输出:", e.stdout[-500:])  # 最后500字符
        if e.stderr:
            print("   错误输出:", e.stderr[-500:])
        return False, None

def main():
    """主函数"""
    print("🌟 批量超类评估工具")
    print("=" * 80)
    print("功能: 使用完整CIFAR-100训练，然后在各个超类上分别评估")
    print("=" * 80)

    # 基础训练参数
    base_args = {
        'epochs': 200,
        'batch_size': 128,
        'lr': 0.1,
        'sup_con_weight': 0.5,
        'temperature': 1.0,
        'num_workers': 16,
        'seed': 1
    }

    # 可用的超类列表
    available_superclasses = list(CIFAR100_SUPERCLASSES.keys())
    print(f"📋 可用超类 ({len(available_superclasses)}个):")
    for i, superclass in enumerate(available_superclasses, 1):
        class_count = len(CIFAR100_SUPERCLASSES[superclass])
        print(f"   {i:2d}. {superclass:<30} ({class_count} 个类别)")

    # GPU选择
    gpu_choice = input(f"\n🖥️ 请选择GPU设备 (默认: 0): ").strip()
    gpu_id = 0
    if gpu_choice.isdigit():
        gpu_id = int(gpu_choice)
    print(f"✅ 使用GPU: {gpu_id}")

    # 用户选择
    choice = input(f"\n🤔 选择评估模式:\n"
                  f"   1. 评估所有超类 (推荐)\n"
                  f"   2. 评估指定超类\n"
                  f"   3. 评估前N个超类\n"
                  f"请输入选择 (1-3): ").strip()

    target_superclasses = []

    if choice == '1':
        target_superclasses = available_superclasses
        print(f"✅ 将评估所有 {len(target_superclasses)} 个超类")

    elif choice == '2':
        superclass_names = input("请输入超类名称 (用逗号分隔): ").strip().split(',')
        for name in superclass_names:
            name = name.strip()
            if name in available_superclasses:
                target_superclasses.append(name)
            else:
                print(f"⚠️ 警告: 未找到超类 '{name}'")

    elif choice == '3':
        try:
            n = int(input(f"请输入要评估的超类数量 (1-{len(available_superclasses)}): "))
            target_superclasses = available_superclasses[:n]
        except ValueError:
            print("❌ 无效输入，使用默认前5个超类")
            target_superclasses = available_superclasses[:5]
    else:
        print("❌ 无效选择，使用默认前5个超类")
        target_superclasses = available_superclasses[:5]

    if not target_superclasses:
        print("❌ 没有选择任何超类，退出")
        return

    print(f"\n🎯 开始批量评估，目标超类: {target_superclasses}")

    # 结果收集
    results = {}
    successful = 0
    failed = 0

    total_start_time = time.time()

    for i, superclass in enumerate(target_superclasses, 1):
        print(f"\n📈 进度: {i}/{len(target_superclasses)}")

        success, output = run_training_and_evaluation(superclass, base_args, gpu_id)

        if success:
            results[superclass] = output
            successful += 1
        else:
            failed += 1

        # 间隔休息（避免资源冲突）
        if i < len(target_superclasses):
            print("😴 休息5秒...")
            time.sleep(5)

    # 总结结果
    total_duration = time.time() - total_start_time

    print(f"\n{'='*80}")
    print("🎉 批量评估完成!")
    print(f"📊 总结:")
    print(f"   成功: {successful} 个超类")
    print(f"   失败: {failed} 个超类")
    print(f"   总用时: {total_duration/60:.1f} 分钟")
    print(f"{'='*80}")

    # 保存结果到文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"superclass_eval_results_{timestamp}.txt"

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("CIFAR-100超类GCD评估结果\n")
        f.write("="*50 + "\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"成功评估: {successful} 个超类\n")
        f.write(f"失败评估: {failed} 个超类\n")
        f.write(f"总用时: {total_duration/60:.1f} 分钟\n\n")

        for superclass, output in results.items():
            f.write(f"\n超类: {superclass}\n")
            f.write("-" * 30 + "\n")
            f.write(f"包含类别: {CIFAR100_SUPERCLASSES[superclass]}\n")
            # 提取关键评估指标
            lines = output.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['ACC:', '评估结果:', '过滤后样本数:']):
                    f.write(f"   {line.strip()}\n")
            f.write("\n")

    print(f"📄 详细结果已保存到: {result_file}")

    # 显示简要结果对比
    if successful > 1:
        print(f"\n📊 简要结果对比:")
        print(f"{'超类名称':<25} {'All ACC':<10} {'Old ACC':<10} {'New ACC':<10}")
        print("-" * 60)

        # 这里需要解析输出提取数值，简化版本先跳过
        print("   (详细结果请查看保存的文件)")

if __name__ == "__main__":
    main()