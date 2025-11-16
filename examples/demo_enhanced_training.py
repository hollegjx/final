#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强训练功能演示脚本
展示新添加的训练功能：轮次分割、时间显示、性能差距、早停机制
"""

import os
import sys
import time
import argparse

# 添加项目根目录路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_utils import (
    EarlyStoppingMonitor,
    PerformanceTracker,
    TrainingSession,
    print_epoch_separator,
    print_performance_summary,
    print_training_start_info,
    print_training_complete_info
)


def demo_early_stopping():
    """演示早停机制"""
    print("Early Stopping Demo")
    print("=" * 50)

    # 创建早停监控器
    early_stopping = EarlyStoppingMonitor(patience=5, metric_name="test_acc")

    # 模拟训练过程
    test_accs = [0.6, 0.65, 0.68, 0.69, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64]

    for epoch, acc in enumerate(test_accs):
        print(f"轮次 {epoch}: 测试准确率 = {acc:.3f}")
        should_stop = early_stopping.update(acc, epoch)

        if should_stop:
            print(f"早停触发！在第{epoch}轮停止")
            break

        time.sleep(0.5)  # 模拟训练时间

    best_acc, best_epoch = early_stopping.get_best_info()
    print(f"\n最佳性能: {best_acc:.3f} (第{best_epoch}轮)")


def demo_performance_tracker():
    """演示性能跟踪器"""
    print("\n🎯 性能跟踪器演示")
    print("=" * 50)

    tracker = PerformanceTracker()

    # 模拟训练过程
    performances = [
        (0.60, 0.55, 0.65),  # epoch 0
        (0.65, 0.62, 0.68),  # epoch 1
        (0.70, 0.68, 0.72),  # epoch 2 (最佳)
        (0.68, 0.66, 0.70),  # epoch 3
        (0.66, 0.64, 0.68),  # epoch 4
    ]

    for epoch, (all_acc, old_acc, new_acc) in enumerate(performances):
        tracker.update(all_acc, old_acc, new_acc, epoch)

        # 计算与最佳性能的差距
        gap = tracker.get_performance_gap(all_acc, old_acc, new_acc)
        elapsed = tracker.get_elapsed_time()

        print(f"\n轮次 {epoch}: All={all_acc:.3f}, Old={old_acc:.3f}, New={new_acc:.3f}")
        print(f"  已用时: {elapsed}")
        print(f"  差距: All={gap['all_acc_gap']:+.3f}, Old={gap['old_acc_gap']:+.3f}, New={gap['new_acc_gap']:+.3f}")

        time.sleep(0.3)

    best_perf = tracker.get_best_performance()
    print(f"\n最佳性能 (第{best_perf['best_epoch']}轮):")
    print(f"  All: {best_perf['best_all_acc']:.3f}")
    print(f"  Old: {best_perf['best_old_acc']:.3f}")
    print(f"  New: {best_perf['best_new_acc']:.3f}")


def demo_training_session():
    """演示完整训练会话"""
    print("\n🎯 完整训练会话演示")
    print("=" * 50)

    # 模拟训练参数
    class MockArgs:
        dataset_name = "cifar100_superclass"
        superclass_name = "mammals"
        num_labeled_classes = 20
        num_unlabeled_classes = 3
        epochs = 10
        batch_size = 128
        lr = 0.1
        sup_con_weight = 0.5

    args = MockArgs()

    # 创建训练会话
    session = TrainingSession(args, enable_early_stopping=True, patience=3)

    # 开始训练
    model_info = {
        'name': 'vit_dino',
        'feat_dim': 768
    }
    session.start_training(model_info)

    # 模拟训练过程
    performances = [
        (0.80, 0.40, 0.60, 0.55, 0.65, 0.62, 0.68, 0.66, 0.70),  # epoch 0
        (0.82, 0.42, 0.65, 0.60, 0.70, 0.67, 0.73, 0.71, 0.75),  # epoch 1
        (0.85, 0.45, 0.70, 0.65, 0.75, 0.72, 0.78, 0.76, 0.80),  # epoch 2 (最佳)
        (0.83, 0.43, 0.68, 0.63, 0.73, 0.70, 0.76, 0.74, 0.78),  # epoch 3
        (0.81, 0.41, 0.66, 0.61, 0.71, 0.68, 0.74, 0.72, 0.76),  # epoch 4
        (0.79, 0.39, 0.64, 0.59, 0.69, 0.66, 0.72, 0.70, 0.74),  # epoch 5 - 应该触发早停
    ]

    for epoch, (train_acc, loss_avg, all_acc, old_acc, new_acc,
                all_acc_test, old_acc_test, new_acc_test) in enumerate(performances):

        # 开始轮次
        session.start_epoch(epoch)

        # 模拟训练时间
        time.sleep(1)

        # 结束轮次，检查是否早停
        should_stop = session.end_epoch(
            epoch=epoch,
            train_acc=train_acc,
            loss_avg=loss_avg,
            all_acc=all_acc,
            old_acc=old_acc,
            new_acc=new_acc,
            all_acc_test=all_acc_test,
            old_acc_test=old_acc_test,
            new_acc_test=new_acc_test
        )

        # 模拟模型保存
        session.save_model_info(f"model_epoch_{epoch}.pt")
        if old_acc_test > 0.75:  # 模拟最佳模型
            session.save_model_info(f"model_best.pt", is_best=True, acc=old_acc_test)

        if should_stop:
            print(f"\n🛑 演示：早停触发在第{epoch+1}轮")
            session.finish_training(epoch, early_stopped=True)
            break
    else:
        session.finish_training(args.epochs - 1, early_stopped=False)

    # 显示最终结果
    best_perf = session.get_best_performance()
    print(f"\n📊 最终最佳性能:")
    for key, value in best_perf.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def demo_output_formatting():
    """演示输出格式"""
    print("\n🎯 输出格式演示")
    print("=" * 50)

    # 演示轮次分隔符
    for epoch in range(3):
        print_epoch_separator(epoch, 10, f"00:0{epoch+1}:30")

        # 模拟一些输出
        print("🔄 训练中...")
        print("📊 计算指标...")
        time.sleep(0.5)

        # 演示性能总结
        performance_gap = {
            'all_acc_gap': -0.02 + epoch * 0.01,
            'old_acc_gap': -0.03 + epoch * 0.015,
            'new_acc_gap': -0.01 + epoch * 0.005
        }

        best_performance = {
            'best_all_acc': 0.75,
            'best_old_acc': 0.72,
            'best_new_acc': 0.78,
            'best_epoch': 1
        }

        print_performance_summary(
            epoch=epoch,
            train_acc=0.80 + epoch * 0.02,
            all_acc=0.65 + epoch * 0.03,
            old_acc=0.62 + epoch * 0.04,
            new_acc=0.68 + epoch * 0.02,
            all_acc_test=0.70 + epoch * 0.02,
            old_acc_test=0.67 + epoch * 0.03,
            new_acc_test=0.73 + epoch * 0.01,
            performance_gap=performance_gap,
            best_performance=best_performance,
            loss_avg=1.5 - epoch * 0.3
        )


def main():
    """主演示函数"""
    parser = argparse.ArgumentParser(description='增强训练功能演示')
    parser.add_argument('--demo', type=str, default='all',
                        choices=['early_stopping', 'performance_tracker', 'training_session', 'output_formatting', 'all'],
                        help='选择要演示的功能')

    args = parser.parse_args()

    print("*** GCD Enhanced Training Features Demo ***")
    print("Demonstrating new features: epoch separation, time display, performance gap, early stopping")
    print("=" * 80)

    if args.demo in ['early_stopping', 'all']:
        demo_early_stopping()

    if args.demo in ['performance_tracker', 'all']:
        demo_performance_tracker()

    if args.demo in ['output_formatting', 'all']:
        demo_output_formatting()

    if args.demo in ['training_session', 'all']:
        demo_training_session()

    print("\n🎉 演示完成！")
    print("现在您可以在实际训练中体验这些新功能了。")
    print("\n💡 使用提示:")
    print("1. 原版GCD训练: python methods/contrastive_training/contrastive_training.py [参数]")
    print("2. 超类训练: python train_superclass.py --superclass_name mammals [参数]")
    print("3. 所有功能都自动启用，包括29轮早停机制")


if __name__ == "__main__":
    main()