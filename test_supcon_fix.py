#!/usr/bin/env python3
"""测试 SupConLoss 权重修复是否正确"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from methods.contrastive_training.contrastive_training import SupConLoss


def test_weighted_loss():
    """验证加权损失的正确性"""
    print("=" * 80)
    print("测试 SupConLoss 权重修复")
    print("=" * 80)

    # 创建损失函数
    criterion = SupConLoss(temperature=0.07)

    # 模拟数据
    batch_size = 4
    n_views = 2
    feat_dim = 128

    # 创建特征 (batch_size, n_views, feat_dim)
    features = torch.randn(batch_size, n_views, feat_dim)
    features = torch.nn.functional.normalize(features, dim=-1)

    # 创建标签
    labels = torch.tensor([0, 0, 1, 1])

    # 创建权重（模拟密度权重）
    # 范围 [0.01, 0.99]，平均值约 0.5
    sample_weights = torch.tensor([0.99, 0.8, 0.3, 0.01])

    print(f"\n📊 测试配置:")
    print(f"   batch_size: {batch_size}")
    print(f"   n_views: {n_views}")
    print(f"   labels: {labels.tolist()}")
    print(f"   weights: {sample_weights.tolist()}")
    print(f"   weights.mean(): {sample_weights.mean():.4f}")
    print(f"   weights.sum(): {sample_weights.sum():.4f}")

    # 测试1: 无权重损失
    print(f"\n{'='*80}")
    print("测试 1: 无权重损失 (baseline)")
    print(f"{'='*80}")

    loss_no_weight = criterion(features, labels=labels)
    print(f"   无权重损失: {loss_no_weight.item():.6f}")

    # 测试2: 有权重损失
    print(f"\n{'='*80}")
    print("测试 2: 有权重损失（修复后）")
    print(f"{'='*80}")

    loss_weighted = criterion(features, labels=labels, sample_weights=sample_weights)
    print(f"   有权重损失: {loss_weighted.item():.6f}")

    # 测试3: 权重全为1（应该等于无权重）
    print(f"\n{'='*80}")
    print("测试 3: 权重全为1（应该等于无权重损失）")
    print(f"{'='*80}")

    weights_ones = torch.ones(batch_size)
    loss_ones = criterion(features, labels=labels, sample_weights=weights_ones)
    print(f"   权重全为1的损失: {loss_ones.item():.6f}")
    print(f"   无权重损失:      {loss_no_weight.item():.6f}")
    print(f"   差异: {abs(loss_ones.item() - loss_no_weight.item()):.8f}")

    if abs(loss_ones.item() - loss_no_weight.item()) < 1e-6:
        print("   ✅ 通过：权重为1时等于无权重损失")
    else:
        print("   ❌ 失败：权重为1时应该等于无权重损失")

    # 测试4: 验证权重上界
    print(f"\n{'='*80}")
    print("测试 4: 验证权重上界（加权损失不应超过原始损失太多）")
    print(f"{'='*80}")

    # 创建权重上界为1的情况
    weights_max_1 = torch.tensor([0.99, 0.99, 0.99, 0.99])
    loss_max_weight = criterion(features, labels=labels, sample_weights=weights_max_1)

    print(f"   无权重损失:        {loss_no_weight.item():.6f}")
    print(f"   最大权重(0.99)损失: {loss_max_weight.item():.6f}")
    print(f"   比值: {loss_max_weight.item() / loss_no_weight.item():.4f}")

    if loss_max_weight.item() <= loss_no_weight.item() * 1.01:  # 允许1%误差
        print("   ✅ 通过：最大权重时损失不超过原始损失")
    else:
        print("   ⚠️  警告：最大权重时损失超过原始损失")

    # 测试5: 验证权重分布影响
    print(f"\n{'='*80}")
    print("测试 5: 不同权重分布的影响")
    print(f"{'='*80}")

    # 高密度样本（权重接近1）
    weights_high = torch.tensor([0.95, 0.95, 0.05, 0.05])
    loss_high = criterion(features, labels=labels, sample_weights=weights_high)

    # 低密度样本（权重接近0）
    weights_low = torch.tensor([0.05, 0.05, 0.95, 0.95])
    loss_low = criterion(features, labels=labels, sample_weights=weights_low)

    print(f"   高密度权重 [0.95, 0.95, 0.05, 0.05] -> 损失: {loss_high.item():.6f}")
    print(f"   低密度权重 [0.05, 0.05, 0.95, 0.95] -> 损失: {loss_low.item():.6f}")
    print(f"   差异: {abs(loss_high.item() - loss_low.item()):.6f}")

    if abs(loss_high.item() - loss_low.item()) > 1e-6:
        print("   ✅ 通过：权重分布影响损失值")
    else:
        print("   ⚠️  警告：权重分布未影响损失值")

    print(f"\n{'='*80}")
    print("✅ 所有测试完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_weighted_loss()
