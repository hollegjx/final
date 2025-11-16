#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
超类模型评估脚本
用于加载预训练模型并重新评估指定超类的性能

=== 调试历程和关键发现 ===

本脚本经历了一个重要的调试过程，发现了导致评估结果不一致的根本原因：

问题背景：
- 训练时模型达到0.8 All ACC，但重新评估只有0.48 All ACC
- 需要找出train_superclass.py和evaluate_superclass_model.py之间的差异

调试过程中排查的原因：

1. 【已排除】模型加载问题
   - 检查了state_dict加载是否正确
   - 确认模型结构一致性

2. 【已排除】K-means实现差异
   - 对比了K-means参数设置 (n_clusters, random_state, n_init)
   - 确认聚类算法实现一致

3. 【已排除】mask构建差异
   - 对比了已知类/未知类mask的构建逻辑
   - 确认 `x.item() in range(len(args.train_classes))` 逻辑一致

4. 【已排除】数据加载顺序问题
   - 确认了batch_size、数据集随机种子等参数一致
   - 排除了数据划分差异

5. 【关键发现】图像尺寸差异 ⭐⭐⭐
   - train_superclass.py: args.image_size = 224
   - evaluate_superclass_model.py: args.image_size = 32 (错误!)

   这个差异导致：
   - 完全不同的数据预处理流程
   - 224x224 vs 32x32 的输入图像尺寸
   - ViT模型接收到完全不同的输入
   - 特征提取结果完全不匹配

修复措施：
- 将evaluate_superclass_model.py的image_size改为224
- 确保与train_superclass.py的数据预处理完全一致

教训总结：
- 图像尺寸是深度学习模型中的关键超参数
- 即使模型权重正确，输入尺寸不匹配也会导致完全错误的结果
- 在复现实验时，必须确保所有预处理参数完全一致

功能：
1. 加载指定超类的完整训练集（包含有标签和无标签样本）和测试集
2. 加载预训练模型（仅base model，与K-means评估保持一致）
3. 进行四种K-means聚类评估：
   - 训练时无标签训练集（与train_superclass.py中的"Train ACC Unlabelled"完全一致）
   - 纯测试集（与train_superclass.py中的"Test ACC"完全一致）
   - 完整训练集（有标签+无标签）
   - 合并数据集（完整训练集+测试集）
4. 输出详细的聚类分布分析，包含匈牙利算法分配结果
5. 提供四种评估结果的对比分析，重点关注与训练时评估的一致性

使用方法：
python test/evaluate_superclass_model.py \
    --superclass_name trees \
    --model_path /path/to/model.pt
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tqdm import tqdm
import argparse
from copy import deepcopy

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.get_datasets import get_class_splits
from data.augmentations import get_transform
from data.cifar100_superclass import get_single_superclass_datasets, SUPERCLASS_NAMES
from models import vision_transformer as vits
from config import dino_pretrain_path
from project_utils.general_utils import str2bool
from project_utils.cluster_and_log_utils import split_cluster_acc_v1, split_cluster_acc_v2, log_accs_from_preds
from project_utils.cluster_utils import cluster_acc


def load_model(model_path, device, feat_dim=768):
    """
    加载基础模型
    """
    print(f"🔄 加载基础模型...")
    print(f"   模型文件: {model_path}")

    # 构建base model
    model = vits.__dict__['vit_base']()

    # 加载预训练权重
    if model_path and os.path.exists(model_path):
        print(f"   加载模型权重: {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print(f"   ⚠️ 模型文件不存在，使用DINO预训练权重")
        state_dict = torch.load(dino_pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 检查模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   模型参数统计: 总参数={total_params:,}, 可训练={trainable_params:,}")

    # 检查前几层权重信息（用于验证模型是否正确加载）
    first_layer_weight = None
    for name, param in model.named_parameters():
        if 'weight' in name:
            first_layer_weight = param.data.flatten()[:10]
            print(f"   第一层权重样本 ({name}): {first_layer_weight}")
            break

    print(f"   ✅ 基础模型加载完成")
    return model



def extract_features(model, data_loader, device):
    """
    提取特征（仅使用base model，与K-means评估保持一致）
    """
    print(f"🔍 提取特征（使用base model，无投影头）...")

    all_feats = []
    all_targets = []
    all_indices = []

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, targets_batch, indices) in enumerate(tqdm(data_loader, desc="提取特征")):
            images = images.to(device)

            # 仅使用基础模型特征提取（与K-means评估保持一致）
            feats = model(images)

            # L2归一化
            feats = torch.nn.functional.normalize(feats, dim=-1)

            all_feats.append(feats.cpu().numpy())
            all_targets.append(targets_batch.numpy())
            all_indices.append(indices.numpy())

    # 合并特征
    all_feats = np.concatenate(all_feats, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    print(f"   ✅ 特征提取完成: {all_feats.shape}")
    return all_feats, all_targets, all_indices


def evaluate_clustering_like_training(features, targets, train_classes, num_labeled_classes, num_unlabeled_classes,
                                     evaluation_name="", args=None):
    """
    完全按照train_superclass.py的test_kmeans_superclass实现K-means聚类评估
    """
    print(f"\n🔬 {evaluation_name} K-means聚类评估（完全复制train_superclass.py逻辑）...")

    # 完全按照train_superclass.py的方式构建mask
    mask = np.array([True if x.item() in range(len(train_classes)) else False for x in targets])

    print(f"   特征维度: {features.shape}")
    print(f"   样本数量: {len(features)}")
    print(f"   已知类样本数: {mask.sum()}")
    print(f"   未知类样本数: {(~mask).sum()}")

    # -----------------------
    # K-MEANS (完全按照train_superclass.py)
    # -----------------------
    print('Fitting K-Means...')
    kmeans = KMeans(n_clusters=num_labeled_classes + num_unlabeled_classes, random_state=0, n_init=10).fit(features)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE (完全按照train_superclass.py)
    # -----------------------
    # 检查是否有未知类
    mask = np.array(mask, dtype=bool)  # 确保mask是numpy数组
    has_unknown_classes = num_unlabeled_classes > 0 and (~mask).sum() > 0

    if has_unknown_classes:
        # 有未知类的情况：正常计算所有指标
        all_acc, old_acc, new_acc = log_accs_from_preds(
            y_true=targets, y_pred=preds, mask=mask,
            T=0, eval_funcs=['v2'], save_name=evaluation_name,
            writer=None, print_output=False
        )
    else:
        # 没有未知类的情况：只计算已知类准确率
        from project_utils.cluster_utils import cluster_acc
        old_acc = cluster_acc(targets, preds)
        all_acc = old_acc  # 当没有未知类时，All ACC = Old ACC
        new_acc = 0.0      # 没有未知类，New ACC为0

        print(f"⚠️  注意: 当前数据中没有未知类样本，仅计算Old ACC")

    # 计算其他指标
    nmi = normalized_mutual_info_score(targets, preds)
    ari = adjusted_rand_score(targets, preds)

    # 输出结果
    print(f"\n📊 {evaluation_name} 聚类结果:")
    print(f"   All ACC: {all_acc:.4f}")
    print(f"   Old ACC: {old_acc:.4f}")
    print(f"   New ACC: {new_acc:.4f}")
    print(f"   NMI: {nmi:.4f}")
    print(f"   ARI: {ari:.4f}")

    return {
        'all_acc': all_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'nmi': nmi,
        'ari': ari,
        'cluster_preds': preds
    }


def print_cluster_distribution_analysis(cluster_preds, targets, known_mask, superclass_name, dataset_name):
    """
    分析并打印聚类标签分布，包含匈牙利算法分配结果
    """
    print(f"\n🔍 {superclass_name} {dataset_name} 聚类标签分布分析:")
    print("=" * 80)

    # 计算匈牙利算法分配
    from scipy.optimize import linear_sum_assignment as linear_assignment

    # 构建混淆矩阵
    y_true = targets.astype(int)
    y_pred = cluster_preds.astype(int)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # 匈牙利算法找最优分配
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    # 创建聚类到真实标签的映射
    cluster_to_label_mapping = {cluster_id: true_label for cluster_id, true_label in ind}

    print(f"🎯 匈牙利算法最优分配:")
    print(f"   聚类ID -> 真实标签: {cluster_to_label_mapping}")
    print()

    unique_clusters = np.unique(cluster_preds)

    # 计算总体统计
    total_samples = len(targets)
    total_known = known_mask.sum()
    total_unknown = (~known_mask).sum()

    print(f"📊 数据集总体统计:")
    print(f"   总样本数: {total_samples}")
    print(f"   已知类样本: {total_known}")
    print(f"   未知类样本: {total_unknown}")
    print()

    for cluster_id in unique_clusters:
        # 找到属于当前聚类的样本
        cluster_mask = cluster_preds == cluster_id
        cluster_targets = targets[cluster_mask]
        cluster_known_mask = known_mask[cluster_mask]

        # 统计已知类和未知类样本数
        known_count = cluster_known_mask.sum()
        unknown_count = (~cluster_known_mask).sum()
        total_count = len(cluster_targets)

        # 获取匈牙利分配的标签
        assigned_label = cluster_to_label_mapping.get(cluster_id, "未分配")

        print(f"聚类 {cluster_id}: {total_count}个样本 (已知类: {known_count}, 未知类: {unknown_count})")
        print(f"   匈牙利分配 -> 标签 {assigned_label}")

        # 统计标签分布
        unique_labels, counts = np.unique(cluster_targets, return_counts=True)
        label_dist = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        print(f"   标签分布: {label_dist}")

        # 计算聚类纯度（最大类别占比）
        if total_count > 0:
            max_count = max(counts)
            purity = max_count / total_count
            dominant_label = unique_labels[np.argmax(counts)]
            print(f"   聚类纯度: {purity:.3f} (主导标签: {dominant_label})")

            # 分析匈牙利分配的正确性
            if assigned_label != "未分配":
                assigned_count = label_dist.get(int(assigned_label), 0)
                assign_accuracy = assigned_count / total_count
                print(f"   分配准确性: {assign_accuracy:.3f} (分配标签{assigned_label}的样本占比)")

        print()


def main():
    parser = argparse.ArgumentParser(description='超类模型评估脚本')

    # 模型和数据参数
    parser.add_argument('--superclass_name', type=str, required=True,
                        help='超类名称', choices=SUPERCLASS_NAMES)
    parser.add_argument('--model_path', type=str, required=True,
                        help='基础模型路径')

    # 数据集参数（与train_superclass.py保持一致）
    parser.add_argument('--prop_train_labels', type=float, default=0.8,
                        help='训练集标签比例（与train_superclass.py一致）')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小（与train_superclass.py一致）')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载器工作线程数')

    # 评估参数（与train_superclass.py保持一致）
    parser.add_argument('--random_state', type=int, default=0,
                        help='K-means随机种子（与train_superclass.py一致）')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID')

    # 模型参数
    parser.add_argument('--feat_dim', type=int, default=768,
                        help='特征维度')

    args = parser.parse_args()

    print("🚀 超类模型评估脚本")
    print("=" * 80)
    print(f"📂 超类名称: {args.superclass_name}")
    print(f"🤖 基础模型: {args.model_path}")
    print(f"🔧 特征提取: 仅使用base model（与K-means评估一致）")

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"💻 设备: {device}")

    # 设置超类参数
    args.dataset_name = 'cifar100_superclass'
    args.eval_funcs = ['v2']  # 添加评估函数参数
    args.writer = None        # 添加writer参数
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print(f"📊 类别信息:")
    print(f"   已知类数量: {args.num_labeled_classes}")
    print(f"   未知类数量: {args.num_unlabeled_classes}")
    print(f"   总类别数: {args.num_labeled_classes + args.num_unlabeled_classes}")

    # 加载数据集
    print(f"\n📦 加载超类数据集...")
    args.image_size = 224  # 与train_superclass.py保持一致
    args.interpolation = 3
    args.crop_pct = 0.875
    args.transform = 'imagenet'  # 与train_superclass.py保持一致
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)

    print(f"   使用变换: {args.transform}")
    print(f"   图像尺寸: {args.image_size}")
    print(f"   插值方式: {args.interpolation}")
    print(f"   裁剪比例: {args.crop_pct}")

    # 数据集参数（与train_superclass.py保持一致）
    args.seed = 1  # 与train_superclass.py默认值一致

    print(f"   数据集随机种子: {args.seed}")

    datasets = get_single_superclass_datasets(
        superclass_name=args.superclass_name,
        train_transform=test_transform,  # 评估时都使用test_transform
        test_transform=test_transform,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
        seed=args.seed  # 使用与训练时相同的seed
    )

    # 构建评估数据集
    test_dataset = datasets['test']
    train_labelled_dataset = datasets['train_labelled']
    train_unlabelled_dataset = datasets['train_unlabelled']

    print(f"   测试集样本数: {len(test_dataset)}")
    print(f"   有标签训练集样本数: {len(train_labelled_dataset)}")
    print(f"   无标签训练集样本数: {len(train_unlabelled_dataset)}")

    # 合并训练集
    from torch.utils.data import ConcatDataset
    train_combined_dataset = ConcatDataset([train_labelled_dataset, train_unlabelled_dataset])
    print(f"   合并后训练集样本数: {len(train_combined_dataset)}")

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    train_loader = DataLoader(train_combined_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # 加载模型（仅base model）
    model = load_model(args.model_path, device, args.feat_dim)

    # 提取训练集特征
    print(f"\n" + "=" * 80)
    print(f"🔍 提取训练集特征")
    print("=" * 80)

    train_features, train_targets, train_indices = extract_features(
        model, train_loader, device)

    # 提取测试集特征
    print(f"\n" + "=" * 80)
    print(f"🔍 提取测试集特征")
    print("=" * 80)

    test_features, test_targets, test_indices = extract_features(
        model, test_loader, device)

    # 1. 训练集K-means聚类评估（完全复制train_superclass.py逻辑）
    print(f"\n" + "=" * 80)
    print(f"🧪 1. 训练集K-means聚类评估")
    print("=" * 80)

    train_results = evaluate_clustering_like_training(
        train_features, train_targets, args.train_classes,
        args.num_labeled_classes, args.num_unlabeled_classes,
        "训练集", args
    )

    train_known_mask = np.array([True if x.item() in range(len(args.train_classes)) else False for x in train_targets])
    print_cluster_distribution_analysis(
        train_results['cluster_preds'], train_targets, train_known_mask,
        args.superclass_name, "训练集"
    )

    # 2. 测试集K-means聚类评估（完全复制train_superclass.py逻辑）
    print(f"\n" + "=" * 80)
    print(f"🧪 2. 测试集K-means聚类评估")
    print("=" * 80)

    test_results = evaluate_clustering_like_training(
        test_features, test_targets, args.train_classes,
        args.num_labeled_classes, args.num_unlabeled_classes,
        "测试集", args
    )

    test_known_mask = np.array([True if x.item() in range(len(args.train_classes)) else False for x in test_targets])
    print_cluster_distribution_analysis(
        test_results['cluster_preds'], test_targets, test_known_mask,
        args.superclass_name, "测试集"
    )

    # 3. 合并数据集K-means聚类评估（完全复制train_superclass.py逻辑）
    print(f"\n" + "=" * 80)
    print(f"🧪 3. 合并数据集K-means聚类评估")
    print("=" * 80)

    combined_features = np.concatenate([train_features, test_features], axis=0)
    combined_targets = np.concatenate([train_targets, test_targets], axis=0)

    combined_results = evaluate_clustering_like_training(
        combined_features, combined_targets, args.train_classes,
        args.num_labeled_classes, args.num_unlabeled_classes,
        "合并数据集", args
    )

    combined_known_mask = np.array([True if x.item() in range(len(args.train_classes)) else False for x in combined_targets])
    print_cluster_distribution_analysis(
        combined_results['cluster_preds'], combined_targets, combined_known_mask,
        args.superclass_name, "合并数据集"
    )

    # 评估结果总结对比
    print(f"\n" + "=" * 80)
    print(f"📊 三种K-means评估结果对比")
    print("=" * 80)
    print(f"超类: {args.superclass_name}")
    print(f"特征提取: 仅使用base model")
    print(f"模型路径: {args.model_path}")
    print()

    # 创建对比表格
    print(f"{'指标':<12} {'训练集':<12} {'测试集':<12} {'合并数据':<12}")
    print("-" * 50)
    print(f"{'All ACC':<12} {train_results['all_acc']:<12.4f} {test_results['all_acc']:<12.4f} {combined_results['all_acc']:<12.4f}")
    print(f"{'Old ACC':<12} {train_results['old_acc']:<12.4f} {test_results['old_acc']:<12.4f} {combined_results['old_acc']:<12.4f}")
    print(f"{'New ACC':<12} {train_results['new_acc']:<12.4f} {test_results['new_acc']:<12.4f} {combined_results['new_acc']:<12.4f}")
    print(f"{'NMI':<12} {train_results['nmi']:<12.4f} {test_results['nmi']:<12.4f} {combined_results['nmi']:<12.4f}")
    print(f"{'ARI':<12} {train_results['ari']:<12.4f} {test_results['ari']:<12.4f} {combined_results['ari']:<12.4f}")

    print()
    print(f"🔍 关键问题分析 - 测试集ACC下降原因:")
    print(f"   训练集样本数: {len(train_features)} | 测试集样本数: {len(test_features)}")

    # 重点分析测试集性能下降
    train_test_diff = test_results['all_acc'] - train_results['all_acc']
    print(f"   测试集 vs 训练集 All ACC差异: {train_test_diff:+.4f}")

    if test_results['all_acc'] < 0.6:
        print(f"   🚨 测试集ACC={test_results['all_acc']:.4f} 明显偏低！可能原因:")
        print(f"      1. 模型权重加载问题")
        print(f"      2. 特征提取方式差异")
        print(f"      3. 数据预处理差异")
        print(f"      4. 随机种子不一致")

    print(f"\n✅ 三种K-means评估完成！")


if __name__ == "__main__":
    main()