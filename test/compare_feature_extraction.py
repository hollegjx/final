#!/usr/bin/env python3
"""
特征提取对比脚本
对比train_superclass.py和evaluate_superclass_model.py两套代码的测试集特征提取结果

使用方法：
python test/compare_feature_extraction.py \
    --superclass_name trees \
    --model_path /path/to/model.pt
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from copy import deepcopy

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.get_datasets import get_class_splits
from data.augmentations import get_transform
from data.cifar100_superclass import get_single_superclass_datasets, SUPERCLASS_NAMES
from methods.contrastive_training.contrastive_training import SupConLoss, ContrastiveLearningViewGenerator
from models import vision_transformer as vits
from config import dino_pretrain_path
from project_utils.general_utils import str2bool

def load_model(model_path, device, feat_dim=768):
    """加载模型（两套代码共用）"""
    print(f"🔄 加载模型: {model_path}")
    model = vits.__dict__['vit_base']()

    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️ 模型文件不存在，使用DINO预训练权重")
        state_dict = torch.load(dino_pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

def extract_features_method1_train_superclass(model, superclass_name, device):
    """
    方法1：完全复制train_superclass.py中的数据读取和特征提取方式
    """
    print(f"\n" + "="*80)
    print(f"🧪 方法1：train_superclass.py的特征提取方式")
    print("="*80)

    # 模拟train_superclass.py中的参数设置
    class Args:
        def __init__(self):
            self.superclass_name = superclass_name
            self.dataset_name = 'cifar100_superclass'
            self.transform = 'imagenet'
            self.image_size = 32
            self.interpolation = 3
            self.crop_pct = 0.875
            self.prop_train_labels = 0.8
            self.seed = 1
            self.batch_size = 128
            self.num_workers = 8
            self.device = device
            self.n_views = 2

    args = Args()

    # 获取类别划分
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print(f"   train_classes: {args.train_classes}")
    print(f"   unlabeled_classes: {args.unlabeled_classes}")

    # 数据变换（完全按照train_superclass.py）
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # 数据集（完全按照train_superclass.py）
    datasets = get_single_superclass_datasets(
        superclass_name=args.superclass_name,
        train_transform=train_transform,
        test_transform=test_transform,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
        seed=args.seed
    )

    # 测试集（完全按照train_superclass.py）
    test_dataset = datasets['test']
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                     batch_size=args.batch_size, shuffle=False)

    print(f"   测试集样本数: {len(test_dataset)}")
    print(f"   batch_size: {args.batch_size}")

    # 特征提取（完全按照train_superclass.py中test_kmeans_superclass的方式）
    model.eval()
    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    with torch.no_grad():
        for batch_idx, (images, label, _) in enumerate(tqdm(test_loader_labelled)):
            images = images.to(args.device)

            # Pass features through base model only (no projection head for evaluation)
            feats = model(images)
            feats = torch.nn.functional.normalize(feats, dim=-1)

            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                             else False for x in label]))

    # 合并特征
    all_feats = np.concatenate(all_feats)
    mask = np.array(mask, dtype=bool)

    print(f"   提取的特征形状: {all_feats.shape}")
    print(f"   标签数量: {len(targets)}")
    print(f"   已知类样本数: {mask.sum()}")
    print(f"   未知类样本数: {(~mask).sum()}")

    return all_feats, targets, mask


def extract_features_method2_evaluate_model(model, superclass_name, device):
    """
    方法2：完全复制evaluate_superclass_model.py中的数据读取和特征提取方式
    """
    print(f"\n" + "="*80)
    print(f"🧪 方法2：evaluate_superclass_model.py的特征提取方式")
    print("="*80)

    # 模拟evaluate_superclass_model.py中的参数设置
    class Args:
        def __init__(self):
            self.superclass_name = superclass_name
            self.dataset_name = 'cifar100_superclass'
            self.transform = 'imagenet'
            self.image_size = 32
            self.interpolation = 3
            self.crop_pct = 0.875
            self.prop_train_labels = 0.8
            self.seed = 1
            self.batch_size = 128  # 与方法1保持一致，排除batch_size影响
            self.num_workers = 8
            self.device = device

    args = Args()

    # 获取类别划分
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print(f"   train_classes: {args.train_classes}")
    print(f"   unlabeled_classes: {args.unlabeled_classes}")

    # 数据变换（按照evaluate_superclass_model.py）
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)

    # 数据集（按照evaluate_superclass_model.py）
    datasets = get_single_superclass_datasets(
        superclass_name=args.superclass_name,
        train_transform=test_transform,  # 评估时都使用test_transform
        test_transform=test_transform,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
        seed=args.seed
    )

    # 测试集（按照evaluate_superclass_model.py）
    test_dataset = datasets['test']
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    print(f"   测试集样本数: {len(test_dataset)}")
    print(f"   batch_size: {args.batch_size}")

    # 特征提取（按照evaluate_superclass_model.py的extract_features方式）
    model.eval()
    all_feats = []
    all_targets = []
    all_indices = []

    print('提取特征（使用base model，无投影头）...')
    with torch.no_grad():
        for batch_idx, (images, targets_batch, indices) in enumerate(tqdm(test_loader, desc="提取特征")):
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

    # 构建mask（按照evaluate_superclass_model.py的方式）
    mask = np.array([True if x.item() in range(len(args.train_classes)) else False for x in all_targets])

    print(f"   提取的特征形状: {all_feats.shape}")
    print(f"   标签数量: {len(all_targets)}")
    print(f"   已知类样本数: {mask.sum()}")
    print(f"   未知类样本数: {(~mask).sum()}")

    return all_feats, all_targets, mask


def compare_features(features1, targets1, mask1, features2, targets2, mask2):
    """比较两套特征提取的结果"""
    print(f"\n" + "="*80)
    print(f"🔍 特征对比分析")
    print("="*80)

    # 基本统计对比
    print(f"方法1 - 特征形状: {features1.shape}, 标签数量: {len(targets1)}, 已知类: {mask1.sum()}")
    print(f"方法2 - 特征形状: {features2.shape}, 标签数量: {len(targets2)}, 已知类: {mask2.sum()}")

    # 检查形状是否一致
    if features1.shape != features2.shape:
        print(f"❌ 特征形状不一致！")
        return False

    if len(targets1) != len(targets2):
        print(f"❌ 标签数量不一致！")
        return False

    # 检查标签是否一致
    labels_match = np.array_equal(targets1, targets2)
    print(f"标签是否一致: {labels_match}")
    if not labels_match:
        print(f"❌ 标签不一致！")
        print(f"方法1前10个标签: {targets1[:10]}")
        print(f"方法2前10个标签: {targets2[:10]}")
        return False

    # 检查mask是否一致
    mask_match = np.array_equal(mask1, mask2)
    print(f"mask是否一致: {mask_match}")
    if not mask_match:
        print(f"❌ mask不一致！")
        return False

    # 检查特征是否完全一致
    features_match = np.allclose(features1, features2, rtol=1e-6, atol=1e-8)
    print(f"特征是否一致（容差1e-6）: {features_match}")

    if features_match:
        print(f"✅ 两套方法的特征提取结果完全一致！")
    else:
        print(f"❌ 特征不一致！")

        # 详细分析差异
        diff = np.abs(features1 - features2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"最大差异: {max_diff}")
        print(f"平均差异: {mean_diff}")
        print(f"差异大于1e-6的元素数量: {np.sum(diff > 1e-6)}")
        print(f"差异大于1e-4的元素数量: {np.sum(diff > 1e-4)}")

        # 显示前几个样本的差异
        print(f"\n前5个样本的特征差异:")
        for i in range(min(5, len(features1))):
            sample_diff = np.abs(features1[i] - features2[i])
            print(f"样本{i}: 最大差异={np.max(sample_diff):.8f}, 平均差异={np.mean(sample_diff):.8f}")

    return features_match


def main():
    parser = argparse.ArgumentParser(description='特征提取对比脚本')

    parser.add_argument('--superclass_name', type=str, required=True,
                        help='超类名称', choices=SUPERCLASS_NAMES)
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID')

    args = parser.parse_args()

    print("🔍 特征提取对比脚本")
    print("="*80)
    print(f"超类: {args.superclass_name}")
    print(f"模型: {args.model_path}")

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    model = load_model(args.model_path, device)

    # 方法1：train_superclass.py的方式
    features1, targets1, mask1 = extract_features_method1_train_superclass(
        model, args.superclass_name, device)

    # 方法2：evaluate_superclass_model.py的方式
    features2, targets2, mask2 = extract_features_method2_evaluate_model(
        model, args.superclass_name, device)

    # 对比结果
    results_match = compare_features(features1, targets1, mask1,
                                   features2, targets2, mask2)

    if results_match:
        print(f"\n🎉 结论：两套代码的特征提取结果完全一致！")
        print(f"   问题可能出现在K-means聚类或ACC计算部分。")
    else:
        print(f"\n⚠️ 结论：两套代码的特征提取结果不一致！")
        print(f"   这就是导致ACC差异的根本原因。")


if __name__ == "__main__":
    main()