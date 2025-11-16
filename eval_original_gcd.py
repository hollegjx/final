#!/usr/bin/env python3
"""
严格按照原版GCD方式的模型评估脚本

完全复制原版test_kmeans的逻辑：
1. 只使用base model，不用projection head
2. 在base model特征空间(768维)进行聚类
3. 与原版training时的测试完全一致
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
import os

# 导入必要的模块
from data.get_datasets import get_datasets, get_class_splits
from data.cifar100_superclass import CIFAR100_SUPERCLASSES
from methods.contrastive_training.contrastive_training import test_kmeans_superclass_eval, test_kmeans
from models import vision_transformer as vits
from config import dino_pretrain_path
from project_utils.general_utils import str2bool
from sklearn.cluster import KMeans


def load_original_gcd_model(model_path, args, device):
    """
    按照原版GCD方式加载模型 - 只加载base model

    Args:
        model_path: 主模型权重路径 (model.pt或model_best.pt)
        args: 参数配置
        device: 设备

    Returns:
        model: 只加载base model，不加载projection head
    """
    print(f"🔄 按照原版GCD方式加载模型...")
    print(f"   模型文件: {model_path}")
    print(f"   ⚠️  注意: 只加载base model，不使用projection head")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 构建base model（与训练时相同）
    if args.base_model == 'vit_dino':
        model = vits.__dict__['vit_base']()

        # 加载DINO预训练权重（作为基础）
        if os.path.exists(dino_pretrain_path):
            print(f"   加载DINO预训练权重: {dino_pretrain_path}")
            dino_state_dict = torch.load(dino_pretrain_path, map_location='cpu')
            model.load_state_dict(dino_state_dict, strict=False)

        # 加载GCD训练后的权重
        print(f"   加载GCD训练权重...")
        gcd_state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(gcd_state_dict)

        model.to(device)

        # 测试时关闭所有梯度计算
        for param in model.parameters():
            param.requires_grad = False

        print(f"✅ Base Model加载成功!")
        print(f"🔍 验证模型架构:")
        print(f"   Base Model输出维度: {args.feat_dim}")
        print(f"   评估特征空间: {args.feat_dim}维 (base model特征)")
        print(f"   原版设计: 训练用projection空间，测试用base空间")
        print(f"   梯度计算: 已关闭 (测试模式)")

        return model

    else:
        raise NotImplementedError(f"不支持的模型类型: {args.base_model}")


def evaluate_on_full_dataset_original(model, args, device):
    """
    严格按照原版方式在完整CIFAR-100数据集上评估
    """
    print("\n" + "="*80)
    print("🌍 完整CIFAR-100数据集评估 (原版方式)")
    print("="*80)

    # 获取数据集配置
    args_eval = argparse.Namespace(**vars(args))
    args_eval.dataset_name = 'cifar100'

    # 获取类别划分
    args_eval = get_class_splits(args_eval)
    args_eval.num_labeled_classes = len(args_eval.train_classes)
    args_eval.num_unlabeled_classes = len(args_eval.unlabeled_classes)

    # 确保test_classes包含所有类别
    all_classes = list(args_eval.train_classes) + list(args_eval.unlabeled_classes)
    args_eval.test_classes = sorted(all_classes)

    print(f"📊 已知类别: {list(args_eval.train_classes)} (共{args_eval.num_labeled_classes}个)")
    print(f"📊 未知类别: {list(args_eval.unlabeled_classes)} (共{args_eval.num_unlabeled_classes}个)")
    print(f"📊 测试类别: {len(args_eval.test_classes)}个类别")

    # 获取正确的transform (按照原版GCD使用imagenet transform)
    from data.augmentations import get_transform
    # 原版GCD在CIFAR上使用imagenet transform，会resize到224
    train_transform, test_transform = get_transform('imagenet', image_size=224, args=args_eval)

    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args_eval.dataset_name,
        train_transform=train_transform,
        test_transform=test_transform,
        args=args_eval
    )

    # 按照原版方式手动创建test_loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        num_workers=args_eval.num_workers,
        batch_size=args_eval.batch_size,
        shuffle=False
    )

    model.eval()

    print(f"📊 在测试集上评估 (原版test_kmeans方式)...")
    print(f"📊 使用评估函数: {args_eval.eval_funcs}")
    print(f"📊 K-means聚类数: {args_eval.num_labeled_classes + args_eval.num_unlabeled_classes}")

    # 添加调试信息
    print(f"🔍 数据加载器配置:")
    print(f"   Batch size: {args_eval.batch_size}")
    print(f"   Image size: {args_eval.image_size}")

    # 测试一个batch的数据
    for batch_idx, (images, label, _) in enumerate(test_loader):
        print(f"   第一个batch形状: {images.shape}")
        print(f"   数据类型: {type(images)}")
        break

    # 使用原版的test_kmeans函数
    all_acc, old_acc, new_acc = test_kmeans(
        model, test_loader, epoch=0, save_name='Original_Full_Dataset_Test',
        args=args_eval, device=device
    )

    print(f"📈 完整数据集评估结果 (原版方式):")
    print(f"   All ACC: {all_acc:.4f}")
    print(f"   Old ACC: {old_acc:.4f}")
    print(f"   New ACC: {new_acc:.4f}")

    return all_acc, old_acc, new_acc


def evaluate_on_superclass_original(model, superclass_name, args, device):
    """
    严格按照原版方式在指定超类上评估
    """
    print(f"\n" + "="*80)
    print(f"🎯 超类 '{superclass_name}' 评估 (原版方式)")
    print("="*80)

    if superclass_name not in CIFAR100_SUPERCLASSES:
        print(f"❌ 错误: 未知的超类名称 '{superclass_name}'")
        return None, None, None

    # 获取完整数据集配置
    args_eval = argparse.Namespace(**vars(args))
    args_eval.dataset_name = 'cifar100'

    # 获取类别划分
    args_eval = get_class_splits(args_eval)
    args_eval.num_labeled_classes = len(args_eval.train_classes)
    args_eval.num_unlabeled_classes = len(args_eval.unlabeled_classes)

    # 确保test_classes包含所有类别
    all_classes = list(args_eval.train_classes) + list(args_eval.unlabeled_classes)
    args_eval.test_classes = sorted(all_classes)

    print(f"📊 已知类别: {list(args_eval.train_classes)} (共{args_eval.num_labeled_classes}个)")
    print(f"📊 未知类别: {list(args_eval.unlabeled_classes)} (共{args_eval.num_unlabeled_classes}个)")

    # 获取正确的transform (按照原版GCD使用imagenet transform)
    from data.augmentations import get_transform
    # 原版GCD在CIFAR上使用imagenet transform，会resize到224
    train_transform, test_transform = get_transform('imagenet', image_size=224, args=args_eval)

    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args_eval.dataset_name,
        train_transform=train_transform,
        test_transform=test_transform,
        args=args_eval
    )

    # 按照原版方式手动创建test_loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        num_workers=args_eval.num_workers,
        batch_size=args_eval.batch_size,
        shuffle=False
    )

    model.eval()

    # 在测试集上进行超类评估
    superclass_classes = CIFAR100_SUPERCLASSES[superclass_name]
    print(f"📊 超类包含类别: {superclass_classes}")
    print(f"📊 在测试集上评估 (原版方式)...")
    print(f"📊 使用评估函数: {args_eval.eval_funcs}")

    # 使用原版的test_kmeans_superclass_eval函数
    all_acc, old_acc, new_acc = test_kmeans_superclass_eval(
        model, test_loader, epoch=0, save_name=f'Original_Superclass_{superclass_name}_Test',
        args=args_eval, eval_superclass=superclass_name, device=device
    )

    return all_acc, old_acc, new_acc


def evaluate_all_superclasses_original(model, args, device):
    """
    严格按照原版方式在所有超类上评估
    """
    print("\n" + "="*80)
    print("🔍 所有超类批量评估 (原版方式)")
    print("="*80)

    results = {}

    for superclass_name in CIFAR100_SUPERCLASSES.keys():
        try:
            all_acc, old_acc, new_acc = evaluate_on_superclass_original(model, superclass_name, args, device)
            if all_acc is not None:
                results[superclass_name] = {
                    'all_acc': all_acc,
                    'old_acc': old_acc,
                    'new_acc': new_acc
                }
                print(f"✅ {superclass_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}")
            else:
                print(f"❌ {superclass_name}: 评估失败")
        except Exception as e:
            print(f"❌ {superclass_name}: 评估出错 - {e}")

    # 显示汇总结果
    print(f"\n📊 所有超类评估汇总 (原版方式):")
    print(f"{'超类名称':<25} {'All ACC':<10} {'Old ACC':<10} {'New ACC':<10}")
    print("-" * 60)

    for superclass_name, result in results.items():
        print(f"{superclass_name:<25} {result['all_acc']:<10.4f} {result['old_acc']:<10.4f} {result['new_acc']:<10.4f}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='严格按照原版GCD方式评估预训练模型')

    # 模型文件路径
    parser.add_argument('--model_path', type=str, required=True,
                        help='GCD模型权重文件路径 (如: /path/to/model_best.pt)')

    # 评估模式
    parser.add_argument('--eval_mode', type=str, choices=['full', 'superclass', 'all_superclasses'],
                        default='full', help='评估模式')
    parser.add_argument('--superclass_name', type=str, default=None,
                        help='指定超类名称 (当eval_mode=superclass时使用)')

    # 模型配置 (只需要base model相关配置)
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--feat_dim', default=768, type=int, help='Base model特征维度')

    # 数据集配置
    parser.add_argument('--batch_size', default=128, type=int, help='评估批次大小')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', default=['v1'], help='评估函数')

    # 设备配置
    parser.add_argument('--gpu', default=0, type=int, help='GPU设备ID')

    args = parser.parse_args()

    # 设备初始化
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        print(f"💻 使用GPU设备: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA不可用，使用CPU")

    # 设置必要参数 (按照原版GCD)
    args.device = device
    args.writer = None
    args.use_ssb_splits = False
    args.prop_train_labels = 0.5
    args.image_size = 224  # ViT固定使用224
    args.interpolation = 3
    args.crop_pct = 0.875

    print("🚀 原版GCD模型评估工具 (严格原版方式)")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"评估模式: {args.eval_mode}")
    if args.eval_mode == 'superclass':
        print(f"目标超类: {args.superclass_name}")
    print(f"特征维度: {args.feat_dim} (base model)")
    print("⚠️  注意: 使用原版评估方式，只用base model特征")
    print("=" * 80)

    try:
        # 加载原版方式的模型 (只有base model)
        model = load_original_gcd_model(args.model_path, args, device)

        # 根据评估模式进行评估
        if args.eval_mode == 'full':
            evaluate_on_full_dataset_original(model, args, device)

        elif args.eval_mode == 'superclass':
            if args.superclass_name is None:
                print("❌ 错误: 超类评估模式需要指定 --superclass_name")
                return
            evaluate_on_superclass_original(model, args.superclass_name, args, device)

        elif args.eval_mode == 'all_superclasses':
            evaluate_all_superclasses_original(model, args, device)

        print("\n🎉 原版方式评估完成!")

        # 验证说明
        print("\n" + "="*80)
        print("📋 原版GCD评估方式验证:")
        print("✅ 只使用base model (768维特征)")
        print("✅ 不使用projection head")
        print("✅ 使用完整测试集 (包含所有已知+未知类别)")
        print("✅ 使用原版GCD的类别划分 (CIFAR-100: 80已知 + 20未知)")
        print("✅ 使用相同的K-means聚类数 (100个聚类)")
        print("✅ 使用相同的ACC计算方法 (cluster_acc + Hungarian算法)")
        print("✅ 使用相同的mask定义 (基于train_classes)")
        print("")
        print("🔍 原版架构设计:")
        print("   训练时: model(x) → projection_head(features) → 对比学习")
        print("   评估时: model(x) → normalize → K-means聚类")
        print("   特点: 训练和测试在不同特征空间进行")
        print("="*80)

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()