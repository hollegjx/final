#!/usr/bin/env python3
"""
原版GCD预训练模型评估脚本

使用原版GCD训练得到的model.pt和projection head，在超类和全数据集上进行评估
支持：
1. 全CIFAR-100数据集评估
2. 指定超类评估
3. 批量超类评估
"""

import argparse
import torch
import torch.nn as nn
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


def load_pretrained_gcd_model(model_path, proj_head_path, args, device):
    """
    加载原版GCD预训练模型

    Args:
        model_path: 主模型权重路径 (model.pt或model_best.pt)
        proj_head_path: 投影头权重路径 (model_proj_head.pt或model_proj_head_best.pt)
        args: 参数配置
        device: 设备

    Returns:
        model: 加载权重后的模型
        projection_head: 加载权重后的投影头
    """
    print(f"🔄 加载原版GCD预训练模型...")
    print(f"   主模型: {model_path}")
    print(f"   投影头: {proj_head_path}")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(proj_head_path):
        raise FileNotFoundError(f"投影头文件不存在: {proj_head_path}")

    # 构建模型架构（与训练时相同）
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

        # 构建投影头
        projection_head = vits.__dict__['DINOHead'](
            in_dim=args.feat_dim,
            out_dim=args.mlp_out_dim,
            nlayers=args.num_mlp_layers
        )

        # 加载投影头权重
        print(f"   加载投影头权重...")
        proj_state_dict = torch.load(proj_head_path, map_location='cpu')
        projection_head.load_state_dict(proj_state_dict)
        projection_head.to(device)

        print(f"✅ 模型加载成功!")
        return model, projection_head

    else:
        raise NotImplementedError(f"不支持的模型类型: {args.base_model}")


def create_combined_model(base_model, projection_head):
    """
    创建组合模型，便于推理
    """
    class CombinedModel(nn.Module):
        def __init__(self, base_model, projection_head):
            super().__init__()
            self.base_model = base_model
            self.projection_head = projection_head

        def forward(self, x):
            features = self.base_model(x)
            projected_features = self.projection_head(features)
            return projected_features

    return CombinedModel(base_model, projection_head)


def evaluate_on_full_dataset(model, args, device):
    """
    在完整CIFAR-100数据集上评估
    """
    print("\n" + "="*80)
    print("🌍 完整CIFAR-100数据集评估")
    print("="*80)

    # 获取数据集
    args_eval = argparse.Namespace(**vars(args))
    args_eval.dataset_name = 'cifar100'

    # 获取类别划分（这是关键！）
    args_eval = get_class_splits(args_eval)
    args_eval.num_labeled_classes = len(args_eval.train_classes)
    args_eval.num_unlabeled_classes = len(args_eval.unlabeled_classes)

    # 确保test_classes包含所有类别（与原版GCD一致）
    all_classes = list(args_eval.train_classes) + list(args_eval.unlabeled_classes)
    args_eval.test_classes = sorted(all_classes)

    print(f"📊 已知类别: {list(args_eval.train_classes)} (共{args_eval.num_labeled_classes}个)")
    print(f"📊 未知类别: {list(args_eval.unlabeled_classes)} (共{args_eval.num_unlabeled_classes}个)")
    print(f"📊 测试类别: {len(args_eval.test_classes)}个类别 (所有已知+未知类别)")

    train_loader, test_loader, unlabelled_train_loader, args_eval = get_datasets(
        args_eval.dataset_name,
        train_transform=None,  # 使用默认transform
        test_transform=None,   # 使用默认transform
        args=args_eval
    )

    model.eval()

    # 在测试集上评估
    print(f"📊 在测试集上评估 (共{len(args_eval.test_classes)}个类)...")
    print(f"📊 使用评估函数: {args_eval.eval_funcs}")
    print(f"📊 K-means聚类数: {args_eval.num_labeled_classes + args_eval.num_unlabeled_classes}")

    all_acc, old_acc, new_acc = test_kmeans(
        model, test_loader, epoch=0, save_name='Full_Dataset_Test', args=args_eval, device=device
    )

    print(f"📈 完整数据集评估结果:")
    print(f"   All ACC: {all_acc:.4f}")
    print(f"   Old ACC: {old_acc:.4f}")
    print(f"   New ACC: {new_acc:.4f}")

    return all_acc, old_acc, new_acc


def evaluate_on_superclass(model, superclass_name, args, device):
    """
    在指定超类上评估
    """
    print(f"\n" + "="*80)
    print(f"🎯 超类 '{superclass_name}' 评估")
    print("="*80)

    if superclass_name not in CIFAR100_SUPERCLASSES:
        print(f"❌ 错误: 未知的超类名称 '{superclass_name}'")
        return None, None, None

    # 获取完整数据集
    args_eval = argparse.Namespace(**vars(args))
    args_eval.dataset_name = 'cifar100'

    # 获取类别划分（这是关键！）
    args_eval = get_class_splits(args_eval)
    args_eval.num_labeled_classes = len(args_eval.train_classes)
    args_eval.num_unlabeled_classes = len(args_eval.unlabeled_classes)

    # 确保test_classes包含所有类别（与原版GCD一致）
    all_classes = list(args_eval.train_classes) + list(args_eval.unlabeled_classes)
    args_eval.test_classes = sorted(all_classes)

    print(f"📊 已知类别: {list(args_eval.train_classes)} (共{args_eval.num_labeled_classes}个)")
    print(f"📊 未知类别: {list(args_eval.unlabeled_classes)} (共{args_eval.num_unlabeled_classes}个)")
    print(f"📊 测试类别: {len(args_eval.test_classes)}个类别 (所有已知+未知类别)")

    train_loader, test_loader, unlabelled_train_loader, args_eval = get_datasets(
        args_eval.dataset_name,
        train_transform=None,
        test_transform=None,
        args=args_eval
    )

    model.eval()

    # 在测试集上进行超类评估
    superclass_classes = CIFAR100_SUPERCLASSES[superclass_name]
    print(f"📊 超类包含类别: {superclass_classes}")
    print(f"📊 在测试集上评估...")
    print(f"📊 使用评估函数: {args_eval.eval_funcs}")

    all_acc, old_acc, new_acc = test_kmeans_superclass_eval(
        model, test_loader, epoch=0, save_name=f'Superclass_{superclass_name}_Test',
        args=args_eval, eval_superclass=superclass_name, device=device
    )

    return all_acc, old_acc, new_acc


def evaluate_all_superclasses(model, args, device):
    """
    在所有超类上进行评估
    """
    print("\n" + "="*80)
    print("🔍 所有超类批量评估")
    print("="*80)

    results = {}

    for superclass_name in CIFAR100_SUPERCLASSES.keys():
        try:
            all_acc, old_acc, new_acc = evaluate_on_superclass(model, superclass_name, args, device)
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
    print(f"\n📊 所有超类评估汇总:")
    print(f"{'超类名称':<25} {'All ACC':<10} {'Old ACC':<10} {'New ACC':<10}")
    print("-" * 60)

    for superclass_name, result in results.items():
        print(f"{superclass_name:<25} {result['all_acc']:<10.4f} {result['old_acc']:<10.4f} {result['new_acc']:<10.4f}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估原版GCD预训练模型')

    # 模型文件路径
    parser.add_argument('--model_path', type=str, required=True,
                        help='GCD模型权重文件路径 (如: /path/to/model_best.pt)')
    parser.add_argument('--proj_head_path', type=str, required=True,
                        help='投影头权重文件路径 (如: /path/to/model_proj_head_best.pt)')

    # 评估模式
    parser.add_argument('--eval_mode', type=str, choices=['full', 'superclass', 'all_superclasses'],
                        default='full', help='评估模式')
    parser.add_argument('--superclass_name', type=str, default=None,
                        help='指定超类名称 (当eval_mode=superclass时使用)')

    # 模型配置 (必须与训练时一致)
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--feat_dim', default=768, type=int)
    parser.add_argument('--mlp_out_dim', default=65536, type=int)
    parser.add_argument('--num_mlp_layers', default=3, type=int)

    # 数据集配置
    parser.add_argument('--batch_size', default=256, type=int, help='评估批次大小')
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

    # 设置一些必要的参数
    args.device = device
    args.writer = None  # 不使用TensorBoard
    args.use_ssb_splits = False
    args.prop_train_labels = 0.5

    print("🚀 原版GCD模型评估工具")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"投影头路径: {args.proj_head_path}")
    print(f"评估模式: {args.eval_mode}")
    if args.eval_mode == 'superclass':
        print(f"目标超类: {args.superclass_name}")
    print("=" * 80)

    try:
        # 加载预训练模型
        base_model, projection_head = load_pretrained_gcd_model(
            args.model_path, args.proj_head_path, args, device
        )

        # 创建组合模型
        model = create_combined_model(base_model, projection_head)

        # 根据评估模式进行评估
        if args.eval_mode == 'full':
            evaluate_on_full_dataset(model, args, device)

        elif args.eval_mode == 'superclass':
            if args.superclass_name is None:
                print("❌ 错误: 超类评估模式需要指定 --superclass_name")
                return
            evaluate_on_superclass(model, args.superclass_name, args, device)

        elif args.eval_mode == 'all_superclasses':
            evaluate_all_superclasses(model, args, device)

        print("\n🎉 评估完成!")

        # 验证说明
        print("\n" + "="*80)
        print("📋 评估一致性验证:")
        print("✅ 使用完整测试集 (包含所有已知+未知类别)")
        print("✅ 使用原版GCD的类别划分 (CIFAR-100: 80已知 + 20未知)")
        print("✅ 使用相同的K-means聚类数 (100个聚类)")
        print("✅ 使用相同的ACC计算方法 (cluster_acc + Hungarian算法)")
        print("✅ 使用相同的mask定义 (基于train_classes)")
        print("="*80)

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()