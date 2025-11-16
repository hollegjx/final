#!/usr/bin/env python3
"""
严格按照原版GCD的test_kmeans函数创建的评估脚本
完全复制原版逻辑，不做任何修改
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
import os

# 导入原版GCD的模块（完全一致）
from data.get_datasets import get_datasets, get_class_splits
from data.cifar100_superclass import CIFAR100_SUPERCLASSES
from methods.contrastive_training.contrastive_training import test_kmeans_superclass_eval
from models import vision_transformer as vits
from config import dino_pretrain_path, exp_root
from project_utils.general_utils import str2bool
from project_utils.cluster_utils import log_accs_from_preds
from sklearn.cluster import KMeans
from data.augmentations import get_transform


def test_kmeans_original(model, test_loader, epoch, save_name, args, device=None):
    """
    完全复制原版GCD的test_kmeans函数，一个字都不改
    """
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        if device is None:
            device = torch.device('cuda:0')  # 默认设备
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, n_init=10).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc


def load_model_original_way(model_path, args, device):
    """
    按照原版GCD的方式加载模型
    """
    print(f"🔄 按照原版GCD方式加载模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 按照原版方式构建模型
    if args.base_model == 'vit_dino':
        model = vits.__dict__['vit_base']()

        # 先加载DINO预训练权重
        if os.path.exists(dino_pretrain_path):
            print(f"   加载DINO预训练权重: {dino_pretrain_path}")
            state_dict = torch.load(dino_pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)

        # 再加载训练后的权重
        print(f"   加载训练权重...")
        trained_state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(trained_state_dict)

        model.to(device)

        print(f"✅ 模型加载成功! 使用768维base model特征进行测试")
        return model
    else:
        raise NotImplementedError(f"不支持的模型类型: {args.base_model}")


def evaluate_full_dataset(model, args, device):
    """
    在完整CIFAR-100数据集上评估
    """
    print("\n" + "="*80)
    print("🌍 完整CIFAR-100数据集评估 (原版GCD方式)")
    print("="*80)

    # 按照原版设置参数
    args_eval = argparse.Namespace(**vars(args))
    args_eval.dataset_name = 'cifar100'

    # 获取类别划分
    args_eval = get_class_splits(args_eval)
    args_eval.num_labeled_classes = len(args_eval.train_classes)
    args_eval.num_unlabeled_classes = len(args_eval.unlabeled_classes)

    print(f"📊 已知类别: {len(args_eval.train_classes)}个")
    print(f"📊 未知类别: {len(args_eval.unlabeled_classes)}个")
    print(f"📊 总聚类数: {args_eval.num_labeled_classes + args_eval.num_unlabeled_classes}")

    # 获取数据集和transform (按照原版contrastive_training.py方式)
    train_transform, test_transform = get_transform(args_eval.transform, image_size=args_eval.image_size, args=args_eval)

    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args_eval.dataset_name, train_transform, test_transform, args_eval
    )

    print(f"🔍 数据配置检查:")
    print(f"   Transform类型: {args_eval.transform}")
    print(f"   图像大小: {args_eval.image_size}")
    print(f"   数据集名称: {args_eval.dataset_name}")

    # 检查实际的图像大小
    sample_image, _, _ = test_dataset[0]
    print(f"   实际图像shape: {sample_image.shape}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=args_eval.num_workers,
        batch_size=args_eval.batch_size,
        shuffle=False
    )

    # 使用原版test_kmeans函数
    all_acc, old_acc, new_acc = test_kmeans_original(
        model, test_loader, epoch=0, save_name='Original_GCD_Full_Test',
        args=args_eval, device=device
    )

    print(f"📈 完整数据集评估结果:")
    print(f"   All ACC: {all_acc:.4f}")
    print(f"   Old ACC: {old_acc:.4f}")
    print(f"   New ACC: {new_acc:.4f}")

    return all_acc, old_acc, new_acc


def evaluate_superclass(model, superclass_name, args, device):
    """
    在指定超类上评估
    """
    print(f"\n" + "="*80)
    print(f"🎯 超类 '{superclass_name}' 评估 (原版GCD方式)")
    print("="*80)

    if superclass_name not in CIFAR100_SUPERCLASSES:
        print(f"❌ 错误: 未知的超类名称 '{superclass_name}'")
        return None, None, None

    # 按照原版设置参数
    args_eval = argparse.Namespace(**vars(args))
    args_eval.dataset_name = 'cifar100'

    # 获取类别划分
    args_eval = get_class_splits(args_eval)
    args_eval.num_labeled_classes = len(args_eval.train_classes)
    args_eval.num_unlabeled_classes = len(args_eval.unlabeled_classes)

    # 获取数据集和transform
    train_transform, test_transform = get_transform(args_eval.transform, image_size=args_eval.image_size, args=args_eval)

    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args_eval.dataset_name, train_transform, test_transform, args_eval
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=args_eval.num_workers,
        batch_size=args_eval.batch_size,
        shuffle=False
    )

    # 使用原版的超类评估函数
    all_acc, old_acc, new_acc = test_kmeans_superclass_eval(
        model, test_loader, epoch=0, save_name=f'Original_GCD_Superclass_{superclass_name}_Test',
        args=args_eval, eval_superclass=superclass_name, device=device
    )

    return all_acc, old_acc, new_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='按照原版GCD方式评估模型')

    # 模型文件路径
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型权重文件路径')

    # 评估模式
    parser.add_argument('--eval_mode', type=str, choices=['full', 'superclass'],
                        default='full', help='评估模式')
    parser.add_argument('--superclass_name', type=str, default=None,
                        help='指定超类名称 (当eval_mode=superclass时使用)')

    # 模型配置 (与原版GCD一致)
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--feat_dim', default=768, type=int)
    parser.add_argument('--image_size', default=224, type=int)  # ViT固定使用224

    # 数据集配置 (与原版GCD一致)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', default=['v1'], help='评估函数')
    parser.add_argument('--transform', type=str, default='imagenet')  # CIFAR用imagenet transform resize到224
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

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

    # 设置必要参数
    args.device = device
    args.writer = None
    args.interpolation = 3
    args.crop_pct = 0.875

    print("🚀 原版GCD评估工具")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"评估模式: {args.eval_mode}")
    if args.eval_mode == 'superclass':
        print(f"目标超类: {args.superclass_name}")
    print("=" * 80)

    try:
        # 按照原版方式加载模型
        model = load_model_original_way(args.model_path, args, device)

        # 根据评估模式进行评估
        if args.eval_mode == 'full':
            evaluate_full_dataset(model, args, device)

        elif args.eval_mode == 'superclass':
            if args.superclass_name is None:
                print("❌ 错误: 超类评估模式需要指定 --superclass_name")
                return
            evaluate_superclass(model, args.superclass_name, args, device)

        print("\n🎉 评估完成!")

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()