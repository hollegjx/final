#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特征提取和保存工具
完全复制test_adaptive_clustering.py的特征提取逻辑，提取并保存特征到指定路径
保存的特征可以方便后续训练文件读取使用
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from copy import deepcopy
import pickle

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augmentations import get_transform
from data.cifar100_superclass import CIFAR100_SUPERCLASSES, get_single_superclass_datasets
from models import vision_transformer as vits
from config import dino_pretrain_path
from project_utils.general_utils import str2bool
from data.data_utils import MergedDataset


def set_deterministic_behavior():
    """设置确定性行为"""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(args, device):
    """加载训练好的模型（完全复制test_adaptive_clustering.py的逻辑）"""
    print(f"🔄 加载模型...")
    print(f"   模型文件: {args.model_path}")
    print(f"   设备: {device}")

    if args.base_model == 'vit_dino':
        model = vits.__dict__['vit_base']()

        # 加载DINO预训练权重
        if os.path.exists(dino_pretrain_path):
            print(f"   加载DINO预训练权重...")
            dino_state_dict = torch.load(dino_pretrain_path, map_location='cpu')
            model.load_state_dict(dino_state_dict, strict=False)

        # 加载训练权重
        print(f"   加载训练权重...")
        gcd_state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(gcd_state_dict)

        model.to(device)

        # 关闭梯度计算
        for param in model.parameters():
            param.requires_grad = False

        print(f"✅ 模型加载成功 (特征维度: {args.feat_dim})")
        return model
    else:
        raise NotImplementedError(f"不支持的模型类型: {args.base_model}")


def extract_features(data_loader, model, device, known_classes=None, dataset_type="unknown", use_l2=True):
    """
    提取特征（完全复制test_adaptive_clustering.py的特征提取逻辑）

    Args:
        data_loader: 数据加载器
        model: 特征提取模型
        device: 设备
        known_classes: 已知类别集合
        dataset_type: 数据集类型标识符
        use_l2: 是否使用L2归一化 (True=L2归一化, False=不使用)

    Returns:
        features: 特征矩阵 (n_samples, feat_dim)
        targets: 真实标签
        known_mask: 已知类别掩码 (True=已知类, False=未知类)
        labeled_mask: 有标签掩码 (True=有标签, False=无标签)
    """
    l2_status = "L2归一化" if use_l2 else "无L2归一化"
    print(f"🔄 提取{dataset_type}特征 ({l2_status})...")

    model.eval()
    all_feats = []
    targets = np.array([])
    mask = np.array([])  # 已知类别掩码
    labeled_mask = np.array([])  # 有标签掩码

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=f"提取{dataset_type}特征")):
            # 解包数据（与test_adaptive_clustering.py完全一致）
            if len(batch_data) == 4:
                images, labels, indices, labeled_or_not = batch_data
                labeled_batch = labeled_or_not.numpy().flatten().astype(bool)
            elif len(batch_data) == 3:
                images, labels, indices = batch_data
                # 测试集全部标记为无标签
                labeled_batch = np.zeros(len(labels), dtype=bool)
            else:
                continue

            # 提取特征
            images = images.to(device)
            feats = model(images)

            # 根据use_l2参数决定是否进行L2归一化
            if use_l2:
                feats = torch.nn.functional.normalize(feats, dim=-1)

            # 收集数据
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, labels.cpu().numpy())
            labeled_mask = np.append(labeled_mask, labeled_batch)

            # 创建已知类别掩码（根据known_classes列表）
            if known_classes is not None:
                batch_mask = np.array([True if x.item() in known_classes else False for x in labels])
            else:
                # 默认：前80个类别是已知类
                batch_mask = np.array([True if x.item() < 80 else False for x in labels])
            mask = np.append(mask, batch_mask)

            # 清理GPU内存
            del images, feats
            torch.cuda.empty_cache()

    # 拼接所有特征
    all_feats = np.concatenate(all_feats, axis=0)
    print(f"✅ {dataset_type}特征提取完成: {all_feats.shape}")

    return all_feats, targets.astype(int), mask.astype(bool), labeled_mask.astype(bool)


def extract_and_save_features(superclass_name, model_path, output_base_dir='/data/gjx/checkpoints/features', use_l2=True):
    """
    提取并保存指定超类的特征

    Args:
        superclass_name: 超类名称
        model_path: 模型路径
        output_base_dir: 输出基础目录
        use_l2: 是否使用L2归一化 (True=L2归一化, False=不使用)
    """
    l2_status = "L2归一化" if use_l2 else "无L2归一化"
    print(f"🚀 开始提取和保存特征 ({l2_status}) - 超类: {superclass_name}")
    print("="*80)

    # 设置参数（完全复制test_adaptive_clustering.py的参数设置）
    class Args:
        def __init__(self):
            self.dataset_name = 'cifar100_superclass'
            self.superclass_name = superclass_name
            self.prop_train_labels = 0.8
            self.image_size = 224
            self.num_workers = 4
            self.batch_size = 64
            self.base_model = 'vit_dino'
            self.feat_dim = 768
            self.model_path = model_path
            self.interpolation = 3
            self.crop_pct = 0.875
            self.seed = 0

    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置确定性行为
    set_deterministic_behavior()

    # 加载模型
    model = load_model(args, device)

    # 获取超类信息（完全复制test_adaptive_clustering.py的逻辑）
    superclass_classes = set(CIFAR100_SUPERCLASSES[superclass_name])
    superclass_known_classes_orig = set([cls for cls in superclass_classes if cls < 80])
    superclass_unknown_classes_orig = set([cls for cls in superclass_classes if cls >= 80])

    # 创建标签映射（与超类数据集保持一致）
    all_classes_sorted = sorted(list(superclass_classes))
    label_mapping = {orig_cls: new_cls for new_cls, orig_cls in enumerate(all_classes_sorted)}

    # 映射后的已知/未知类别ID
    known_classes_mapped = set([label_mapping[cls] for cls in superclass_known_classes_orig])
    unknown_classes_mapped = set([label_mapping[cls] for cls in superclass_unknown_classes_orig])

    print(f"📊 超类信息:")
    print(f"   原始已知类: {sorted(list(superclass_known_classes_orig))} -> 映射后: {sorted(list(known_classes_mapped))}")
    print(f"   原始未知类: {sorted(list(superclass_unknown_classes_orig))} -> 映射后: {sorted(list(unknown_classes_mapped))}")

    # 获取数据（完全复制test_adaptive_clustering.py的数据获取逻辑）
    train_transform, test_transform = get_transform('imagenet', image_size=args.image_size, args=args)

    datasets = get_single_superclass_datasets(
        superclass_name=superclass_name,
        train_transform=train_transform,
        test_transform=test_transform,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
        seed=args.seed
    )

    # 准备数据集和加载器（复制train_and_test模式）
    train_dataset = datasets['train_labelled']
    unlabelled_train_dataset = datasets['train_unlabelled']
    test_dataset = datasets['test']

    # 创建MergedDataset
    merged_train_dataset = MergedDataset(
        labelled_dataset=deepcopy(train_dataset),
        unlabelled_dataset=deepcopy(unlabelled_train_dataset)
    )

    # 创建数据加载器
    train_loader = DataLoader(merged_train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 提取特征
    print("📊 提取训练集特征...")
    train_feats, train_targets, train_known_mask, train_labeled_mask = extract_features(
        train_loader, model, device, known_classes_mapped, "训练集", use_l2=use_l2
    )

    print("📊 提取测试集特征...")
    test_feats, test_targets, test_known_mask, test_labeled_mask = extract_features(
        test_loader, model, device, known_classes_mapped, "测试集", use_l2=use_l2
    )

    # 合并训练集和测试集（与test_adaptive_clustering.py一致）
    all_feats = np.concatenate([train_feats, test_feats], axis=0)
    all_targets = np.concatenate([train_targets, test_targets], axis=0)
    all_known_mask = np.concatenate([train_known_mask, test_known_mask], axis=0)
    all_labeled_mask = np.concatenate([train_labeled_mask, test_labeled_mask], axis=0)

    print(f"📊 合并后数据统计:")
    print(f"   总样本数: {len(all_feats)}")
    print(f"   已知类样本: {np.sum(all_known_mask)}")
    print(f"   未知类样本: {np.sum(~all_known_mask)}")
    print(f"   有标签样本: {np.sum(all_labeled_mask)}")
    print(f"   无标签样本: {np.sum(~all_labeled_mask)}")

    # 创建输出目录
    output_dir = os.path.join(output_base_dir, superclass_name)
    os.makedirs(output_dir, exist_ok=True)

    # 准备保存的数据
    feature_data = {
        # 完整数据（训练+测试）
        'all_features': all_feats,
        'all_targets': all_targets,
        'all_known_mask': all_known_mask,
        'all_labeled_mask': all_labeled_mask,

        # 分离的训练集和测试集数据
        'train_features': train_feats,
        'train_targets': train_targets,
        'train_known_mask': train_known_mask,
        'train_labeled_mask': train_labeled_mask,
        'test_features': test_feats,
        'test_targets': test_targets,
        'test_known_mask': test_known_mask,
        'test_labeled_mask': test_labeled_mask,

        # 元信息
        'superclass_name': superclass_name,
        'known_classes_mapped': sorted(list(known_classes_mapped)),
        'unknown_classes_mapped': sorted(list(unknown_classes_mapped)),
        'known_classes_orig': sorted(list(superclass_known_classes_orig)),
        'unknown_classes_orig': sorted(list(superclass_unknown_classes_orig)),
        'label_mapping': label_mapping,
        'all_classes_sorted': all_classes_sorted,

        # 数据集分割信息
        'train_size': len(train_feats),
        'test_size': len(test_feats),
        'total_size': len(all_feats),

        # 提取参数
        'model_path': model_path,
        'feat_dim': args.feat_dim,
        'image_size': args.image_size,
        'prop_train_labels': args.prop_train_labels
    }

    # 保存特征数据
    feature_file = os.path.join(output_dir, 'features.pkl')
    with open(feature_file, 'wb') as f:
        pickle.dump(feature_data, f)
    print(f"✅ 特征数据已保存到: {feature_file}")

    # 保存文本摘要
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"超类特征提取摘要\n")
        f.write(f"================\n")
        f.write(f"超类名称: {superclass_name}\n")
        f.write(f"模型路径: {model_path}\n")
        from datetime import datetime
        f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"数据统计:\n")
        f.write(f"  训练集样本数: {len(train_feats)}\n")
        f.write(f"  测试集样本数: {len(test_feats)}\n")
        f.write(f"  总样本数: {len(all_feats)}\n")
        f.write(f"  特征维度: {all_feats.shape[1]}\n\n")

        f.write(f"类别信息:\n")
        f.write(f"  原始已知类: {sorted(list(superclass_known_classes_orig))}\n")
        f.write(f"  原始未知类: {sorted(list(superclass_unknown_classes_orig))}\n")
        f.write(f"  映射后已知类: {sorted(list(known_classes_mapped))}\n")
        f.write(f"  映射后未知类: {sorted(list(unknown_classes_mapped))}\n\n")

        f.write(f"标签统计:\n")
        f.write(f"  已知类样本: {np.sum(all_known_mask)} ({np.sum(all_known_mask)/len(all_feats)*100:.1f}%)\n")
        f.write(f"  未知类样本: {np.sum(~all_known_mask)} ({np.sum(~all_known_mask)/len(all_feats)*100:.1f}%)\n")
        f.write(f"  有标签样本: {np.sum(all_labeled_mask)} ({np.sum(all_labeled_mask)/len(all_feats)*100:.1f}%)\n")
        f.write(f"  无标签样本: {np.sum(~all_labeled_mask)} ({np.sum(~all_labeled_mask)/len(all_feats)*100:.1f}%)\n")

    print(f"✅ 摘要信息已保存到: {summary_file}")

    print(f"\n🎯 特征提取完成!")
    print(f"   超类: {superclass_name}")
    print(f"   输出目录: {output_dir}")
    print(f"   特征文件: features.pkl")
    print(f"   摘要文件: summary.txt")

    return feature_data


def load_saved_features(superclass_name, output_base_dir='/data/gjx/checkpoints/features'):
    """
    加载已保存的特征数据

    Args:
        superclass_name: 超类名称
        output_base_dir: 特征保存的基础目录

    Returns:
        feature_data: 特征数据字典
    """
    feature_file = os.path.join(output_base_dir, superclass_name, 'features.pkl')

    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"特征文件不存在: {feature_file}")

    print(f"📥 加载特征数据: {feature_file}")
    with open(feature_file, 'rb') as f:
        feature_data = pickle.load(f)

    print(f"✅ 特征数据加载完成: {superclass_name}")
    print(f"   总样本数: {feature_data['total_size']}")
    print(f"   特征维度: {feature_data['all_features'].shape[1]}")

    return feature_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='提取并保存超类特征（逻辑与 test_adaptive_clustering.py 保持一致）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必要参数
    parser.add_argument(
        '--model_path',
        type=str,
        default='/data1/jiangzhen/gjx/exp/newgpc/final/metric_learn_gcd/log/(14.09.2025_|_56.443)/checkpoints/model.pt',
        help='训练好的超类模型权重路径。必须是与 test_adaptive_clustering.py 兼容的 ViT-DINO checkpoint。'
    )
    parser.add_argument(
        '--superclass_name',
        type=str,
        default='trees',
        help='目标超类名称（CIFAR-100 超类之一，例如 trees、flowers 等）。'
    )

    # 可选参数
    parser.add_argument(
        '--load_only',
        type=str2bool,
        default=False,
        help='仅加载已存在的 features.pkl 而不重新提取。当缓存已存在时可快速查看摘要。'
    )
    parser.add_argument(
        '--l2',
        type=str2bool,
        default=True,
        help='是否在保存前对特征做 L2 归一化。该开关会影响默认的输出目录（features vs features_nol2）。'
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default=None,
        help='特征缓存的根目录，内部会自动创建 <root>/<superclass>/features.pkl。未指定时按 --l2 自动落在 /data/gjx/checkpoints/features[_nol2]。'
    )

    args = parser.parse_args()

    # 根据l2参数选择输出目录
    if args.output_base_dir:
        output_base_dir = args.output_base_dir
    else:
        output_base_dir = '/data/gjx/checkpoints/features' if args.l2 else '/data/gjx/checkpoints/features_nol2'

    print("特征提取和保存工具")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"超类名称: {args.superclass_name}")
    print(f"特征归一化: {'L2归一化' if args.l2 else '无L2归一化'}")
    print(f"输出目录: {output_base_dir}")
    print(f"仅加载模式: {args.load_only}")
    print("="*80)

    try:
        if args.load_only:
            # 仅加载模式
            feature_data = load_saved_features(args.superclass_name, output_base_dir=output_base_dir)
            print(f"\n🎯 特征加载完成!")
        else:
            # 提取并保存模式
            feature_data = extract_and_save_features(
                superclass_name=args.superclass_name,
                model_path=args.model_path,
                output_base_dir=output_base_dir,
                use_l2=args.l2
            )

        # 显示一些基本统计信息
        print(f"\n📈 特征数据统计:")
        print(f"   特征形状: {feature_data['all_features'].shape}")
        print(f"   已知类别: {len(feature_data['known_classes_mapped'])}个")
        print(f"   未知类别: {len(feature_data['unknown_classes_mapped'])}个")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
