#!/usr/bin/env python3
"""
超类特征缓存脚本
负责加载指定超类的最佳模型，提取训练/测试特征并写入标准缓存目录
完全复制test_feature.py的特征提取逻辑以确保性能一致性
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 将项目根目录加入sys.path，方便脚本独立运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from collections import OrderedDict
from utils.data import FeatureLoader
from config import checkpoint_root as DEFAULT_CHECKPOINT_ROOT
from config import feature_cache_dir as DEFAULT_FEATURE_CACHE_DIR
from config import dino_pretrain_path
from data.augmentations import get_transform
from data.cifar100_superclass import CIFAR100_SUPERCLASSES, SUPERCLASS_NAMES, get_single_superclass_datasets
from data.data_utils import MergedDataset
from models import vision_transformer as vits
from project_utils.general_utils import str2bool


def set_deterministic_behavior():
    """设置确定性行为（完全复制test_feature.py）"""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_acc_from_filename(filename: str) -> Optional[float]:
    """
    根据文件名解析ACC值，兼容多种命名格式
    """
    match = re.search(r'acc([0-9]+(?:\.[0-9]+)?)', filename)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    match = re.search(r'allacc_(\d+)', filename)
    if match:
        try:
            return float(match.group(1)) / 100.0
        except ValueError:
            return None

    return None


def find_best_superclass_model(superclass_name: str,
                               checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT) -> Tuple[str, float]:
    """
    自动扫描超类模型目录以找到ACC最高的模型权重
    """
    model_dir = os.path.join(checkpoint_root, superclass_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"超类模型目录不存在: {model_dir}")

    candidates: Set[str] = set()
    for pattern in ('model_best_acc*.pt', 'allacc_*.pt', '*.pt'):
        pattern_paths = glob.glob(os.path.join(model_dir, pattern))
        candidates.update(pattern_paths)

    if not candidates:
        raise FileNotFoundError(f"在目录中未找到任何模型文件: {model_dir}")

    best_path: Optional[str] = None
    best_acc = -1.0

    for path in sorted(candidates):
        acc = parse_acc_from_filename(os.path.basename(path))
        if acc is None:
            continue
        if acc > best_acc:
            best_acc = acc
            best_path = path

    if best_path is None:
        raise ValueError(f"无法从以下文件解析ACC: {', '.join(sorted(candidates))}")

    return best_path, best_acc


def _unwrap_model_weights(raw_state) -> OrderedDict:
    """
    将可能包含完整训练状态的检查点解包为纯模型权重。
    """
    if isinstance(raw_state, dict):
        # 来自 save_training_state 的完整检查点
        if "model" in raw_state and isinstance(raw_state["model"], (dict, OrderedDict)):
            print("   检测到完整训练检查点，自动提取其中的模型参数用于特征提取")
            return raw_state["model"]
        # 通用模式：state_dict 包裹
        if "state_dict" in raw_state and isinstance(raw_state["state_dict"], (dict, OrderedDict)):
            return raw_state["state_dict"]
    if isinstance(raw_state, OrderedDict):
        return raw_state
    raise RuntimeError(
        "无法从给定文件中解析模型参数，请确认传入的是模型权重或 save_training_state 产生的检查点。"
    )


def load_superclass_model(model_path: str, device: torch.device, feat_dim: int = 768) -> torch.nn.Module:
    """
    加载训练好的ViT基础模型（完全复制test_feature.py的逻辑）
    采用两阶段加载：先加载DINO预训练基座，再加载GCD训练权重
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"🔄 加载模型: {model_path}")
    print(f"   设备: {device}")

    # 创建ViT模型
    model = vits.__dict__['vit_base']()

    # 【关键】第一阶段：加载DINO预训练权重作为基座
    if os.path.exists(dino_pretrain_path):
        print(f"   加载DINO预训练权重: {dino_pretrain_path}")
        dino_state_dict = torch.load(dino_pretrain_path, map_location='cpu')
        model.load_state_dict(dino_state_dict, strict=False)
    else:
        print(f"⚠️  DINO预训练权重不存在: {dino_pretrain_path}")

    # 【关键】第二阶段：加载GCD训练权重
    print(f"   加载训练权重: {model_path}")
    raw_state = torch.load(model_path, map_location='cpu')
    gcd_state_dict = _unwrap_model_weights(raw_state)
    model.load_state_dict(gcd_state_dict)

    model.to(device)

    # 关闭梯度计算
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载完成 (参数量: {total_params:,}, 特征维度: {feat_dim})")
    return model


def extract_features(data_loader, model, device, known_classes=None, dataset_type="unknown", use_l2=True):
    """
    提取特征（完全复制test_feature.py的特征提取逻辑）

    Args:
        data_loader: 数据加载器
        model: 特征提取模型
        device: 设备
        known_classes: 已知类别集合
        dataset_type: 数据集类型标识符
        use_l2: 是否使用L2归一化 (True=L2归一化, False=不使用)

    Returns:
        tuple(np.ndarray, ...):
            features: 特征矩阵 (n_samples, feat_dim)
            targets: 真实标签
            known_mask: 已知类别掩码 (True=已知类, False=未知类)
            labeled_mask: 有标签掩码 (True=有标签, False=无标签)
            indices: 数据集全局索引
    """
    l2_status = "L2归一化" if use_l2 else "无L2归一化"
    print(f"🔄 提取{dataset_type}特征 ({l2_status})...")

    model.eval()
    all_feats = []
    targets = np.array([])
    mask = np.array([])  # 已知类别掩码
    labeled_mask = np.array([])  # 有标签掩码
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=f"提取{dataset_type}特征")):
            # 解包数据（与test_feature.py完全一致）
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

            # 记录样本索引
            if isinstance(indices, torch.Tensor):
                batch_indices = indices.cpu().numpy()
            else:
                batch_indices = np.atleast_1d(np.asarray(indices))
            all_indices.append(batch_indices.astype(np.int64))

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
    all_indices = np.concatenate(all_indices, axis=0).astype(np.int64)
    print(f"✅ {dataset_type}特征提取完成: {all_feats.shape}")

    return all_feats, targets.astype(int), mask.astype(bool), labeled_mask.astype(bool), all_indices


def load_superclass_datasets(superclass_name: str,
                             batch_size: int,
                             num_workers: int,
                             prop_train_labels: float = 0.8,
                             seed: int = 0,
                             image_size: int = 224):
    """
    构建与test_feature.py一致的数据加载器（完全复制逻辑）
    """
    # 【关键】使用与test_feature.py完全一致的参数设置
    class Args:
        def __init__(self):
            self.dataset_name = 'cifar100_superclass'
            self.superclass_name = superclass_name
            self.prop_train_labels = prop_train_labels
            self.image_size = image_size
            self.num_workers = num_workers
            self.batch_size = batch_size
            self.base_model = 'vit_dino'
            self.feat_dim = 768
            self.interpolation = 3
            self.crop_pct = 0.875
            self.seed = seed

    args = Args()

    # 【关键】硬编码使用'imagenet' transform，与test_feature.py保持一致
    train_transform, test_transform = get_transform('imagenet', image_size=args.image_size, args=args)

    datasets = get_single_superclass_datasets(
        superclass_name=superclass_name,
        train_transform=train_transform,
        test_transform=test_transform,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
        seed=args.seed
    )

    # 创建MergedDataset（与test_feature.py一致）
    merged_train = MergedDataset(
        labelled_dataset=deepcopy(datasets['train_labelled']),
        unlabelled_dataset=deepcopy(datasets['train_unlabelled'])
    )

    train_loader = DataLoader(
        merged_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    # 【关键】手动构建label_mapping，与test_feature.py保持一致
    superclass_classes = set(CIFAR100_SUPERCLASSES[superclass_name])
    superclass_known_classes_orig = set([cls for cls in superclass_classes if cls < 80])
    superclass_unknown_classes_orig = set([cls for cls in superclass_classes if cls >= 80])

    # 创建标签映射（与超类数据集保持一致）
    all_classes_sorted = sorted(list(superclass_classes))
    label_mapping = {orig_cls: new_cls for new_cls, orig_cls in enumerate(all_classes_sorted)}

    # 映射后的已知/未知类别ID
    known_label_ids = set([label_mapping[cls] for cls in superclass_known_classes_orig])
    unknown_label_ids = set([label_mapping[cls] for cls in superclass_unknown_classes_orig])

    dataset_stats = {
        'train_labelled_len': len(datasets['train_labelled']),
        'train_unlabelled_len': len(datasets['train_unlabelled']),
        'train_total': len(datasets['train_labelled']) + len(datasets['train_unlabelled']),
        'test_len': len(datasets['test']),
        'known_label_ids': known_label_ids,
        'unknown_label_ids': unknown_label_ids,
        'label_mapping': label_mapping,
        'all_classes_sorted': all_classes_sorted,
        'known_classes_orig': sorted(list(superclass_known_classes_orig)),
        'unknown_classes_orig': sorted(list(superclass_unknown_classes_orig))
    }

    print(f"📊 超类信息:")
    print(f"   原始已知类: {sorted(list(superclass_known_classes_orig))} -> 映射后: {sorted(list(known_label_ids))}")
    print(f"   原始未知类: {sorted(list(superclass_unknown_classes_orig))} -> 映射后: {sorted(list(unknown_label_ids))}")

    return train_loader, test_loader, dataset_stats


def extract_and_cache_features(model: torch.nn.Module,
                               train_loader: DataLoader,
                               test_loader: DataLoader,
                               superclass_name: str,
                               dataset_stats: Dict,
                               cache_loader: FeatureLoader,
                               use_l2: bool,
                               device: torch.device,
                               model_path: str):
    """
    通过extract_features提取特征并写入标准缓存（完全复制test_feature.py的逻辑）
    """
    # 提取特征（使用复制的extract_features函数）
    print("📊 提取训练集特征...")
    train_feats, train_targets, train_known_mask, train_labeled_mask, train_indices = extract_features(
        train_loader, model, device, dataset_stats['known_label_ids'], "训练集", use_l2=use_l2
    )

    print("📊 提取测试集特征...")
    test_feats, test_targets, test_known_mask, test_labeled_mask, test_indices = extract_features(
        test_loader, model, device, dataset_stats['known_label_ids'], "测试集", use_l2=use_l2
    )

    # 合并训练集和测试集（与test_feature.py一致）
    all_feats = np.concatenate([train_feats, test_feats], axis=0)
    all_targets = np.concatenate([train_targets, test_targets], axis=0)
    all_known_mask = np.concatenate([train_known_mask, test_known_mask], axis=0)
    all_labeled_mask = np.concatenate([train_labeled_mask, test_labeled_mask], axis=0)
    all_indices = np.concatenate([train_indices, test_indices], axis=0)

    print(f"📊 合并后数据统计:")
    print(f"   总样本数: {len(all_feats)}")
    print(f"   已知类样本: {np.sum(all_known_mask)}")
    print(f"   未知类样本: {np.sum(~all_known_mask)}")
    print(f"   有标签样本: {np.sum(all_labeled_mask)}")
    print(f"   无标签样本: {np.sum(~all_labeled_mask)}")

    # 准备保存的数据（与test_feature.py的数据结构完全一致）
    feature_dict = {
        # 完整数据（训练+测试）
        'all_features': all_feats,
        'all_targets': all_targets,
        'all_known_mask': all_known_mask,
        'all_labeled_mask': all_labeled_mask,
        'all_indices': all_indices,

        # 分离的训练集和测试集数据
        'train_features': train_feats,
        'train_targets': train_targets,
        'train_known_mask': train_known_mask,
        'train_labeled_mask': train_labeled_mask,
        'train_indices': train_indices,
        'test_features': test_feats,
        'test_targets': test_targets,
        'test_known_mask': test_known_mask,
        'test_labeled_mask': test_labeled_mask,
        'test_indices': test_indices,

        # 元信息
        'superclass_name': superclass_name,
        'known_classes_mapped': dataset_stats['known_label_ids'],
        'unknown_classes_mapped': dataset_stats['unknown_label_ids'],
        'known_classes_orig': dataset_stats['known_classes_orig'],
        'unknown_classes_orig': dataset_stats['unknown_classes_orig'],
        'label_mapping': dataset_stats['label_mapping'],
        'all_classes_sorted': dataset_stats['all_classes_sorted'],

        # 数据集分割信息
        'train_size': len(train_feats),
        'test_size': len(test_feats),
        'total_size': len(all_feats),

        # 提取参数
        'model_path': model_path,
        'feat_dim': 768,
        'image_size': 224
    }

    # 保存到缓存文件
    cache_file_path = cache_loader.get_cache_path(superclass_name, use_l2=use_l2)
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)

    with open(cache_file_path, 'wb') as handle:
        pickle.dump(feature_dict, handle, protocol=4)

    total_samples = len(feature_dict['all_features'])
    feat_dim = feature_dict['all_features'].shape[1]
    size_mb = os.path.getsize(cache_file_path) / (1024 ** 2)

    print(f"💾 已写入缓存: {cache_file_path}")
    print(f"   样本数: {total_samples}, 特征维度: {feat_dim}, 文件大小: {size_mb:.2f} MB")

    return cache_file_path, feature_dict


def cache_single_superclass(superclass_name: str,
                            model_path: Optional[str] = None,
                            auto_find_best: bool = True,
                            checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
                            cache_dir: str = DEFAULT_FEATURE_CACHE_DIR,
                            batch_size: int = 64,
                            num_workers: int = 4,
                            gpu: int = 0,
                            use_l2: bool = True,
                            overwrite: bool = False,
                            prop_train_labels: float = 0.8,
                            seed: int = 0,
                            image_size: int = 224) -> Dict[str, object]:
    """
    为单个超类提取并缓存特征（完全复制test_feature.py的逻辑）
    """
    print("\n" + "=" * 80)
    print(f"🎯 开始处理超类: {superclass_name}")
    print("=" * 80)

    # 设置确定性行为（与test_feature.py一致）
    set_deterministic_behavior()

    cache_loader = FeatureLoader(cache_base_dir=cache_dir)

    if cache_loader.check_cache_exists(superclass_name, use_l2=use_l2) and not overwrite:
        cache_path = cache_loader.get_cache_path(superclass_name, use_l2=use_l2)
        print(f"⚠️  缓存已存在且未启用覆盖: {cache_path}")
        return {'status': 'skipped', 'superclass_name': superclass_name, 'reason': 'cache_exists'}

    if model_path is None:
        if not auto_find_best:
            raise ValueError("未提供model_path且auto_find_best被禁用")
        model_path, best_acc = find_best_superclass_model(superclass_name, checkpoint_root)
        print(f"🔍 自动找到最佳模型: {os.path.basename(model_path)} (ACC={best_acc:.4f})")
    else:
        best_acc = None
        model_path = os.path.abspath(model_path)

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {device}")

    # 【关键】使用新的模型加载函数（两阶段加载）
    model = load_superclass_model(model_path, device, feat_dim=768)

    # 【关键】使用新的数据集加载函数（完全复制test_feature.py逻辑）
    train_loader, test_loader, dataset_stats = load_superclass_datasets(
        superclass_name=superclass_name,
        batch_size=batch_size,
        num_workers=num_workers,
        prop_train_labels=prop_train_labels,
        seed=seed,
        image_size=image_size
    )

    print("📊 数据集统计:")
    print(f"   训练集(有标记): {dataset_stats['train_labelled_len']} | "
          f"训练集(无标记): {dataset_stats['train_unlabelled_len']}")
    print(f"   训练集总计: {dataset_stats['train_total']} | 测试集: {dataset_stats['test_len']}")
    print(f"   已知类数: {len(dataset_stats['known_label_ids'])} | "
          f"未知类数: {len(dataset_stats['unknown_label_ids'])}")

    # 【关键】使用新的特征提取和缓存函数（完全复制test_feature.py逻辑）
    cache_file_path, feature_dict = extract_and_cache_features(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        superclass_name=superclass_name,
        dataset_stats=dataset_stats,
        cache_loader=cache_loader,
        use_l2=use_l2,
        device=device,
        model_path=model_path
    )

    print("🧪 正在验证缓存可读性...")
    loaded = cache_loader.load(superclass_name, use_l2=use_l2, silent=True)
    if loaded is None:
        raise RuntimeError("缓存验证失败，pickle文件可能损坏")

    print(f"🎉 超类 {superclass_name} 缓存完成")
    return {
        'status': 'success',
        'superclass_name': superclass_name,
        'model_path': model_path,
        'cache_file_path': cache_file_path,
        'n_samples': len(feature_dict['all_features']),
        'feat_dim': feature_dict['all_features'].shape[1],
        'best_acc': best_acc
    }


def cache_all_superclasses(superclass_names: Optional[List[str]] = None, **kwargs):
    """
    批量为多个超类生成特征缓存
    """
    names = superclass_names or SUPERCLASS_NAMES
    print(f"\n🌟 即将处理 {len(names)} 个超类: {names}")

    results = []
    for idx, name in enumerate(names, start=1):
        print(f"\n➡️  进度 [{idx}/{len(names)}] - {name}")
        try:
            result = cache_single_superclass(superclass_name=name, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"❌ 超类 {name} 缓存失败: {exc}")
            result = {'status': 'failed', 'superclass_name': name, 'error': str(exc)}
        results.append(result)

    success = sum(1 for r in results if r.get('status') == 'success')
    skipped = sum(1 for r in results if r.get('status') == 'skipped')
    failed = sum(1 for r in results if r.get('status') == 'failed')

    print("\n" + "=" * 80)
    print("📈 批量缓存统计:")
    print(f"   成功: {success} | 跳过: {skipped} | 失败: {failed}")
    print("=" * 80)

    return results


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行解析器
    """
    parser = argparse.ArgumentParser(
        description='超类特征缓存脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--superclass_name', type=str, default=None,
                        help="单个超类名称，如'trees'")
    parser.add_argument('--all_superclasses', action='store_true',
                        help='是否处理全部15个超类')

    parser.add_argument('--model_path', type=str, default=None,
                        help='手动指定模型文件路径')
    parser.add_argument('--auto_find_best', type=str2bool, default=True,
                        help='是否自动搜索ACC最高的模型')
    parser.add_argument('--checkpoint_root', type=str, default=DEFAULT_CHECKPOINT_ROOT,
                        help='超类模型checkpoint根目录')

    parser.add_argument('--cache_dir', type=str,
                        default=DEFAULT_FEATURE_CACHE_DIR,
                        help='特征缓存输出根目录')
    parser.add_argument('--use_l2', type=str2bool, default=True,
                        help='是否对特征进行L2归一化后再缓存')
    parser.add_argument('--overwrite', action='store_true',
                        help='是否覆盖已存在的缓存')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='特征提取批大小（与test_feature.py一致）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader的工作线程数（与test_feature.py一致）')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的GPU编号')
    parser.add_argument('--prop_train_labels', type=float, default=0.8,
                        help='训练集中有标签样本占比')
    parser.add_argument('--seed', type=int, default=0,
                        help='数据划分随机种子（与test_feature.py一致：0）')
    parser.add_argument('--image_size', type=int, default=224,
                        help='图像大小（与test_feature.py一致）')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.all_superclasses and args.superclass_name is None:
        parser.error("必须指定 --superclass_name 或开启 --all_superclasses")

    common_kwargs = dict(
        model_path=args.model_path,
        auto_find_best=args.auto_find_best,
        checkpoint_root=args.checkpoint_root,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu=args.gpu,
        use_l2=args.use_l2,
        overwrite=args.overwrite,
        prop_train_labels=args.prop_train_labels,
        seed=args.seed,
        image_size=args.image_size
    )

    if args.all_superclasses:
        cache_all_superclasses(**common_kwargs)
    else:
        cache_single_superclass(superclass_name=args.superclass_name, **common_kwargs)

    print("\n✅ 所有任务完成!")


if __name__ == '__main__':
    main()
