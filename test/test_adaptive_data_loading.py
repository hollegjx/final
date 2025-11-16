#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试目标：
1. 验证能否正确读取指定超类的训练集和测试集数据
2. 检查数据格式是否符合kmeans方案的要求（包含labeled_or_not标记）
3. 分析已知标签样本和未知标签样本的分布情况
4. 确认训练集中的有标签/无标签比例符合prop_train_labels设置
5. 验证测试集是否正确包含已知类和未知类样本
6. 为自适应聚类算法提供正确的数据输入格式验证

关键验证点：
- 训练集中已知类的有标签样本比例应约为prop_train_labels
- 训练集中未知类样本应全部为无标签
- 测试集应包含已知类和未知类样本
- 数据格式应为(features, labels, indices, labeled_or_not)
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.get_datasets import get_datasets, get_class_splits
from data.augmentations import get_transform
from data.cifar100_superclass import CIFAR100_SUPERCLASSES, get_single_superclass_datasets
from models import vision_transformer as vits
from config import dino_pretrain_path, exp_root
from project_utils.general_utils import str2bool


def load_model(args, device):
    """
    加载训练好的模型（模仿eval_original_gcd.py）
    """
    print(f"   模型文件: {args.model_path}")
    print(f"   设备: {device}")

    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    # 构建base model
    if args.base_model == 'vit_dino':
        model = vits.__dict__['vit_base']()

        # 加载DINO预训练权重（作为基础）
        if os.path.exists(dino_pretrain_path):
            print(f"   加载DINO预训练权重...")
            dino_state_dict = torch.load(dino_pretrain_path, map_location='cpu')
            model.load_state_dict(dino_state_dict, strict=False)

        # 加载GCD训练后的权重
        print(f"   加载训练权重...")
        gcd_state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(gcd_state_dict)

        model.to(device)
        model.eval()

        # 关闭梯度计算
        for param in model.parameters():
            param.requires_grad = False

        print(f"✅ 模型加载成功 (特征维度: {args.feat_dim})")
        return model

    else:
        raise NotImplementedError(f"不支持的模型类型: {args.base_model}")


def test_superclass_data_loading(superclass_name='trees', prop_train_labels=0.8):
    """
    测试指定超类的数据加载和标签分析

    Args:
        superclass_name: 超类名称
        prop_train_labels: 训练集有标签样本比例
    """
    print(f"🧪 测试超类 '{superclass_name}' 数据加载")
    print("=" * 80)

    # 设置参数（模仿训练脚本使用超类数据集）
    class Args:
        def __init__(self):
            self.dataset_name = 'cifar100_superclass'  # 使用超类数据集
            self.superclass_name = superclass_name
            self.prop_train_labels = prop_train_labels
            self.image_size = 224  # 模仿原版GCD使用imagenet transform
            self.num_workers = 4
            self.batch_size = 64  # 减少内存使用
            self.base_model = 'vit_dino'
            self.feat_dim = 768  # ViT-Base特征维度
            self.model_path = os.path.join(
                exp_root,
                'metric_learn_gcd/log/(14.09.2025_|_56.443)/checkpoints/model.pt'
            )
            self.interpolation = 3  # 添加缺失的参数
            self.crop_pct = 0.875
            self.seed = 0  # 添加随机种子

    args = Args()

    # 验证超类是否存在
    if superclass_name not in CIFAR100_SUPERCLASSES:
        print(f"❌ 错误: 未知的超类名称 '{superclass_name}'")
        return None, None

    # 获取超类信息
    superclass_classes = set(CIFAR100_SUPERCLASSES[superclass_name])
    print(f"📊 超类 '{superclass_name}' 包含类别: {sorted(list(superclass_classes))}")

    # 加载模型（模仿eval_original_gcd.py）
    print(f"\n🔄 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)

    # 获取数据变换（模仿训练脚本）
    print(f"🔄 获取数据变换...")
    train_transform, test_transform = get_transform('imagenet', image_size=args.image_size, args=args)

    # 使用超类专用数据集函数（模仿训练脚本）
    print(f"🔄 加载超类数据集...")
    datasets = get_single_superclass_datasets(
        superclass_name=superclass_name,
        train_transform=train_transform,
        test_transform=test_transform,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
        seed=args.seed
    )

    # 获取训练和测试数据集
    train_dataset = datasets['train_labelled']  # 有标签训练数据
    unlabelled_train_dataset = datasets['train_unlabelled']  # 无标签训练数据
    test_dataset = datasets['test']  # 测试数据

    # 创建MergedDataset（模仿训练脚本）
    from data.data_utils import MergedDataset
    from copy import deepcopy
    merged_train_dataset = MergedDataset(
        labelled_dataset=deepcopy(train_dataset),
        unlabelled_dataset=deepcopy(unlabelled_train_dataset)
    )

    # 创建数据加载器（分开训练集和测试集）
    train_loader = DataLoader(
        merged_train_dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=False
    )

    # 获取超类的已知/未知类别划分（原始ID）
    superclass_known_classes_orig = set([cls for cls in superclass_classes if cls < 80])
    superclass_unknown_classes_orig = set([cls for cls in superclass_classes if cls >= 80])

    # 获取标签映射（超类数据集使用了连续标签映射）
    all_classes_sorted = sorted(list(superclass_classes))
    label_mapping = {orig_cls: new_cls for new_cls, orig_cls in enumerate(all_classes_sorted)}

    # 映射后的类别ID（0,1,2,3,4）
    superclass_known_classes = set([label_mapping[cls] for cls in superclass_known_classes_orig])
    superclass_unknown_classes = set([label_mapping[cls] for cls in superclass_unknown_classes_orig])

    print(f"📊 超类内类别划分:")
    print(f"   原始已知类: {sorted(list(superclass_known_classes_orig))} -> 映射后: {sorted(list(superclass_known_classes))}")
    print(f"   原始未知类: {sorted(list(superclass_unknown_classes_orig))} -> 映射后: {sorted(list(superclass_unknown_classes))}")
    print(f"   标签映射: {label_mapping}")
    print(f"   有标签样本比例: {args.prop_train_labels}")

    print(f"✅ 数据集加载完成")
    print(f"   train_labelled大小: {len(train_dataset)}")
    print(f"   train_unlabelled大小: {len(unlabelled_train_dataset)}")
    print(f"   merged_train_dataset大小: {len(merged_train_dataset)}")
    print(f"   test_dataset大小: {len(test_dataset)}")

    # 分析训练集数据（使用模型提取特征）
    print(f"\n📊 分析训练集数据...")
    train_analysis = analyze_data_with_model(train_loader, "训练集", model, device, superclass_classes, superclass_known_classes, superclass_unknown_classes)

    # 分析测试集数据（使用模型提取特征）
    print(f"\n📊 分析测试集数据...")
    test_analysis = analyze_data_with_model(test_loader, "测试集", model, device, superclass_classes, superclass_known_classes, superclass_unknown_classes)

    # 验证数据一致性
    print(f"\n🔍 数据一致性验证:")
    validate_data_consistency(train_analysis, test_analysis, args.prop_train_labels)

    return train_analysis, test_analysis


def analyze_data_with_model(data_loader, dataset_name, model, device, superclass_classes, superclass_known_classes, superclass_unknown_classes):
    """
    使用模型分析数据加载器中的标签分布（只分析超类样本）
    """
    print(f"   正在分析{dataset_name}...")

    all_labels = []
    all_labeled_masks = []
    all_features = []
    batch_formats = []
    superclass_sample_count = 0

    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # 记录批次格式
                batch_formats.append(len(batch_data))

                # 解包数据
                if len(batch_data) == 4:
                    images, labels, indices, labeled_or_not = batch_data
                    labeled_mask = labeled_or_not.numpy().flatten()
                elif len(batch_data) == 3:
                    images, labels, indices = batch_data
                    # 测试集没有labeled_or_not，全部标记为无标签（聚类时看不到标签）
                    labeled_mask = np.zeros(len(labels))  # 测试集全部无标签
                else:
                    print(f"   ⚠️ 异常批次格式: {len(batch_data)}元素")
                    continue

                # 数据已经被过滤到超类，直接处理所有样本
                labels_np = labels.numpy()

                # 静默处理所有batch，只在最后显示统计信息

                # 使用模型提取特征
                if len(images) > 0:
                    images = images.to(device)
                    with torch.no_grad():  # 确保不计算梯度
                        features = model(images)
                        features_cpu = features.cpu().numpy()

                    all_features.append(features_cpu)
                    all_labels.extend(labels_np)
                    all_labeled_masks.extend(labeled_mask)
                    superclass_sample_count += len(labels_np)

                    # 清理GPU内存
                    del images, features
                    torch.cuda.empty_cache()

                # 分析所有批次以获得完整数据分布
                # 不限制批次数，确保统计完整

    except Exception as e:
        print(f"   ❌ 数据加载错误: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 如果没有超类样本，返回空分析
    if superclass_sample_count == 0:
        print(f"   ⚠️ 未找到属于超类的样本")
        return None

    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_labeled_masks = np.array(all_labeled_masks)

    # 拼接特征
    if all_features:
        all_features = np.concatenate(all_features, axis=0)

    print(f"   超类样本总数: {superclass_sample_count}")
    print(f"   特征维度: {all_features.shape if all_features.size > 0 else 'N/A'}")

    # 分析标签分布
    analysis = {}
    analysis['total_samples'] = len(all_labels)
    analysis['batch_formats'] = set(batch_formats)
    analysis['features'] = all_features

    # 分析已知类和未知类（在超类范围内）
    known_class_mask = np.isin(all_labels, list(superclass_known_classes))
    unknown_class_mask = np.isin(all_labels, list(superclass_unknown_classes))

    # 已知类样本分析
    known_samples = np.sum(known_class_mask)
    known_labeled = np.sum(known_class_mask & (all_labeled_masks == 1))
    known_unlabeled = np.sum(known_class_mask & (all_labeled_masks == 0))

    # 最终统计信息

    # 未知类样本分析
    unknown_samples = np.sum(unknown_class_mask)
    unknown_labeled = np.sum(unknown_class_mask & (all_labeled_masks == 1))
    unknown_unlabeled = np.sum(unknown_class_mask & (all_labeled_masks == 0))

    # 总体标签分析
    total_labeled = np.sum(all_labeled_masks == 1)
    total_unlabeled = np.sum(all_labeled_masks == 0)

    analysis.update({
        'known_samples': known_samples,
        'known_labeled': known_labeled,
        'known_unlabeled': known_unlabeled,
        'unknown_samples': unknown_samples,
        'unknown_labeled': unknown_labeled,
        'unknown_unlabeled': unknown_unlabeled,
        'total_labeled': total_labeled,
        'total_unlabeled': total_unlabeled,
        'superclass_known_classes': superclass_known_classes,
        'superclass_unknown_classes': superclass_unknown_classes
    })

    # 输出分析结果
    print(f"   批次格式: {analysis['batch_formats']}")
    print(f"   超类中已知类: {sorted(list(superclass_known_classes))}")
    print(f"   超类中未知类: {sorted(list(superclass_unknown_classes))}")
    print(f"   总有标签: {total_labeled}, 总无标签: {total_unlabeled}")
    print(f"   已知类样本: {known_samples} (有标签: {known_labeled}, 无标签: {known_unlabeled})")
    print(f"   未知类样本: {unknown_samples} (有标签: {unknown_labeled}, 无标签: {unknown_unlabeled})")

    if known_samples > 0:
        known_labeled_ratio = known_labeled / known_samples
        print(f"   已知类有标签比例: {known_labeled_ratio:.3f}")

    return analysis


def validate_fused_data_consistency(fused_analysis, expected_prop_train_labels, train_labelled_size, train_unlabelled_size, test_size):
    """
    验证融合数据集的一致性（模仿kmeans评估方式）
    """
    if fused_analysis is None:
        print("   ❌ 无法验证：融合数据分析失败")
        return

    print(f"   检查项目:")

    # 期望的数据分布
    expected_total = train_labelled_size + train_unlabelled_size + test_size
    expected_train_known_samples = int(train_labelled_size / expected_prop_train_labels)  # 训练集中已知类总数
    expected_train_known_labeled = train_labelled_size  # 训练集中已知类有标签数
    expected_test_unlabeled = test_size  # 测试集全部无标签（需要聚类预测）

    print(f"   📊 期望分布:")
    print(f"     总样本: {expected_total}")
    print(f"     训练集已知类: {expected_train_known_samples} (有标签: {expected_train_known_labeled})")
    print(f"     测试集: {test_size} (全部无标签，待聚类预测)")

    print(f"   📊 实际分布:")
    print(f"     总样本: {fused_analysis['total_samples']}")
    print(f"     已知类样本: {fused_analysis['known_samples']} (聚类算法知道这些来自训练见过的类)")
    print(f"     未知类样本: {fused_analysis['unknown_samples']} (聚类算法不知道这些来自新类)")
    print(f"     总有标签: {fused_analysis['total_labeled']} (聚类时可以利用的标签信息)")
    print(f"     总无标签: {fused_analysis['total_unlabeled']} (聚类时需要预测的样本)")

    # 1. 检查总样本数
    if fused_analysis['total_samples'] == expected_total:
        print(f"   ✅ 融合数据集大小正确: {expected_total}")
    else:
        print(f"   ❌ 融合数据集大小异常: {fused_analysis['total_samples']} (期望: {expected_total})")

    # 2. 检查已知类未知类分布
    if fused_analysis['unknown_labeled'] == 0:
        print(f"   ✅ 未知类全部无标签: {fused_analysis['unknown_unlabeled']}")
    else:
        print(f"   ❌ 未知类有标签样本异常: {fused_analysis['unknown_labeled']}")

    # 3. 检查有标签样本总数（应该只包括训练集的有标签部分，测试集全部无标签）
    expected_total_labeled = expected_train_known_labeled  # 只有训练集有标签部分
    expected_total_unlabeled = train_unlabelled_size + test_size  # 训练集无标签部分 + 测试集全部

    if abs(fused_analysis['total_labeled'] - expected_total_labeled) <= 50:  # 允许一些误差
        print(f"   ✅ 总有标签样本数正确: {fused_analysis['total_labeled']} ≈ {expected_total_labeled}")
    else:
        print(f"   ⚠️ 总有标签样本数: {fused_analysis['total_labeled']} (期望约: {expected_total_labeled})")

    if abs(fused_analysis['total_unlabeled'] - expected_total_unlabeled) <= 50:  # 允许一些误差
        print(f"   ✅ 总无标签样本数正确: {fused_analysis['total_unlabeled']} ≈ {expected_total_unlabeled}")
    else:
        print(f"   ⚠️ 总无标签样本数: {fused_analysis['total_unlabeled']} (期望约: {expected_total_unlabeled})")

    # 4. 检查批次格式
    if 4 in fused_analysis['batch_formats']:
        print(f"   ✅ 融合数据集包含labeled_or_not信息")
    else:
        print(f"   ❌ 融合数据集缺少labeled_or_not信息")

    print(f"   🎯 数据集已准备好用于自适应聚类算法!")
    print(f"   💡 聚类算法接收:")
    print(f"      - 特征: 768维特征向量")
    print(f"      - 已知/未知标识: 区分训练见过的类vs新类")
    print(f"      - 有标签/无标签标识: 区分有监督信息vs需要预测的样本")
    print(f"      - 真实标签仅用于最终评估ACC，聚类过程中不可见")


def validate_data_consistency(train_analysis, test_analysis, expected_prop_train_labels):
    """
    验证数据一致性
    """
    if train_analysis is None or test_analysis is None:
        print("   ❌ 无法验证：数据分析失败")
        return

    print(f"   检查项目:")

    # 1. 检查训练集有标签比例
    if train_analysis['known_samples'] > 0:
        actual_prop = train_analysis['known_labeled'] / train_analysis['known_samples']
        expected_range = (expected_prop_train_labels - 0.1, expected_prop_train_labels + 0.1)

        if expected_range[0] <= actual_prop <= expected_range[1]:
            print(f"   ✅ 训练集已知类有标签比例: {actual_prop:.3f} (期望: {expected_prop_train_labels})")
        else:
            print(f"   ❌ 训练集已知类有标签比例异常: {actual_prop:.3f} (期望: {expected_prop_train_labels})")

    # 2. 检查训练集未知类是否全部无标签
    if train_analysis['unknown_samples'] > 0:
        if train_analysis['unknown_labeled'] == 0:
            print(f"   ✅ 训练集未知类全部无标签: {train_analysis['unknown_unlabeled']}")
        else:
            print(f"   ❌ 训练集未知类有标签样本异常: {train_analysis['unknown_labeled']}")

    # 3. 检查测试集标签分布（应该全部无标签）
    test_total = test_analysis['total_samples']
    if test_total > 0:
        if test_analysis['total_unlabeled'] == test_total:
            print(f"   ✅ 测试集全部无标签: {test_total} (聚类时看不到真实标签)")
        elif test_analysis['total_labeled'] == test_total:
            print(f"   ❌ 测试集全部有标签: {test_total} (聚类时不应该看到标签)")
        else:
            print(f"   ⚠️ 测试集标签分布: 有标签{test_analysis['total_labeled']}, 无标签{test_analysis['total_unlabeled']}")

    # 4. 检查数据格式
    train_formats = train_analysis.get('batch_formats', set())
    test_formats = test_analysis.get('batch_formats', set())

    if 4 in train_formats:
        print(f"   ✅ 训练集包含labeled_or_not信息")
    else:
        print(f"   ⚠️ 训练集缺少labeled_or_not信息")

    if 4 in test_formats or 3 in test_formats:
        print(f"   ✅ 测试集格式正常")
    else:
        print(f"   ❌ 测试集格式异常")


def main():
    """主测试函数"""
    print("自适应聚类数据加载测试")
    print("=" * 80)

    # 测试不同超类
    superclasses_to_test = ['trees', 'flowers', 'mammals']

    for superclass in superclasses_to_test:
        try:
            print(f"\n{'='*20} 测试超类: {superclass} {'='*20}")
            train_analysis, test_analysis = test_superclass_data_loading(
                superclass_name=superclass,
                prop_train_labels=0.8
            )
            print(f"✓ {superclass} 测试完成")

        except Exception as e:
            print(f"✗ {superclass} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n数据加载测试完成！")
    print("下一步：基于此数据格式实现自适应聚类算法")


if __name__ == "__main__":
    main()
