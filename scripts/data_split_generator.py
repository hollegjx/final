#!/usr/bin/env python3
"""
GCD项目超类数据划分生成器
基于DCCL项目的level_a_superclass_splitter.py，适配GCD项目结构
"""

import os
import json
import numpy as np
from collections import defaultdict, Counter
from torchvision import datasets
import datetime
import argparse
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入超类定义
from data.cifar100_superclass import CIFAR100_SUPERCLASSES, CLASS_TO_SUPERCLASS, SUPERCLASS_NAMES


def simple_train_test_split(indices, targets, test_size=0.5, random_state=42):
    """简单的分层划分函数，替代sklearn"""
    import random
    random.seed(random_state)

    # 按类别分组
    class_indices = {}
    for i, (idx, target) in enumerate(zip(indices, targets)):
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(idx)

    train_indices = []
    test_indices = []

    # 对每个类别按比例划分
    for target, cls_indices in class_indices.items():
        random.shuffle(cls_indices)
        n_test = int(len(cls_indices) * test_size)
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    return train_indices, test_indices


class GCDSuperclassSplitter:
    """GCD项目超类数据划分器"""

    def __init__(self, output_dir='./data_splits', cifar100_root='./data'):
        self.output_dir = output_dir
        self.cifar100_root = cifar100_root

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # GCD设定：前80个类为已知类，后20个类为未知类
        self.known_classes = list(range(80))
        self.unknown_classes = list(range(80, 100))

        print(f"🚀 GCD超类数据划分器初始化")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📊 已知类: {len(self.known_classes)}个 (0-79)")
        print(f"📊 未知类: {len(self.unknown_classes)}个 (80-99)")

    def load_cifar100_data(self):
        """加载CIFAR-100数据"""
        print("\n📂 加载CIFAR-100数据...")

        try:
            # 加载训练集和测试集
            cifar100_train = datasets.CIFAR100(root=self.cifar100_root, train=True, download=False)
            cifar100_test = datasets.CIFAR100(root=self.cifar100_root, train=False, download=False)
        except Exception as e:
            print(f"❌ 加载CIFAR-100数据失败: {e}")
            print("💡 尝试下载数据...")
            cifar100_train = datasets.CIFAR100(root=self.cifar100_root, train=True, download=True)
            cifar100_test = datasets.CIFAR100(root=self.cifar100_root, train=False, download=True)

        # 合并训练集和测试集的标签和索引 (全局索引)
        all_targets = []
        all_indices = []
        all_is_train = []

        # 训练集: 索引 0-49999
        for i, target in enumerate(cifar100_train.targets):
            all_targets.append(target)
            all_indices.append(i)
            all_is_train.append(True)

        # 测试集: 索引 50000-59999
        for i, target in enumerate(cifar100_test.targets):
            all_targets.append(target)
            all_indices.append(50000 + i)
            all_is_train.append(False)

        print(f"✅ 数据加载完成: 总共 {len(all_targets)} 个样本")
        return all_targets, all_indices, all_is_train

    def get_superclass_samples(self, all_targets, all_indices, all_is_train, superclass_id):
        """获取指定超类的所有样本"""
        superclass_name = SUPERCLASS_NAMES[superclass_id]

        # 获取当前超类包含的所有类别
        superclass_classes = CIFAR100_SUPERCLASSES[superclass_name]

        # 筛选当前超类的样本
        superclass_targets = []
        superclass_indices = []
        superclass_is_train = []

        for target, idx, is_train in zip(all_targets, all_indices, all_is_train):
            if target in superclass_classes:
                superclass_targets.append(target)
                superclass_indices.append(idx)
                superclass_is_train.append(is_train)

        # 分离已知类和未知类
        known_classes_in_superclass = [cls for cls in superclass_classes if cls in self.known_classes]
        unknown_classes_in_superclass = [cls for cls in superclass_classes if cls in self.unknown_classes]

        print(f"\n🎯 超类 {superclass_id}: {superclass_name}")
        print(f"  总样本数: {len(superclass_indices)}")
        print(f"  包含类别: {superclass_classes}")
        print(f"  已知类别: {known_classes_in_superclass} ({len(known_classes_in_superclass)}个)")
        print(f"  未知类别: {unknown_classes_in_superclass} ({len(unknown_classes_in_superclass)}个)")

        return {
            'superclass_id': superclass_id,
            'superclass_name': superclass_name,
            'all_classes': superclass_classes,
            'known_classes': known_classes_in_superclass,
            'unknown_classes': unknown_classes_in_superclass,
            'targets': superclass_targets,
            'indices': superclass_indices,
            'is_train': superclass_is_train
        }

    def create_superclass_split(self, superclass_data):
        """为单个超类创建GCD兼容的划分"""
        superclass_id = superclass_data['superclass_id']
        superclass_name = superclass_data['superclass_name']

        print(f"\n📊 为超类 {superclass_id} ({superclass_name}) 创建GCD划分...")

        # 分离训练集来源和测试集来源的样本
        train_set_known_indices = []  # 来自训练集的已知类样本
        train_set_unknown_indices = []  # 来自训练集的未知类样本
        test_set_indices = []  # 来自测试集的所有超类样本（用作最终测试集）

        for target, idx, is_train in zip(superclass_data['targets'],
                                       superclass_data['indices'],
                                       superclass_data['is_train']):
            if is_train:  # 来自CIFAR100训练集 (索引0-49999)
                if target in superclass_data['known_classes']:
                    train_set_known_indices.append(idx)
                elif target in superclass_data['unknown_classes']:
                    train_set_unknown_indices.append(idx)
            else:  # 来自CIFAR100测试集 (索引50000-59999)
                test_set_indices.append(idx)  # 测试集包含所有超类样本

        print(f"  训练集中已知类样本: {len(train_set_known_indices)}")
        print(f"  训练集中未知类样本: {len(train_set_unknown_indices)}")
        print(f"  测试集中超类样本: {len(test_set_indices)}")

        # 从训练集的已知类样本中划分训练/验证集
        if len(train_set_known_indices) > 0:
            # 加载训练集获取标签
            try:
                cifar100_train = datasets.CIFAR100(root=self.cifar100_root, train=True, download=False)
            except:
                cifar100_train = datasets.CIFAR100(root=self.cifar100_root, train=True, download=True)

            # 获取训练集已知类样本的标签用于分层划分
            train_known_targets = []
            for idx in train_set_known_indices:
                train_known_targets.append(cifar100_train.targets[idx])

            # 8:2划分为训练/验证集（与GCD标准一致）
            train_indices, val_indices = simple_train_test_split(
                train_set_known_indices,
                train_known_targets,
                test_size=0.2,  # 20%验证集
                random_state=42
            )
        else:
            train_indices, val_indices = [], []

        # 测试集使用CIFAR100原始测试集中的超类样本
        test_indices = test_set_indices

        # 为GCD兼容性保留这些字段
        labeled_indices = train_indices + val_indices  # 所有有标签的训练数据
        unlabeled_indices = train_set_unknown_indices  # 训练集中的未知类样本
        unknown_indices = test_set_indices  # 测试集中的所有样本

        # 创建详细的划分结果
        split_result = {
            'superclass_info': {
                'superclass_id': superclass_id,
                'superclass_name': superclass_name,
                'all_classes': superclass_data['all_classes'],
                'known_classes': superclass_data['known_classes'],
                'unknown_classes': superclass_data['unknown_classes']
            },
            'split_statistics': {
                'train_samples': len(train_indices),
                'val_samples': len(val_indices),
                'test_samples': len(test_indices),
                'labeled_samples': len(labeled_indices),
                'unlabeled_samples': len(unlabeled_indices),
                'unknown_samples': len(unknown_indices),
                'total_samples': len(superclass_data['indices'])
            },
            'data_splits': {
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'labeled_indices': labeled_indices,
                'unlabeled_indices': unlabeled_indices,
                'unknown_indices': unknown_indices
            }
        }

        print(f"  🎯 最终划分 (GCD兼容):")
        print(f"    训练集: {len(train_indices)} 样本 (来自CIFAR100训练集的已知类)")
        print(f"    验证集: {len(val_indices)} 样本 (来自CIFAR100训练集的已知类)")
        print(f"    测试集: {len(test_indices)} 样本 (来自CIFAR100原始测试集的所有超类样本)")
        print(f"    无标签: {len(unlabeled_indices)} 样本 (来自CIFAR100训练集的未知类)")

        return split_result

    def save_superclass_split(self, split_result):
        """保存单个超类的划分结果"""
        superclass_id = split_result['superclass_info']['superclass_id']
        superclass_name = split_result['superclass_info']['superclass_name']

        # 为每个超类创建独立的JSON文件
        output_file = os.path.join(self.output_dir, f'superclass_{superclass_id:02d}_{superclass_name}.json')

        # 确保所有数据都是JSON可序列化的
        json_data = {
            'superclass_info': split_result['superclass_info'],
            'split_statistics': split_result['split_statistics'],
            'data_splits': {
                'train_indices': [int(idx) for idx in split_result['data_splits']['train_indices']],
                'val_indices': [int(idx) for idx in split_result['data_splits']['val_indices']],
                'test_indices': [int(idx) for idx in split_result['data_splits']['test_indices']],
                'labeled_indices': [int(idx) for idx in split_result['data_splits']['labeled_indices']],
                'unlabeled_indices': [int(idx) for idx in split_result['data_splits']['unlabeled_indices']],
                'unknown_indices': [int(idx) for idx in split_result['data_splits']['unknown_indices']]
            },
            'metadata': {
                'split_version': 'gcd_superclass_v1.0',
                'gcd_config': 'GCD-compatible splits for superclass training',
                'train_val_ratio': '8:2 for labeled samples',
                'random_seed': 42,
                'created_time': datetime.datetime.now().isoformat(),
                'dataset': 'CIFAR-100',
                'framework': 'Generalized Category Discovery'
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"💾 超类 {superclass_id} 划分结果已保存: {output_file}")
        return output_file

    def create_all_superclass_splits(self):
        """为所有15个超类创建划分"""
        print("🚀 开始为所有超类创建GCD兼容划分")
        print("=" * 80)

        # 1. 加载数据
        all_targets, all_indices, all_is_train = self.load_cifar100_data()

        # 2. 为每个超类创建划分
        saved_files = []
        summary_stats = []

        for superclass_id in range(15):  # 0-14
            # 获取超类样本
            superclass_data = self.get_superclass_samples(all_targets, all_indices, all_is_train, superclass_id)

            # 创建划分
            split_result = self.create_superclass_split(superclass_data)

            # 保存结果
            output_file = self.save_superclass_split(split_result)
            saved_files.append(output_file)

            # 收集统计信息
            summary_stats.append(split_result['split_statistics'])

        # 3. 创建总结文件
        self.create_summary_file(summary_stats, saved_files)

        print(f"\n✅ 所有15个超类的GCD兼容划分创建完成!")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📊 生成文件数: {len(saved_files)}")

        return saved_files

    def create_summary_file(self, summary_stats, saved_files):
        """创建总结文件"""
        summary = {
            'gcd_superclass_splits_summary': {
                'total_superclasses': 15,
                'superclass_names': SUPERCLASS_NAMES,
                'gcd_config': {
                    'known_classes': self.known_classes,
                    'unknown_classes': self.unknown_classes,
                    'labeled_ratio': 0.8,
                    'train_val_ratio': 0.8
                },
                'files_created': [os.path.basename(f) for f in saved_files],
                'statistics_by_superclass': []
            }
        }

        total_train = total_val = total_test = 0

        for i, stats in enumerate(summary_stats):
            summary['gcd_superclass_splits_summary']['statistics_by_superclass'].append({
                'superclass_id': i,
                'superclass_name': SUPERCLASS_NAMES[i],
                'train_samples': stats['train_samples'],
                'val_samples': stats['val_samples'],
                'test_samples': stats['test_samples'],
                'total_samples': stats['total_samples']
            })

            total_train += stats['train_samples']
            total_val += stats['val_samples']
            total_test += stats['test_samples']

        summary['gcd_superclass_splits_summary']['total_statistics'] = {
            'total_train_samples': total_train,
            'total_val_samples': total_val,
            'total_test_samples': total_test,
            'grand_total': total_train + total_val + total_test
        }

        summary_file = os.path.join(self.output_dir, 'gcd_superclass_splits_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📋 总结文件已保存: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GCD项目超类数据划分生成器')
    parser.add_argument('--output_dir', type=str, default='./data_splits',
                        help='输出目录路径')
    parser.add_argument('--cifar100_root', type=str, default='./data',
                        help='CIFAR-100数据集根目录')

    args = parser.parse_args()

    print("🎯 GCD项目超类数据划分生成器")
    print("=" * 80)

    # 创建划分器
    splitter = GCDSuperclassSplitter(
        output_dir=args.output_dir,
        cifar100_root=args.cifar100_root
    )

    # 创建所有超类的划分
    saved_files = splitter.create_all_superclass_splits()

    print(f"\n🎉 划分完成!")
    print(f"📁 所有文件保存在: {splitter.output_dir}")
    print(f"📊 15个超类的数据划分已生成，可用于GCD训练")

    return saved_files


if __name__ == "__main__":
    main()