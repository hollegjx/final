"""
CIFAR-100超类数据集实现 - 为GCD项目定制版本
基于DCCL项目的15个超类划分方案，适配GCD项目的数据结构
"""

import json
import numpy as np
import os
from torchvision.datasets import CIFAR100
from copy import deepcopy
from data.data_utils import subsample_instances
from config import cifar_100_root

# CIFAR-100 15个超类映射（与DCCL项目保持一致）
CIFAR100_SUPERCLASSES = {
    'trees': [47, 52, 56, 59, 96],
    'flowers': [54, 62, 70, 82, 92],
    'fruits_vegetables': [0, 51, 53, 57, 83],
    'mammals': [3, 4, 21, 31, 34, 36, 38, 43, 50, 55, 63, 64, 65, 66, 72, 74, 75, 15, 19, 42, 80, 88, 97],
    'marine_animals': [1, 26, 27, 30, 32, 45, 67, 73, 95, 91],
    'insects_arthropods': [6, 7, 14, 24, 77, 18, 79, 99],
    'reptiles': [44, 78, 29, 93],
    'humans': [11, 35, 46, 2, 98],
    'furniture': [5, 20, 25, 84, 94],
    'containers': [16, 9, 10, 28, 61],
    'vehicles': [8, 13, 48, 58, 69, 81, 85, 89, 90],
    'electronic_devices': [22, 39, 40, 41, 86, 87],
    'buildings': [12, 17, 37, 76],
    'terrain': [33, 49, 60, 68, 71],
    'weather_phenomena': [23]
}

# 创建类别到超类的映射
CLASS_TO_SUPERCLASS = {}
SUPERCLASS_NAMES = list(CIFAR100_SUPERCLASSES.keys())
for superclass_id, (superclass_name, class_list) in enumerate(CIFAR100_SUPERCLASSES.items()):
    for class_id in class_list:
        CLASS_TO_SUPERCLASS[class_id] = superclass_id


class CustomCIFAR100Superclass(CIFAR100):
    """
    支持超类划分的自定义CIFAR-100数据集
    与GCD项目的CustomCIFAR100保持兼容
    """

    def __init__(self, *args, target_transform=None, **kwargs):
        self.verbose = kwargs.pop('verbose', True)
        super(CustomCIFAR100Superclass, self).__init__(*args, **kwargs)

        self.original_target_transform = target_transform
        self.uq_idxs = np.array(range(len(self)))

        # 标签映射字典（用于超类内部训练时的连续标签映射）
        self.label_mapping = None
        self.reverse_label_mapping = None

        # 标记这是一个超类数据集，不应该被外部的target_transform覆盖
        self.is_superclass_dataset = True

    @property
    def target_transform(self):
        """获取target_transform"""
        return self._target_transform if hasattr(self, '_target_transform') else None

    @target_transform.setter
    def target_transform(self, value):
        """设置target_transform，但如果已有label_mapping则忽略外部设置"""
        if hasattr(self, 'label_mapping') and self.label_mapping is not None:
            # 如果已经有了标签映射，忽略外部的target_transform设置
            if self.verbose:
                print(f"忽略外部target_transform设置，使用内部标签映射")
            return
        self._target_transform = value

    def get_superclass_label(self, class_id):
        """获取类别对应的超类标签"""
        return CLASS_TO_SUPERCLASS.get(class_id, -1)

    def create_label_mapping(self, class_list):
        """
        为指定的类别列表创建连续标签映射
        Args:
            class_list: 原始类别ID列表，如 [47, 52, 56, 59]
        """
        # 排序确保映射一致性
        sorted_classes = sorted(class_list)

        # 创建原始标签到连续标签的映射
        self.label_mapping = {original_class: idx for idx, original_class in enumerate(sorted_classes)}

        # 创建反向映射（连续标签到原始标签）
        self.reverse_label_mapping = {idx: original_class for idx, original_class in enumerate(sorted_classes)}

        if self.verbose:
            print(f"   创建标签映射: {self.label_mapping}")
        return self.label_mapping

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        # 应用标签重新映射（如果存在）
        if self.label_mapping is not None and label in self.label_mapping:
            label = self.label_mapping[label]
            # 如果有自己的标签映射，就不要再应用外部的target_transform了
        else:
            # 只有在没有自己的标签映射时才应用target_transform
            if self.target_transform is not None:
                try:
                    label = self.target_transform(label)
                except KeyError as e:
                    print(f"KeyError: 标签 {label} 不在target_transform映射中")
                    print(f"可用的映射键: {list(self.target_transform.__closure__[0].cell_contents.keys()) if hasattr(self.target_transform, '__closure__') else 'N/A'}")
                    raise e
            elif self.original_target_transform is not None:
                label = self.original_target_transform(label)

        # 返回格式与GCD项目一致：(img, label, uq_idx)
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_classes_superclass(dataset, include_classes):
    """
    按指定类别筛选数据集，适配GCD项目结构
    """
    include_classes = np.array(include_classes)

    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    # 找到属于指定类别的样本索引
    class_mask = np.isin(targets, include_classes)
    indices = np.where(class_mask)[0]

    # 创建新的数据集
    new_dataset = deepcopy(dataset)

    if hasattr(new_dataset, 'data'):
        new_dataset.data = dataset.data[indices]
    if hasattr(new_dataset, 'targets'):
        new_dataset.targets = [dataset.targets[i] for i in indices]
    if hasattr(new_dataset, 'uq_idxs'):
        new_dataset.uq_idxs = dataset.uq_idxs[indices]

    return new_dataset


def subsample_dataset_superclass(dataset, idxs):
    """
    按索引子采样数据集，适配GCD项目结构
    """
    if len(idxs) > 0:
        if hasattr(dataset, 'data'):
            dataset.data = dataset.data[idxs]
        if hasattr(dataset, 'targets'):
            dataset.targets = np.array(dataset.targets)[idxs].tolist()
        if hasattr(dataset, 'uq_idxs'):
            dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def get_train_val_indices_superclass(train_dataset, val_split=0.2):
    """
    获取训练验证集索引，适配超类数据集
    """
    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cifar100_superclass_datasets(train_transform, test_transform, train_classes,
                                   prop_train_labels=0.8, split_train_val=False, seed=0,
                                   verbose=True):
    """
    获取基于超类的CIFAR-100数据集，与GCD项目的接口保持一致

    Args:
        train_transform: 训练数据变换
        test_transform: 测试数据变换
        train_classes: 训练类别列表（已知类）
        prop_train_labels: 有标签样本比例
        split_train_val: 是否分离训练验证集
        seed: 随机种子
        verbose: 是否打印数据加载信息
    """
    np.random.seed(seed)

    # 使用定制的CIFAR-100超类数据集
    whole_training_set = CustomCIFAR100Superclass(
        root=cifar_100_root, transform=train_transform, train=True, verbose=verbose
    )

    test_dataset = CustomCIFAR100Superclass(
        root=cifar_100_root, transform=test_transform, train=False, verbose=verbose
    )

    # 标准的类别筛选流程（与GCD项目一致）
    train_dataset_labelled = subsample_classes_superclass(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset_superclass(train_dataset_labelled, subsample_indices)

    # 获取无标签数据
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset_superclass(
        deepcopy(whole_training_set), np.array(list(unlabelled_indices))
    )

    # 训练验证集分离（如果需要）
    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices_superclass(train_dataset_labelled)
        train_dataset_labelled_split = subsample_dataset_superclass(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset_superclass(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

        train_dataset_labelled = train_dataset_labelled_split
        val_dataset_labelled = val_dataset_labelled_split
    else:
        val_dataset_labelled = None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset
    }

    return all_datasets


def get_superclass_splits():
    """
    获取15个超类的已知/未知类划分
    返回每个超类中的已知类和未知类列表
    """
    superclass_splits = {}

    for superclass_name, class_list in CIFAR100_SUPERCLASSES.items():
        # 按照GCD设定：前80个为已知类，后20个为未知类
        known_classes = [cls for cls in class_list if cls < 80]
        unknown_classes = [cls for cls in class_list if cls >= 80]

        superclass_splits[superclass_name] = {
            'known_classes': known_classes,
            'unknown_classes': unknown_classes,
            'superclass_id': SUPERCLASS_NAMES.index(superclass_name)
        }

    return superclass_splits


def get_single_superclass_datasets(superclass_name, train_transform, test_transform,
                                 prop_train_labels=0.8, split_train_val=False, seed=0,
                                 verbose: bool = True):
    """
    获取单个超类的数据集，用于超类内部的GCD训练

    Args:
        superclass_name: 超类名称，如 'trees', 'flowers' 等
        其他参数同 get_cifar100_superclass_datasets

    Returns:
        包含该超类所有类别的数据集，按GCD设置划分已知/未知类
    """
    if superclass_name not in CIFAR100_SUPERCLASSES:
        raise ValueError(f"未知超类名称: {superclass_name}")

    # 获取该超类包含的所有类别
    superclass_classes = CIFAR100_SUPERCLASSES[superclass_name]

    # 按GCD设定划分已知类和未知类
    known_classes = [cls for cls in superclass_classes if cls < 80]
    unknown_classes = [cls for cls in superclass_classes if cls >= 80]

    if verbose:
        print(f"超类 '{superclass_name}' 包含类别: {superclass_classes}")
        print(f"已知类 ({len(known_classes)}): {known_classes}")
        print(f"未知类 ({len(unknown_classes)}): {unknown_classes}")

    # 如果该超类没有已知类或未知类，给出警告
    if len(known_classes) == 0 and verbose:
        print(f"警告：超类 '{superclass_name}' 没有已知类（类别ID < 80）")
    if len(unknown_classes) == 0 and verbose:
        print(f"警告：超类 '{superclass_name}' 没有未知类（类别ID >= 80）")

    # 使用标准的CIFAR-100超类数据获取函数
    all_classes = known_classes + unknown_classes
    datasets = get_cifar100_superclass_datasets(
        train_transform=train_transform,
        test_transform=test_transform,
        train_classes=known_classes,  # 只有已知类作为训练类
        prop_train_labels=prop_train_labels,
        split_train_val=split_train_val,
        seed=seed,
        verbose=verbose
    )

    # 过滤数据集，只保留该超类的样本，并创建标签映射
    filtered_datasets = {}
    for split_name, dataset in datasets.items():
        if dataset is not None:
            if verbose:
                print(f"\n📊 处理{split_name}数据集:")
            # 过滤样本，只保留属于当前超类的样本
            filtered_dataset = filter_dataset_by_classes(dataset, all_classes, split_name, verbose=verbose)

            # 为过滤后的数据集创建连续标签映射
            if hasattr(filtered_dataset, 'create_label_mapping'):
                filtered_dataset.create_label_mapping(all_classes)

            filtered_datasets[split_name] = filtered_dataset
        else:
            filtered_datasets[split_name] = None

    return filtered_datasets


def filter_dataset_by_classes(dataset, target_classes, split_name="数据集", verbose: bool = True):
    """
    过滤数据集，只保留指定类别的样本
    """
    target_classes = set(target_classes)

    # 获取所有样本的标签
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    else:
        all_labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # 找到属于目标类别的样本索引
    valid_mask = np.isin(all_labels, list(target_classes))
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        if verbose:
            print(f"   ⚠️ 警告: {split_name}中没有找到属于类别 {target_classes} 的样本")
        return dataset

    # 创建过滤后的数据集
    filtered_dataset = deepcopy(dataset)

    if hasattr(filtered_dataset, 'data'):
        filtered_dataset.data = dataset.data[valid_indices]
    if hasattr(filtered_dataset, 'targets'):
        filtered_dataset.targets = [dataset.targets[i] for i in valid_indices]
    if hasattr(filtered_dataset, 'uq_idxs'):
        filtered_dataset.uq_idxs = dataset.uq_idxs[valid_indices]

    if verbose:
        print(f"   过滤后{split_name}大小: {len(filtered_dataset)} (原始: {len(dataset)})")

    return filtered_dataset


# 为了与GCD项目的get_datasets.py集成，添加以下函数
def subsample_classes(dataset, include_classes):
    """
    与GCD项目中的subsample_classes保持兼容的接口
    """
    return subsample_classes_superclass(dataset, include_classes)


if __name__ == '__main__':
    # 测试超类数据集功能
    from torchvision import transforms

    # 简单的变换
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.ToTensor()

    # 测试单个超类数据集
    superclass_name = 'trees'
    datasets = get_single_superclass_datasets(
        superclass_name=superclass_name,
        train_transform=train_transform,
        test_transform=test_transform,
        prop_train_labels=0.8,
        split_train_val=False,
        seed=0
    )

    print(f'\n超类 "{superclass_name}" 数据集统计:')
    for k, v in datasets.items():
        if v is not None:
            print(f'{k}: {len(v)} 样本')

    # 测试所有超类的划分
    print(f'\n所有15个超类的划分:')
    superclass_splits = get_superclass_splits()
    for name, split_info in superclass_splits.items():
        print(f'{name} (ID: {split_info["superclass_id"]}): '
              f'{len(split_info["known_classes"])} 已知类, '
              f'{len(split_info["unknown_classes"])} 未知类')
