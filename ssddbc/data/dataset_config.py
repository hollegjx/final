#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集配置模块
独立定义数据集相关配置，不依赖外部训练代码
"""

# CIFAR-100超类配置（使用自定义的超类划分，与data/cifar100_superclass.py保持一致）
CIFAR100_SUPERCLASS_CONFIG = {
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


def get_superclass_info(superclass_name):
    """
    获取指定超类的详细信息

    Args:
        superclass_name: 超类名称

    Returns:
        info_dict: 包含超类信息的字典
            - 'name': 超类名称
            - 'classes': 该超类包含的原始CIFAR-100类别ID列表
            - 'known_classes': 已知类别（原始ID < 80）
            - 'unknown_classes': 未知类别（原始ID >= 80）
            - 'label_mapping': 原始类别ID到映射后ID的字典
            - 'known_classes_mapped': 映射后的已知类别ID
            - 'unknown_classes_mapped': 映射后的未知类别ID
    """
    if superclass_name not in CIFAR100_SUPERCLASS_CONFIG:
        raise ValueError(f"未知的超类名称: {superclass_name}")

    # 获取该超类包含的所有类别
    superclass_classes = CIFAR100_SUPERCLASS_CONFIG[superclass_name]

    # 区分已知类和未知类（CIFAR-100标准：前80个类别为已知类）
    known_classes_orig = [cls for cls in superclass_classes if cls < 80]
    unknown_classes_orig = [cls for cls in superclass_classes if cls >= 80]

    # 创建标签映射（将原始类别ID映射到0-N）
    all_classes_sorted = sorted(superclass_classes)
    label_mapping = {orig_cls: new_cls for new_cls, orig_cls in enumerate(all_classes_sorted)}

    # 映射后的已知/未知类别ID
    known_classes_mapped = [label_mapping[cls] for cls in known_classes_orig]
    unknown_classes_mapped = [label_mapping[cls] for cls in unknown_classes_orig]

    return {
        'name': superclass_name,
        'classes': superclass_classes,
        'known_classes': known_classes_orig,
        'unknown_classes': unknown_classes_orig,
        'label_mapping': label_mapping,
        'known_classes_mapped': set(known_classes_mapped),
        'unknown_classes_mapped': set(unknown_classes_mapped)
    }


def get_all_superclass_names():
    """
    获取所有超类的名称列表

    Returns:
        list: 所有超类名称
    """
    return list(CIFAR100_SUPERCLASS_CONFIG.keys())


def validate_superclass_name(superclass_name):
    """
    验证超类名称是否有效

    Args:
        superclass_name: 超类名称

    Returns:
        bool: 是否有效
    """
    return superclass_name in CIFAR100_SUPERCLASS_CONFIG
