#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checkpoint管理工具模块
提供模型保存、超参数记录、旧文件清理等统一接口
"""

import os
import argparse
from datetime import datetime
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


def save_hyperparameters_to_txt(
    txt_path: str,
    args: argparse.Namespace,
    current_epoch: int,
    metrics: Dict[str, float],
    model_path: str
) -> None:
    """
    保存超参数到txt文件

    Args:
        txt_path: 超参数文件路径
        args: 训练参数命名空间
        current_epoch: 当前训练轮数
        metrics: 性能指标字典（如{'all_acc': 0.79, 'old_acc': 0.76}）
        model_path: 对应的模型文件路径
    """
    # 生成当前时间戳
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(txt_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write("=" * 80 + "\n")
        f.write("训练超参数记录\n")
        f.write("=" * 80 + "\n")
        f.write(f"模型文件: {os.path.basename(model_path)}\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"训练轮数: {current_epoch}/{getattr(args, 'epochs', 'N/A')}\n")
        f.write("\n")

        # 性能指标
        if metrics:
            f.write("性能指标:\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  - {key}: {value*100:.2f}%\n")
                else:
                    f.write(f"  - {key}: {value}\n")
            f.write("\n")

        # 数据集参数
        f.write("=" * 80 + "\n")
        f.write("数据集参数\n")
        f.write("=" * 80 + "\n")
        dataset_params = [
            'dataset_name', 'superclass_name', 'batch_size',
            'prop_train_labels', 'seed', 'num_workers'
        ]
        for param in dataset_params:
            if hasattr(args, param):
                value = getattr(args, param)
                f.write(f"{param:25s}: {value}\n")
        f.write("\n")

        # 模型参数
        f.write("=" * 80 + "\n")
        f.write("模型参数\n")
        f.write("=" * 80 + "\n")
        model_params = [
            'base_model', 'feat_dim', 'image_size',
            'mlp_out_dim', 'num_mlp_layers', 'grad_from_block'
        ]
        for param in model_params:
            if hasattr(args, param):
                value = getattr(args, param)
                f.write(f"{param:25s}: {value}\n")
        f.write("\n")

        # 训练参数
        f.write("=" * 80 + "\n")
        f.write("训练参数\n")
        f.write("=" * 80 + "\n")
        training_params = [
            'epochs', 'lr', 'weight_decay', 'momentum',
            'warmup_teacher_temp', 'teacher_temp', 'warmup_teacher_temp_epochs'
        ]
        for param in training_params:
            if hasattr(args, param):
                value = getattr(args, param)
                f.write(f"{param:25s}: {value}\n")
        f.write("\n")

        # 损失函数参数
        f.write("=" * 80 + "\n")
        f.write("损失函数参数\n")
        f.write("=" * 80 + "\n")
        loss_params = [
            'sup_weight', 'contrast_weight', 'temperature',
            'contrast_loss_weight', 'sup_con_weight'
        ]
        for param in loss_params:
            if hasattr(args, param):
                value = getattr(args, param)
                f.write(f"{param:25s}: {value}\n")
        f.write("\n")

        # 其他参数（过滤掉内部属性和已记录的参数）
        recorded_params = set(dataset_params + model_params + training_params + loss_params)
        other_params = []
        for attr in dir(args):
            if not attr.startswith('_') and attr not in recorded_params:
                value = getattr(args, attr)
                # 过滤掉方法和复杂对象
                if not callable(value) and not isinstance(value, (type, type(None))):
                    other_params.append((attr, value))

        if other_params:
            f.write("=" * 80 + "\n")
            f.write("其他参数\n")
            f.write("=" * 80 + "\n")
            for param, value in sorted(other_params):
                f.write(f"{param:25s}: {value}\n")
            f.write("\n")


def save_best_checkpoint_with_hyperparams(
    model: nn.Module,
    checkpoint_dir: str,
    filename: str,
    args: argparse.Namespace,
    current_epoch: int,
    metrics: Dict[str, float],
    old_checkpoint_path: Optional[str] = None
) -> Tuple[str, str]:
    """
    保存最优checkpoint并生成对应的超参数文件，同时删除旧文件

    Args:
        model: 要保存的模型
        checkpoint_dir: checkpoint保存目录
        filename: 新的文件名（如 "allacc_79_date_2025_11_13_16_10.pt"）
        args: 训练参数命名空间
        current_epoch: 当前训练轮数
        metrics: 性能指标字典
        old_checkpoint_path: 旧的checkpoint路径（用于删除）

    Returns:
        (model_path, txt_path): 新生成的模型文件路径和超参数文件路径

    Raises:
        RuntimeError: 如果模型保存失败
    """
    # 确保目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 构建新文件路径
    model_path = os.path.join(checkpoint_dir, filename)
    txt_filename = filename.replace('.pt', '.txt').replace('.pth', '.txt')
    txt_path = os.path.join(checkpoint_dir, txt_filename)

    try:
        # 第1步：保存模型
        print(f"💾 保存最优模型: {filename}")
        torch.save(model.state_dict(), model_path)

        # 第2步：保存超参数
        print(f"📝 生成超参数记录: {txt_filename}")
        save_hyperparameters_to_txt(
            txt_path=txt_path,
            args=args,
            current_epoch=current_epoch,
            metrics=metrics,
            model_path=model_path
        )

        # 第3步：删除旧文件（如果存在）
        if old_checkpoint_path and os.path.exists(old_checkpoint_path):
            try:
                # 删除旧的.pt文件
                print(f"🗑️  删除旧模型: {os.path.basename(old_checkpoint_path)}")
                os.remove(old_checkpoint_path)

                # 删除对应的.txt文件
                old_txt_path = old_checkpoint_path.replace('.pt', '.txt').replace('.pth', '.txt')
                if os.path.exists(old_txt_path):
                    print(f"🗑️  删除旧超参数记录: {os.path.basename(old_txt_path)}")
                    os.remove(old_txt_path)
            except OSError as e:
                # 删除失败不应中断训练，仅打印警告
                print(f"⚠️  删除旧文件失败（不影响训练）: {e}")

        print(f"✅ Checkpoint保存成功")
        return model_path, txt_path

    except Exception as e:
        # 保存失败时清理可能生成的不完整文件
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except:
                pass
        if os.path.exists(txt_path):
            try:
                os.remove(txt_path)
            except:
                pass
        raise RuntimeError(f"保存checkpoint失败: {e}")


def generate_checkpoint_filename(
    prefix: str,
    accuracy: float,
    timestamp: Optional[str] = None
) -> str:
    """
    生成标准化的checkpoint文件名

    Args:
        prefix: 文件名前缀（如 "allacc", "best_acc"）
        accuracy: 准确率（0-1之间的浮点数）
        timestamp: 时间戳字符串，如果为None则自动生成

    Returns:
        标准化的文件名（如 "allacc_79_date_2025_11_13_16_10.pt"）

    Examples:
        >>> generate_checkpoint_filename("allacc", 0.7920)
        'allacc_79_date_2025_11_13_16_10.pt'
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

    acc_int = int(accuracy * 100)
    filename = f"{prefix}_{acc_int}_date_{timestamp}.pt"
    return filename
