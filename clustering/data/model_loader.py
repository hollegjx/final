#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型加载模块
处理模型的加载和初始化
注意：此模块仍需依赖models和config（无法完全避免）
"""

import os
import sys
import torch

# 添加项目路径以导入models和config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from models import vision_transformer as vits
    from config import dino_pretrain_path
except ImportError:
    print("⚠️  警告: 无法导入models或config模块")
    print("   模型加载功能需要项目根目录的models/和config.py")
    vits = None
    dino_pretrain_path = None


def set_deterministic_behavior():
    """
    设置确定性行为
    只设置torch相关的种子以保持模型加载和特征提取的一致性
    """
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    # 设置确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelLoader:
    """
    模型加载器类
    负责加载训练好的模型
    """

    def __init__(self, model_path, base_model='vit_dino', feat_dim=768, device='cuda'):
        """
        初始化模型加载器

        Args:
            model_path: 训练好的模型权重路径
            base_model: 基础模型类型（默认'vit_dino'）
            feat_dim: 特征维度（默认768）
            device: 计算设备（默认'cuda'）
        """
        self.model_path = model_path
        self.base_model = base_model
        self.feat_dim = feat_dim
        self.device = device if torch.cuda.is_available() else 'cpu'

    def load(self, silent=False):
        """
        加载模型

        Args:
            silent: 是否静默模式

        Returns:
            model: 加载好的PyTorch模型

        Raises:
            ImportError: 如果models或config模块不可用
            FileNotFoundError: 如果模型文件不存在
            NotImplementedError: 如果模型类型不支持
        """
        if vits is None:
            raise ImportError("models模块不可用，无法加载模型")

        if not silent:
            print(f"🔄 加载模型...")
            print(f"   模型文件: {self.model_path}")
            print(f"   设备: {self.device}")

        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 构建base model
        if self.base_model == 'vit_dino':
            model = vits.__dict__['vit_base']()

            # 加载DINO预训练权重（如果存在）
            if dino_pretrain_path and os.path.exists(dino_pretrain_path):
                if not silent:
                    print(f"   加载DINO预训练权重...")
                dino_state_dict = torch.load(dino_pretrain_path, map_location='cpu')
                model.load_state_dict(dino_state_dict, strict=False)

            # 加载训练权重
            if not silent:
                print(f"   加载训练权重...")
            gcd_state_dict = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(gcd_state_dict)

            model.to(self.device)

            # 关闭梯度计算（推理模式）
            for param in model.parameters():
                param.requires_grad = False

            if not silent:
                print(f"✅ 模型加载成功 (特征维度: {self.feat_dim})")

            return model

        else:
            raise NotImplementedError(f"不支持的模型类型: {self.base_model}")


def load_model_simple(model_path, device='cuda', silent=False):
    """
    简化的模型加载函数（向后兼容）

    Args:
        model_path: 模型权重路径
        device: 计算设备
        silent: 是否静默模式

    Returns:
        model: 加载好的模型
    """
    loader = ModelLoader(model_path=model_path, device=device)
    return loader.load(silent=silent)
