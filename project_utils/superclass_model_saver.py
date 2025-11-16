#!/usr/bin/env python3
"""
超类模型保存管理器
专门处理超类训练过程中的模型保存逻辑，避免影响其他训练文件

保存规则：
1. 模型保存在 /data1/jiangzhen/gjx/checkpoints/gcdsuperclass/{superclass_name}/ 目录下
2. 文件名格式：allacc_{整数ACC}_date_{YYYY_M_D_H_M}.pt
3. 只保存当前训练过程中的最佳模型，删除之前的保存
4. 同步生成超参数记录文件（.txt格式）
"""

import os
import torch
import glob
from datetime import datetime
from typing import Optional, Tuple
from config import superclass_model_root


# 导入超参数保存工具
try:
    from utils.checkpoint import save_hyperparameters_to_txt
except ImportError:
    # 如果导入失败，提供一个空实现（向后兼容）
    def save_hyperparameters_to_txt(*args, **kwargs):
        pass


class SuperclassModelSaver:
    """超类模型保存管理器"""

    def __init__(self, superclass_name: str, args=None):
        """
        初始化超类模型保存器

        Args:
            superclass_name: 超类名称，如 'trees'
            args: 训练参数（argparse.Namespace），用于生成超参数记录
        """
        self.superclass_name = superclass_name
        self.args = args  # 保存训练参数引用
        self.save_dir = os.path.join(superclass_model_root, superclass_name)
        self.best_acc = -1.0
        self.current_best_model_path = None
        self.current_best_proj_path = None

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"🗂️  超类模型保存器初始化: {superclass_name}")
        print(f"   保存目录: {self.save_dir}")

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳字符串，格式：YYYY_M_D_H_M"""
        now = datetime.now()
        return f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}"

    def _generate_model_filename(self, acc: float) -> str:
        """
        生成模型文件名

        Args:
            acc: 准确率（浮点数）

        Returns:
            文件名字符串，如 'allacc_80_date_2025_9_21_14_00.pt'
        """
        acc_int = int(round(acc * 100))  # 转换为整数百分比
        timestamp = self._get_current_timestamp()
        return f"allacc_{acc_int}_date_{timestamp}.pt"

    def _remove_previous_best_models(self):
        """删除之前保存的最佳模型和对应的超参数文件"""
        if self.current_best_model_path and os.path.exists(self.current_best_model_path):
            try:
                os.remove(self.current_best_model_path)
                print(f"🗑️  删除之前的最佳模型: {os.path.basename(self.current_best_model_path)}")

                # 删除对应的超参数文件
                txt_path = self.current_best_model_path.replace('.pt', '.txt')
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    print(f"🗑️  删除对应的超参数记录: {os.path.basename(txt_path)}")
            except OSError as e:
                print(f"⚠️  删除旧文件时出错（不影响训练）: {e}")

        if self.current_best_proj_path and os.path.exists(self.current_best_proj_path):
            try:
                os.remove(self.current_best_proj_path)
                print(f"🗑️  删除之前的投影头: {os.path.basename(self.current_best_proj_path)}")
            except OSError as e:
                print(f"⚠️  删除旧投影头时出错（不影响训练）: {e}")

    def save_best_model(self, model, projection_head, acc: float, metadata: Optional[dict] = None, current_epoch: int = 0) -> Tuple[str, str]:
        """
        保存当前最佳模型并生成超参数记录

        Args:
            model: 主模型
            projection_head: 投影头模型
            acc: 当前准确率
            metadata: 附加的训练信息（可选），包含性能指标
            current_epoch: 当前训练轮数

        Returns:
            tuple: (主模型路径, 投影头路径)
        """
        if acc <= self.best_acc:
            print(f"⏭️  当前ACC {acc:.4f} 未超过最佳 {self.best_acc:.4f}，跳过保存")
            return self.current_best_model_path, self.current_best_proj_path

        # 删除之前的最佳模型
        self._remove_previous_best_models()

        # 生成新的文件名
        model_filename = self._generate_model_filename(acc)
        proj_filename = model_filename.replace('.pt', '_proj_head.pt')

        model_path = os.path.join(self.save_dir, model_filename)
        proj_path = os.path.join(self.save_dir, proj_filename)

        try:
            # 保存模型
            torch.save(model.state_dict(), model_path)
            torch.save(projection_head.state_dict(), proj_path)

            # 【新功能】保存超参数记录
            if self.args is not None:
                txt_filename = model_filename.replace('.pt', '.txt')
                txt_path = os.path.join(self.save_dir, txt_filename)

                # 准备性能指标字典
                metrics = {'all_acc': acc}
                if metadata:
                    metrics.update(metadata)

                try:
                    save_hyperparameters_to_txt(
                        txt_path=txt_path,
                        args=self.args,
                        current_epoch=current_epoch,
                        metrics=metrics,
                        model_path=model_path
                    )
                    print(f"📝 生成超参数记录: {txt_filename}")
                except Exception as e:
                    print(f"⚠️  生成超参数记录失败（不影响模型保存）: {e}")

            # 更新记录
            self.best_acc = acc
            self.current_best_model_path = model_path
            self.current_best_proj_path = proj_path

            print(f"💾 保存新的最佳模型:")
            print(f"   ACC: {acc:.4f} -> 主模型: {model_filename}")
            print(f"   投影头: {proj_filename}")

            return model_path, proj_path

        except Exception as e:
            # 保存失败时清理可能生成的不完整文件
            for path in [model_path, proj_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            raise RuntimeError(f"保存模型失败: {e}")

    def get_best_model_info(self) -> dict:
        """
        获取当前最佳模型信息

        Returns:
            dict: 包含最佳模型信息的字典
        """
        return {
            'superclass_name': self.superclass_name,
            'best_acc': self.best_acc,
            'model_path': self.current_best_model_path,
            'proj_path': self.current_best_proj_path,
            'save_dir': self.save_dir
        }

    def list_saved_models(self) -> list:
        """
        列出当前超类保存的所有模型

        Returns:
            list: 模型文件列表
        """
        pattern = os.path.join(self.save_dir, "allacc_*.pt")
        model_files = glob.glob(pattern)
        return sorted(model_files)

    def cleanup_old_models(self, keep_latest: int = 1):
        """
        清理旧模型，只保留最新的几个

        Args:
            keep_latest: 保留最新的模型数量
        """
        model_files = self.list_saved_models()
        if len(model_files) > keep_latest:
            files_to_remove = model_files[:-keep_latest]
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # 同时删除对应的投影头文件
                    proj_file = file_path.replace('.pt', '_proj_head.pt')
                    if os.path.exists(proj_file):
                        os.remove(proj_file)
                    print(f"🧹 清理旧模型: {os.path.basename(file_path)}")


def create_superclass_model_saver(superclass_name: str, args=None) -> SuperclassModelSaver:
    """
    创建超类模型保存器的工厂函数

    Args:
        superclass_name: 超类名称
        args: 训练参数（argparse.Namespace），用于生成超参数记录

    Returns:
        SuperclassModelSaver: 配置好的模型保存器
    """
    return SuperclassModelSaver(superclass_name, args=args)


# 使用示例
if __name__ == "__main__":
    # 测试用例
    saver = create_superclass_model_saver("trees")
    print("超类模型保存器创建成功!")
    print(f"保存目录: {saver.save_dir}")

    # 模拟保存
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

    model = DummyModel()
    proj_head = DummyModel()

    # 模拟几次保存
    accs = [0.75, 0.78, 0.82, 0.79, 0.85]
    for acc in accs:
        print(f"\n--- 尝试保存 ACC: {acc:.4f} ---")
        model_path, proj_path = saver.save_best_model(model, proj_head, acc)

    print(f"\n最终最佳模型信息:")
    info = saver.get_best_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
