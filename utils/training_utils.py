#!/usr/bin/env python3
"""
训练工具模块
提供训练过程中的辅助功能：时间显示、性能监控、早停机制等
"""

import time
import datetime
from typing import Any, Dict, Optional, Tuple


class EarlyStoppingMonitor:
    """早停机制监控器"""

    def __init__(self, patience: int = 20, min_delta: float = 0.0001,
                 metric_name: str = "all_acc", verbose: bool = True):
        """
        Args:
            patience: 最大容忍轮数，默认20轮
            min_delta: 最小改善阈值
            metric_name: 监控的指标名称
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.verbose = verbose

        self.best_metric = -float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

        if self.verbose:
            print(f"🚀 启用早停机制: 监控{metric_name}, 容忍{patience}轮无改善")

    def update(self, current_metric: float, current_epoch: int) -> bool:
        """
        更新监控状态

        Args:
            current_metric: 当前轮次的指标值
            current_epoch: 当前轮次

        Returns:
            bool: 是否应该停止训练
        """
        # 检查是否有改善
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.best_epoch = current_epoch
            self.counter = 0
            if self.verbose:
                print(f"🎯 {self.metric_name}新最佳: {current_metric:.4f} (第{current_epoch}轮)")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏰ {self.metric_name}无改善: {self.counter}/{self.patience} 轮")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"🛑 早停触发: {self.metric_name}已{self.patience}轮无改善")

        return self.should_stop

    def get_best_info(self) -> Tuple[float, int]:
        """获取最佳指标信息"""
        return self.best_metric, self.best_epoch

    def state_dict(self) -> Dict[str, Any]:
        """导出当前早停监控器状态，便于断点续训。"""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "metric_name": self.metric_name,
            "verbose": self.verbose,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从字典恢复早停监控器状态。"""
        if not state:
            return
        self.best_metric = state.get("best_metric", self.best_metric)
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        self.counter = state.get("counter", self.counter)
        self.should_stop = state.get("should_stop", self.should_stop)


class PerformanceTracker:
    """性能跟踪器"""

    def __init__(self):
        self.best_all_acc = -float('inf')
        self.best_old_acc = -float('inf')
        self.best_new_acc = -float('inf')
        self.best_epoch = 0
        self.start_time = time.time()

    def update(self, all_acc: float, old_acc: float, new_acc: float, epoch: int):
        """更新性能记录"""
        if all_acc > self.best_all_acc:
            self.best_all_acc = all_acc
            self.best_old_acc = old_acc
            self.best_new_acc = new_acc
            self.best_epoch = epoch

    def get_performance_gap(self, current_all_acc: float, current_old_acc: float, current_new_acc: float) -> Dict[str, float]:
        """计算与最佳性能的差距"""
        return {
            'all_acc_gap': self.best_all_acc - current_all_acc,
            'old_acc_gap': self.best_old_acc - current_old_acc,
            'new_acc_gap': self.best_new_acc - current_new_acc
        }

    def get_elapsed_time(self) -> str:
        """获取已用时间"""
        elapsed = time.time() - self.start_time
        return str(datetime.timedelta(seconds=int(elapsed)))

    def get_best_performance(self) -> Dict[str, float]:
        """获取最佳性能"""
        return {
            'best_all_acc': self.best_all_acc,
            'best_old_acc': self.best_old_acc,
            'best_new_acc': self.best_new_acc,
            'best_epoch': self.best_epoch
        }

    def state_dict(self) -> Dict[str, Any]:
        """导出性能跟踪器状态，用于断点续训。"""
        elapsed = time.time() - self.start_time
        return {
            "best_all_acc": self.best_all_acc,
            "best_old_acc": self.best_old_acc,
            "best_new_acc": self.best_new_acc,
            "best_epoch": self.best_epoch,
            # 保存已用时间，恢复时通过偏移量近似还原
            "elapsed_seconds": elapsed,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从字典恢复性能跟踪器状态。"""
        if not state:
            return
        self.best_all_acc = state.get("best_all_acc", self.best_all_acc)
        self.best_old_acc = state.get("best_old_acc", self.best_old_acc)
        self.best_new_acc = state.get("best_new_acc", self.best_new_acc)
        self.best_epoch = state.get("best_epoch", self.best_epoch)

        elapsed = state.get("elapsed_seconds", None)
        if isinstance(elapsed, (int, float)) and elapsed >= 0:
            # 重新设置 start_time，使得 get_elapsed_time() 近似连续
            self.start_time = time.time() - float(elapsed)
        


def print_epoch_separator(epoch: int, total_epochs: int, elapsed_time: str):
    """打印轮次分隔符"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print("\n" + "=" * 80)
    print(f"🕐 第 {epoch+1}/{total_epochs} 轮 | 时间: {timestamp} | 已用时: {elapsed_time}")
    print("=" * 80)


def print_performance_summary(
    epoch: int,
    train_acc: float,
    all_acc: float,
    old_acc: float,
    new_acc: float,
    all_acc_test: float,
    old_acc_test: float,
    new_acc_test: float,
    performance_gap: Dict[str, float],
    best_performance: Dict[str, float],
    loss_avg: float
):
    """打印性能总结"""
    print(f"\n📊 第{epoch+1}轮性能总结:")
    print(f"┌─ 训练损失: {loss_avg:.4f}")
    print(f"├─ 训练准确率: {train_acc:.4f}")
    print(f"├─ 训练集评估: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}")
    print(f"└─ 测试集评估: All {all_acc_test:.4f} | Old {old_acc_test:.4f} | New {new_acc_test:.4f}")

    print(f"\n🎯 与最佳模型差距:")
    print(f"├─ All ACC差距: {performance_gap['all_acc_gap']:+.4f}")
    print(f"├─ Old ACC差距: {performance_gap['old_acc_gap']:+.4f}")
    print(f"└─ New ACC差距: {performance_gap['new_acc_gap']:+.4f}")

    print(f"\n🏆 历史最佳 (第{best_performance['best_epoch']}轮):")
    print(f"├─ Best All ACC: {best_performance['best_all_acc']:.4f}")
    print(f"├─ Best Old ACC: {best_performance['best_old_acc']:.4f}")
    print(f"└─ Best New ACC: {best_performance['best_new_acc']:.4f}")


def print_training_start_info(args, model_info: Optional[Dict] = None):
    """打印训练开始信息"""
    print("\n" + "🚀" * 20 + " 训练开始 " + "🚀" * 20)
    print(f"📊 数据集: {args.dataset_name}")
    if hasattr(args, 'superclass_name') and args.superclass_name:
        print(f"🎯 超类: {args.superclass_name}")
    print(f"🔢 已知类数: {args.num_labeled_classes}")
    print(f"🔢 未知类数: {args.num_unlabeled_classes}")
    print(f"📈 总轮数: {args.epochs}")
    print(f"📏 批次大小: {args.batch_size}")
    print(f"📐 学习率: {args.lr}")
    print(f"⚖️  监督对比权重: {args.sup_con_weight}")

    if model_info:
        print(f"🤖 模型: {model_info.get('name', 'Unknown')}")
        print(f"🧠 特征维度: {model_info.get('feat_dim', 'Unknown')}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"⏰ 开始时间: {timestamp}")
    print("=" * 80)


def print_training_complete_info(
    total_epochs: int,
    elapsed_time: str,
    best_performance: Dict[str, float],
    early_stopped: bool = False,
    stopped_epoch: int = None
):
    """打印训练完成信息"""
    print("\n" + "🎉" * 20 + " 训练完成 " + "🎉" * 20)

    if early_stopped:
        print(f"🛑 早停于第{stopped_epoch}轮 (总计划{total_epochs}轮)")
    else:
        print(f"✅ 完成全部{total_epochs}轮训练")

    print(f"⏱️  总用时: {elapsed_time}")
    print(f"🏆 最佳性能 (第{best_performance['best_epoch']}轮):")
    print(f"├─ Best All ACC: {best_performance['best_all_acc']:.4f}")
    print(f"├─ Best Old ACC: {best_performance['best_old_acc']:.4f}")
    print(f"└─ Best New ACC: {best_performance['best_new_acc']:.4f}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"⏰ 结束时间: {timestamp}")
    print("=" * 80)


def print_model_save_info(model_path: str, is_best: bool = False, acc: float = None):
    """打印模型保存信息"""
    if is_best:
        print(f"💾 保存最佳模型: {model_path} (ACC: {acc:.4f})")
    else:
        print(f"💾 保存模型: {model_path}")


class TrainingSession:
    """训练会话管理器，整合所有训练辅助功能"""

    def __init__(self, args, enable_early_stopping: bool = True, patience: int = 20, quiet: bool = False):
        self.args = args
        self.performance_tracker = PerformanceTracker()
        self.quiet = quiet

        self.early_stopping = None
        if enable_early_stopping:
            self.early_stopping = EarlyStoppingMonitor(
                patience=patience,
                metric_name="test_all_acc",
                verbose=not self.quiet
            )

        self.start_epoch_time = None

    def start_training(self, model_info: Optional[Dict] = None):
        """开始训练"""
        if not self.quiet:
            print_training_start_info(self.args, model_info)

    def start_epoch(self, epoch: int):
        """开始新轮次"""
        self.start_epoch_time = time.time()
        elapsed_time = self.performance_tracker.get_elapsed_time()
        if not self.quiet:
            print_epoch_separator(epoch, self.args.epochs, elapsed_time)

    def end_epoch(
        self,
        epoch: int,
        train_acc: float,
        loss_avg: float,
        all_acc: float,
        old_acc: float,
        new_acc: float,
        all_acc_test: float,
        old_acc_test: float,
        new_acc_test: float
    ) -> bool:
        """
        结束当前轮次，返回是否应该早停

        Returns:
            bool: 是否应该早停
        """
        # 更新性能跟踪
        self.performance_tracker.update(all_acc_test, old_acc_test, new_acc_test, epoch)

        # 计算性能差距
        performance_gap = self.performance_tracker.get_performance_gap(
            all_acc_test, old_acc_test, new_acc_test
        )

        # 获取最佳性能
        best_performance = self.performance_tracker.get_best_performance()

        # 打印性能总结
        if not self.quiet:
            print_performance_summary(
                epoch, train_acc, all_acc, old_acc, new_acc,
                all_acc_test, old_acc_test, new_acc_test,
                performance_gap, best_performance, loss_avg
            )

        # 检查早停
        should_stop = False
        if self.early_stopping:
            should_stop = self.early_stopping.update(all_acc_test, epoch)

        return should_stop

    def finish_training(self, final_epoch: int, early_stopped: bool = False):
        """完成训练"""
        elapsed_time = self.performance_tracker.get_elapsed_time()
        best_performance = self.performance_tracker.get_best_performance()

        if not self.quiet:
            print_training_complete_info(
                total_epochs=self.args.epochs,
                elapsed_time=elapsed_time,
                best_performance=best_performance,
                early_stopped=early_stopped,
                stopped_epoch=final_epoch if early_stopped else None
            )

    def save_model_info(self, model_path: str, is_best: bool = False, acc: float = None):
        """记录模型保存"""
        if not self.quiet:
            print_model_save_info(model_path, is_best, acc)

    def get_best_performance(self) -> Dict[str, float]:
        """获取最佳性能"""
        return self.performance_tracker.get_best_performance()

    def state_dict(self) -> Dict[str, Any]:
        """导出训练会话状态，用于断点续训。"""
        state: Dict[str, Any] = {
            "performance_tracker": self.performance_tracker.state_dict(),
            "early_stopping": None,
        }
        if self.early_stopping is not None:
            state["early_stopping"] = self.early_stopping.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从字典恢复训练会话状态。"""
        if not state:
            return

        perf_state = state.get("performance_tracker")
        if perf_state is not None:
            self.performance_tracker.load_state_dict(perf_state)

        es_state = state.get("early_stopping")
        if es_state is not None and self.early_stopping is not None:
            self.early_stopping.load_state_dict(es_state)
