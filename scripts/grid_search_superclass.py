#!/usr/bin/env python3
"""
超类网格搜索脚本
用于在CIFAR-100超类任务中对lr与sup_con_weight进行网格搜索，并智能管理模型文件
"""

import argparse
import itertools
import os
import sys
from copy import deepcopy
from typing import Dict, Optional, Set, Tuple

# 添加项目根目录到路径，复用训练脚本中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from data.cifar100_superclass import SUPERCLASS_NAMES
from data.get_datasets import get_class_splits
from project_utils.general_utils import init_experiment
from project_utils.superclass_model_saver import SuperclassModelSaver
from scripts.train_superclass import (
    train_single_superclass,
    build_superclass_train_parser
)

# 网格搜索固定的学习率与 sup_con_weight 网格
DEFAULT_LR_GRID = [0.1, 0.05, 0.01, 0.001]
DEFAULT_SUP_CON_GRID = [round(0.2 + 0.05 * i, 2) for i in range(13)]  # 0.2~0.8 步长0.05


class SuperclassGridSearchSaver(SuperclassModelSaver):
    """网格搜索专用保存器：只记录指标与日志，不落地模型文件"""

    def __init__(self, superclass_name: str):
        super().__init__(superclass_name)
        self.run_context: Optional[Dict] = None
        self.current_run_best = -1.0
        self.current_run_metadata: Dict = {}
        self.log_file_path = os.path.join(self.save_dir, 'grid_search_log.txt')
        self._init_log_file()

    def _init_log_file(self):
        """确保日志文件存在并写入表头"""
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"# Grid Search Log for {self.superclass_name}\n")
                log_file.write("# Format: lr sup_con_weight all_acc old_acc new_acc\n")

    def _append_search_log(self, lr: float, sup_con_weight: float,
                           all_acc: float, old_acc: float, new_acc: float):
        """将单次组合的最优指标追加到日志"""
        log_line = (
            f"lr:{lr} "
            f"sup_con_weight:{sup_con_weight} "
            f"all_acc:{all_acc:.4f} "
            f"old_acc:{old_acc:.4f} "
            f"new_acc:{new_acc:.4f}\n"
        )
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(log_line)
        except OSError as exc:
            print(f"⚠️ 写入搜索日志失败: {exc}")

    def load_completed_runs(self) -> Set[Tuple[float, float]]:
        """读取日志文件，返回已完成组合集合"""
        completed: Set[Tuple[float, float]] = set()
        if not os.path.exists(self.log_file_path):
            return completed

        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as log_file:
                for line in log_file:
                    record = line.strip()
                    if not record or record.startswith('#'):
                        continue

                    try:
                        parts = dict(segment.split(':', 1) for segment in record.split())
                        if 'lr' not in parts or 'sup_con_weight' not in parts:
                            continue
                        lr_val = round(float(parts['lr']), 4)
                        sup_val = round(float(parts['sup_con_weight']), 4)
                        completed.add((lr_val, sup_val))
                    except (ValueError, KeyError):
                        continue
        except OSError as exc:
            print(f"⚠️ 读取搜索日志失败: {exc}")

        return completed

    def start_new_run(self, context: Dict):
        """开始新的超参数组合，重置本轮缓存"""
        self.run_context = context
        self.current_run_best = -1.0
        self.current_run_metadata = {}

    def save_best_model(self, model, projection_head, acc: float, metadata: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
        """网格搜索模式：不落地模型，仅缓存最优指标"""
        if self.run_context is None:
            raise RuntimeError("保存前必须先调用 start_new_run()")

        metadata = metadata or {}
        if acc > self.current_run_best:
            self.current_run_best = acc
            self.current_run_metadata = metadata
            if acc > self.best_acc:
                self.best_acc = acc

        return None, None

    def finalize_run(self) -> Tuple[str, Optional[Dict]]:
        """结束当前组合，仅写入日志"""
        if self.run_context is None or self.current_run_best < 0:
            self.current_run_best = -1.0
            self.current_run_metadata = {}
            return "skipped", None

        hyperparams = self.run_context.get('hyperparams', {})
        self._append_search_log(
            lr=hyperparams.get('lr', 0.0),
            sup_con_weight=hyperparams.get('sup_con_weight', 0.0),
            all_acc=self.current_run_metadata.get('all_acc_test', 0.0),
            old_acc=self.current_run_metadata.get('old_acc_test', 0.0),
            new_acc=self.current_run_metadata.get('new_acc_test', 0.0)
        )

        logged_metadata = self.current_run_metadata
        self.current_run_best = -1.0
        self.current_run_metadata = {}
        return "logged", logged_metadata

    def cleanup_keep_best_only(self):
        """网格搜索模式无需清理模型文件"""
        return


class GridSearchManager:
    """协调多超类网格搜索流程"""

    def __init__(self, args):
        self.args = args
        self.device = self._prepare_device(args.gpu)
        self.args.device = self.device
        self.learning_rates = DEFAULT_LR_GRID
        self.sup_con_weights = DEFAULT_SUP_CON_GRID

    @staticmethod
    def _prepare_device(gpu_index: int):
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_index}')
            torch.cuda.set_device(gpu_index)
            print(f"🎛️  使用GPU: cuda:{gpu_index}")
        else:
            device = torch.device('cpu')
            print("⚠️ CUDA不可用，使用CPU执行网格搜索")
        return device

    def run(self):
        summary: Dict[str, Dict] = {}

        print(f"\n{'=' * 70}")
        print("📋 CIFAR-100 可用超类列表（共15个）:")
        for idx, name in enumerate(SUPERCLASS_NAMES, start=1):
            print(f"  {idx:2d}. {name}")
        print(f"{'=' * 70}")
        print(f"本次搜索超类: {', '.join(self.args.superclasses)}")
        print(f"{'=' * 70}")

        for superclass in self.args.superclasses:
            if superclass not in SUPERCLASS_NAMES:
                print(f"❌ 超类 '{superclass}' 非法，跳过")
                continue

            print(f"\n{'=' * 70}")
            print(f"🔍 开始超类 '{superclass}' 的网格搜索")
            saver = SuperclassGridSearchSaver(superclass)

            best_info = self._search_single_superclass(superclass, saver)
            saver.cleanup_keep_best_only()
            if best_info['acc'] >= 0:
                print(
                    f"📌 超类 '{superclass}' 最佳组合: lr={best_info['lr']} "
                    f"sup_con_weight={best_info['sup_con_weight']} "
                    f"(需使用该超参数单独训练以导出模型)"
                )
            else:
                print(f"⚠️  超类 '{superclass}' 未能得到有效的组合结果")
            summary[superclass] = best_info

        self._print_summary(summary)

    def _search_single_superclass(self, superclass: str, saver: SuperclassGridSearchSaver) -> Dict:
        best_result = {
            'acc': -1.0,
            'lr': None,
            'sup_con_weight': None,
            'model_path': None,
            'params_path': None
        }

        def _quantize(value: float) -> float:
            return round(float(value), 4)

        combos = list(itertools.product(self.learning_rates, self.sup_con_weights))
        combo_keys = {(_quantize(lr), _quantize(sup)) for lr, sup in combos}
        total_combos = len(combos)

        completed_runs = saver.load_completed_runs()
        completed_in_grid = completed_runs & combo_keys
        completed_count = len(completed_in_grid)
        pending_count = max(total_combos - completed_count, 0)

        historical_best = self._restore_best_from_log(saver) if completed_count else None
        if historical_best and historical_best['acc'] > best_result['acc']:
            best_result = historical_best

        if completed_count:
            print(
                f"ℹ️  超类 '{superclass}' 已完成 {completed_count}/{total_combos} 个组合，本次预计训练 {pending_count} 个组合"
            )
        else:
            print(f"ℹ️  未检测到 '{superclass}' 的历史网格搜索记录，将完整遍历 {total_combos} 个组合")

        skipped_count = 0

        outer_pbar = tqdm(
            combos,
            desc=f"[{superclass}] 网格搜索",
            position=0,
            leave=True,
            dynamic_ncols=True
        )
        try:
            for lr, sup_weight in outer_pbar:
                quantized_lr = _quantize(lr)
                quantized_sup = _quantize(sup_weight)

                outer_pbar.set_postfix({
                    'lr': f"{lr:.3f}",
                    'sup': f"{sup_weight:.3f}",
                    'best_acc': f"{best_result['acc']:.4f}" if best_result['acc'] >= 0 else "N/A"
                })

                if (quantized_lr, quantized_sup) in completed_in_grid:
                    skipped_count += 1
                    outer_pbar.update(1)
                    outer_pbar.set_postfix({
                        'lr': f"{lr:.3f}",
                        'sup': f"{sup_weight:.3f}",
                        'best_acc': f"{best_result['acc']:.4f}" if best_result['acc'] >= 0 else "N/A",
                        'status': 'skipped'
                    })
                    tqdm.write(
                        f"⏭️  组合已完成，跳过 lr={lr:.4f} sup_con_weight={sup_weight:.4f}"
                    )
                    continue

                run_args = self._prepare_run_args(superclass, lr, sup_weight)

                saver.start_new_run({
                    'hyperparams': self._collect_hparams(run_args)
                })

                # 训练并记录是否成功
                training_success = False
                try:
                    _, _, best_acc = train_single_superclass(
                        run_args,
                        model_saver=saver,
                        progress_parent=outer_pbar
                    )
                    training_success = True
                except Exception as exc:
                    tqdm.write(f"⚠️  训练失败，跳过该组合: {exc}")
                    if hasattr(run_args, 'writer') and run_args.writer:
                        run_args.writer.close()
                    best_acc = -1.0

                # ✅ 无论训练是否成功，都调用finalize_run写入日志或复位状态
                status, _ = saver.finalize_run()

                # 只有训练成功时才更新最佳结果
                if training_success:
                    tqdm.write(f"   ✅ 组合完成，最优all_acc_test={best_acc:.4f}，标记为{status}")

                    if best_acc > best_result['acc']:
                        best_result.update({
                            'acc': best_acc,
                            'lr': lr,
                            'sup_con_weight': sup_weight,
                            'model_path': None,
                            'params_path': None
                        })
                        outer_pbar.set_postfix({
                            'lr': f"{lr:.3f}",
                            'sup': f"{sup_weight:.3f}",
                            'best_acc': f"{best_acc:.4f}"
                        })

                    if hasattr(run_args, 'writer') and run_args.writer:
                        run_args.writer.close()
        finally:
            outer_pbar.close()

        if skipped_count:
            print(f"⏭️  超类 '{superclass}' 共跳过 {skipped_count} 个组合")
        if best_result['acc'] < 0 and skipped_count == total_combos:
            restored = self._restore_best_from_log(saver)
            if restored:
                best_result = restored
                print(
                    f"ℹ️  所有组合均由日志恢复，best_all_acc={best_result['acc']:.4f} "
                    f"lr={best_result['lr']} sup_con_weight={best_result['sup_con_weight']}"
                )

        return best_result

    def _prepare_run_args(self, superclass: str, lr: float, sup_weight: float):
        run_args = deepcopy(self.args)
        run_args.superclass_name = superclass
        run_args.train_all_superclasses = False
        run_args.lr = lr
        run_args.sup_con_weight = sup_weight
        run_args.dataset_name = 'cifar100_superclass'
        run_args.is_grid_search = True

        run_args = get_class_splits(run_args)
        run_args.num_labeled_classes = len(run_args.train_classes)
        run_args.num_unlabeled_classes = len(run_args.unlabeled_classes)

        run_args.exp_name = self._build_experiment_name(superclass, lr, sup_weight)
        init_experiment(run_args, runner_name=['grid_search_superclass'])
        run_args.device = self.device

        return run_args

    @staticmethod
    def _collect_hparams(args) -> Dict:
        return {
            'lr': args.lr,
            'sup_con_weight': args.sup_con_weight,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'grad_from_block': args.grad_from_block,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'temperature': args.temperature,
            'n_views': args.n_views,
            'contrast_unlabel_only': args.contrast_unlabel_only,
            'seed': args.seed
        }

    @staticmethod
    def _build_experiment_name(superclass: str, lr: float, sup_weight: float) -> str:
        def _sanitize(value: float) -> str:
            text = f"{value:.4f}".rstrip('0').rstrip('.')
            return text.replace('.', 'p') if text else "0"

        lr_tag = _sanitize(lr)
        sup_tag = _sanitize(sup_weight)
        return f"grid_{superclass}_lr{lr_tag}_sup{sup_tag}"

    @staticmethod
    def _print_summary(summary: Dict[str, Dict]):
        if not summary:
            print("\n⚠️ 未获得任何有效结果")
            return

        print(f"\n{'=' * 70}")
        print("📋 网格搜索汇总（模型未保存，请用最佳超参数重新训练获取最终模型）")
        for superclass, info in summary.items():
            if info['acc'] < 0:
                print(f" - {superclass}: 无有效模型")
                continue
            print(f" - {superclass}: best_all_acc={info['acc']:.4f} | lr={info['lr']} | sup_con_weight={info['sup_con_weight']}")
            if info['model_path']:
                print(f"   模型: {info['model_path']}")
            if info['params_path']:
                print(f"   参数: {info['params_path']}")

    @staticmethod
    def _restore_best_from_log(saver: SuperclassGridSearchSaver) -> Optional[Dict]:
        """当全部组合被跳过时，从日志中恢复最佳指标"""
        if not os.path.exists(saver.log_file_path):
            return None

        best_entry: Optional[Dict] = None
        try:
            with open(saver.log_file_path, 'r', encoding='utf-8') as log_file:
                for line in log_file:
                    record = line.strip()
                    if not record or record.startswith('#'):
                        continue

                    try:
                        parts = dict(segment.split(':', 1) for segment in record.split())
                        required = ('lr', 'sup_con_weight', 'all_acc')
                        if not all(key in parts for key in required):
                            continue
                        all_acc = float(parts['all_acc'])
                        lr_val = float(parts['lr'])
                        sup_val = float(parts['sup_con_weight'])
                    except (ValueError, KeyError):
                        continue

                    if best_entry is None or all_acc > best_entry['acc']:
                        best_entry = {
                            'acc': all_acc,
                            'lr': lr_val,
                            'sup_con_weight': sup_val,
                            'model_path': None,
                            'params_path': None
                        }
        except OSError as exc:
            print(f"⚠️  读取日志恢复最佳结果失败: {exc}")
            return None

        return best_entry


def parse_args():
    base_parser = build_superclass_train_parser(add_help=False)  # 禁用父parser的帮助
    parser = argparse.ArgumentParser(
        description='CIFAR-100超类网格搜索工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[base_parser],
        add_help=True  # 当前parser启用帮助
    )

    parser.add_argument('--superclasses', nargs='+', default=SUPERCLASS_NAMES,
                        help='需要执行网格搜索的超类列表')

    # 批量搜索默认训练200轮（早停会提前结束）
    parser.set_defaults(epochs=200)

    return parser.parse_args()


def main():
    args = parse_args()
    args.train_all_superclasses = False
    manager = GridSearchManager(args)
    manager.run()


if __name__ == "__main__":
    main()
