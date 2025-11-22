#!/usr/bin/env python3
"""
超类训练脚本 - 基于GCD项目的contrastive_training.py
支持15个超类的独立训练和数据划分
"""

import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from data.cifar100_superclass import get_single_superclass_datasets, get_superclass_splits, SUPERCLASS_NAMES
from data.data_utils import MergedDataset
from copy import deepcopy

from tqdm import tqdm

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path, feature_cache_dir

# 从原始训练脚本导入必要的类和函数
from methods.contrastive_training.contrastive_training import (
    SupConLoss, ContrastiveLearningViewGenerator, info_nce_logits
)

# 导入训练工具
from utils.training_utils import TrainingSession
from utils.checkpoint_utils import save_training_state, load_training_state
from utils.pseudo_labels import load_pseudo_label_cache
from utils.best_model_tracker import BestModelTracker

# 导入超类模型保存器
from project_utils.superclass_model_saver import create_superclass_model_saver
from scripts._feature_cache_runner import run_cache_features

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


from typing import Optional


def get_gamma(epoch: int, warmup_epochs: int = 50, total_epochs: int = 200,
              update_interval: Optional[int] = None, ramp_intervals: int = 2) -> float:
    """
    伪标签权重调度函数 γ(epoch)。

    - epoch < warmup_epochs: γ = 0.0
    - 若提供 update_interval，则在 warmup 后的前 ramp_intervals 个伪标签更新间隔内线性升至 1.0，
      之后保持 1.0（默认在前两个间隔内完成增长）。
    - 若未提供 update_interval，则退回原有的线性调度：warmup→total_epochs 期间升至 1.0。
    """
    if epoch < warmup_epochs:
        return 0.0

    if update_interval is not None and update_interval > 0 and ramp_intervals > 0:
        ramp_total = max(ramp_intervals * update_interval, 1)
        progress = (epoch - warmup_epochs + 1) / ramp_total
        progress = max(0.0, min(1.0, progress))
        return float(progress)

    # 兼容老的线性策略
    remaining_epochs = max(total_epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / remaining_epochs
    progress = max(0.0, min(1.0, progress))
    return float(progress)


def train_superclass(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args,
                     pseudo_cache=None, model_saver=None, progress_parent=None, feature_cache_args=None):
    """
    超类训练函数，基于原始的train函数修改

    Returns:
        tuple: (model, projection_head, best_all_acc_test)
    """
    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss()

    # 初始化最佳模型追踪器（使用 JSON 文件持久化全局最佳信息）
    best_model_tracker = BestModelTracker(args.exp_root)
    best_test_acc_lab = best_model_tracker.get_best_acc()

    is_grid_search = getattr(args, 'is_grid_search', False)

    if not is_grid_search and best_test_acc_lab > 0:
        print(f"📊 从 JSON 恢复全局最佳 ACC: {best_test_acc_lab:.4f} (epoch {best_model_tracker.get_best_epoch()})")

    # 初始化训练会话管理器
    training_session = TrainingSession(args, enable_early_stopping=False, patience=20, quiet=is_grid_search)
    model_info = {
        'name': args.base_model,
        'feat_dim': getattr(args, 'feat_dim', 'Unknown')
    }
    training_session.start_training(model_info)
    setattr(args, "pseudo_cache", pseudo_cache)
    pseudo_weight_mode = getattr(args, "pseudo_weight_mode", "none")
    valid_weight_modes = {"none", "density", "inverse_density"}
    if pseudo_weight_mode not in valid_weight_modes:
        raise ValueError(f"不支持的 pseudo_weight_mode: {pseudo_weight_mode}")
    pseudo_for_labeled_mode = getattr(args, "pseudo_for_labeled_mode", "off")
    if pseudo_for_labeled_mode not in {"off", "all"}:
        raise ValueError(f"不支持的 pseudo_for_labeled_mode: {pseudo_for_labeled_mode}")
    effective_weight_mode = pseudo_weight_mode
    if pseudo_cache is None:
        if pseudo_weight_mode != "none" and not is_grid_search:
            print("⚠️  未提供伪标签文件，忽略 pseudo_weight_mode 设置")
        effective_weight_mode = "none"
    elif pseudo_weight_mode in {"density", "inverse_density"} and not pseudo_cache.has_density_weights:
        if not is_grid_search:
            print("⚠️  伪标签文件缺少密度信息，无法按密度相关加权，退回均匀权重")
        effective_weight_mode = "none"
    args.pseudo_weight_mode_effective = effective_weight_mode
    if pseudo_cache is not None and not is_grid_search:
        if effective_weight_mode == "density":
            mode_desc = "按密度加权"
        elif effective_weight_mode == "inverse_density":
            mode_desc = "反向密度加权"
        else:
            mode_desc = "均匀权重"
        print(f"   伪标签损失权重模式: {mode_desc}")
        mode_scope = "已标注+未标注一起" if pseudo_for_labeled_mode == "all" else "仅未标注"
        print(f"   伪标签样本范围: {mode_scope}")

    epoch_progress = None
    if is_grid_search and progress_parent is not None:
        epoch_progress = tqdm(
            total=args.epochs,
            desc=f"[{args.superclass_name}] 训练",
            position=1,
            leave=False,
            dynamic_ncols=True
        )

    # 初始化超类模型保存器（传入args以支持超参数记录）
    if model_saver is None:
        model_saver = create_superclass_model_saver(args.superclass_name, args=args)
    if not is_grid_search:
        print(f"🗂️  超类模型保存器已初始化，保存目录: {model_saver.save_dir}")

    # 支持从检查点恢复训练
    start_epoch = 0
    if getattr(args, "resume_from_ckpt", None):
        if not is_grid_search:
            print(f"📂 从检查点恢复训练: {args.resume_from_ckpt}")
        loaded_epoch, extra_state = load_training_state(
            args.resume_from_ckpt,
            model=model,
            projection_head=projection_head,
            optimizer=optimizer,
            scheduler=exp_lr_scheduler,
            training_session=training_session,
        )
        start_epoch = loaded_epoch + 1

        # 验证 T_max 是否一致（防止用户手动修改 --epochs 参数）
        saved_T_max = extra_state.get("T_max")
        if saved_T_max is not None and saved_T_max != args.epochs:
            raise ValueError(
                f"❌ Checkpoint 保存时的 epochs={saved_T_max}，"
                f"但当前 --epochs={args.epochs}，T_max 不一致！\n"
                f"提示：请使用相同的 --epochs 参数恢复训练，或删除 --resume_from_ckpt 参数从头训练。"
            )

        if not is_grid_search:
            print(f"   ✅ 恢复到第 {start_epoch} 轮继续训练")
            print(f"   ✅ Scheduler 状态已恢复 (T_max={args.epochs}, 学习率连续)")

    if pseudo_cache and getattr(args, "writer", None):
        args.writer.add_scalar('Pseudo/file_core_ratio', pseudo_cache.core_ratio, start_epoch)
        args.writer.add_scalar('Pseudo/file_num_core', pseudo_cache.num_core, start_epoch)
        train_stats = getattr(pseudo_cache, "training_stats", None)
        if train_stats:
            args.writer.add_scalar('Pseudo/train_core_ratio', train_stats.get('core_ratio', 0.0), start_epoch)
            args.writer.add_scalar('Pseudo/train_core_count', train_stats.get('core_count', 0), start_epoch)
        args.writer.add_scalar('Pseudo/weight_mode_flag', 1 if args.pseudo_weight_mode_effective == 'density' else 0, start_epoch)
        args.writer.add_text('Pseudo/weight_mode', args.pseudo_weight_mode_effective, start_epoch)

    for epoch in range(start_epoch, args.epochs):
        if epoch_progress:
            epoch_progress.set_description(f"[{args.superclass_name}] Epoch {epoch+1}/{args.epochs}")

        # 开始新轮次
        training_session.start_epoch(epoch)

        loss_record = AverageMeter()
        contrastive_loss_record = AverageMeter()
        sup_con_loss_record = AverageMeter()
        pseudo_loss_record = AverageMeter()
        pseudo_sample_record = AverageMeter()
        train_acc_record = AverageMeter()

        # 定义轮次级别的权重变量，避免批次内重复定义
        epoch_contrastive_weight = 1 - args.sup_con_weight
        epoch_sup_con_weight = args.sup_con_weight

        projection_head.train()
        model.train()

        if not is_grid_search:
            train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        else:
            train_iter = train_loader

        for batch_idx, batch in enumerate(train_iter):

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(args.device), mask_lab.to(args.device).bool()
            mask_lab_np = mask_lab.detach().cpu().numpy()
            images = torch.cat(images, dim=0).to(args.device)

            # Extract features with base model
            features = model(images)

            # Pass features through projection head
            features = projection_head(features)

            # L2-normalize features
            features = torch.nn.functional.normalize(features, dim=-1)

            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = features

            contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # Supervised contrastive loss
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            if f1.size(0) > 0:  # 只有当有标记样本时才计算监督对比损失
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            else:
                sup_con_loss = torch.tensor(0.0).to(args.device)

            # 伪标签对比损失
            pseudo_loss = torch.tensor(0.0, device=args.device)
            pseudo_sample_count = 0
            if pseudo_cache is not None:
                if isinstance(uq_idxs, torch.Tensor):
                    batch_indices_np = uq_idxs.detach().cpu().numpy().astype(np.int64)
                else:
                    batch_indices_np = np.asarray(uq_idxs, dtype=np.int64)
                pseudo_labels_np = pseudo_cache.lookup_labels(batch_indices_np)
                valid_mask_np = pseudo_labels_np >= 0
                if pseudo_for_labeled_mode != "all":
                    # 仅未标注样本参与伪标签损失
                    unlabeled_mask_np = ~mask_lab_np
                    valid_mask_np = np.logical_and(valid_mask_np, unlabeled_mask_np)
                if valid_mask_np.any():
                    pseudo_labels_tensor = torch.from_numpy(
                        pseudo_labels_np[valid_mask_np]
                    ).long().to(args.device)
                    pseudo_mask_tensor = torch.from_numpy(valid_mask_np).to(args.device)
                    f1_pseudo, f2_pseudo = [f[pseudo_mask_tensor] for f in features.chunk(2)]
                    if f1_pseudo.size(0) > 0:
                        pseudo_feats = torch.cat([f1_pseudo.unsqueeze(1), f2_pseudo.unsqueeze(1)], dim=1)
                        if args.pseudo_weight_mode_effective == 'density':
                            weight_values = pseudo_cache.lookup_weights(batch_indices_np, mode="density")[valid_mask_np]
                        elif args.pseudo_weight_mode_effective == 'inverse_density':
                            weight_values = pseudo_cache.lookup_weights(batch_indices_np, mode="inverse_density")[valid_mask_np]
                        else:
                            weight_values = np.ones(pseudo_labels_tensor.size(0), dtype=np.float32)
                        pseudo_weights_tensor = torch.from_numpy(weight_values).to(args.device)
                        pseudo_loss = sup_con_crit(
                            pseudo_feats,
                            labels=pseudo_labels_tensor,
                            sample_weights=pseudo_weights_tensor,
                        )
                        pseudo_sample_count = int(pseudo_labels_tensor.size(0))

            gamma = get_gamma(
                epoch,
                warmup_epochs=args.warmup_epochs,
                total_epochs=args.epochs,
                update_interval=getattr(args, "update_interval", None),
                ramp_intervals=2,
            )

            # 总损失 = 无监督对比损失 + 监督对比损失 + γ·pseudo_loss_weight·伪标签损失
            loss = epoch_contrastive_weight * contrastive_loss + epoch_sup_con_weight * sup_con_loss
            if pseudo_cache is not None and pseudo_sample_count > 0:
                pseudo_loss_weight = getattr(args, 'pseudo_loss_weight', 1.0)
                loss = loss + gamma * pseudo_loss_weight * pseudo_loss

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            # 记录各种损失
            loss_record.update(loss.item(), class_labels.size(0))
            contrastive_loss_record.update(contrastive_loss.item(), class_labels.size(0))
            sup_con_loss_record.update(sup_con_loss.item(), class_labels.size(0))
            if pseudo_cache is not None:
                pseudo_loss_record.update(pseudo_loss.item(), max(pseudo_sample_count, 1))
                pseudo_sample_record.update(pseudo_sample_count, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每50个batch输出一次实时损失
            if batch_idx % 50 == 0 and batch_idx > 0 and not is_grid_search:
                print(f"   Batch {batch_idx}: loss = {epoch_contrastive_weight:.2f}*{contrastive_loss.item():.4f} + {epoch_sup_con_weight:.2f}*{sup_con_loss.item():.4f} = {loss.item():.4f}")

        with torch.no_grad():
            if not is_grid_search:
                print('🔍 Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test_kmeans_superclass(model, unlabelled_train_loader,
                                                        epoch=epoch, save_name='Train ACC Unlabelled',
                                                        args=args)

            if not is_grid_search:
                print('🔍 Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_kmeans_superclass(model, test_loader,
                                                                   epoch=epoch, save_name='Test ACC',
                                                                   args=args)

        # ----------------
        # 输出详细损失分解
        # ----------------
        lr_value = get_mean_lr(optimizer)

        if not is_grid_search:
            pseudo_loss_weight = getattr(args, 'pseudo_loss_weight', 1.0)
            print(f"\n📊 Epoch {epoch+1}/{args.epochs} 损失分解:")
            print(f"   总损失 = {epoch_contrastive_weight:.2f}*对比损失 + {epoch_sup_con_weight:.2f}*监督对比损失 + γ*λ*伪标签损失")
            print(
                f"   总损失 = {epoch_contrastive_weight:.2f}*{contrastive_loss_record.avg:.4f} "
                f"+ {epoch_sup_con_weight:.2f}*{sup_con_loss_record.avg:.4f} "
                f"+ {gamma:.2f}*{pseudo_loss_weight:.2f}*{pseudo_loss_record.avg:.4f} = {loss_record.avg:.4f}"
            )
            print(f"   对比学习损失: {contrastive_loss_record.avg:.4f}")
            print(f"   监督对比损失: {sup_con_loss_record.avg:.4f}")
            if pseudo_cache is not None:
                print(
                    f"   伪标签损失: {pseudo_loss_record.avg:.4f} "
                    f"(每批伪标签样本≈{pseudo_sample_record.avg:.1f})"
                )
            print(f"   训练准确率: {train_acc_record.avg:.4f}")
            print(f"   学习率: {lr_value:.6f}")
        elif epoch_progress:
            epoch_progress.update(1)
            epoch_progress.set_postfix({
                'Loss': f"{loss_record.avg:.4f}",
                'TrainAcc': f"{train_acc_record.avg:.4f}",
                'TestAll': f"{all_acc_test:.4f}",
                'Best': f"{best_test_acc_lab:.4f}"
            })

        # ----------------
        # LOG & HOOKS
        # ----------------
        # 1) 记录当前损失与学习率
        args.writer.add_scalar('Loss/Total', loss_record.avg, epoch)
        args.writer.add_scalar('Loss/Contrastive', contrastive_loss_record.avg, epoch)
        args.writer.add_scalar('Loss/SupCon', sup_con_loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        # 2) 计算并记录当前 epoch 的 γ（伪标签损失权重）
        gamma = get_gamma(
            epoch,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            update_interval=getattr(args, "update_interval", None),
            ramp_intervals=2,
        )
        pseudo_loss_weight = getattr(args, 'pseudo_loss_weight', 1.0)
        args.writer.add_scalar('Pseudo/gamma', gamma, epoch)
        args.writer.add_scalar('Pseudo/loss_weight', pseudo_loss_weight, epoch)
        args.writer.add_scalar('Pseudo/effective_weight', gamma * pseudo_loss_weight, epoch)
        args.writer.add_scalar('Pseudo/for_labeled_mode_flag', 1.0 if pseudo_for_labeled_mode == "all" else 0.0, epoch)
        if pseudo_cache is not None:
            args.writer.add_scalar('Loss/Pseudo', pseudo_loss_record.avg, epoch)
            args.writer.add_scalar('Pseudo/samples_per_batch', pseudo_sample_record.avg, epoch)
        if not is_grid_search:
            print(f"   伪标签权重 γ(epoch={epoch}) = {gamma:.4f}, λ = {pseudo_loss_weight:.2f}, 有效权重 = {gamma * pseudo_loss_weight:.4f}")

        # 3) 伪标签刷新钩子（已由离线 Stage2 取代，避免在训练内部重复聚类）

        # 使用训练会话管理器处理轮次结束
        should_stop = training_session.end_epoch(
            epoch=epoch,
            train_acc=train_acc_record.avg,
            loss_avg=loss_record.avg,
            all_acc=all_acc,
            old_acc=old_acc,
            new_acc=new_acc,
            all_acc_test=all_acc_test,
            old_acc_test=old_acc_test,
            new_acc_test=new_acc_test
        )

        # Step schedule
        exp_lr_scheduler.step()

        # 保存模型 - 使用新的超类模型保存机制
        # 使用超类模型保存器保存最佳模型 (仅保存最佳，不再保存每轮模型)
        if all_acc_test > best_test_acc_lab:
            # 使用新的保存机制，会自动管理文件命名、删除旧模型、生成超参数记录
            best_model_path, best_proj_path = model_saver.save_best_model(
                model=model,
                projection_head=projection_head,
                acc=all_acc_test,
                current_epoch=epoch + 1,  # 传入当前轮数以记录训练进度
                metadata={
                    'epoch': epoch + 1,
                    'train_loss': loss_record.avg,
                    'all_acc_test': all_acc_test,
                    'old_acc_test': old_acc_test,
                    'new_acc_test': new_acc_test,
                    'train_acc': train_acc_record.avg
                }
            )
            best_test_acc_lab = all_acc_test

            # 更新 JSON 文件记录全局最佳信息
            stage_name = "stage3" if getattr(args, "resume_from_ckpt", None) else "stage1"
            best_model_tracker.update_if_better(
                new_acc=all_acc_test,
                epoch=epoch + 1,
                model_path=os.path.relpath(best_model_path, args.exp_root),
                proj_path=os.path.relpath(best_proj_path, args.exp_root),
                metadata={
                    'all_acc': all_acc_test,
                    'old_acc': old_acc_test,
                    'new_acc': new_acc_test,
                    'train_loss': loss_record.avg,
                    'train_acc': train_acc_record.avg
                },
                hyperparameters={
                    'lr': args.lr,
                    'sup_con_weight': args.sup_con_weight,
                    'grad_from_block': args.grad_from_block,
                    'total_epochs': args.epochs
                },
                stage=stage_name
            )

        # 可选：保存完整训练状态检查点，便于后续验证断点续训
        if getattr(args, "save_ckpt_every", 0) and (epoch + 1) % args.save_ckpt_every == 0:
            ckpt_dir = os.path.join(args.exp_root, "checkpoints", args.superclass_name)
            ckpt_name = f"ckpt_epoch_{epoch+1:03d}.pt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            extra_state = {
                "superclass_name": getattr(args, "superclass_name", None),
                "T_max": args.epochs,  # 记录当前的 T_max，用于恢复时验证
            }
            if not is_grid_search:
                print(f"💾 保存训练检查点: {ckpt_path}")
            save_training_state(
                ckpt_path,
                epoch=epoch,
                model=model,
                projection_head=projection_head,
                optimizer=optimizer,
                scheduler=exp_lr_scheduler,
                training_session=training_session,
                extra_state=extra_state,
            )

        stop_at_epoch = getattr(args, "stop_at_epoch", None)
        reached_stop = stop_at_epoch is not None and (epoch + 1) >= stop_at_epoch

        if should_stop or reached_stop:
            if not is_grid_search:
                reason = "早停触发" if should_stop else f"到达 stop_at_epoch={stop_at_epoch}"
                print(f"\n🛑 超类 '{args.superclass_name}' {reason}，在第{epoch+1}轮停止训练")
            training_session.finish_training(epoch, early_stopped=True)
            if reached_stop and feature_cache_args and getattr(args, "save_features_and_exit", False):
                _run_feature_cache_stage(args, feature_cache_args)
            break

    else:
        # 正常完成所有轮次
        training_session.finish_training(args.epochs - 1, early_stopped=False)

    if epoch_progress:
        epoch_progress.close()

    # 训练结束，显示最佳模型信息
    if not is_grid_search:
        print(f"\n📊 超类 '{args.superclass_name}' 训练完成")

        # 使用持久化的 BestModelTracker（与 pipeline 保持一致）
        best_acc = best_model_tracker.get_best_acc()
        best_epoch = best_model_tracker.get_best_epoch()

        print(f"🏆 最佳模型信息:")
        if best_acc > 0:
            print(f"   最佳ACC: {best_acc:.4f} (epoch {best_epoch})")
            # 显示详细的准确率分解
            tracker_data = best_model_tracker.load()  # 使用 load() 方法而不是 data 属性
            if tracker_data.get('metadata'):
                metadata = tracker_data['metadata']
                old_acc = metadata.get('old_acc', 0)
                new_acc = metadata.get('new_acc', 0)
                if old_acc > 0 or new_acc > 0:
                    print(f"   Old ACC: {old_acc:.4f}")
                    print(f"   New ACC: {new_acc:.4f}")
        else:
            print("   尚未记录最佳模型")

    return model, projection_head, best_test_acc_lab

def _run_feature_cache_stage(args, feature_cache_args):
    ckpt_dir = os.path.join(args.exp_root, "checkpoints", args.superclass_name)
    latest_ckpt = None
    if os.path.isdir(ckpt_dir):
        candidates = sorted(
            [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.startswith("ckpt_epoch_")],
            reverse=True)
        if candidates:
            latest_ckpt = candidates[0]
    if not latest_ckpt:
        raise FileNotFoundError(f"在 {ckpt_dir} 找不到 ckpt_epoch_*.pt，无法导出特征")

    run_cache_features(
        superclass_name=feature_cache_args["superclass_name"],
        model_path=latest_ckpt,
        cache_dir=feature_cache_args["cache_dir"],
        batch_size=feature_cache_args["batch_size"],
        num_workers=feature_cache_args["num_workers"],
        gpu=feature_cache_args["gpu"],
        prop_train_labels=feature_cache_args["prop_train_labels"],
        seed=feature_cache_args["seed"],
    )


def test_kmeans_superclass(model, test_loader, epoch, save_name, args):
    """
    超类K-means测试函数，基于原始的test_kmeans修改
    """
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    is_grid_search = getattr(args, 'is_grid_search', False)

    if not is_grid_search:
        print('Collating features...')
        test_iter = tqdm(test_loader, desc='Collating features')
    else:
        test_iter = test_loader

    # First extract all features
    for batch_idx, (images, label, _) in enumerate(test_iter):

        images = images.to(args.device)

        # Pass features through base model only (no projection head for evaluation)
        feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    if not is_grid_search:
        print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, n_init=10).fit(all_feats)
    preds = kmeans.labels_
    if not is_grid_search:
        print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    # 检查是否有未知类
    mask = np.array(mask, dtype=bool)  # 确保mask是numpy数组
    has_unknown_classes = args.num_unlabeled_classes > 0 and (~mask).sum() > 0

    if has_unknown_classes:
        # 有未知类的情况：正常计算所有指标
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                        writer=args.writer)
    else:
        # 没有未知类的情况：只计算已知类准确率
        from project_utils.cluster_utils import cluster_acc
        old_acc = cluster_acc(targets, preds)
        all_acc = old_acc  # 当没有未知类时，All ACC = Old ACC
        new_acc = 0.0      # 没有未知类，New ACC为0

        if not is_grid_search:
            print(f"⚠️  注意: 超类 '{args.superclass_name}' 中没有未知类样本，仅计算Old ACC")

        # 写入TensorBoard日志
        if args.writer is not None:
            args.writer.add_scalars(f'{save_name}_v1',
                                   {'Old': old_acc, 'New': new_acc, 'All': all_acc}, epoch)

    return all_acc, old_acc, new_acc



def train_all_superclasses(args):
    """
    训练所有15个超类
    """
    print("🌟 开始训练所有15个超类")
    print("=" * 80)

    superclass_splits = get_superclass_splits()

    for superclass_name in SUPERCLASS_NAMES:
        print(f"\n🎯 准备训练超类: {superclass_name}")

        # 检查该超类是否有足够的已知类和未知类
        split_info = superclass_splits[superclass_name]
        if len(split_info['known_classes']) == 0:
            print(f"⚠️  跳过超类 '{superclass_name}': 没有已知类")
            continue
        if len(split_info['unknown_classes']) == 0:
            print(f"⚠️  跳过超类 '{superclass_name}': 没有未知类")
            continue

        # 设置当前超类的参数
        current_args = argparse.Namespace(**vars(args))
        current_args.superclass_name = superclass_name
        current_args.dataset_name = 'cifar100_superclass'

        # 获取类别划分
        current_args = get_class_splits(current_args)
        current_args.num_labeled_classes = len(current_args.train_classes)
        current_args.num_unlabeled_classes = len(current_args.unlabeled_classes)

        # 创建实验目录
        exp_name = f'superclass_{superclass_name}_{current_args.model_name}'
        current_args.exp_name = exp_name
        init_experiment(current_args, runner_name=['superclass_train'])

        try:
            # 训练单个超类
            train_single_superclass(current_args)
        except Exception as e:
            print(f"❌ 训练超类 '{superclass_name}' 时出错: {e}")
            continue

    print("\n🎉 所有超类训练完成!")



def train_single_superclass(args, model_saver=None, progress_parent=None):
    """
    训练单个超类

    Args:
        args: 训练配置
        model_saver: 可选，覆盖默认的超类模型保存器
    """
    device = args.device
    is_grid_search = getattr(args, 'is_grid_search', False)
    pseudo_cache = None
    pseudo_path = getattr(args, "pseudo_labels_path", None)
    if pseudo_path:
        pseudo_path = os.path.abspath(pseudo_path)
        if not os.path.isfile(pseudo_path):
            raise FileNotFoundError(f"伪标签文件不存在: {pseudo_path}")
        pseudo_cache = load_pseudo_label_cache(pseudo_path)
        file_superclass = pseudo_cache.packet.metadata.get("superclass")
        if file_superclass and file_superclass != args.superclass_name:
            raise ValueError(
                f"伪标签文件属于超类 '{file_superclass}'，与当前 '{args.superclass_name}' 不匹配"
            )
        if not is_grid_search:
            print("📥 载入伪标签:")
            print(f"   文件: {pseudo_path}")
            print(
                f"   样本总数: {pseudo_cache.num_total}, 核心点: {pseudo_cache.num_core} "
                f"({pseudo_cache.core_ratio * 100:.1f}%)"
            )
            if pseudo_cache.packet.metadata:
                meta_ckpt = pseudo_cache.packet.metadata.get("ckpt_path")
                if meta_ckpt:
                    print(f"   来源 ckpt: {meta_ckpt}")

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path

        model = vits.__dict__['vit_base']()

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        if args.warmup_model_dir is not None:
            if not getattr(args, 'is_grid_search', False):
                print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:
        raise NotImplementedError

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS - 使用超类特定的数据集
    # --------------------
    # 直接使用超类特定的数据集获取函数
    datasets = get_single_superclass_datasets(
        superclass_name=args.superclass_name,
        train_transform=train_transform,
        test_transform=test_transform,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
        seed=args.seed,
        verbose=not is_grid_search
    )

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)

    if pseudo_cache:
        labelled_uq = getattr(train_dataset.labelled_dataset, 'uq_idxs', None)
        unlabelled_uq = getattr(train_dataset.unlabelled_dataset, 'uq_idxs', None)
        if labelled_uq is None or unlabelled_uq is None:
            print("⚠️  当前数据集不包含 uq_idxs 信息，无法匹配伪标签索引")
        else:
            train_indices = np.concatenate([labelled_uq, unlabelled_uq]).astype(np.int64)
            subset_stats = pseudo_cache.subset_stats(train_indices)
            if subset_stats["missing"] > 0:
                missing_sample = train_indices[pseudo_cache.missing_mask(train_indices)][0]
                raise ValueError(
                    f"伪标签文件缺少 {subset_stats['missing']} 个训练样本。示例索引: {missing_sample}"
                )
            pseudo_cache.training_stats = subset_stats
            if not is_grid_search:
                print(
                    f"💡 训练样本伪标签覆盖: {subset_stats['covered']}/{subset_stats['total']} "
                    f"({subset_stats['core_ratio']*100:.1f}% 核心点比例)"
                )

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projection_head.to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    return train_superclass(
        projection_head,
        model,
        train_loader,
        test_loader_labelled,
        test_loader_unlabelled,
        args,
        pseudo_cache=pseudo_cache,
        model_saver=model_saver,
        progress_parent=progress_parent,
        feature_cache_args={
            "superclass_name": args.superclass_name,
            "cache_dir": args.feature_cache_dir if args.feature_cache_dir else feature_cache_dir,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "gpu": args.gpu,
            "prop_train_labels": args.prop_train_labels,
            "seed": args.seed,
        } if getattr(args, "save_features_and_exit", False) else None,
    )


def build_superclass_train_parser(add_help=True):
    parser = argparse.ArgumentParser(
            description='Train GCD on CIFAR-100 superclasses',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=add_help)

    # =====================================================================
    # 数据集和训练基础参数
    # =====================================================================
    parser.add_argument('--batch_size', default=128, type=int,
                        help='训练批次大小。默认值: 128。较大的batch_size可以提高训练稳定性，但需要更多GPU内存')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='数据加载的工作线程数。默认值: 8。根据CPU核心数调整，过大可能导致内存开销增加')
    parser.add_argument('--eval_funcs', nargs='+', default=['v1', 'v2'],
                        help='使用的评估函数列表。默认值: [v1, v2]。v1为标准聚类准确率，v2为匈牙利算法匹配准确率')

    # =====================================================================
    # 模型架构参数
    # =====================================================================
    parser.add_argument('--warmup_model_dir', type=str, default=None,
                        help='预训练权重文件路径。默认值: None。如果提供，将从该路径加载预训练模型权重进行微调')
    parser.add_argument('--model_name', type=str, default='vit_dino',
                        help='模型名称标识。默认值: vit_dino。格式为{模型架构}_{预训练方法}，用于实验命名和日志记录')
    parser.add_argument('--dataset_name', type=str, default='cifar100_superclass',
                        help='数据集名称。默认值: cifar100_superclass。固定使用CIFAR-100超类数据集，不建议修改')
    parser.add_argument('--prop_train_labels', type=float, default=0.8,
                        help='已知类样本中用于训练的比例。默认值: 0.8。剩余0.2用作验证集，符合GCD标准划分')

    # =====================================================================
    # 超类训练模式参数
    # =====================================================================
    parser.add_argument('--superclass_name', type=str, default=None,
                        help='指定要训练的单个超类名称。默认值: None。例如: aquatic_mammals, fish, flowers等。'
                             '可用超类请查看CIFAR-100超类定义')
    parser.add_argument('--train_all_superclasses', action='store_true',
                        help='批量训练所有15个超类的标志。默认值: False。启用后将依次训练所有超类，忽略--superclass_name参数')

    # =====================================================================
    # 优化器和学习率参数
    # =====================================================================
    parser.add_argument('--grad_from_block', type=int, default=11,
                        help='ViT模型从第几个block开始计算梯度。默认值: 11。ViT-Base共12个blocks (0-11)，'
                             '仅微调最后几层可以减少计算量并防止过拟合')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='初始学习率。默认值: 0.1。使用余弦退火调度器，训练过程中会逐渐降低至lr*1e-3')
    parser.add_argument('--save_best_thresh', type=float, default=None,
                        help='保存最佳模型的准确率阈值。默认值: None。如果设置，仅当准确率超过该阈值时才保存模型')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='学习率衰减因子。默认值: 0.1。用于StepLR调度器（当前使用CosineAnnealing，此参数暂未使用）')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD优化器的动量系数。默认值: 0.9。有助于加速收敛并减少振荡')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减系数(L2正则化)。默认值: 1e-4。防止模型过拟合')
    parser.add_argument('--epochs', default=20, type=int,
                        help='训练总轮数。默认值: 20。配合早停机制(patience=20)，实际训练可能提前结束')

    # =====================================================================
    # 路径和实验配置参数
    # =====================================================================
    parser.add_argument('--exp_root', type=str, default=exp_root,
                        help=f'实验输出根目录。默认值: {exp_root}。所有模型、日志、TensorBoard文件将保存在此目录下')
    parser.add_argument('--transform', type=str, default='imagenet',
                        help='数据增强方案名称。默认值: imagenet。使用ImageNet标准的数据增强策略')
    parser.add_argument('--seed', default=1, type=int,
                        help='随机种子。默认值: 1。用于保证实验可复现性')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的GPU设备编号。默认值: 0。如果有多块GPU，可以指定使用哪一块(0, 1, 2...)')

    # =====================================================================
    # 训练检查点参数（用于断点续训与离线SSDDBC集成）
    # =====================================================================
    parser.add_argument('--resume_from_ckpt', type=str, default=None,
                        help='可选：从指定检查点文件恢复训练（保存了模型、优化器、调度器和训练会话状态）。')
    parser.add_argument('--save_ckpt_every', type=int, default=0,
                        help='可选：每隔多少个epoch保存一次完整训练检查点（0表示不自动保存）。')
    parser.add_argument('--pseudo_labels_path', type=str, default=None,
                        help='可选：离线 SSDDBC 生成的伪标签文件(.npz)路径。提供后训练会加载伪标签进行阶段2/3实验。')
    parser.add_argument('--pseudo_weight_mode', type=str, default='none',
                        choices=['none', 'density', 'inverse_density'],
                        help='伪标签损失权重模式：none=均匀，density=高密度高权重，inverse_density=低密度高权重（权重裁剪到0.2-0.8）。')
    parser.add_argument('--pseudo_loss_weight', type=float, default=1.0,
                        help='伪标签损失的整体权重系数，最终权重 = γ * pseudo_loss_weight（默认: 1.0）')
    parser.add_argument('--pseudo_for_labeled_mode', type=str, default='off',
                        choices=['off', 'all'],
                        help="伪标签损失的样本范围：off=仅未标注样本，all=已标注+未标注一起使用")
    parser.add_argument('--warmup_epochs', type=int, default=50,
                        help='伪标签权重预热轮数，epoch < warmup_epochs 时 γ=0（默认: 50）')
    parser.add_argument('--reuse_log_dir', type=str, default=None,
                        help='可选：复用既有 TensorBoard 日志目录，便于在多阶段训练中拼接曲线。')
    parser.add_argument('--stop_at_epoch', type=int, default=None,
                        help='可选：在指定 epoch 后提前停止训练，常用于阶段1预热。')
    parser.add_argument('--save_features_and_exit', action='store_true',
                        help='可选：在 stop_at_epoch 停止时自动触发特征缓存脚本。')
    parser.add_argument('--feature_cache_dir', type=str, default=None,
                        help='可选：特征缓存目录。未指定时使用 config.py 中的默认值。')
    parser.add_argument('--update_interval', type=int, default=None,
                        help='伪标签更新间隔（用于 γ 在前两个间隔内完成上升；未指定则使用线性调度）')

    # =====================================================================
    # 对比学习框架参数
    # =====================================================================
    parser.add_argument('--base_model', type=str, default='vit_dino',
                        help='基础模型架构。默认值: vit_dino。当前仅支持vit_dino (ViT-Base with DINO预训练)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='对比学习温度系数。默认值: 1.0。控制softmax的平滑度，较小的值使分布更尖锐')
    parser.add_argument('--sup_con_weight', type=float, default=0.5,
                        help='监督对比损失的权重。默认值: 0.5。总损失 = (1-w)*对比损失 + w*监督对比损失。'
                             '范围[0,1]，0表示纯无监督，1表示纯监督')
    parser.add_argument('--n_views', default=2, type=int,
                        help='对比学习的视图数量。默认值: 2。每个样本生成n个增强视图用于对比学习')
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False,
                        help='是否仅对未标记样本计算对比损失。默认值: False。True时对比损失仅用于未知类样本')

    return parser


def main():
    parser = build_superclass_train_parser()
    args = parser.parse_args()

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA不可用，使用CPU")

    print("🚀 CIFAR-100超类GCD训练脚本")
    print("=" * 80)
    print(f"💻 使用设备: {device}")

    # 将device添加到args中，方便其他函数使用
    args.device = device

    # 检查训练模式
    if args.train_all_superclasses:
        print("🌟 模式: 训练所有15个超类")
        train_all_superclasses(args)
    elif args.superclass_name:
        print(f"🎯 模式: 训练单个超类 '{args.superclass_name}'")

        # 验证超类名称
        if args.superclass_name not in SUPERCLASS_NAMES:
            print(f"❌ 错误: 未知的超类名称 '{args.superclass_name}'")
            print(f"📋 可用的超类: {SUPERCLASS_NAMES}")
            sys.exit(1)

        # 获取类别划分
        args = get_class_splits(args)
        args.num_labeled_classes = len(args.train_classes)
        args.num_unlabeled_classes = len(args.unlabeled_classes)

        # 初始化实验 / 复用日志
        exp_name = f'superclass_{args.superclass_name}_{args.model_name}'
        args.exp_name = exp_name
        reuse_log_dir = getattr(args, "reuse_log_dir", None)
        if reuse_log_dir:
            reuse_log_dir = os.path.abspath(reuse_log_dir)
            if not os.path.isdir(reuse_log_dir):
                raise FileNotFoundError(f"reuse_log_dir 不存在: {reuse_log_dir}")
            args.log_dir = reuse_log_dir
            args.model_dir = os.path.join(args.log_dir, 'checkpoints')
            os.makedirs(args.model_dir, exist_ok=True)
            args.model_path = os.path.join(args.model_dir, 'model.pt')
            if getattr(args, "writer", None):
                try:
                    args.writer.close()
                except Exception:
                    pass
            args.writer = SummaryWriter(log_dir=args.log_dir)
            if not getattr(args, 'is_grid_search', False):
                print(f"🔁 复用 TensorBoard 日志目录: {args.log_dir}")
        else:
            init_experiment(args, runner_name=['superclass_train'])
        print(f'Using evaluation function {args.eval_funcs[0]} to print results')

        # 训练
        train_single_superclass(args)
    else:
        print("❌ 错误: 请指定以下选项之一:")
        print("   --superclass_name <名称>  : 训练单个超类")
        print("   --train_all_superclasses  : 训练所有15个超类")
        print(f"📋 可用的超类: {SUPERCLASS_NAMES}")
        sys.exit(1)

    print("\n🎉 训练完成!")


if __name__ == "__main__":
    main()
