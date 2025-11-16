import argparse
import os
import sys

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from data.cifar100_superclass import CIFAR100_SUPERCLASSES

from tqdm import tqdm

from torch.nn import functional as F

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path

# 添加训练工具
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.training_utils import TrainingSession

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def print_kmeans_cluster_distribution(cluster_preds, mapped_labels, original_labels, known_mask, superclass_name, superclass_classes):
    """
    输出K-means聚类内部的标签分布详情 (原版GCD)

    Args:
        cluster_preds: K-means聚类预测结果
        mapped_labels: 映射后的标签 (0,1,2,...)
        original_labels: 原始CIFAR-100标签
        known_mask: 已知标签掩码
        superclass_name: 超类名称
        superclass_classes: 超类包含的原始类别
    """
    print(f"\n🔍 K-means聚类内部标签分布详情 - 超类 '{superclass_name}':")
    print("=" * 80)

    n_clusters = len(set(cluster_preds))

    # 为每个聚类分析标签分布
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_preds == cluster_id
        cluster_size = cluster_mask.sum()

        if cluster_size == 0:
            continue

        print(f"\n📊 聚类 {cluster_id} (大小: {cluster_size})")
        print("-" * 50)

        # 获取该聚类的标签信息
        cluster_mapped_labels = mapped_labels[cluster_mask]
        cluster_original_labels = original_labels[cluster_mask]
        cluster_known_mask = known_mask[cluster_mask]

        # 统计已知/未知样本
        known_count = cluster_known_mask.sum()
        unknown_count = cluster_size - known_count

        print(f"已知样本: {known_count}/{cluster_size} ({known_count/cluster_size*100:.1f}%)")
        print(f"未知样本: {unknown_count}/{cluster_size} ({unknown_count/cluster_size*100:.1f}%)")

        # 统计原始标签分布
        from collections import Counter
        original_label_counts = Counter(cluster_original_labels)

        print(f"\n原始标签分布:")
        for orig_label, count in sorted(original_label_counts.items()):
            percentage = count / cluster_size * 100
            # 计算该类别中已知和未知的数量
            class_mask = cluster_original_labels == orig_label
            known_in_class = (cluster_known_mask[class_mask]).sum()
            unknown_in_class = count - known_in_class

            print(f"  类别{orig_label}: {count}样本 ({percentage:.1f}%) "
                  f"[已知:{known_in_class}, 未知:{unknown_in_class}]")

        # 统计映射后标签分布
        mapped_label_counts = Counter(cluster_mapped_labels)
        print(f"\n超类内标签分布:")
        for mapped_label, count in sorted(mapped_label_counts.items()):
            percentage = count / cluster_size * 100
            # 找到对应的原始标签
            orig_label = superclass_classes[mapped_label]
            print(f"  超类标签{mapped_label} (原始{orig_label}): {count}样本 ({percentage:.1f}%)")

        # 计算聚类纯度
        if mapped_label_counts:
            most_common_mapped = mapped_label_counts.most_common(1)[0]
            purity = most_common_mapped[1] / cluster_size
            dominant_mapped_label = most_common_mapped[0]
            dominant_orig_label = superclass_classes[dominant_mapped_label]

            print(f"\n聚类纯度: {purity:.3f} (主导类别: {dominant_orig_label})")

        # 判断聚类类型
        if known_count == 0:
            cluster_type = "🔴 纯未知聚类"
        elif unknown_count == 0:
            cluster_type = "🟢 纯已知聚类"
        else:
            cluster_type = "🟡 混合聚类"

        print(f"聚类类型: {cluster_type}")

    # 输出总体统计
    print(f"\n📈 K-means总体聚类统计:")
    print("-" * 50)

    pure_known_clusters = 0
    pure_unknown_clusters = 0
    mixed_clusters = 0

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_preds == cluster_id
        if cluster_mask.sum() == 0:
            continue

        cluster_known_mask = known_mask[cluster_mask]
        known_count = cluster_known_mask.sum()
        unknown_count = cluster_mask.sum() - known_count

        if known_count == 0:
            pure_unknown_clusters += 1
        elif unknown_count == 0:
            pure_known_clusters += 1
        else:
            mixed_clusters += 1

    print(f"总聚类数: {n_clusters}")
    print(f"纯已知聚类: {pure_known_clusters}")
    print(f"纯未知聚类: {pure_unknown_clusters}")
    print(f"混合聚类: {mixed_clusters}")

    # 计算平均聚类纯度
    total_purity = 0
    valid_clusters = 0

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_preds == cluster_id
        if cluster_mask.sum() == 0:
            continue

        cluster_mapped_labels = mapped_labels[cluster_mask]
        label_counts = Counter(cluster_mapped_labels)

        if label_counts:
            purity = max(label_counts.values()) / cluster_mask.sum()
            total_purity += purity
            valid_clusters += 1

    if valid_clusters > 0:
        avg_purity = total_purity / valid_clusters
        print(f"平均聚类纯度: {avg_purity:.3f}")

    print("=" * 80)


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, args):

    b_ = 0.5 * int(features.size(0))
    device = features.device  # 从features中获取device

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args):

    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss()
    best_test_acc_lab = 0

    # 初始化训练会话管理器
    training_session = TrainingSession(args, enable_early_stopping=True, patience=20)
    model_info = {
        'name': args.base_model,
        'feat_dim': getattr(args, 'feat_dim', 'Unknown')
    }
    training_session.start_training(model_info)

    for epoch in range(args.epochs):

        # 开始新轮次
        training_session.start_epoch(epoch)

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        projection_head.train()
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(device)

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
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]

            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            # Total loss
            loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        with torch.no_grad():

            print('🔍 Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test_kmeans(model, unlabelled_train_loader,
                                                    epoch=epoch, save_name='Train ACC Unlabelled',
                                                    args=args, device=device)

            print('🔍 Testing on disjoint test set...')
            if args.eval_superclass:
                # 超类评估模式
                all_acc_test, old_acc_test, new_acc_test = test_kmeans_superclass_eval(
                    model, test_loader, epoch=epoch, save_name='Test ACC',
                    args=args, eval_superclass=args.eval_superclass, device=device)
            else:
                # 全数据集评估模式
                all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader,
                                                                       epoch=epoch, save_name='Test ACC',
                                                                       args=args, device=device)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

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

        # 保存模型
        torch.save(model.state_dict(), args.model_path)
        training_session.save_model_info(args.model_path)

        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')

        if old_acc_test > best_test_acc_lab:
            torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
            training_session.save_model_info(args.model_path[:-3] + f'_best.pt', is_best=True, acc=old_acc_test)

            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')

            best_test_acc_lab = old_acc_test

        # 检查早停
        if should_stop:
            print(f"\n🛑 早停触发，在第{epoch+1}轮停止训练")
            training_session.finish_training(epoch, early_stopped=True)
            break

    else:
        # 正常完成所有轮次
        training_session.finish_training(args.epochs - 1, early_stopped=False)


def test_kmeans_superclass_eval(model, test_loader, epoch, save_name, args, eval_superclass=None, device=None):
    """
    支持超类评估的测试函数
    如果指定了eval_superclass，则只在该超类内进行评估
    否则进行全数据集评估
    """
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        if device is None:
            device = torch.device('cuda:0')  # 默认设备
        images = images.to(device)
        feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    all_feats = np.concatenate(all_feats)

    if eval_superclass and eval_superclass in CIFAR100_SUPERCLASSES:
        print(f'🎯 开始超类 "{eval_superclass}" 评估...')

        # 获取超类包含的所有类别
        superclass_classes = CIFAR100_SUPERCLASSES[eval_superclass]

        # 过滤到指定超类的样本
        superclass_mask = np.isin(targets, superclass_classes)
        if superclass_mask.sum() == 0:
            print(f"⚠️ 警告: 测试集中没有找到超类 '{eval_superclass}' 的样本")
            return 0.0, 0.0, 0.0

        filtered_feats = all_feats[superclass_mask]
        filtered_targets = targets[superclass_mask]
        filtered_mask = mask[superclass_mask]

        # 重新映射标签到连续的0-n范围
        unique_labels = sorted(set(filtered_targets))
        label_mapping = {old_label: new_idx for new_idx, old_label in enumerate(unique_labels)}
        mapped_targets = np.array([label_mapping[label] for label in filtered_targets])

        # 更新mask以反映新的标签映射
        known_classes_in_superclass = [cls for cls in superclass_classes if cls in args.train_classes]
        mapped_known_indices = [label_mapping[cls] for cls in known_classes_in_superclass if cls in label_mapping]
        new_mask = np.array([True if mapped_targets[i] in mapped_known_indices else False
                            for i in range(len(mapped_targets))])

        # 聚类参数
        n_clusters = len(unique_labels)

        print(f'   超类包含类别: {superclass_classes}')
        print(f'   过滤后样本数: {len(filtered_feats)}')
        print(f'   实际类别数: {n_clusters}')
        print(f'   已知类别数: {len(mapped_known_indices)}')
        print(f'   未知类别数: {n_clusters - len(mapped_known_indices)}')

        # 检查是否存在未知类
        num_unknown_classes = n_clusters - len(mapped_known_indices)
        has_unknown_classes = num_unknown_classes > 0

        if not has_unknown_classes:
            print(f"⚠️  注意: 超类 '{eval_superclass}' 中所有类别都是已知类，将仅计算Old ACC")

    else:
        # 全数据集评估
        print('🌍 开始全数据集评估...')
        filtered_feats = all_feats
        mapped_targets = targets
        new_mask = mask
        n_clusters = args.num_labeled_classes + args.num_unlabeled_classes
        has_unknown_classes = args.num_unlabeled_classes > 0

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(filtered_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    eval_name = f"{save_name}_{eval_superclass}" if eval_superclass else save_name

    if has_unknown_classes:
        # 有未知类的情况：正常计算所有指标
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=mapped_targets, y_pred=preds, mask=new_mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=eval_name,
                                                        writer=args.writer)

        if eval_superclass:
            print(f"📊 超类 '{eval_superclass}' 评估结果 (包含未知类):")
            print(f"   All ACC: {all_acc:.4f}")
            print(f"   Old ACC: {old_acc:.4f}")
            print(f"   New ACC: {new_acc:.4f}")

            # 显示K-means聚类内部分布
            print_kmeans_cluster_distribution(preds, mapped_targets, targets[superclass_mask],
                                            new_mask, eval_superclass, list(superclass_classes))
    else:
        # 没有未知类的情况：只计算已知类准确率
        from project_utils.cluster_utils import cluster_acc
        old_acc = cluster_acc(mapped_targets, preds)
        all_acc = old_acc  # 当没有未知类时，All ACC = Old ACC
        new_acc = 0.0      # 没有未知类，New ACC为0

        if eval_superclass:
            print(f"📊 超类 '{eval_superclass}' 评估结果 (仅已知类):")
            print(f"   Old ACC (仅已知类): {old_acc:.4f}")
            print(f"   All ACC: {all_acc:.4f} (等于Old ACC)")
            print(f"   New ACC: {new_acc:.4f} (无未知类)")

            # 显示K-means聚类内部分布
            print_kmeans_cluster_distribution(preds, mapped_targets, targets[superclass_mask],
                                            new_mask, eval_superclass, list(superclass_classes))

        # 写入TensorBoard日志
        if args.writer is not None:
            args.writer.add_scalars(f'{eval_name}_v1',
                                   {'Old': old_acc, 'New': new_acc, 'All': all_acc}, epoch)

    return all_acc, old_acc, new_acc


def test_kmeans(model, test_loader,
                epoch, save_name,
                args, device=None):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        if device is None:
            device = torch.device('cuda:0')  # 默认设备
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, n_init=10).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
    parser.add_argument('--gpu', default=0, type=int, help='指定使用的GPU设备ID')

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    # 超类评估参数
    parser.add_argument('--eval_superclass', type=str, default=None,
                        help='指定用于评估的超类名称，如果为None则使用全数据集评估。可选: trees, flowers, vehicles_1, vehicles_2, 等')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()

    # 设置GPU设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        print(f"💻 使用GPU设备: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA不可用，使用CPU")

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

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
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)


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

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projection_head.to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    train(projection_head, model, train_loader, test_loader_labelled, test_loader_unlabelled, args)