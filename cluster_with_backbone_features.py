#!/usr/bin/env python3
"""
使用骨干网络特征进行自适应密度聚类
集成到GCD训练文件中的聚类方案
"""

import torch
import numpy as np
import argparse
from tqdm import tqdm
from adaptive_density_clustering import AdaptiveDensityClustering, evaluate_clustering_results
# 可选特征增强模块（如果不存在则跳过）
try:
    from feature_enhancement import FeatureEnhancer, compute_class_separability
    FEATURE_ENHANCEMENT_AVAILABLE = True
except ImportError:
    print("⚠️ 特征增强模块不可用，将跳过特征增强功能")
    FEATURE_ENHANCEMENT_AVAILABLE = False
from data.get_datasets import get_datasets, get_class_splits
from data.cifar100_superclass import CIFAR100_SUPERCLASSES
from models import vision_transformer as vits
from config import dino_pretrain_path
from project_utils.general_utils import str2bool
from project_utils.cluster_and_log_utils import log_accs_from_preds


def extract_backbone_features(model, dataloader, device):
    """
    提取骨干网络特征

    Args:
        model: 训练好的骨干网络
        dataloader: 数据加载器
        device: 设备

    Returns:
        features: 特征矩阵 [n_samples, feat_dim]
        labels: 真实标签
        indices: 样本索引
    """
    model.eval()
    all_features = []
    all_labels = []
    all_indices = []

    print("🔄 提取骨干网络特征...")

    with torch.no_grad():
        for batch_idx, (images, labels, indices) in enumerate(tqdm(dataloader)):
            images = images.to(device)

            # 提取backbone特征（不使用projection head）
            features = model(images)
            features = torch.nn.functional.normalize(features, dim=-1)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_indices.extend(indices.numpy())

    # 合并所有特征
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    indices = np.array(all_indices)

    print(f"✅ 提取完成! 特征形状: {features.shape}")

    return features, labels, indices


def create_known_unknown_mask(labels, train_classes):
    """
    创建已知/未知类别掩码

    Args:
        labels: 真实标签
        train_classes: 训练时的已知类别

    Returns:
        mask: 布尔掩码，True表示已知类别
        known_labels: 已知标签数组，-1表示未知
    """
    mask = np.array([label in train_classes for label in labels])
    known_labels = np.where(mask, labels, -1)

    print(f"📊 数据分布:")
    print(f"   已知类别样本: {mask.sum()}")
    print(f"   未知类别样本: {(~mask).sum()}")
    print(f"   已知类别: {sorted(list(train_classes))}")

    return mask, known_labels


def adaptive_density_clustering_test(model, test_loader, args, device):
    """
    使用自适应密度聚类进行测试

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        args: 参数配置
        device: 设备

    Returns:
        clustering_results: 聚类结果字典
    """
    print("\n" + "="*80)
    print("🧠 自适应密度聚类测试")
    print("="*80)

    # 提取特征
    features, true_labels, sample_indices = extract_backbone_features(
        model, test_loader, device
    )

    # 创建已知/未知掩码
    mask, known_labels = create_known_unknown_mask(true_labels, args.train_classes)

    # 特征增强（如果启用且可用）
    if args.enable_feature_enhancement and FEATURE_ENHANCEMENT_AVAILABLE:
        print(f"\n🔧 特征增强处理...")
        print(f"   增强方法: {args.enhancement_method}")

        # 计算原始可分性
        original_separability = compute_class_separability(features, true_labels)

        # 初始化特征增强器
        enhancer = FeatureEnhancer(
            enhancement_method=args.enhancement_method,
            push_strength=args.push_strength,
            pull_strength=args.pull_strength
        )

        # 执行特征增强
        enhanced_features = enhancer.fit_transform(features, known_labels)

        # 计算增强后可分性
        enhanced_separability = compute_class_separability(enhanced_features, true_labels)

        print(f"   原始可分性: {original_separability:.3f}")
        print(f"   增强可分性: {enhanced_separability:.3f}")
        if original_separability > 0:
            improvement = enhanced_separability / original_separability
            print(f"   改进倍数: {improvement:.2f}x")

        features = enhanced_features
        print(f"✅ 特征增强完成!")
    elif args.enable_feature_enhancement and not FEATURE_ENHANCEMENT_AVAILABLE:
        print("⚠️ 特征增强已启用但模块不可用，跳过特征增强步骤")

    # 初始化聚类器（使用命令行参数或自适应值）
    # 计算自适应参数
    adaptive_k = max(5, min(20, int(len(features) * 0.1)))
    adaptive_min_size = 3

    # 使用命令行参数或自适应值
    k_neighbors = args.k_neighbors if args.k_neighbors is not None else adaptive_k
    min_cluster_size = args.min_cluster_size if args.min_cluster_size is not None else adaptive_min_size

    clusterer = AdaptiveDensityClustering(
        k_neighbors=k_neighbors,
        density_percentile=args.density_percentile,
        lambda_weight=args.lambda_weight,
        min_cluster_size=min_cluster_size,
        standardize=args.standardize_features
    )

    print(f"🔧 聚类参数:")
    print(f"   k近邻数: {clusterer.k}")
    print(f"   密度阈值: {clusterer.density_percentile}分位数")
    print(f"   权重系数λ: {clusterer.lambda_weight}")
    print(f"   最小聚类大小: {clusterer.min_cluster_size}")

    # 执行聚类（使用新的start_new_clustering方法）
    # 将特征分为训练和查询部分（模拟原始函数的输入方式）
    mid_point = len(features) // 2
    train_x = features[:mid_point]
    query_x = features[mid_point:]
    train_y = true_labels[:mid_point]
    query_y = true_labels[mid_point:]

    print(f"🔄 使用新版本聚类算法...")
    print(f"   训练样本: {len(train_x)}")
    print(f"   查询样本: {len(query_x)}")

    # 调用新的聚类方法
    acc, nmi, ari, sh = clusterer.start_new_clustering(train_x, train_y, query_x, query_y)

    # 获取聚类预测结果
    cluster_predictions = clusterer.cluster_assignments

    # 基础评估（使用新算法内部计算的指标）
    basic_metrics = {
        'accuracy': acc,
        'nmi': nmi,
        'ari': ari,
        'silhouette': sh,
        'n_clusters_predicted': len(clusterer.clusters),
        'n_clusters_true': len(set(true_labels))
    }

    # GCD风格评估
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=true_labels,
        y_pred=cluster_predictions,
        mask=mask,
        eval_funcs=args.eval_funcs,
        save_name='Adaptive_Density_Clustering_NewVersion',
        writer=args.writer
    )

    # 识别未知聚类
    unknown_clusters = clusterer.get_unknown_clusters(known_labels)

    # 分析聚类质量
    cluster_analysis = analyze_cluster_quality(
        clusterer.clusters, features, true_labels, known_labels
    )

    results = {
        'cluster_predictions': cluster_predictions,
        'all_acc': all_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'basic_metrics': basic_metrics,
        'unknown_clusters': unknown_clusters,
        'cluster_analysis': cluster_analysis,
        'n_clusters_found': len(clusterer.clusters)
    }

    print(f"\n📈 GCD评估结果:")
    print(f"   All ACC: {all_acc:.4f}")
    print(f"   Old ACC: {old_acc:.4f}")
    print(f"   New ACC: {new_acc:.4f}")
    print(f"   发现聚类数: {len(clusterer.clusters)}")
    print(f"   未知聚类数: {len(unknown_clusters)}")

    return results


def analyze_cluster_quality(clusters, features, true_labels, known_labels):
    """
    分析聚类质量

    Args:
        clusters: 聚类列表
        features: 特征矩阵
        true_labels: 真实标签
        known_labels: 已知标签

    Returns:
        analysis: 分析结果字典
    """
    analysis = {
        'cluster_sizes': [],
        'cluster_purities': [],
        'known_cluster_count': 0,
        'unknown_cluster_count': 0,
        'mixed_cluster_count': 0
    }

    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue

        cluster_points = list(cluster)
        cluster_size = len(cluster_points)
        analysis['cluster_sizes'].append(cluster_size)

        # 计算纯度
        cluster_true_labels = true_labels[cluster_points]
        most_common_label = np.bincount(cluster_true_labels).argmax()
        purity = np.sum(cluster_true_labels == most_common_label) / cluster_size
        analysis['cluster_purities'].append(purity)

        # 分析聚类类型
        cluster_known_labels = known_labels[cluster_points]
        has_known = np.any(cluster_known_labels != -1)
        has_unknown = np.any(cluster_known_labels == -1)

        if has_known and not has_unknown:
            analysis['known_cluster_count'] += 1
        elif has_unknown and not has_known:
            analysis['unknown_cluster_count'] += 1
        else:
            analysis['mixed_cluster_count'] += 1

    # 计算平均值
    if analysis['cluster_sizes']:
        analysis['avg_cluster_size'] = np.mean(analysis['cluster_sizes'])
        analysis['avg_purity'] = np.mean(analysis['cluster_purities'])
    else:
        analysis['avg_cluster_size'] = 0
        analysis['avg_purity'] = 0

    print(f"\n🔍 聚类质量分析:")
    print(f"   平均聚类大小: {analysis['avg_cluster_size']:.2f}")
    print(f"   平均纯度: {analysis['avg_purity']:.4f}")
    print(f"   纯已知聚类: {analysis['known_cluster_count']}")
    print(f"   纯未知聚类: {analysis['unknown_cluster_count']}")
    print(f"   混合聚类: {analysis['mixed_cluster_count']}")

    return analysis


def load_trained_model(model_path, args, device):
    """
    加载训练好的模型
    """
    print(f"🔄 加载训练模型: {model_path}")

    if args.base_model == 'vit_dino':
        model = vits.__dict__['vit_base']()

        # 加载DINO预训练权重
        if hasattr(args, 'dino_pretrain_path'):
            dino_state_dict = torch.load(dino_pretrain_path, map_location='cpu')
            model.load_state_dict(dino_state_dict, strict=False)
            print(f"   加载DINO预训练权重")

        # 加载训练后的权重
        trained_state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(trained_state_dict)
        print(f"   加载训练权重")

        model.to(device)

        # 设置为评估模式并关闭梯度
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        print(f"✅ 模型加载成功!")
        return model

    else:
        raise NotImplementedError(f"不支持的模型类型: {args.base_model}")


def filter_superclass_data(data_loader, superclass_name):
    """
    过滤出指定超类的数据

    Args:
        data_loader: 数据加载器
        superclass_name: 超类名称

    Returns:
        filtered_data: (features, labels, indices, label_masks)
    """
    if superclass_name not in CIFAR100_SUPERCLASSES:
        print(f"❌ 错误: 未知的超类名称 '{superclass_name}'")
        return None

    superclass_classes = set(CIFAR100_SUPERCLASSES[superclass_name])
    print(f"📊 超类 '{superclass_name}' 包含类别: {sorted(list(superclass_classes))}")

    filtered_images = []
    filtered_labels = []
    filtered_indices = []
    filtered_label_masks = []

    try:
        for batch_idx, batch_data in enumerate(data_loader):
            # 尝试解包数据
            try:
                if len(batch_data) == 4:
                    # 4元素格式：(images, labels, indices, labeled_or_not)
                    images, labels, indices, labeled_or_not = batch_data  # 提取标签mask
                elif len(batch_data) == 3:
                    images, labels, indices = batch_data
                    # 没有标签mask，假设都是有标签的
                    labeled_or_not = torch.ones(len(labels), dtype=torch.long)
                elif len(batch_data) == 2:
                    images, labels = batch_data
                    # 生成默认索引和标签mask
                    batch_size = len(labels)
                    start_idx = batch_idx * data_loader.batch_size
                    indices = torch.arange(start_idx, start_idx + batch_size)
                    labeled_or_not = torch.ones(len(labels), dtype=torch.long)
                else:
                    print(f"⚠️ 跳过异常batch，数据元素数量: {len(batch_data)}")
                    continue
            except Exception as e:
                print(f"⚠️ 解包batch数据时出错: {e}")
                continue

            # 过滤出属于该超类的样本
            try:
                mask = torch.tensor([label.item() in superclass_classes for label in labels])

                if mask.any():
                    filtered_images.append(images[mask])
                    filtered_labels.extend(labels[mask].tolist())
                    filtered_indices.extend(indices[mask].tolist())
                    filtered_label_masks.extend(labeled_or_not[mask].tolist())
            except Exception as e:
                print(f"⚠️ 过滤数据时出错: {e}")
                continue

    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return None

    if not filtered_images:
        print(f"❌ 超类 '{superclass_name}' 中没有找到任何样本")
        return None

    # 合并所有批次
    all_images = torch.cat(filtered_images, dim=0)
    all_labels = torch.tensor(filtered_labels)
    all_indices = torch.tensor(filtered_indices)
    all_label_masks = torch.tensor(filtered_label_masks)

    print(f"✅ 超类数据过滤完成: {len(all_labels)} 个样本")
    print(f"   有标签样本: {all_label_masks.sum().item()}")
    print(f"   无标签样本: {(all_label_masks == 0).sum().item()}")

    return all_images, all_labels, all_indices, all_label_masks


def evaluate_superclass_clustering(model, superclass_name, args, device):
    """
    在指定超类上进行聚类评估 - 增强版本A：合并训练和测试数据

    Args:
        model: 训练好的模型
        superclass_name: 超类名称
        args: 参数配置
        device: 设备

    Returns:
        results: 聚类结果
    """
    print(f"\n" + "="*80)
    print(f"🎯 超类 '{superclass_name}' 增强版自适应密度聚类测试")
    print("="*80)

    # 获取完整数据集（训练+测试）
    from data.augmentations import get_transform
    train_transform, test_transform = get_transform('imagenet', image_size=args.image_size, args=args)

    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )

    # 创建训练和测试数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=False
    )

    # 过滤超类数据 - 训练集
    train_superclass_data = filter_superclass_data(train_loader, superclass_name)
    if train_superclass_data is None:
        print(f"❌ 训练集中未找到超类 '{superclass_name}' 的数据")
        return None

    train_images, train_labels, train_indices, train_label_masks = train_superclass_data

    # 过滤超类数据 - 测试集
    test_superclass_data = filter_superclass_data(test_loader, superclass_name)
    if test_superclass_data is None:
        print(f"❌ 测试集中未找到超类 '{superclass_name}' 的数据")
        return None

    test_images, test_labels, test_indices, test_label_masks = test_superclass_data

    print(f"📊 超类数据统计:")
    print(f"   训练集样本: {len(train_labels)}")
    print(f"   测试集样本: {len(test_labels)}")

    # 创建训练和测试数据加载器
    train_superclass_dataset = torch.utils.data.TensorDataset(train_images, train_labels, train_indices)
    train_superclass_loader = torch.utils.data.DataLoader(
        train_superclass_dataset, batch_size=args.batch_size, shuffle=False
    )

    test_superclass_dataset = torch.utils.data.TensorDataset(test_images, test_labels, test_indices)
    test_superclass_loader = torch.utils.data.DataLoader(
        test_superclass_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 提取训练集特征
    train_features, train_true_labels, train_sample_indices = extract_backbone_features(
        model, train_superclass_loader, device
    )

    # 提取测试集特征
    test_features, test_true_labels, test_sample_indices = extract_backbone_features(
        model, test_superclass_loader, device
    )

    # 重新映射标签到连续的超类内标签
    superclass_classes = sorted(list(CIFAR100_SUPERCLASSES[superclass_name]))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(superclass_classes)}

    train_mapped_labels = np.array([label_mapping[label] for label in train_true_labels])
    test_mapped_labels = np.array([label_mapping[label] for label in test_true_labels])

    print(f"📊 标签映射: {label_mapping}")
    print(f"📊 超类包含类别: {len(superclass_classes)} 个")

    # 初始化增强版聚类器
    n_classes_in_superclass = len(superclass_classes)
    total_samples = len(train_features) + len(test_features)

    # 计算自适应参数
    adaptive_k = max(3, min(10, int(total_samples * 0.05)))
    adaptive_min_size = max(2, int(total_samples * 0.01))

    # 使用命令行参数或自适应值
    k_neighbors = args.k_neighbors if args.k_neighbors is not None else adaptive_k
    min_cluster_size = args.min_cluster_size if args.min_cluster_size is not None else adaptive_min_size

    clusterer = AdaptiveDensityClustering(
        k_neighbors=k_neighbors,
        density_percentile=args.density_percentile,
        lambda_weight=args.lambda_weight,
        min_cluster_size=min_cluster_size,
        standardize=args.standardize_features,
        unknown_threshold=0.3  # 新增：未知检测阈值
    )

    print(f"🔧 增强版超类聚类参数:")
    print(f"   超类类别数: {n_classes_in_superclass}")
    print(f"   总样本数: {total_samples}")
    print(f"   k近邻数: {clusterer.k}")
    print(f"   密度阈值: {clusterer.density_percentile}分位数")
    print(f"   λ权重: {clusterer.lambda_weight}")
    print(f"   最小聚类大小: {clusterer.min_cluster_size}")
    print(f"   未知检测阈值: {clusterer.unknown_threshold}")

    # 执行增强版聚类（合并训练和测试数据）
    print(f"🚀 执行增强版聚类算法...")

    # 调用增强版方法（传入标签掩码信息）
    test_predictions, test_acc, test_nmi, test_ari = clusterer.enhanced_fit_predict(
        train_features, train_mapped_labels, train_label_masks,
        test_features, test_mapped_labels, test_label_masks,
        set(args.train_classes)  # 传入已知类别集合
    )

    # 基础评估（使用增强算法的测试集评估指标）
    basic_metrics = {
        'test_accuracy': test_acc,
        'test_nmi': test_nmi,
        'test_ari': test_ari,
        'n_clusters_predicted': len(clusterer.clusters),
        'n_classes_true': n_classes_in_superclass,
        'train_samples': len(train_features),
        'test_samples': len(test_features)
    }

    # 为GCD评估创建掩码（基于测试集）
    # 检查测试集中是否有未知类别样本
    test_train_classes_mapped = set()
    for orig_class in args.train_classes:
        if orig_class in label_mapping:
            test_train_classes_mapped.add(label_mapping[orig_class])

    test_mask = np.array([label in test_train_classes_mapped for label in test_mapped_labels])
    test_known_labels = np.where(test_mask, test_mapped_labels, -1)

    has_unknown_classes = (~test_mask).sum() > 0

    print(f"📊 测试集标签分布:")
    print(f"   已知类别样本: {test_mask.sum()}")
    print(f"   未知类别样本: {(~test_mask).sum()}")

    if has_unknown_classes:
        # 有未知类的情况：正常计算所有指标（只针对测试集）
        all_acc, old_acc, new_acc = log_accs_from_preds(
            y_true=test_mapped_labels,
            y_pred=test_predictions,
            mask=test_mask,
            eval_funcs=args.eval_funcs,
            save_name=f'Enhanced_Superclass_{superclass_name}_TestOnly',
            writer=args.writer
        )
    else:
        # 没有未知类的情况：只计算已知类准确率
        from project_utils.cluster_utils import cluster_acc
        old_acc = cluster_acc(test_mapped_labels, test_predictions)
        all_acc = old_acc  # 当没有未知类时，All ACC = Old ACC
        new_acc = 0.0      # 没有未知类，New ACC为0

        print(f"⚠️ 超类 '{superclass_name}' 测试集中没有未知类别样本")

    # 识别未知聚类（基于全体数据的已知标签信息）
    # 创建全体数据的已知标签掩码
    all_features = np.concatenate([train_features, test_features], axis=0)
    all_true_labels = np.concatenate([train_mapped_labels, test_mapped_labels], axis=0)

    # 为全体数据创建已知标签掩码
    all_train_classes_mapped = set()
    for orig_class in args.train_classes:
        if orig_class in label_mapping:
            all_train_classes_mapped.add(label_mapping[orig_class])

    all_mask = np.array([label in all_train_classes_mapped for label in all_true_labels])
    all_known_labels = np.where(all_mask, all_true_labels, -1)

    unknown_clusters = clusterer.get_unknown_clusters(all_known_labels)

    # 分析聚类质量（基于全体数据）
    cluster_analysis = analyze_cluster_quality(
        clusterer.clusters, all_features, all_true_labels, all_known_labels
    )

    results = {
        'superclass_name': superclass_name,
        'n_classes': n_classes_in_superclass,
        'n_train_samples': len(train_features),
        'n_test_samples': len(test_features),
        'n_total_samples': len(all_features),
        'test_predictions': test_predictions,
        'all_acc': all_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'basic_metrics': basic_metrics,
        'unknown_clusters': unknown_clusters,
        'cluster_analysis': cluster_analysis,
        'n_clusters_found': len(clusterer.clusters)
    }

    print(f"\n📈 超类 '{superclass_name}' 增强版聚类结果:")
    print(f"   训练样本数: {len(train_features)}")
    print(f"   测试样本数: {len(test_features)}")
    print(f"   总样本数: {len(all_features)}")
    print(f"   真实类别数: {n_classes_in_superclass}")
    print(f"   发现聚类数: {len(clusterer.clusters)}")
    print(f"   未知聚类数: {len(unknown_clusters)}")
    print(f"   ")
    print(f"📊 测试集评估结果:")
    print(f"   All ACC: {all_acc:.4f}")
    print(f"   Old ACC: {old_acc:.4f}")
    print(f"   New ACC: {new_acc:.4f}")
    print(f"   Test ACC: {test_acc:.4f}")
    print(f"   Test NMI: {test_nmi:.4f}")
    print(f"   Test ARI: {test_ari:.4f}")

    # 输出每个聚类的详细标签占比（基于全体数据）
    print_enhanced_cluster_analysis(clusterer.clusters, all_true_labels, all_known_labels,
                                   len(train_features), superclass_classes, superclass_name)

    return results


def evaluate_all_superclasses_clustering(model, args, device):
    """
    批量评估所有超类的聚类性能

    Args:
        model: 训练好的模型
        args: 参数配置
        device: 设备

    Returns:
        all_results: 所有超类的评估结果
    """
    print("\n" + "="*80)
    print("🔍 所有超类批量聚类评估")
    print("="*80)

    all_results = {}

    for superclass_name in CIFAR100_SUPERCLASSES.keys():
        try:
            print(f"\n{'='*20} 评估超类: {superclass_name} {'='*20}")

            result = evaluate_superclass_clustering(model, superclass_name, args, device)

            if result is not None:
                all_results[superclass_name] = result
                print(f"✅ {superclass_name}: All {result['all_acc']:.4f} | "
                      f"Old {result['old_acc']:.4f} | New {result['new_acc']:.4f}")
            else:
                print(f"❌ {superclass_name}: 评估失败")

        except Exception as e:
            print(f"❌ {superclass_name}: 评估出错 - {e}")
            import traceback
            traceback.print_exc()

    # 显示汇总结果
    print(f"\n📊 所有超类聚类评估汇总:")
    print(f"{'超类名称':<25} {'样本数':<8} {'类别数':<8} {'聚类数':<8} {'All ACC':<10} {'Old ACC':<10} {'New ACC':<10}")
    print("-" * 90)

    total_samples = 0
    avg_all_acc = 0
    valid_results = 0

    for superclass_name, result in all_results.items():
        print(f"{superclass_name:<25} {result['n_samples']:<8} {result['n_classes']:<8} "
              f"{result['n_clusters_found']:<8} {result['all_acc']:<10.4f} "
              f"{result['old_acc']:<10.4f} {result['new_acc']:<10.4f}")

        total_samples += result['n_samples']
        avg_all_acc += result['all_acc']
        valid_results += 1

    if valid_results > 0:
        avg_all_acc /= valid_results
        print("-" * 90)
        print(f"{'平均值':<25} {total_samples:<8} {'-':<8} {'-':<8} {avg_all_acc:<10.4f}")

    return all_results


def print_enhanced_cluster_analysis(clusters, true_labels, known_labels, train_size, superclass_classes, superclass_name):
    """
    输出增强版聚类分析，详细区分训练集和测试集样本

    Args:
        clusters: 聚类列表
        true_labels: 真实标签数组 (训练集+测试集)
        known_labels: 已知标签掩码 (-1表示未知)
        train_size: 训练集大小
        superclass_classes: 超类包含的类别列表
        superclass_name: 超类名称
    """
    print(f"\n🔍 增强版聚类分析 - 超类 '{superclass_name}':")
    print("=" * 100)

    # 为每个聚类进行详细分析
    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue

        cluster_points = list(cluster)
        cluster_size = len(cluster_points)

        print(f"\n📊 聚类 {cluster_id} (总样本数: {cluster_size})")
        print("-" * 80)

        # 分离训练集和测试集样本
        train_points = [p for p in cluster_points if p < train_size]
        test_points = [p for p in cluster_points if p >= train_size]

        print(f"样本分布: 训练集 {len(train_points)} 个, 测试集 {len(test_points)} 个")

        # 分析每个类别的详细情况
        from collections import defaultdict

        # 统计每个类别的情况
        class_stats = defaultdict(lambda: {
            'train_known': 0, 'train_unknown': 0, 'test_known': 0, 'test_unknown': 0
        })

        # 处理训练集样本
        for point in train_points:
            true_label = true_labels[point]
            is_known = known_labels[point] != -1

            if is_known:
                class_stats[true_label]['train_known'] += 1
            else:
                class_stats[true_label]['train_unknown'] += 1

        # 处理测试集样本
        for point in test_points:
            true_label = true_labels[point]
            is_known = known_labels[point] != -1

            if is_known:
                class_stats[true_label]['test_known'] += 1
            else:
                class_stats[true_label]['test_unknown'] += 1

        # 输出详细统计
        print(f"\n类别详细分析:")
        print(f"{'类别':<8} {'训练已知':<10} {'训练未知':<10} {'测试已知':<10} {'测试未知':<10} {'总计':<8} {'占比':<8}")
        print("-" * 70)

        total_samples = 0
        for class_label in sorted(class_stats.keys()):
            stats = class_stats[class_label]
            class_total = sum(stats.values())
            percentage = class_total / cluster_size * 100
            total_samples += class_total

            print(f"{class_label:<8} {stats['train_known']:<10} {stats['train_unknown']:<10} "
                  f"{stats['test_known']:<10} {stats['test_unknown']:<10} "
                  f"{class_total:<8} {percentage:<7.1f}%")

        # 汇总统计
        total_train_known = sum(stats['train_known'] for stats in class_stats.values())
        total_train_unknown = sum(stats['train_unknown'] for stats in class_stats.values())
        total_test_known = sum(stats['test_known'] for stats in class_stats.values())
        total_test_unknown = sum(stats['test_unknown'] for stats in class_stats.values())

        print("-" * 70)
        print(f"{'汇总':<8} {total_train_known:<10} {total_train_unknown:<10} "
              f"{total_test_known:<10} {total_test_unknown:<10} "
              f"{cluster_size:<8} {'100.0%':<8}")

        # 聚类特征分析
        print(f"\n聚类特征:")

        # 计算主导类别
        dominant_class = max(class_stats.keys(), key=lambda x: sum(class_stats[x].values()))
        dominant_count = sum(class_stats[dominant_class].values())
        purity = dominant_count / cluster_size

        print(f"  主导类别: {dominant_class} ({dominant_count}/{cluster_size} = {purity:.3f})")

        # 判断聚类类型
        if total_train_known > 0 and total_test_unknown > 0:
            cluster_type = "🟡 混合聚类 (包含训练已知 + 测试未知)"
        elif total_train_known > 0 and total_test_unknown == 0:
            cluster_type = "🟢 已知类别聚类 (主要为训练已知样本)"
        elif total_train_known == 0 and total_test_unknown > 0:
            cluster_type = "🔴 潜在新类别聚类 (主要为测试未知样本)"
        elif total_train_unknown > 0:
            cluster_type = "🟠 训练未知聚类 (包含训练时未知样本)"
        else:
            cluster_type = "⚪ 其他类型聚类"

        print(f"  聚类类型: {cluster_type}")

        # 新类别发现潜力
        if total_test_unknown > 0:
            new_class_potential = total_test_unknown / cluster_size
            print(f"  新类别发现潜力: {new_class_potential:.3f} ({total_test_unknown}个测试未知样本)")

    # 全局统计
    print(f"\n📈 全局聚类统计:")
    print("=" * 100)

    valid_clusters = [c for c in clusters if len(c) > 0]

    global_train_known = 0
    global_train_unknown = 0
    global_test_known = 0
    global_test_unknown = 0

    pure_known_clusters = 0
    pure_unknown_clusters = 0
    mixed_clusters = 0
    potential_new_class_clusters = 0

    for cluster in valid_clusters:
        cluster_points = list(cluster)

        train_points = [p for p in cluster_points if p < train_size]
        test_points = [p for p in cluster_points if p >= train_size]

        cluster_train_known = sum(1 for p in train_points if known_labels[p] != -1)
        cluster_train_unknown = len(train_points) - cluster_train_known
        cluster_test_known = sum(1 for p in test_points if known_labels[p] != -1)
        cluster_test_unknown = len(test_points) - cluster_test_known

        global_train_known += cluster_train_known
        global_train_unknown += cluster_train_unknown
        global_test_known += cluster_test_known
        global_test_unknown += cluster_test_unknown

        # 聚类类型分类
        if cluster_train_known > 0 and cluster_test_unknown > 0:
            mixed_clusters += 1
        elif cluster_train_known > 0 and cluster_test_unknown == 0:
            pure_known_clusters += 1
        elif cluster_train_known == 0 and cluster_test_unknown > 0:
            potential_new_class_clusters += 1

    print(f"总聚类数: {len(valid_clusters)}")
    print(f"  🟢 纯已知类别聚类: {pure_known_clusters}")
    print(f"  🔴 潜在新类别聚类: {potential_new_class_clusters}")
    print(f"  🟡 混合类型聚类: {mixed_clusters}")
    print(f"")
    print(f"样本分布统计:")
    print(f"  训练集已知样本: {global_train_known}")
    print(f"  训练集未知样本: {global_train_unknown}")
    print(f"  测试集已知样本: {global_test_known}")
    print(f"  测试集未知样本: {global_test_unknown}")
    print(f"  ")
    print(f"新类别发现评估:")
    total_test_samples = global_test_known + global_test_unknown
    if total_test_samples > 0:
        unknown_ratio = global_test_unknown / total_test_samples
        print(f"  测试集未知样本比例: {unknown_ratio:.3f} ({global_test_unknown}/{total_test_samples})")
        print(f"  潜在新类别聚类比例: {potential_new_class_clusters/len(valid_clusters):.3f} ({potential_new_class_clusters}/{len(valid_clusters)})")

def print_cluster_label_distribution(clusters, mapped_labels, true_labels, superclass_classes, known_labels, superclass_name):
    """
    输出每个聚类内部的标签占比详情

    Args:
        clusters: 聚类列表
        mapped_labels: 映射后的标签 (0,1,2,...)
        true_labels: 原始标签 (CIFAR-100标签)
        superclass_classes: 超类包含的原始类别
        known_labels: 已知标签掩码
        superclass_name: 超类名称
    """
    print(f"\n🔍 聚类内部标签分布详情 - 超类 '{superclass_name}':")
    print("=" * 80)

    # 创建原始标签到类别名称的映射（如果需要）
    cifar100_class_names = {
        # 这里可以添加CIFAR-100的类别名称，暂时用数字
    }

    # 为每个聚类分析标签分布
    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue

        cluster_points = list(cluster)
        cluster_size = len(cluster_points)

        print(f"\n📊 聚类 {cluster_id} (大小: {cluster_size})")
        print("-" * 50)

        # 获取该聚类的标签信息
        cluster_mapped_labels = mapped_labels[cluster_points]
        cluster_true_labels = true_labels[cluster_points]
        cluster_known_labels = known_labels[cluster_points]

        # 统计已知/未知样本
        known_count = np.sum(cluster_known_labels != -1)
        unknown_count = cluster_size - known_count

        print(f"已知样本: {known_count}/{cluster_size} ({known_count/cluster_size*100:.1f}%)")
        print(f"未知样本: {unknown_count}/{cluster_size} ({unknown_count/cluster_size*100:.1f}%)")

        # 统计原始标签分布
        from collections import Counter
        true_label_counts = Counter(cluster_true_labels)

        print(f"\n原始标签分布:")
        for orig_label, count in sorted(true_label_counts.items()):
            percentage = count / cluster_size * 100
            known_in_this_class = np.sum((cluster_true_labels == orig_label) &
                                       (cluster_known_labels != -1))
            unknown_in_this_class = count - known_in_this_class

            # 获取类别名称
            class_name = cifar100_class_names.get(orig_label, f"类别{orig_label}")

            print(f"  {class_name} (标签{orig_label}): {count}样本 ({percentage:.1f}%) "
                  f"[已知:{known_in_this_class}, 未知:{unknown_in_this_class}]")

        # 统计映射后标签分布
        mapped_label_counts = Counter(cluster_mapped_labels)
        print(f"\n超类内标签分布:")
        for mapped_label, count in sorted(mapped_label_counts.items()):
            percentage = count / cluster_size * 100
            # 找到对应的原始标签
            orig_label = superclass_classes[mapped_label]
            print(f"  超类标签{mapped_label} (原始{orig_label}): {count}样本 ({percentage:.1f}%)")

        # 计算聚类纯度
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
    print(f"\n📈 总体聚类统计:")
    print("-" * 50)

    total_samples = len(mapped_labels)
    pure_known_clusters = 0
    pure_unknown_clusters = 0
    mixed_clusters = 0

    for cluster in clusters:
        if len(cluster) == 0:
            continue

        cluster_points = list(cluster)
        cluster_known_labels = known_labels[cluster_points]

        known_count = np.sum(cluster_known_labels != -1)
        unknown_count = len(cluster_points) - known_count

        if known_count == 0:
            pure_unknown_clusters += 1
        elif unknown_count == 0:
            pure_known_clusters += 1
        else:
            mixed_clusters += 1

    print(f"总聚类数: {len([c for c in clusters if len(c) > 0])}")
    print(f"纯已知聚类: {pure_known_clusters}")
    print(f"纯未知聚类: {pure_unknown_clusters}")
    print(f"混合聚类: {mixed_clusters}")

    # 计算平均聚类纯度
    total_purity = 0
    valid_clusters = 0

    for cluster in clusters:
        if len(cluster) == 0:
            continue

        cluster_points = list(cluster)
        cluster_mapped_labels = mapped_labels[cluster_points]
        label_counts = Counter(cluster_mapped_labels)

        if label_counts:
            purity = max(label_counts.values()) / len(cluster_points)
            total_purity += purity
            valid_clusters += 1

    if valid_clusters > 0:
        avg_purity = total_purity / valid_clusters
        print(f"平均聚类纯度: {avg_purity:.3f}")

    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用自适应密度聚类测试骨干网络')

    # 模型配置
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--feat_dim', default=768, type=int)

    # 数据集配置
    parser.add_argument('--dataset_name', type=str, default='cifar100')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    # 评估配置
    parser.add_argument('--eval_funcs', nargs='+', default=['v1'], help='评估函数')
    parser.add_argument('--gpu', default=0, type=int, help='GPU设备ID')

    # 评估模式配置
    parser.add_argument('--eval_mode', type=str, choices=['full', 'superclass', 'all_superclasses'],
                        default='full', help='评估模式')
    parser.add_argument('--superclass_name', type=str, default=None,
                        help='指定超类名称进行测试')

    # 聚类算法超参数配置
    parser.add_argument('--k_neighbors', type=int, default=None,
                        help='k近邻数量 (默认自适应: max(3, min(10, 样本数*0.1)))')
    parser.add_argument('--density_percentile', type=int, default=70,
                        help='密度阈值百分位数 (默认70)')
    parser.add_argument('--lambda_weight', type=float, default=0.7,
                        help='原型置信度权重 (默认0.7)')
    parser.add_argument('--min_cluster_size', type=int, default=None,
                        help='最小聚类大小 (默认自适应: max(2, 样本数*0.01))')
    parser.add_argument('--standardize_features', type=str2bool, default=True,
                        help='是否标准化特征 (默认True)')

    # 特征增强参数
    parser.add_argument('--enable_feature_enhancement', type=str2bool, default=False,
                        help='是否启用特征增强 (默认False)')
    parser.add_argument('--enhancement_method', type=str, default='multi_scale',
                        choices=['prototype_push', 'contrastive_separation', 'dimension_weighting', 'multi_scale'],
                        help='特征增强方法')
    parser.add_argument('--push_strength', type=float, default=0.1,
                        help='原型推离强度')
    parser.add_argument('--pull_strength', type=float, default=0.2,
                        help='原型拉近强度')

    args = parser.parse_args()

    # 设备配置
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        print(f"💻 使用GPU设备: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA不可用，使用CPU")

    # 设置必要参数
    args.device = device
    args.writer = None
    args.interpolation = 3
    args.crop_pct = 0.875

    # 获取类别划分
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print(f"📊 类别信息:")
    print(f"   已知类别数: {args.num_labeled_classes}")
    print(f"   未知类别数: {args.num_unlabeled_classes}")

    # 加载训练好的模型
    model = load_trained_model(args.model_path, args, device)

    # 根据评估模式执行相应的测试
    if args.eval_mode == 'full':
        # 获取数据集
        from data.augmentations import get_transform
        train_transform, test_transform = get_transform('imagenet', image_size=args.image_size, args=args)

        train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
            args.dataset_name, train_transform, test_transform, args
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, num_workers=args.num_workers,
            batch_size=args.batch_size, shuffle=False
        )

        # 完整数据集评估
        results = adaptive_density_clustering_test(model, test_loader, args, device)

        print("\n🎉 完整数据集聚类测试完成!")
        print(f"最终结果: All ACC: {results['all_acc']:.4f} | "
              f"Old ACC: {results['old_acc']:.4f} | "
              f"New ACC: {results['new_acc']:.4f}")

    elif args.eval_mode == 'superclass':
        # 单个超类评估
        if args.superclass_name is None:
            print("❌ 错误: 超类评估模式需要指定 --superclass_name")
            return

        results = evaluate_superclass_clustering(model, args.superclass_name, args, device)

        if results is not None:
            print(f"\n🎉 超类 '{args.superclass_name}' 聚类测试完成!")
            print(f"最终结果: All ACC: {results['all_acc']:.4f} | "
                  f"Old ACC: {results['old_acc']:.4f} | "
                  f"New ACC: {results['new_acc']:.4f}")

    elif args.eval_mode == 'all_superclasses':
        # 所有超类批量评估
        results = evaluate_all_superclasses_clustering(model, args, device)

        print(f"\n🎉 所有超类聚类测试完成!")
        print(f"共评估了 {len(results)} 个超类")


if __name__ == "__main__":
    main()