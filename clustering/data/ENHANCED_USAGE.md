# 增强型数据提供器使用指南

## 概述

`EnhancedDataProvider` 和 `EnhancedDataset` 是在原有数据读取系统基础上的增强版本，提供：

1. **更全面的信息**：包含所有聚类所需的元数据
2. **更高的效率**：预计算常用信息，避免重复计算
3. **更简洁的API**：一次调用获取所有数据，减少代码冗余

---

## 对比：旧 vs 新

### ❌ 旧方式（存在问题）

```python
# test_superclass.py 中的旧代码

# 问题1: 重复的数据提取逻辑
data_provider = DataProvider(cache_base_dir='/data/gjx/checkpoints/features')
cache_info = data_provider.get_cache_info(superclass_name, use_l2=use_l2)

if cache_info['exists']:
    feature_dict, source = data_provider.get_features(...)
    # 提取数据（12行重复代码）
    all_feats = feature_dict['all_features']
    all_targets = feature_dict['all_targets']
    # ... 10 more lines
else:
    # 再次提取数据（12行重复代码）
    all_feats = feature_dict['all_features']
    # ... 10 more lines

# 问题2: 多次计算相同的信息
train_size = len(train_feats) if use_train_and_test else None  # Line 176
# ...
test_start_idx = len(train_feats)  # Line 127
# ...
test_start_idx = len(train_feats)  # Line 174 (再次计算)

# 问题3: 缺失信息
# - 没有样本索引（indices）
# - 没有特征维度信息
# - 没有数据来源标记
```

### ✅ 新方式（优化后）

```python
from clustering.data import EnhancedDataProvider

# 一次调用获取所有数据
provider = EnhancedDataProvider(cache_base_dir='/data/gjx/checkpoints/features')
dataset = provider.load_dataset(
    dataset_name='trees',
    model_path='/path/to/model.pt',  # 仅缓存不存在时需要
    use_l2=True,
    silent=False
)

# 所有信息都已预计算，直接使用
print(f"数据来源: {dataset.source}")
print(f"训练集大小: {dataset.train_size}")
print(f"测试集起始索引: {dataset.test_start_idx}")
print(f"特征维度: {dataset.feat_dim}")

# 获取聚类输入（一行代码）
X, targets, known_mask, labeled_mask, train_size = dataset.get_clustering_input()

# 获取测试集子集（一行代码）
test_data = dataset.get_test_subset(predictions)
```

---

## EnhancedDataset 完整功能

### 1. 基础数据（与原来相同）

```python
dataset.all_features       # 所有特征
dataset.all_targets        # 所有标签
dataset.all_known_mask     # 已知/未知掩码
dataset.all_labeled_mask   # 有标签/无标签掩码

dataset.train_features     # 训练集特征
dataset.train_targets      # 训练集标签
dataset.train_known_mask   # 训练集已知/未知掩码
dataset.train_labeled_mask # 训练集标签掩码

dataset.test_features      # 测试集特征
dataset.test_targets       # 测试集标签
dataset.test_known_mask    # 测试集已知/未知掩码
dataset.test_labeled_mask  # 测试集标签掩码
```

### 2. 元信息（新增）

```python
dataset.dataset_name       # 数据集名称: 'trees'
dataset.use_l2             # 是否L2归一化: True/False
dataset.source             # 数据来源: 'cache' or 'extraction'
dataset.feat_dim           # 特征维度: 768
```

### 3. 预计算信息（新增，避免重复计算）

```python
# 样本数量
dataset.n_samples          # 总样本数
dataset.train_size         # 训练集样本数
dataset.test_size          # 测试集样本数
dataset.test_start_idx     # 测试集在合并数据中的起始索引

# 划分标记
dataset.has_train_test_split  # 是否有训练/测试划分: True/False

# 统计信息
dataset.n_known            # 已知类样本数
dataset.n_unknown          # 未知类样本数
dataset.n_labeled          # 有标签样本数
dataset.n_unlabeled        # 无标签样本数

# 类别信息
dataset.n_classes          # 总类别数
dataset.n_known_classes    # 已知类别数
dataset.n_unknown_classes  # 未知类别数
```

### 4. 便捷方法

#### 4.1 获取聚类输入

```python
X, targets, known_mask, labeled_mask, train_size = dataset.get_clustering_input()

# 直接用于聚类
predictions, n_clusters, unknown_clusters = adaptive_density_clustering(
    X, targets, known_mask, labeled_mask,
    k=10, density_percentile=75, train_size=train_size, ...
)
```

#### 4.2 获取测试集子集

```python
# 用于ACC计算（自动处理训练/测试划分）
test_data = dataset.get_test_subset(predictions)

# test_data 包含:
# - features: 测试集特征
# - targets: 测试集标签
# - known_mask: 测试集已知/未知掩码
# - predictions: 测试集预测结果
# - n_samples: 测试集样本数

# 直接用于ACC计算
all_acc, old_acc, new_acc = split_cluster_acc_v2(
    test_data['targets'],
    test_data['predictions'],
    test_data['known_mask']
)
```

#### 4.3 打印摘要

```python
dataset.print_summary(silent=False)

# 输出示例:
# 📊 数据集信息:
#    名称: trees
#    数据来源: cache
#    L2归一化: 是
#    特征维度: 768
#
# 📊 样本统计:
#    总样本数: 3000
#    训练集: 2500 样本
#    测试集: 500 样本
#    已知类样本: 2400 (80.0%)
#    未知类样本: 600 (20.0%)
#    有标签样本: 2000 (66.7%)
#    无标签样本: 1000 (33.3%)
#
# 📊 类别统计:
#    总类别数: 5
#    已知类别数: 4
#    未知类别数: 1
```

---

## 完整示例：test_superclass.py 优化后

### 优化前（104行重复代码）

```python
def test_adaptive_clustering_on_superclass(superclass_name, model_path, ...):
    data_provider = DataProvider(...)
    cache_info = data_provider.get_cache_info(superclass_name, use_l2=use_l2)

    if cache_info['exists']:
        feature_dict, source = data_provider.get_features(...)
        all_feats = feature_dict['all_features']
        all_targets = feature_dict['all_targets']
        # ... 10 more lines (12行提取代码)
    else:
        model_loader = ModelLoader(...)
        model = model_loader.load(...)
        dataset_loader = DatasetLoader(...)
        data_loaders = dataset_loader.load(...)
        feature_dict, source = data_provider.get_features(...)
        all_feats = feature_dict['all_features']
        all_targets = feature_dict['all_targets']
        # ... 10 more lines (12行重复提取代码)

    # 手动打印统计信息（8行）
    print(f"数据来源: {source}")
    print(f"总样本数: {len(all_feats)}")
    # ...

    train_size = len(train_feats) if use_train_and_test else None

    # 聚类
    clustering_result = adaptive_density_clustering(
        all_feats, all_targets, all_known_mask, all_labeled_mask,
        train_size=train_size, ...
    )

    # 计算测试集范围（重复逻辑）
    if use_train_and_test:
        test_start_idx = len(train_feats)
        test_predictions = predictions[test_start_idx:]
        test_targets_for_acc = all_targets[test_start_idx:]
        # ...
    else:
        test_predictions = predictions
        # ...

    # ACC计算
    all_acc, old_acc, new_acc = split_cluster_acc_v2(...)

    # 再次计算test_start_idx（重复）
    if use_train_and_test:
        test_start_idx = len(train_feats)
        test_features_for_kmeans = all_feats[test_start_idx:]
    # ...
```

### 优化后（代码减少50%+）

```python
from clustering.data import EnhancedDataProvider

def test_adaptive_clustering_on_superclass(superclass_name, model_path, ...):
    # 步骤1: 获取超类配置
    superclass_info = get_superclass_info(superclass_name)

    # 步骤2: 加载数据（一次调用，自动处理缓存）
    provider = EnhancedDataProvider(cache_base_dir='/data/gjx/checkpoints/features')
    dataset = provider.load_dataset(
        dataset_name=superclass_name,
        model_path=model_path,
        use_l2=use_l2,
        use_train_and_test=use_train_and_test,
        silent=silent
    )

    # 打印摘要（一行代码替代8行）
    dataset.print_summary(silent=silent)

    # 步骤3: 获取聚类输入（一行代码）
    X, targets, known_mask, labeled_mask, train_size = dataset.get_clustering_input()

    # 步骤4: 运行聚类
    clustering_result = adaptive_density_clustering(
        X, targets, known_mask, labeled_mask,
        k=k, density_percentile=density_percentile,
        train_size=train_size, ...
    )

    # 步骤5: 获取测试集数据并计算ACC（一行代码替代10行）
    predictions, n_clusters, unknown_clusters = clustering_result
    test_data = dataset.get_test_subset(predictions)

    all_acc, old_acc, new_acc = split_cluster_acc_v2(
        test_data['targets'],
        test_data['predictions'],
        test_data['known_mask']
    )

    # 步骤6: K-means基线（直接使用test_data）
    kmeans_baseline = test_kmeans_baseline(
        test_data['features'],
        test_data['targets'],
        test_data['known_mask'], ...
    )

    # 步骤7: 返回结果
    return {
        'all_acc': all_acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'n_clusters': n_clusters,
        'dataset': dataset,  # 包含所有元信息
        ...
    }
```

---

## 优势总结

### 1. 消除重复逻辑

| 旧方式 | 新方式 | 减少代码 |
|--------|--------|----------|
| 两次12行数据提取 | 一次 | -12行 |
| 3次计算train_size | 预计算 | -2行 |
| 2次计算test_start_idx | 预计算 | -2行 |
| 8行统计打印 | 1行方法调用 | -7行 |
| 10行测试集提取 | 1行方法调用 | -9行 |
| **总计** | **总计** | **-32行** |

### 2. 添加缺失信息

| 信息 | 旧方式 | 新方式 |
|------|--------|--------|
| 样本索引 | ❌ 无 | ✅ 支持（预留） |
| 数据来源 | ⚠️ 变量 | ✅ dataset.source |
| 特征维度 | ❌ 无 | ✅ dataset.feat_dim |
| train_size | ⚠️ 需计算 | ✅ dataset.train_size |
| test_start_idx | ⚠️ 需计算 | ✅ dataset.test_start_idx |
| 统计信息 | ❌ 无 | ✅ 完整统计 |

### 3. 提高效率

- ✅ 避免重复计算（train_size, test_start_idx等）
- ✅ 一次API调用获取所有数据
- ✅ 预计算统计信息（n_known, n_classes等）
- ✅ 便捷方法减少样板代码

---

## 迁移指南

### 步骤1: 导入新模块

```python
# 旧代码
from clustering.data import DataProvider, ModelLoader, DatasetLoader

# 新代码
from clustering.data import EnhancedDataProvider
```

### 步骤2: 替换数据加载逻辑

```python
# 旧代码（30+行）
data_provider = DataProvider(...)
cache_info = data_provider.get_cache_info(...)
if cache_info['exists']:
    feature_dict, source = data_provider.get_features(...)
    all_feats = feature_dict['all_features']
    # ... 20+ more lines
else:
    # ... 30+ more lines

# 新代码（5行）
provider = EnhancedDataProvider(cache_base_dir='...')
dataset = provider.load_dataset(
    dataset_name=superclass_name,
    model_path=model_path,
    use_l2=use_l2
)
```

### 步骤3: 使用预计算信息

```python
# 旧代码
train_size = len(train_feats) if use_train_and_test else None
test_start_idx = len(train_feats)

# 新代码
train_size = dataset.train_size
test_start_idx = dataset.test_start_idx
```

### 步骤4: 使用便捷方法

```python
# 旧代码（获取聚类输入）
clustering_result = adaptive_density_clustering(
    all_feats, all_targets, all_known_mask, all_labeled_mask,
    train_size=train_size, ...
)

# 新代码
X, targets, known_mask, labeled_mask, train_size = dataset.get_clustering_input()
clustering_result = adaptive_density_clustering(
    X, targets, known_mask, labeled_mask,
    train_size=train_size, ...
)
```

```python
# 旧代码（获取测试集）
if use_train_and_test:
    test_start_idx = len(train_feats)
    test_predictions = predictions[test_start_idx:]
    test_targets_for_acc = all_targets[test_start_idx:]
    test_known_mask_for_acc = all_known_mask[test_start_idx:]
else:
    test_predictions = predictions
    test_targets_for_acc = all_targets
    test_known_mask_for_acc = all_known_mask

# 新代码
test_data = dataset.get_test_subset(predictions)
# 直接使用 test_data['targets'], test_data['predictions'], test_data['known_mask']
```

---

## 兼容性

- ✅ 完全向后兼容（旧代码仍可使用）
- ✅ 数据格式相同（与缓存兼容）
- ✅ 可逐步迁移（不需要一次性修改所有代码）

---

## 更新历史

- **2025-01-20**: 初始版本，创建EnhancedDataProvider和EnhancedDataset
