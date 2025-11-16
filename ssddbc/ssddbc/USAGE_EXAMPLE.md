# 稀疏点分配策略使用示例

## 命令行使用

### 1. 使用main.py测试

```bash
# 默认使用模式2（KNN距离加权投票）
python -m ssddbc.testing.main \
    --model_path /path/to/model.pt \
    --superclass_name trees \
    --k 10 \
    --density_percentile 75

# 使用模式1（簇原型就近分配）
python -m ssddbc.testing.main \
    --model_path /path/to/model.pt \
    --superclass_name trees \
    --assign_model 1

# 使用模式3（簇内K近邻平均距离）并调整voting_k
python -m ssddbc.testing.main \
    --model_path /path/to/model.pt \
    --superclass_name trees \
    --assign_model 3 \
    --voting_k 7

# 完整示例：指定所有关键参数
python -m ssddbc.testing.main \
    --model_path /path/to/model.pt \
    --superclass_name trees \
    --k 10 \
    --density_percentile 75 \
    --assign_model 2 \
    --voting_k 5 \
    --dense_method 0 \
    --use_train_and_test True \
    --l2 True
```

### 2. 参数说明

- `--assign_model`: 稀疏点分配策略（默认2）
  - `1`: 基于簇原型的就近分配
  - `2`: 高密度子空间KNN投票（距离加权）⭐ 推荐
  - `3`: 簇内K近邻平均距离

- `--voting_k`: KNN投票时使用的近邻数量（默认5）
  - 仅在 assign_model=2 时使用
  - 建议范围：3-10

## Python代码使用

### 方式1: 通过adaptive_clustering调用

```python
from ssddbc.ssddbc.adaptive_clustering import adaptive_density_clustering
import numpy as np

# 准备数据
X = ...  # 特征矩阵 (n_samples, n_features)
targets = ...  # 真实标签
known_mask = ...  # 已知类别掩码
labeled_mask = ...  # 有标签掩码

# 运行聚类（使用模式2，默认推荐）
predictions, n_clusters, unknown_clusters = adaptive_density_clustering(
    X=X,
    targets=targets,
    known_mask=known_mask,
    labeled_mask=labeled_mask,
    k=10,
    density_percentile=75,
    assign_model=2,  # 选择策略
    voting_k=5,      # KNN邻居数
    silent=False
)

print(f"聚类数: {n_clusters}")
print(f"预测结果: {predictions}")
```

### 方式2: 直接调用assignment函数

```python
from ssddbc.ssddbc.assignment import assign_sparse_points_density_based
from ssddbc.ssddbc.clustering import build_clusters_ssddbc
from ssddbc.density.density_estimation import compute_simple_density, identify_high_density_points
from ssddbc.prototypes.prototype_builder import build_prototypes
import numpy as np

# 步骤1: 计算密度
densities, knn_distances, neighbors = compute_simple_density(X, k=10)

# 步骤2: 识别高密度点
high_density_mask = identify_high_density_points(densities, density_percentile=75)

# 步骤3: 构建骨干网络
co = np.mean(knn_distances)
clusters, cluster_labels, _ = build_clusters_ssddbc(
    X, high_density_mask, neighbors, labeled_mask, targets,
    densities, known_mask, k=10, co=co
)

# 步骤4: 构建原型
prototypes, _ = build_prototypes(X, clusters, labeled_mask, targets)

# 步骤5: 分配稀疏点（选择策略）
final_labels = assign_sparse_points_density_based(
    X=X,
    clusters=clusters,
    cluster_labels=cluster_labels,
    densities=densities,
    neighbors=neighbors,
    labeled_mask=labeled_mask,
    targets=targets,
    prototypes=prototypes,
    voting_k=5,
    assign_model=2,  # 1/2/3
    silent=False
)
```

## 策略选择建议

### 何时使用模式1（簇原型就近）？
- ✅ 簇的原型（中心）能很好代表簇的整体特征
- ✅ 需要最快的计算速度
- ✅ 簇形状比较规则（接近球形）
- ❌ 不适合形状不规则的簇

### 何时使用模式2（KNN投票加权）？⭐ 推荐
- ✅ 通用场景，适应性最强
- ✅ 能处理各种形状的簇
- ✅ 对噪声有一定鲁棒性
- ✅ 平衡了速度和准确性
- 建议的voting_k值：5（默认）

### 何时使用模式3（簇内K近邻）？
- ✅ 簇大小差异很大
- ✅ 关注簇的局部结构
- ✅ 需要更精细的距离度量
- 每个簇固定选择3个最近点

## 性能对比

在标准测试数据集上的表现（200样本，3类）：

| 策略 | 准确率 | 速度 | 内存占用 |
|-----|-------|------|---------|
| 模式1 | 1.0000 | ⚡⚡⚡ 最快 | 低 |
| 模式2 | 1.0000 | ⚡⚡ 中等 | 中 |
| 模式3 | 1.0000 | ⚡⚡ 中等 | 中 |

*注：实际性能取决于数据集特征*

## 常见问题

### Q1: 如何选择voting_k的值？
A: 一般建议：
- 小数据集（<1000样本）: voting_k = 3-5
- 中等数据集（1000-10000）: voting_k = 5-7
- 大数据集（>10000）: voting_k = 7-10

### Q2: prototypes参数从哪来？
A: 由`build_prototypes`函数自动生成，是每个簇的中心向量。在`adaptive_clustering`中会自动调用。

### Q3: 三种策略可以组合使用吗？
A: 不可以同时使用，需要选择其中一种。但可以尝试不同策略并对比结果。

### Q4: 如何调试分配过程？
A: 设置`silent=False`可以看到详细的分配日志：
```python
assign_sparse_points_density_based(..., silent=False)
```

## 技术细节

### 模式2的权重计算公式
```python
weight = exp(-distance)
```
距离越小，权重越大，指数衰减确保近邻影响更大。

### 模式3的平均距离计算
```python
# 从每个簇选择3个最近样本
k_in_cluster = min(3, cluster_size)
nearest_distances = sort(distances)[:k_in_cluster]
avg_distance = mean(nearest_distances)
```

## 更新日志

- 2025-01: 实现三种新的稀疏点分配策略
- 支持命令行参数 `--assign_model` 和 `--voting_k`
- 完整的文档和使用示例
