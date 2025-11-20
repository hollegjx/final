# 稀疏点分配策略说明

## 概述

本模块提供了3种不同的稀疏点分配策略，用于将低密度点分配到高密度骨干网络的簇中。

## 策略说明

### 模式1: 基于簇原型的就近分配 (assign_model=1)

**原理:**
- 计算稀疏点到每个簇原型的欧氏距离
- 将稀疏点分配到距离最近的簇

**特点:**
- 最简单直接
- 计算效率最高
- 适用于簇原型能很好代表簇的情况

**实现位置:** `clustering/ssddbc/assignment.py:103-127`

---

### 模式2: 高密度子空间KNN投票（距离加权） (assign_model=2) ⭐ 默认

**原理:**
- 在高密度点中找K个最近邻（默认k=5）
- 使用 `exp(-distance)` 作为权重进行加权投票
- 每个邻居对其所在簇的贡献分数为 `exp(-distance)`
- 选择总分最高的簇

**特点:**
- 平衡了距离和邻居投票
- 距离越近的邻居权重越大
- 对噪声有一定鲁棒性
- 默认推荐策略

**实现位置:** `clustering/ssddbc/assignment.py:129-172`

---

### 模式3: 簇内K近邻平均距离 (assign_model=3)

**原理:**
- 从每个簇中找3个最近的样本
- 计算这3个样本到稀疏点的平均距离
- 将稀疏点分配到平均距离最小的簇

**特点:**
- 更关注簇的局部结构
- 对每个簇都考察其内部最近样本
- 适用于簇大小差异较大的情况

**实现位置:** `clustering/ssddbc/assignment.py:174-212`

---

## 使用方法

### 通过adaptive_clustering调用

```python
from clustering.ssddbc.adaptive_clustering import adaptive_density_clustering

predictions, n_clusters, unknown_clusters = adaptive_density_clustering(
    X=X,
    targets=targets,
    known_mask=known_mask,
    labeled_mask=labeled_mask,
    k=10,
    density_percentile=75,
    assign_model=2,  # 选择策略: 1/2/3
    voting_k=5,      # 模式2的KNN邻居数量
    silent=False
)
```

### 直接调用assignment函数

```python
from clustering.ssddbc.assignment import assign_sparse_points_density_based

final_labels = assign_sparse_points_density_based(
    X=X,
    clusters=clusters,
    cluster_labels=cluster_labels,
    densities=densities,
    neighbors=neighbors,
    labeled_mask=labeled_mask,
    targets=targets,
    prototypes=prototypes,  # 必需参数
    voting_k=5,
    assign_model=2,         # 1/2/3
    silent=False
)
```

## 参数说明

- `assign_model`: int, 分配策略编号
  - 1: 基于簇原型的就近分配
  - 2: 高密度子空间KNN投票（默认）
  - 3: 簇内K近邻平均距离

- `voting_k`: int, default=5
  - 模式2中KNN投票使用的邻居数量
  - 模式1和3不使用此参数

- `prototypes`: ndarray, 必需
  - 每个簇的原型（中心）特征向量
  - 由`build_prototypes`函数生成

## 特殊处理

### 孤立点处理
- 检测高密度子空间中可能存在的孤立点（未分配到任何簇的高密度点）
- 发出警告提示

### 兜底策略
- 如果某些稀疏点分配失败（理论上不应该发生）
- 自动使用最近簇原型进行分配
- 确保所有点都能被分配

### 空簇检查
- 避免空簇导致的计算错误
- 自动跳过空簇

## 性能对比

| 策略 | 计算复杂度 | 内存占用 | 适用场景 |
|-----|-----------|---------|---------|
| 模式1 | O(n*k) | 低 | 簇原型代表性强 |
| 模式2 | O(n*m*k) | 中 | 通用场景（推荐） |
| 模式3 | O(n*k*3) | 中 | 簇大小差异大 |

*注: n=稀疏点数, m=高密度点数, k=簇数*

## 更新历史

- 2025-01: 实现3种新的分配策略，替换原有的复杂投票逻辑
- 添加孤立点检测和兜底策略
- 移除表情符号以兼容Windows终端
