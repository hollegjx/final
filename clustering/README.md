# Clustering 模块文档

## 📋 目录

- [模块概述](#模块概述)
- [文件夹结构](#文件夹结构)
- [核心模块详细说明](#核心模块详细说明)
  - [1. Data 独立数据模块](#1-data-独立数据模块)
  - [2. Utils 工具模块](#2-utils-工具模块)
  - [3. Density 密度计算模块](#3-density-密度计算模块)
  - [4. SSDDBC 算法核心模块](#4-ssddbc-算法核心模块)
  - [5. Evaluation 评估模块](#5-evaluation-评估模块)
  - [6. Unknown 未知类识别模块](#6-unknown-未知类识别模块)
  - [7. Baseline 基线方法模块](#7-baseline-基线方法模块)
  - [8. Prototypes 原型构建模块](#8-prototypes-原型构建模块)
  - [9. Testing 测试模块](#9-testing-测试模块)
- [快速开始](#快速开始)
- [命令行使用](#命令行使用)
- [依赖关系](#依赖关系)
- [更新日志](#更新日志)

---

## 模块概述

本模块实现了基于密度的半监督聚类算法（SS-DDBC），用于广义类别发现（Generalized Category Discovery, GCD）任务。

**核心功能**：
1. 密度估计和高密度点识别（支持多种密度计算方法）
2. 基于密度的半监督聚类构建（含标签冲突解决）
3. 灵活的截止距离计算（co模式）
4. 稀疏点分配（多种策略可选）
5. 未知类别识别
6. 聚类结果分析和评估（含损失函数）
7. 骨干网络详细日志记录（可选）

**特点**：
- ✅ **独立性强**: `clustering/data`模块提供独立的数据读取系统
- ✅ **模块化**: 清晰的职责分离，易于维护和扩展
- ✅ **灵活配置**: 丰富的参数系统，支持多种算法变体
- ✅ **完善文档**: 每个模块都有详细的专题文档

---

## 文件夹结构

```
clustering/
├── README.md                           # 本文档
├── REFACTORING_SUMMARY.md              # 重构总结
├── __init__.py                         # 主模块导入
│
├── data/                               # 独立数据读取模块 ⭐ NEW
│   ├── __init__.py
│   ├── README.md                       # 数据模块详细文档
│   ├── dataset_config.py               # 数据集配置（独立定义CIFAR100超类）
│   ├── feature_loader.py               # 特征缓存加载器
│   ├── feature_extractor.py            # 模型特征提取器
│   ├── data_provider.py                # 统一数据提供接口
│   ├── model_loader.py                 # 模型加载器（封装models依赖）
│   └── dataset_loader.py               # 数据集加载器（封装data依赖）
│
├── utils/                              # 工具模块
│   ├── __init__.py
│   ├── co_calculation.py               # Co截止距离计算（3种模式）
│   ├── dense_logger.py                 # 骨干网络详细日志记录器 ⭐ NEW
│   └── CO_MODES.md                     # Co计算模式文档
│
├── density/                            # 密度计算模块
│   ├── __init__.py
│   ├── density_estimation.py           # 密度估计（4种方法）
│   └── DENSITY_METHODS.md              # 密度计算方法文档
│
├── ssddbc/                             # SS-DDBC算法核心
│   ├── __init__.py
│   ├── conflict.py                     # 冲突解决
│   ├── clustering.py                   # 聚类构建
│   ├── analysis.py                     # 结果分析
│   ├── assignment.py                   # 稀疏点分配（3种策略）
│   ├── merging.py                      # 聚类合并
│   ├── adaptive_clustering.py          # 自适应聚类主算法
│   ├── debug_single_clusters.py        # 单样本聚类调试
│   ├── debug_high_density.py           # 高密度点调试
│   ├── ASSIGNMENT_STRATEGIES.md        # 分配策略文档
│   ├── README_DEBUG.md                 # 调试功能文档
│   └── USAGE_EXAMPLE.md                # 使用示例
│
├── evaluation/                         # 评估模块 ⭐ NEW
│   ├── __init__.py
│   ├── loss_function.py                # 损失函数（L1+L2综合损失）
│   └── LOSS_FUNCTION.md                # 损失函数文档
│
├── unknown/                            # 未知类识别模块
│   ├── __init__.py
│   └── detection.py                    # 未知类检测
│
├── baseline/                           # 基线方法模块
│   ├── __init__.py
│   └── kmeans.py                       # K-means基线
│
├── prototypes/                         # 原型构建模块
│   ├── __init__.py
│   └── prototype_builder.py            # 原型构建
│
└── testing/                            # 测试模块
    ├── __init__.py
    ├── test_superclass.py              # 超类测试
    ├── main.py                         # 主程序入口
    ├── grid_search.py                  # 网格搜索
    ├── heatmap.py                      # 热力图绘制
    └── PARAMETERS_GUIDE.md             # 参数完整指南 ⭐
```

---

## 核心模块详细说明

### 1. Data 独立数据模块

**目的**: 提供独立的数据读取系统，最小化对外部训练代码的依赖

**核心类/函数**:
- `DataProvider`: 统一数据提供接口（自动选择缓存或实时提取）
- `FeatureLoader`: 特征缓存加载器
- `FeatureExtractor`: 模型特征提取器
- `ModelLoader`: 模型加载器（封装models和config依赖）
- `DatasetLoader`: 数据集加载器（封装data/依赖）
- `get_superclass_info()`: 获取超类配置信息

**详细文档**: `clustering/data/README.md`

**快速示例**:
```python
from clustering.data import DataProvider, get_superclass_info

# 获取特征数据（自动处理缓存）
provider = DataProvider()
feature_dict, source = provider.get_features(
    dataset_name='trees',
    use_l2=True
)

# 获取超类信息
info = get_superclass_info('trees')
known_classes = info['known_classes_mapped']
```

---

### 2. Utils 工具模块

#### 📁 `utils/co_calculation.py`

**Co截止距离计算**，支持3种模式：
- **模式1**: 手动指定固定co值
- **模式2**: K近邻平均距离（通用默认）
- **模式3**: 相对自适应距离（每点自适应）

**核心函数**:
- `compute_co_value(co_mode, knn_distances, densities, neighbors, k, co_manual)`: 计算co值
- `apply_co_filter(neighbors, distances, co)`: 应用co过滤
- `get_co_mode_description(co_mode)`: 获取模式描述

**详细文档**: `clustering/utils/CO_MODES.md`

#### 📁 `utils/dense_logger.py` ⭐ NEW

**骨干网络详细日志记录器**，用于记录高密度点聚类过程的详细信息。

**核心类/函数**:
- `DenseNetworkLogger`: 日志记录器类
- `init_logger(log_dir, enabled)`: 初始化全局logger
- `get_logger()`: 获取全局logger实例
- `reset_logger()`: 重置logger

**记录内容**:
- 点序号、密度值、相对co值
- 已知/未知状态、是否有标签
- 来自训练集/测试集
- 聚类动作（创建/扩展/合并/移动/拒绝）
- 邻居详细信息

**使用**:
```bash
# 启用详细日志
python -m clustering.testing.main --superclass_name trees --detail_dense True
# 日志保存在 /data/gjx/checkpoints/log/dense_network_trees_*.txt
```

---

### 3. Density 密度计算模块

#### 📁 `density/density_estimation.py`

**密度计算**，支持4种方法（通过`dense_method`参数选择）：
- **方法0**: 平均距离倒数（通用默认）
- **方法1**: 中位数距离倒数（抗噪声）
- **方法2**: 归一化倒数密度（归一化到[0,1]）
- **方法3**: 指数密度（强调局部密集度）

**核心函数**:
- `compute_simple_density(X, k)`: 方法0
- `compute_median_density(X, k)`: 方法1
- `compute_normalized_inverse_density(X, k)`: 方法2
- `compute_exponential_density(X, k)`: 方法3
- `compute_relative_density(densities, neighbors, k)`: 计算相对密度
- `identify_high_density_points(densities, percentile, use_relative)`: 识别高密度点

**详细文档**: `clustering/density/DENSITY_METHODS.md`

---

### 4. SSDDBC 算法核心模块

#### 📁 `ssddbc/clustering.py`

**聚类构建**，完全按照SS-DDBC算法流程。

**核心函数**:
- `build_clusters_ssddbc(X, high_density_mask, neighbors, labeled_mask, targets, densities, known_mask, k, co, silent, logger, train_size)`: 构建聚类

**算法流程**:
```
For each high-density point xi (按密度降序):
    If xi is not assigned to any cluster:
        Create a new cluster pi
        For each KNN_co neighbor xj:
            If xj is not assigned:
                Add xj to pi (扩展)
            Else If xj ∈ pj:
                检查标签冲突:
                - 样本级冲突（硬约束）→ 拒绝
                - 簇级冲突（软约束）→ 密度判断
                - 无冲突 → 直接合并
```

**新增**: 支持`logger`参数，记录详细聚类过程（`detail_dense=True`时）

#### 📁 `ssddbc/assignment.py`

**稀疏点分配**，支持3种策略（通过`assign_model`参数选择）：
- **策略1**: 簇原型就近（速度最快）
- **策略2**: KNN投票加权（推荐默认，考虑邻域信息）
- **策略3**: 簇内K近邻平均距离（最精细但慢）

**核心函数**:
- `assign_sparse_points_density_based(X, clusters, cluster_labels, densities, neighbors, labeled_mask, targets, label_threshold, purity_threshold, train_size, silent, prototypes, prototype_true_labels, voting_k, assign_model)`: 分配稀疏点

**详细文档**: `clustering/ssddbc/ASSIGNMENT_STRATEGIES.md`

#### 📁 `ssddbc/adaptive_clustering.py`

**完整的SS-DDBC算法流程**，整合所有步骤。

**核心函数**:
- `adaptive_density_clustering(X, targets, known_mask, labeled_mask, k, density_percentile, lambda_weight, simple_ssddbc, random_state, train_size, co_mode, co_manual, single_detail, detail_sample, eval_dense, eval_version, analyze_dense, silent, dense_method, assign_model, voting_k, detail_dense)`: 完整聚类流程

**算法步骤**:
1. 计算样本密度（根据`dense_method`选择）
2. 识别高密度点
3. 计算co截止距离（根据`co_mode`选择）
4. 构建聚类（含冲突处理，可选`detail_dense`日志）
5. 建立原型
6. 分配稀疏点（根据`assign_model`选择）
7. 识别未知类聚类

**新增参数**:
- `detail_dense`: 是否记录骨干网络详细日志
- `dense_method`: 密度计算方法（0-3）
- `assign_model`: 稀疏点分配策略（1-3）
- `voting_k`: KNN投票邻居数量
- `co_mode`: co计算模式（1-3）
- `co_manual`: 手动指定co值

#### 📁 `ssddbc/analysis.py`

**结果分析**。

**核心函数**:
- `analyze_ssddbc_clustering_result(clusters, cluster_labels, labeled_mask, targets, known_mask)`: 分析聚类构建结果
- `analyze_cluster_composition(predictions, targets, known_mask, labeled_mask, unknown_clusters)`: 分析聚类组成
- `evaluate_high_density_clustering(cluster_labels, targets, known_mask, eval_version, X, silent)`: 评估高密度点聚类

---

### 5. Evaluation 评估模块 ⭐ NEW

#### 📁 `evaluation/loss_function.py`

**损失函数计算**，用于优化参数。

**损失定义**:
- **L1 (监督损失)**: `1 - accuracy`（使用匈牙利算法匹配）
- **L2 (无监督损失)**: `1 - DBCV`（基于密度的聚类验证）
- **综合损失**: `L = w1 × L1 + w2 × L2`

**核心函数**:
- `compute_dbcv_score(X, labels)`: 计算DBCV分数
- `compute_supervised_loss_l1(predictions, targets, labeled_mask, loss_type)`: 计算L1
- `compute_unsupervised_loss_l2(X, predictions, loss_type)`: 计算L2
- `compute_total_loss(X, predictions, targets, labeled_mask, l1_weight, l2_weight, l1_type, l2_type, silent)`: 计算综合损失

**详细文档**: `clustering/evaluation/LOSS_FUNCTION.md`

**使用**: 聚类完成后自动计算并显示（在ACC之前输出）

---

### 6. Unknown 未知类识别模块

#### 📁 `unknown/detection.py`

**未知类识别**。

**核心函数**:
- `identify_unknown_clusters(clusters, labeled_mask)`: 从聚类列表识别
- `identify_unknown_clusters_from_predictions(predictions, labeled_mask)`: 从预测结果识别

**判断标准**: 不包含有标签样本的聚类标记为潜在未知类

---

### 7. Baseline 基线方法模块

#### 📁 `baseline/kmeans.py`

**K-means基线对比**。

**核心函数**:
- `test_kmeans_baseline(test_features, test_targets, test_known_mask, n_clusters, random_state, eval_version, kmeans_merge, train_features, train_targets, train_known_mask)`: K-means聚类

**支持**:
- 仅测试集模式
- 合并训练+测试集模式（`kmeans_merge=True`）
- 与SS-DDBC使用相同的评估指标

---

### 8. Prototypes 原型构建模块

#### 📁 `prototypes/prototype_builder.py`

**原型构建**。

**核心函数**:
- `build_prototypes(X, clusters, labeled_mask, targets)`: 基于partial-clustering结果建立原型

**返回**:
- 每个聚类的原型（聚类中心）
- 每个聚类的主导标签（-1表示未知类）

---

### 9. Testing 测试模块

#### 📁 `testing/main.py`

**主程序命令行入口**。

**核心函数**:
- `main()`: 主测试函数

**支持的命令行参数** (详见`PARAMETERS_GUIDE.md`):

**基础参数**:
- `--model_path`: 模型路径
- `--superclass_name`: 超类名称
- `--use_train_and_test`: 是否合并训练+测试集
- `--l2`: 是否使用L2归一化

**算法参数**:
- `--k`: K近邻数量（默认10）
- `--density_percentile`: 高密度点百分位阈值（默认75）
- `--lambda_weight`: 冲突解决权重（默认0.7）
- `--dense_method`: 密度计算方法（0-3，默认0）⭐
- `--assign_model`: 稀疏点分配策略（1-3，默认2）⭐
- `--voting_k`: KNN投票邻居数量（默认5）⭐
- `--co_mode`: co计算模式（1-3，默认2）⭐
- `--co_manual`: 手动指定co值 ⭐

**调试参数**:
- `--detail_dense`: 骨干网络详细日志（默认False）⭐ NEW
- `--single_detail`: 单样本聚类详细分析（默认False）
- `--detail_sample`: 样本匹配度分析（默认False）
- `--eval_dense`: 仅评估高密度点（默认False）
- `--analyze_dense`: 高密度点类内类间距离分析（默认False）
- `--silent`: 静默模式（默认False）

**其他参数**:
- `--eval_version`: 评估版本（'v1'或'v2'，默认'v1'）
- `--run_kmeans_baseline`: 是否运行K-means对比（默认False）
- `--grid_search`: 是否启用网格搜索（默认False）

#### 📁 `testing/test_superclass.py`

**超类测试函数**。

**核心函数**:
- `test_adaptive_clustering_on_superclass(superclass_name, model_path, ...)`: 在指定超类上测试

**功能**:
- 自动处理缓存/实时提取
- 运行完整聚类流程
- 计算损失函数（L1+L2）⭐ NEW
- 计算评估指标（ACC、NMI、ARI）
- 可选K-means对比

#### 📁 `testing/grid_search.py`

**网格搜索**。

**核心函数**:
- `grid_search_parameters(...)`: 网格搜索最优参数
- `run_single_test(...)`: 运行单个参数组合

#### 📁 `testing/heatmap.py`

**参数热力图可视化**。

**核心函数**:
- `load_existing_results(...)`: 加载已有搜索结果
- `run_parameter_grid_search(...)`: 运行网格搜索
- `create_heatmap(...)`: 创建单指标热力图
- `create_multiple_heatmaps(...)`: 创建多指标热力图

---

## 快速开始

### 安装依赖

```bash
pip install numpy torch scikit-learn tqdm kDBCV
```

### 基本使用（Python API）

```python
from clustering import adaptive_density_clustering
from clustering.data import DataProvider

# 1. 获取特征数据
provider = DataProvider()
feature_dict, source = provider.get_features(
    dataset_name='trees',
    use_l2=True
)

# 2. 提取数据
all_feats = feature_dict['all_features']
all_targets = feature_dict['all_targets']
all_known_mask = feature_dict['all_known_mask']
all_labeled_mask = feature_dict['all_labeled_mask']

# 3. 运行聚类
predictions, n_clusters, unknown_clusters = adaptive_density_clustering(
    X=all_feats,
    targets=all_targets,
    known_mask=all_known_mask,
    labeled_mask=all_labeled_mask,
    k=10,
    density_percentile=75,
    dense_method=0,      # 平均距离
    assign_model=2,      # KNN投票
    co_mode=2,           # K近邻平均距离
    detail_dense=False   # 不记录详细日志
)

print(f"聚类数量: {n_clusters}")
print(f"潜在未知类: {len(unknown_clusters)}个")
```

---

## 命令行使用

### 基本聚类测试

```bash
python -m clustering.testing.main \
    --model_path /data1/jiangzhen/gjx/exp/newgpc/final/metric_learn_gcd/log/(...)/checkpoints/model.pt \
    --superclass_name trees \
    --k 10 \
    --density_percentile 75
```

### 启用详细日志

```bash
python -m clustering.testing.main \
    --superclass_name trees \
    --detail_dense True
# 日志保存在 /data/gjx/checkpoints/log/dense_network_trees_*.txt
```

### 尝试不同算法配置

```bash
# 使用中位数密度 + 簇原型分配
python -m clustering.testing.main \
    --superclass_name trees \
    --dense_method 1 \
    --assign_model 1

# 使用相对自适应co + KNN投票分配
python -m clustering.testing.main \
    --superclass_name trees \
    --co_mode 3 \
    --assign_model 2 \
    --voting_k 7
```

### 网格搜索最优参数

```bash
python -m clustering.testing.main \
    --superclass_name trees \
    --grid_search True \
    --k_min 5 --k_max 15 \
    --dp_min 60 --dp_max 90 --dp_step 5
```

### K-means基线对比

```bash
python -m clustering.testing.main \
    --superclass_name trees \
    --run_kmeans_baseline True
```

---

## 依赖关系

### 外部依赖
- `numpy`: 数组操作和数值计算
- `torch`: PyTorch深度学习框架
- `scikit-learn`: 机器学习库（KNN、K-means、评估指标）
- `tqdm`: 进度条显示
- `kDBCV`: DBCV聚类验证指标（用于L2损失）

### 内部依赖（仍需外部训练代码）
- `models.vision_transformer`: ViT-DINO模型定义
- `config.dino_pretrain_path`: DINO预训练权重路径
- `data.augmentations`: 数据增强（实时提取时）
- `data.get_datasets`: 数据集获取（实时提取时）
- `project_utils.cluster_and_log_utils`: ACC评估工具

**注意**: 这些依赖已被封装在`clustering/data`模块中，主聚类逻辑不直接接触。

### 模块间依赖

```
clustering/
├── data/ (最小外部依赖，封装models/config/data依赖)
├── utils/ (独立)
├── density/ (独立)
├── evaluation/ (依赖kDBCV)
├── ssddbc/
│   ├── clustering.py → 依赖 utils/dense_logger
│   ├── assignment.py (独立)
│   └── adaptive_clustering.py → 整合所有模块
├── unknown/ (独立)
├── baseline/ (依赖 project_utils)
├── prototypes/ (独立)
└── testing/
    └── test_superclass.py → 依赖 data/, evaluation/
```

---

## 更新日志

### 2025-01-20 v5 (最新)
- ✅ 新增 `clustering/data/` 独立数据模块
  - `DataProvider`: 统一数据提供接口
  - `FeatureLoader`, `FeatureExtractor`: 特征加载和提取
  - `ModelLoader`, `DatasetLoader`: 模型和数据集加载器
  - `get_superclass_info()`: 独立超类配置
- ✅ 新增 `clustering/utils/dense_logger.py` 骨干网络详细日志记录器
  - 支持 `--detail_dense` 参数
  - 记录高密度点聚类过程到txt文件
- ✅ 新增 `clustering/evaluation/loss_function.py` 损失函数模块
  - L1监督损失（基于accuracy）
  - L2无监督损失（基于DBCV）
  - 综合损失（可加权）
- ✅ 删除 `clustering/utils/model_utils.py` 和 `cache_utils.py`（功能迁移到data模块）
- ✅ 新增参数: `detail_dense`, `dense_method`, `assign_model`, `voting_k`, `co_mode`, `co_manual`
- ✅ 完善文档: 新增多个专题MD文档

### 2025-10-12 v4
- 新增 `clustering/utils/co_calculation.py` - Co截止距离计算（3种模式）
- 新增 `clustering/density/` - 多种密度计算方法（4种）
- 新增 `clustering/ssddbc/assignment.py` - 稀疏点分配策略（3种）
- 完善参数系统和帮助文档

### 2025-10-12 v3
- 新增 `testing/grid_search.py` - 网格搜索
- 新增 `testing/heatmap.py` - 参数热力图
- 修复导入错误

### 2025-10-12 v2
- 新增 `adaptive_clustering.py` - 完整SS-DDBC流程
- 新增 `analyze_cluster_composition()` - 聚类组成分析
- 新增 `testing/` 模块

### 2025-10-12 v1
- 初始版本，包含基础聚类功能

---

## 相关文档

- **数据模块**: `clustering/data/README.md`
- **Co计算模式**: `clustering/utils/CO_MODES.md`
- **密度计算方法**: `clustering/density/DENSITY_METHODS.md`
- **分配策略**: `clustering/ssddbc/ASSIGNMENT_STRATEGIES.md`
- **损失函数**: `clustering/evaluation/LOSS_FUNCTION.md`
- **参数完整指南**: `clustering/testing/PARAMETERS_GUIDE.md`
- **重构总结**: `clustering/REFACTORING_SUMMARY.md`

---

## 联系方式

如有问题或建议，请联系项目维护者。
