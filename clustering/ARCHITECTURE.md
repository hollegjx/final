# Clustering Module Architecture

## 目录结构

```
clustering/
├── ssddbc/              # SS-DDBC核心聚类算法
├── density/             # 密度计算方法
├── data/                # 数据加载与特征提取
├── evaluation/          # 评估指标与损失函数
├── grid_search/         # 参数网格搜索与可视化
├── testing/             # 测试入口与主函数
├── information/         # 信息分析模块
├── utils/               # 工具函数
└── baseline/            # 基线算法（K-means）
```

---

## 核心算法流程 (ssddbc/)

### adaptive_clustering.py - 主算法入口

**函数**: `adaptive_density_clustering(X, targets, known_mask, labeled_mask, k, density_percentile, ...)`

**返回**:
- `predictions`: 聚类标签
- `n_clusters`: 簇数量
- `unknown_clusters`: 未知类簇索引
- `prototypes`: 簇原型
- `updated_clusters`: 完整簇（核心点+稀疏点）
- `cluster_category_labels`: 簇类别标签映射
- `neighbors`: K近邻索引（避免重复计算）
- `core_clusters`: 纯核心点簇（用于聚类质量第一项）

**7步流程**:

1. **密度计算** → `compute_*_density(X, k)`
   - 4种方法：平均距离倒数(0)、中位数倒数(1)、归一化倒数(2)、指数密度(3)
   - 返回：`densities, knn_distances, neighbors`

2. **高密度点筛选** → `identify_high_density_points(densities, percentile)`
   - 基于百分位阈值筛选核心点

3. **核心簇构建** → `build_clusters_ssddbc(...)`
   - SS-DDBC算法构建核心点簇
   - 处理标签冲突、密度约束
   - 孤立簇合并（规模≤3）
   - **保存`core_clusters`**: 此时只包含核心点（含孤立簇合并）

4. **原型构建** → `build_prototypes(X, clusters, ...)`
   - 每个簇计算中心点作为原型

5. **稀疏点分配**
   - **标签引导** (可选): 有标签稀疏点优先分配到对应簇
   - **3种分配策略**:
     - 簇原型就近(1): 最快，距离最近原型
     - KNN投票加权(2): 推荐，考虑邻域投票
     - 簇内K近邻(3): 最精细，计算簇内平均距离

6. **孤立簇合并** (全空间)
   - 合并稀疏点分配后的小簇

7. **未知类识别**
   - 基于簇纯度识别潜在未知类

---

## 密度计算 (density/)

### density_estimation.py

| 方法ID | 名称 | 公式 | 适用场景 |
|--------|------|------|----------|
| 0 | 平均距离倒数 | `1 / avg(knn_dist)` | 通用默认 |
| 1 | 中位数倒数 | `1 / median(knn_dist)` | 抗噪声数据 |
| 2 | 归一化倒数 | 归一化到[0,1] | 密度值标准化 |
| 3 | 指数密度 | `exp(-avg(knn_dist))` | 强调局部密集 |

**关键函数**:
- `compute_relative_density(densities, neighbors, k)`: 计算相对密度（点密度/邻居平均密度）
- `identify_high_density_points(densities, percentile)`: 基于百分位筛选

---

## 数据加载 (data/)

### EnhancedDataProvider

**功能**: 一站式数据加载，自动缓存特征

**核心方法**:
```python
dataset = provider.load_dataset(
    dataset_name='trees',
    model_path=model_path,
    use_l2=True,              # L2归一化
    use_train_and_test=True   # 合并训练+测试集
)

# 便捷接口
X, targets, known_mask, labeled_mask, train_size = dataset.get_clustering_input()
test_data = dataset.get_test_subset(predictions)
```

**缓存机制**:
- 路径: `/data/gjx/checkpoints/features/{dataset}/{model_hash}/`
- 文件: `train_features.npy`, `test_features.npy`

---

## 评估模块 (evaluation/)

### loss_function.py - 损失函数

**L1监督损失**:
```python
L1 = 1 - accuracy  # 基于匈牙利算法的簇-标签匹配
```

**L2聚类质量损失** (新增):
```python
L2 = separation_score - penalty_score
```
- 由 `cluster_quality.py` 实现

**综合损失**:
```python
L = l1_weight × L1 + l2_weight × L2
```

### cluster_quality.py - 聚类质量评估

**公式**:
```
Score = exp(-(1/M) × Σd(Pi,Pj)) - (1/N) × Σ[(1/k) × Σ(I(xi,xj) × exp(-I(xi,xj) × d(xi,xj)))]
        \_____第一项_________/       \___________________第二项_____________________/
```

**第一项: 簇间分离度**
- 使用 `core_clusters` (只包含核心点)
- 3种距离计算方法:
  - `nearest_k(1)`: 最近k对点平均距离
  - `all_pairs(2)`: 所有点对平均距离
  - `prototype(3)`: 原型距离（默认）

**第二项: 局部密度惩罚**
- 使用所有点（核心点+稀疏点）
- `I(xi,xj)`: 同簇=1, 异簇=-1

**关键**:
- 复用 `neighbors` 避免重复计算KNN
- 第一项和第二项使用不同的簇定义

---

## 网格搜索 (grid_search/)

### batch_runner.py - 批量网格搜索

**功能**: 并行/串行网格搜索，自动保存结果

**命令**:
```bash
python -m clustering.grid_search.batch_runner \
    --superclasses trees \
    --k_min 3 --k_max 21 \
    --density_min 40 --density_max 100 --density_step 5 \
    --use_cluster_quality \
    --cluster_distance_method 1 \
    --l2_components "separation silhouette"
```

**输出**: `/data/gjx/checkpoints/search/{superclass}/{timestamp}.txt`

注意：batch_runner 仅负责“参数网格 + 原始数据收集”，不涉及权重配置。
- 已移除: `--separation_weight`、`--penalty_weight`、`--l2_component_weights`
- 保留: `--l2_components`（决定收集哪些 L2 组件值，例如 separation、silhouette）
- 全量超类：可使用 `--count_all` 一键启用 15 个内置超类（覆盖 `--superclasses` 与 `--superclass_file`）。

### heatmap.py - 热力图可视化

**功能**: 从搜索结果生成热力图

**命令**:
```bash
python -m clustering.grid_search.heatmap \
    --superclass_name trees \
    --cluster_quality_heatmaps true
```

**输出图片**:
1. `all_acc_heatmap_*.png` - ACC热力图
2. `labeled_acc_colored_by_all_acc_*.png` - Labeled ACC混合图
3. `quality_score_colored_by_all_acc_*.png` - 质量分数（背景色all_acc，显示quality_score）
4. `separation_score_colored_by_all_acc_*.png` - 分离度

---

## L1+L2 联合权重探索 (l1l2_search/)

### run_l1l2_exploration.py - 权重穷举与热力图

**功能**: 基于 batch_runner 的结果，按权重总和穷举所有整数组合 `(w_l1, w_sep, w_sil)`，为每组权重生成三类热力图（all/new/old）。

**推荐命令（任务隔离）**:
```bash
python -m clustering.grid_search.l1l2_search.run_l1l2_exploration \
  --task_folder 4class_11_06_21_06 \
  --output_dir /data/gjx/checkpoints/search \
  --output_dir_heatmap /data/gjx/checkpoints/l1l2_search \
  --weight_sum 10
```

兼容命令（旧行为）：提供 `--search_dir` 直接指向某个任务路径，任务名将从末级目录自动推断。

**输出结构**: `/data/gjx/checkpoints/l1l2_search/{task_folder}/{superclass}/`
- `all/ new/ old/`: 背景为各自 ACC 的权重热力图，文件名包含 ACC 与权重：`{metric}_{acc:.4f}_{w_l1}_{w_sep}_{w_sil}.png`
- `single_metrics/`: 单指标热力图（all_acc/new_acc/old_acc）与三组件热力图（背景为 new_acc）
- `l1l2_weights_summary.txt`: 汇总报告（跨权重配置统计）

默认在每个超类处理完成后清理中间 txt 报告（`l1l2_report_wl1X_sepY_silZ.txt`），仅保留汇总与 PNG。若需保留中间报告，可添加 `--keep_reports`。

动态组件与自适应加权：
- 不再强制 `separation/silhouette/penalty` 同时存在；仅要求存在 `l1_loss` 且至少一个 L2 组件。
- 对可用的 L2 组件（逐样本）动态重分配 L2 总权重（`w_sep + w_sil`），缺失组件自动忽略。
- 组件值提取兼容新旧格式：
  - 旧：`separation_score`、`penalty_score`、`silhouette`
  - 新：`l2_components.<name>.value`（如 `l2_components.separation.value`）

### run_l1l2_region_search.py - 跨超类权重区域搜索

**功能**: 基于已生成的权重热力图文件名（不重复计算），按给定 ACC 阈值在多个超类之间求“共同有效”的权重交集，并生成报告。

**命令（任务隔离）**:
```bash
python -m clustering.grid_search.l1l2_search.run_l1l2_region_search \
  --acc_mode all \
  --output_dir /data/gjx/checkpoints/l1l2_search \
  --task_folder 4class_11_06_21_06 \
  --trees 0.77 --humans 0.70 --vehicles 0.84 --buildings 0.95
```

提示：若未提供 `--task_folder`，工具会自动列出 `output_dir` 下所有有效任务，供用户选择。

**输出**: `/data/gjx/checkpoints/findL/<num_common>_{task_folder}.txt`
- 列出重合的权重 `(w_l1,w_sep,w_sil)` 及各超类在该权重下的 ACC 值
- 支持 `acc_mode` 选择 `all/new/old`
5. `penalty_score_colored_by_all_acc_*.png` - 密度惩罚

**Top3标注**:
- quality_score: 标注最高的3个
- separation_score: 标注最高的3个
- penalty_score: 标注最低的3个（越小越好）

---

## 测试入口 (testing/)

### main.py - 命令行入口

**单组参数运行**:
```bash
python -m clustering.testing.main \
    --superclass_name trees \
    --k 10 \
    --density_percentile 75 \
    --dense_method 1 \
    --assign_model 2 \
    --co_mode 2 \
    --use_cluster_quality \
    --cluster_distance_method 1
```

**关键参数**:
- `--dense_method {0,1,2,3}`: 密度计算方法
- `--assign_model {1,2,3}`: 稀疏点分配策略
- `--co_mode {1,2,3}`: co截止距离计算模式
- `--label_guide`: 标签引导模式
- `--use_cluster_quality`: 启用聚类质量评估

### test_superclass.py - 测试函数

**函数**: `test_adaptive_clustering_on_superclass(...)`

**流程**:
1. 加载数据 (`EnhancedDataProvider`)
2. 运行聚类 (`adaptive_density_clustering`)
3. 计算ACC（测试集）
4. 计算损失 (`compute_total_loss`)
5. 计算labeled_acc
6. 可选：错误样本分析

---

## 信息分析 (information/)

### labeled_acc_calculation.py
- `compute_labeled_acc_with_unknown_penalty()`: 有标签样本准确率（考虑unknown_clusters惩罚）

### error_sample_analysis.py
- `analyze_error_samples()`: 测试集错误样本详细分析
- 输出: `/data/gjx/checkpoints/logs/error_samples_{dataset}_{timestamp}.txt`

### dense_logger.py
- 骨干网络聚类详细日志记录
- 启用: `--detail_dense true`

---

## 工具模块 (utils/)

### co_calculation.py
- `compute_co_value(co_mode, ...)`: 计算co截止距离
  - Mode 1: 手动指定
  - Mode 2: K近邻平均距离（默认）
  - Mode 3: 相对自适应距离

---

## 性能优化要点

1. **KNN复用**: `adaptive_clustering` 返回 `neighbors`，传递给 `compute_local_density_penalty`，避免重复计算
   - 单次节省: 0.5-1秒 (CIFAR-100)
   - 网格搜索100组合节省: 50-100秒

2. **特征缓存**: `EnhancedDataProvider` 自动缓存提取的特征

3. **并行网格搜索**: `batch_runner` 支持多进程并行

4. **静默模式**: `--silent true` 关闭所有打印输出，加速网格搜索

---

## 常见配置

| 场景 | 推荐配置 |
|------|---------|
| 通用场景 | `dense_method=0, assign_model=2, co_mode=2` |
| 噪声数据 | `dense_method=1, density_percentile=80-90` |
| 稀疏数据 | `k=5-10, density_percentile=60-70` |
| 稠密数据 | `k=15-20, density_percentile=75-85` |
| 标签少 | `label_guide=false` |
| 标签多 | `label_guide=true` |

---

## 数据流向图

```
数据加载 (EnhancedDataProvider)
    ↓
密度计算 (compute_*_density) → neighbors, densities
    ↓
核心簇构建 (build_clusters_ssddbc) → core_clusters (纯核心点)
    ↓
原型构建 (build_prototypes)
    ↓
稀疏点分配 (assign_sparse_points) → updated_clusters (核心点+稀疏点)
    ↓
损失计算 (compute_total_loss)
    ├─ L1: 使用 predictions
    └─ L2: 使用 core_clusters (第一项) + predictions (第二项)
```

---

## 版本记录

- **v1.0.25**: 添加聚类质量评估指标、修复网格搜索、分离热力图
- 核心算法返回 `neighbors` 避免KNN重复计算
- 核心算法返回 `core_clusters` 用于聚类质量第一项
