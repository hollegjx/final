# SS-DDBC 超参数完整指南

本文档详细说明了SS-DDBC聚类算法的所有超参数，包括每个参数的意义、取值范围、默认值以及使用建议。

## 📋 目录

1. [核心算法参数](#核心算法参数)
2. [密度和聚类参数](#密度和聚类参数)
3. [低密度点分配参数](#低密度点分配参数)
4. [调试和分析参数](#调试和分析参数)
5. [网格搜索参数](#网格搜索参数)
6. [快速参考表](#快速参考表)

---

## 核心算法参数

### `--k` (K近邻数量)
**默认值:** 10
**类型:** int
**范围:** [3-30]

**作用:**
- 用于密度计算：每个点的密度基于其K个最近邻居的距离
- 用于邻居发现：在聚类扩展时考虑K个近邻

**选择建议:**
- **较小值 (5-10):** 适合稀疏数据或类别边界清晰的数据
- **中等值 (10-15):** 通用默认，适合大多数场景
- **较大值 (15-20):** 适合稠密数据或需要更平滑的密度估计

**示例:**
```bash
# 稀疏数据
python -m clustering.testing.main --k 7

# 稠密数据
python -m clustering.testing.main --k 15
```

---

### `--density_percentile` (高密度点百分位阈值)
**默认值:** 75
**类型:** int
**范围:** [20-100]

**作用:**
- 决定哪些点被选为高密度核心点
- 值为P表示选择相对密度排名前P%的点作为核心

**选择建议:**
- **较低值 (60-70):** 选择更多核心点，适合干净数据，可能包含一些噪声
- **中等值 (75-80):** 通用默认，平衡覆盖率和纯度
- **较高值 (85-95):** 选择更少但更可靠的核心点，适合噪声数据

**影响:**
- **值越高:** 聚类数越少，核心点越纯但覆盖率可能不足
- **值越低:** 聚类数越多，覆盖率高但可能包含噪声聚类

**示例:**
```bash
# 噪声数据，选择更可靠的核心
python -m clustering.testing.main --density_percentile 90

# 干净数据，选择更多核心
python -m clustering.testing.main --density_percentile 65
```

---

### `--lambda_weight` (冲突解决权重)
**默认值:** 0.7
**类型:** float
**范围:** [0.0-1.0]

**作用:**
- 在标签冲突时平衡密度和距离的重要性
- 公式: `score = λ × density_score + (1-λ) × distance_score`

**选择建议:**
- **接近1.0 (0.8-0.9):** 更信任密度高的点，忽略距离
- **中等值 (0.6-0.8):** 默认推荐，平衡考虑
- **接近0.5 (0.5-0.6):** 更平衡密度和距离

**通常保持默认即可，除非有特殊需求。**

---

## 密度和聚类参数

### `--dense_method` (密度计算方法)
**默认值:** 0
**类型:** int
**选项:** [0, 1, 2, 3]

**选项说明:**
- **0 - 平均距离倒数 (默认):**
  - 公式: `ρ = 1 / mean(KNN_distances)`
  - 通用默认选择
  - 计算最快

- **1 - 中位数距离倒数:**
  - 公式: `ρ = 1 / median(KNN_distances)`
  - 对极端值不敏感
  - 适合有噪声的数据

- **2 - 归一化倒数:**
  - 密度值归一化到[0, 1]
  - 便于跨数据集对比
  - 便于设置统一阈值

- **3 - 指数密度:**
  - 公式: `ρ = exp(-mean(KNN_distances))`
  - 强调局部密集程度
  - 密度值范围[0, 1]

**详见:** `clustering/density/DENSITY_METHODS.md`

**示例:**
```bash
# 噪声数据，使用中位数
python -m clustering.testing.main --dense_method 1

# 需要归一化，使用方法2
python -m clustering.testing.main --dense_method 2
```

---

### `--co_mode` (co截止距离计算模式)
**默认值:** 2
**类型:** int
**选项:** [1, 2, 3]

**选项说明:**
- **1 - 手动指定:**
  - 需要配合`--co_manual`指定具体值
  - 适合已知最佳co值的情况
  - 用于精确复现实验

- **2 - K近邻平均距离 (默认):**
  - 公式: `co = mean(所有点的K近邻平均距离)`
  - 全局固定值
  - 通用默认选择

- **3 - 相对自适应距离:**
  - 公式: `co'(x_i) = (K + ρ̄) / (K + neighbor_avg) × co_base`
  - 每个点有自己的co值
  - 适合数据分布高度不均匀的情况

**详见:** `clustering/utils/CO_MODES.md`

**示例:**
```bash
# 手动指定
python -m clustering.testing.main --co_mode 1 --co_manual 2.5

# 自适应co（不均匀分布）
python -m clustering.testing.main --co_mode 3
```

---

### `--co_manual` (手动指定的co值)
**默认值:** None
**类型:** float
**范围:** 通常[1.0-5.0]，取决于特征空间尺度

**作用:**
- 仅当`--co_mode 1`时使用
- 直接指定co截止距离

**如何确定:**
- 通过网格搜索找到最佳值
- 基于特征空间的先验知识
- 观察K近邻距离分布

**示例:**
```bash
python -m clustering.testing.main --co_mode 1 --co_manual 3.2
```

---

## 低密度点分配参数

### `--assign_model` (分配策略)
**默认值:** 2
**类型:** int
**选项:** [1, 2, 3]

**选项说明:**
- **1 - 簇原型就近分配:**
  - 最简单最快
  - 只考虑到簇原型的距离
  - 可能不够精确

- **2 - KNN投票加权 (默认推荐):**
  - 考虑邻域信息
  - 基于voting_k个近邻的投票
  - 平衡速度和精度

- **3 - 簇内K近邻平均距离:**
  - 最精细的方法
  - 计算到每个簇内所有点的平均距离
  - 最慢但最准确

**详见:** `clustering/ssddbc/ASSIGNMENT_STRATEGIES.md` (如果存在)

**示例:**
```bash
# 最快速度
python -m clustering.testing.main --assign_model 1

# 最高精度（慢）
python -m clustering.testing.main --assign_model 3
```

---

### `--voting_k` (投票近邻数量)
**默认值:** 5
**类型:** int
**范围:** [3-15]

**作用:**
- 仅当`--assign_model 2`时生效
- 决定考虑多少个近邻的投票

**选择建议:**
- **较小值 (3-5):** 更快，适合局部结构清晰的数据
- **中等值 (5-7):** 默认推荐
- **较大值 (10-15):** 更稳定，但计算慢

**示例:**
```bash
python -m clustering.testing.main --assign_model 2 --voting_k 7
```

---

## 调试和分析参数

### `--use_train_and_test`
**默认值:** True
**类型:** bool

**作用:**
- **True (默认):** 合并训练集和测试集进行聚类，在更大数据集上构建更鲁棒的聚类
- **False:** 仅在测试集上聚类，更快但可能不稳定

---

### `--l2`
**默认值:** True
**类型:** bool

**作用:**
- **True (推荐):** 使用L2归一化特征，与eval_original_gcd保持一致
- **False:** 使用原始特征（不推荐）

---

### `--eval_dense`
**默认值:** False
**类型:** bool

**作用:**
- **True:** 只构建和评估高密度聚类（骨干网络），跳过低密度点分配
- **False (默认):** 完整流程，分配所有点

**用途:** 评估核心聚类质量，调试高密度点选择

---

### `--silent`
**默认值:** False
**类型:** bool

**作用:**
- **True:** 静默模式，关闭所有调试输出，只显示最终结果
- **False (默认):** 显示详细的聚类过程信息

**用途:** 网格搜索加速、批量实验

---

### `--single_detail`
**默认值:** False
**类型:** bool

**作用:**
- **True:** 详细分析只包含1个样本的聚类，输出K近邻信息
- **False (默认):** 不分析

**用途:** 调试孤立点问题

---

### `--detail_sample`
**默认值:** False
**类型:** bool

**作用:**
- **True:** 上帝视角分析，显示所有样本到真实类别原型的距离
- **False (默认):** 不分析

**用途:** 算法调试，理解聚类失败的原因

---

### `--analyze_dense`
**默认值:** False
**类型:** bool

**作用:**
- **True:** 分析高密度点之间的真实类内类间距离分布
- **False (默认):** 不分析

**用途:** 理解数据结构，验证高密度点选择的合理性

---

### `--run_kmeans_baseline`
**默认值:** False
**类型:** bool

**作用:**
- **True:** 同时运行K-means基线并输出对比结果
- **False (默认):** 只运行SS-DDBC

---

### `--eval_version`
**默认值:** v1
**类型:** str
**选项:** [v1, v2]

**作用:**
- **v1 (默认):** 原始匈牙利算法匹配
- **v2:** 新版本匹配策略

**通常保持默认即可。**

---

## 网格搜索参数

### `--grid_search`
**默认值:** False
**类型:** bool

**作用:**
- **True:** 启用网格搜索，自动搜索k和density_percentile的最佳组合
- **False (默认):** 单次运行

**结果:** 保存为CSV文件，包含所有参数组合的性能

---

### `--k_min` / `--k_max`
**默认值:** 3 / 21
**类型:** int

**作用:**
- 网格搜索时k值的范围
- 搜索范围为`[k_min, k_max)`，步长为2

**建议:**
- `k_min`: [3-10]
- `k_max`: [15-30]

---

### `--dp_min` / `--dp_max`
**默认值:** 20 / 100
**类型:** int

**作用:**
- 网格搜索时density_percentile的范围
- 搜索范围为`[dp_min, dp_max]`，步长由`--dp_step`指定

**建议:**
- `dp_min`: [20-50]
- `dp_max`: [80-100]

---

### `--dp_step`
**默认值:** 5
**类型:** int

**作用:**
- 网格搜索时density_percentile的步长

**建议:**
- 较粗搜索: 10
- 默认: 5
- 精细搜索: 2 (耗时)

---

## 快速参考表

### 最常用的参数组合

#### 1. 通用默认（推荐起点）
```bash
python -m clustering.testing.main \
  --superclass_name trees \
  --k 10 \
  --density_percentile 75 \
  --dense_method 0 \
  --assign_model 2 \
  --co_mode 2
```

#### 2. 噪声数据
```bash
python -m clustering.testing.main \
  --superclass_name trees \
  --k 12 \
  --density_percentile 85 \
  --dense_method 1 \
  --assign_model 2 \
  --co_mode 2
```

#### 3. 不均匀分布数据
```bash
python -m clustering.testing.main \
  --superclass_name trees \
  --k 10 \
  --density_percentile 75 \
  --dense_method 0 \
  --assign_model 2 \
  --co_mode 3
```

#### 4. 网格搜索最佳参数
```bash
python -m clustering.testing.main \
  --superclass_name trees \
  --grid_search True \
  --k_min 5 \
  --k_max 21 \
  --dp_min 60 \
  --dp_max 90 \
  --dp_step 5 \
  --silent True
```

#### 5. 快速调试模式
```bash
python -m clustering.testing.main \
  --superclass_name trees \
  --use_train_and_test False \
  --silent False \
  --single_detail True
```

---

## 参数调优建议

### 阶段1: 确定基本参数
1. 使用默认参数运行一次
2. 观察聚类数量和准确率
3. 如果聚类数过多 → 提高`density_percentile`
4. 如果聚类数过少 → 降低`density_percentile`

### 阶段2: 网格搜索优化
```bash
python -m clustering.testing.main \
  --grid_search True \
  --silent True \
  --superclass_name trees
```

### 阶段3: 精细调整
1. 根据网格搜索结果选择最佳的k和density_percentile
2. 尝试不同的`dense_method`（如果数据有噪声，尝试方法1）
3. 尝试不同的`co_mode`（如果数据分布不均匀，尝试模式3）
4. 调整`assign_model`和`voting_k`（影响相对较小）

---

## 常见问题 FAQ

**Q: 聚类数量太多怎么办？**
A: 提高`--density_percentile`，例如从75提高到85-90

**Q: 聚类数量太少怎么办？**
A: 降低`--density_percentile`，例如从75降低到60-65

**Q: 如何处理噪声数据？**
A: 使用`--dense_method 1`（中位数密度）+ 较高的`--density_percentile`（85-90）

**Q: 如何加速网格搜索？**
A: 使用`--silent True` + 增大`--dp_step`（例如10）+ 减小搜索范围

**Q: co_mode选择哪个？**
A:
- 默认使用模式2（K近邻平均）
- 数据分布不均匀时使用模式3（相对自适应）
- 已知最佳值时使用模式1（手动指定）

**Q: assign_model选择哪个？**
A:
- 默认使用模式2（KNN投票）
- 需要极致速度时使用模式1
- 需要最高精度时使用模式3

---

## 更新日志

- 2025-01-20: 初始版本，添加所有参数的详细说明
- 2025-01-20: 添加co_mode和co_manual参数说明
- 2025-01-20: 增强所有参数的help信息

---

**相关文档:**
- 密度计算方法: `clustering/density/DENSITY_METHODS.md`
- co计算模式: `clustering/utils/CO_MODES.md`
- 分配策略: `clustering/ssddbc/ASSIGNMENT_STRATEGIES.md` (如果存在)
