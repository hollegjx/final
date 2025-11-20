# 自适应K值选择方案设计文档

## 📋 目录
1. [背景与动机](#背景与动机)
2. [问题分析](#问题分析)
3. [方案对比](#方案对比)
4. [推荐方案详解](#推荐方案详解)
5. [实现路线图](#实现路线图)

---

## 🎯 背景与动机

### 当前状态
- ✅ 已实现：co距离自适应（使用k近邻平均距离的平均值）
- ❌ 未实现：k值自适应选择

### 目标
实现k值的自适应选择，通过评估聚类质量来自动确定最优k值。

### 特殊需求
**保护有意义的小簇**：
- GCD任务中，小簇可能捕捉了某个类别的局部特征或子结构
- 这些小簇虽然样本少，但在后续低密度样本分配时可能有重要作用
- 需要避免评估指标过度惩罚小簇

### 实际案例
```
📊 常规聚类 #2 - 大小: 7  (小簇，但类别纯度高)
   真实类别分布: {3: 6, 4: 1}
   已知类样本分布: {3: 6}
   → 这个小簇捕捉到了类别3的某种特征
```

---

## 🔍 问题分析

### 评估指标的局限性

#### 1. **轮廓系数 (Silhouette Coefficient)**
**公式**：
```
silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))

a(i) = 样本i到同簇其他点的平均距离（簇内距离）
b(i) = 样本i到最近其他簇的平均距离（簇间距离）
```

**优点**：
- ✅ 不需要真实标签
- ✅ 直接衡量簇内紧密度和簇间分离度

**缺点**：
- ❌ **偏向大簇和凸形簇**
- ❌ 对小簇不友好：小簇样本到大簇的距离虽远，但簇内距离也可能不小
- ❌ 可能导致算法倾向于"消灭"小簇

#### 2. **NMI (Normalized Mutual Information)**
**优点**：
- ✅ 衡量聚类结果与真实标签的信息一致性
- ✅ 对聚类数量不敏感

**缺点**：
- ❌ **需要完整的真实标签**
- ❌ GCD任务中未知类无标签，无法直接使用

#### 3. **ARI (Adjusted Rand Index)**
**优点**：
- ✅ 衡量配对一致性
- ✅ 调整了随机因素

**缺点**：
- ❌ **需要完整的真实标签**
- ❌ 与NMI有相同的局限性

### GCD任务的特殊性

1. **标签不完整**：
   - 已知类：有部分标签（80%训练集有标签）
   - 未知类：完全无标签

2. **小簇的重要性**：
   - 可能捕捉类别的局部特征
   - 在低密度样本分配时提供多样性
   - 不应被过度惩罚

3. **需要平衡**：
   - 无监督指标（不需要标签）
   - 有监督信息（利用已知类标签）
   - 小簇保护机制

---

## 📊 方案对比

### 方案1：基于轮廓系数的网格搜索

**核心思路**：
```python
for k in range(3, 20):
    predictions = run_clustering(X, k=k)
    score = silhouette_score(X, predictions)
    # 选择score最高的k
```

**优点**：
- ✅ 不需要真实标签
- ✅ 直接衡量聚类质量

**缺点**：
- ❌ 计算成本高（需要多次完整聚类）
- ❌ 对小簇不友好
- ❌ 偏向凸形簇

**推荐度**：⭐⭐⭐

---

### 方案2：基于密度稳定性的肘部法则

**核心思路**：
```python
for k in range(3, 20):
    densities = compute_density(X, k)
    n_high_density = count_high_density_points(densities)
    # 寻找n_high_density变化的拐点
```

**优点**：
- ✅ 不需要重复运行完整聚类
- ✅ 计算成本低
- ✅ 符合密度聚类的逻辑

**缺点**：
- ⚠️ 拐点可能不明显
- ⚠️ 需要额外的拐点检测算法

**推荐度**：⭐⭐⭐⭐

---

### 方案3：基于簇内方差的评估

**核心思路**：
```python
for k in range(3, 20):
    predictions, clusters = run_clustering(X, k=k)
    intra_distance = compute_intra_cluster_distance(X, clusters)
    # k越大 → 簇越多 → 簇内距离越小
    # 寻找簇内距离下降趋势变缓的k值
```

**优点**：
- ✅ 不需要真实标签
- ✅ 直接衡量簇的紧密度
- ✅ 可以结合co距离使用

**缺点**：
- ⚠️ 需要运行多次完整聚类
- ⚠️ 仍然可能对小簇不友好

**推荐度**：⭐⭐⭐

---

### 方案4：混合方案 - 两阶段优化（推荐）

**核心思路**：
```
阶段1（粗筛选）：使用密度稳定性快速缩小k范围
  k_range: 3-20 → candidate_k: [5, 7, 10, 12, 15]

阶段2（精筛选）：在候选k中使用轮廓系数精确选择
  使用混合指标：轮廓系数 + 簇纯度 + 小簇保护
```

**优点**：
- ✅ 平衡计算成本和准确性
- ✅ 结合多个指标的优势
- ✅ 适合大数据集

**缺点**：
- ⚠️ 实现相对复杂

**推荐度**：⭐⭐⭐⭐⭐

---

## 💡 推荐方案详解

### 方案：混合指标 + 小簇保护机制

#### 核心设计

```python
def adaptive_k_selection_for_gcd(X, labeled_mask, targets, k_range=(3, 20)):
    """
    为GCD任务设计的k自适应选择

    评估维度：
    1. 轮廓系数 (40%) - 无监督聚类质量
    2. Davies-Bouldin指数 (30%) - 簇分离度
    3. 已知类簇纯度 (20%) - 利用已知类信息
    4. 小簇保护奖励 (10%) - 保护有意义的小簇
    """
```

#### 指标详解

##### 1. 轮廓系数 (Silhouette Coefficient) - 权重40%

**计算**：
```python
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(X, labels)
```

**取值范围**：[-1, 1]
- 1.0：完美聚类
- 0.0：簇重叠
- 负值：样本分配错误

**作用**：主要评估指标，衡量无监督聚类质量

---

##### 2. Davies-Bouldin指数 - 权重30%

**计算**：
```python
from sklearn.metrics import davies_bouldin_score
db_score = davies_bouldin_score(X, labels)
# 注意：DB指数越小越好，需要取负值
normalized_db = -db_score / 10  # 归一化
```

**特点**：
- 越小越好（簇间分离度高，簇内紧密度高）
- 对小簇相对友好
- 不需要真实标签

**作用**：辅助评估，补充轮廓系数的不足

---

##### 3. 已知类簇纯度 - 权重20%

**计算**：
```python
def compute_cluster_purity_for_known(clusters, labeled_mask, targets):
    """
    计算已知类部分的簇纯度

    仅考虑有标签的样本
    """
    total_purity = 0

    for cluster in clusters:
        cluster_indices = list(cluster)
        cluster_labeled_mask = labeled_mask[cluster_indices]

        if np.sum(cluster_labeled_mask) == 0:
            continue  # 跳过无标签样本的簇

        # 获取簇中有标签样本的真实标签
        cluster_targets = targets[cluster_indices][cluster_labeled_mask]

        # 计算主导类占比
        unique, counts = np.unique(cluster_targets, return_counts=True)
        purity = np.max(counts) / len(cluster_targets)
        total_purity += purity

    return total_purity / len([c for c in clusters if has_labeled_samples(c)])
```

**取值范围**：[0, 1]
- 1.0：所有簇的已知类样本完全纯净
- 0.0：完全混乱

**作用**：利用GCD任务中的已知类信息，提高评估准确性

---

##### 4. 小簇保护奖励 - 权重10%

**计算**：
```python
def evaluate_small_clusters(clusters, labeled_mask, targets,
                           size_threshold=10, purity_threshold=0.8):
    """
    评估小簇质量，给予额外加分

    标准：
    1. 簇大小 < size_threshold
    2. 已知类样本纯度 >= purity_threshold

    奖励机制：
    - 高纯度小簇 → 额外加分
    - 鼓励保留捕捉局部特征的小簇
    """
    bonus = 0

    for cluster in clusters:
        cluster_size = len(cluster)
        if cluster_size >= size_threshold:
            continue  # 只评估小簇

        cluster_indices = list(cluster)
        cluster_labeled_mask = labeled_mask[cluster_indices]

        if np.sum(cluster_labeled_mask) == 0:
            continue  # 跳过无标签小簇

        # 计算已知类样本纯度
        cluster_targets = targets[cluster_indices][cluster_labeled_mask]
        unique, counts = np.unique(cluster_targets, return_counts=True)
        purity = np.max(counts) / len(cluster_targets)

        # 高纯度小簇给予奖励
        if purity >= purity_threshold:
            # 纯度越高，奖励越多
            bonus += 0.1 * (purity - purity_threshold) / (1.0 - purity_threshold)

    return bonus
```

**取值范围**：[0, 1]

**作用**：
- 保护有意义的小簇
- 避免过度合并或消除捕捉局部特征的簇
- 特别适合GCD任务

---

#### 综合评分公式

```python
final_score = (0.4 * silhouette +
               0.3 * normalized_db +
               0.2 * purity +
               0.1 * small_cluster_bonus)
```

**权重说明**：
- **40% 轮廓系数**：主要指标，评估整体聚类质量
- **30% DB指数**：补充指标，关注簇分离度
- **20% 簇纯度**：利用已知类信息
- **10% 小簇保护**：特殊奖励机制

**权重可调**：根据具体任务调整，例如：
- 更重视纯度 → 提高purity权重至30-40%
- 更重视小簇保护 → 提高bonus权重至15-20%

---

### 两阶段优化流程

#### 阶段1：粗筛选（密度稳定性）

**目的**：快速缩小k范围，降低计算成本

**方法**：
```python
def coarse_selection_by_density(X, k_range=(3, 20),
                                density_percentile=75,
                                n_candidates=5):
    """
    基于密度稳定性的粗筛选

    观察指标：
    1. 高密度点数量
    2. 密度分布方差
    3. 预期聚类数量
    """
    k_values = []
    metrics = []

    for k in range(k_range[0], k_range[1] + 1):
        # 计算密度
        densities, _, _ = compute_simple_density(X, k)

        # 识别高密度点
        threshold = np.percentile(densities, density_percentile)
        n_high_density = np.sum(densities >= threshold)

        # 密度方差
        density_var = np.var(densities)

        k_values.append(k)
        metrics.append({
            'k': k,
            'n_high_density': n_high_density,
            'density_var': density_var
        })

    # 寻找高密度点数量变化的拐点
    n_high_density_values = [m['n_high_density'] for m in metrics]
    elbow_indices = find_elbow_points(k_values, n_high_density_values,
                                     n_points=n_candidates)

    candidate_k = [k_values[i] for i in elbow_indices]
    return candidate_k
```

**拐点检测**：
```python
def find_elbow_points(x, y, n_points=5):
    """
    使用二阶导数检测拐点

    拐点特征：曲线曲率变化最大的位置
    """
    # 计算一阶导数
    dy = np.diff(y)

    # 计算二阶导数
    d2y = np.diff(dy)

    # 找到二阶导数绝对值最大的n_points个位置
    abs_d2y = np.abs(d2y)
    elbow_indices = np.argsort(abs_d2y)[-n_points:]

    # 加1是因为diff减少了一个元素
    return sorted(elbow_indices + 1)
```

---

#### 阶段2：精筛选（混合指标）

**目的**：在候选k中精确选择最优k

**方法**：
```python
def fine_selection_by_hybrid_metrics(X, candidate_k, labeled_mask, targets):
    """
    使用混合指标在候选k中精确选择
    """
    best_k = None
    best_score = -float('inf')
    results = []

    for k in candidate_k:
        # 运行完整聚类
        labels, clusters = run_clustering(X, k=k)

        # 计算各项指标
        silhouette = silhouette_score(X, labels)
        db_score = -davies_bouldin_score(X, labels) / 10
        purity = compute_cluster_purity_for_known(clusters, labeled_mask, targets)
        bonus = evaluate_small_clusters(clusters, labeled_mask, targets)

        # 综合评分
        final_score = (0.4 * silhouette +
                      0.3 * db_score +
                      0.2 * purity +
                      0.1 * bonus)

        results.append({
            'k': k,
            'score': final_score,
            'silhouette': silhouette,
            'db_score': db_score,
            'purity': purity,
            'bonus': bonus
        })

        if final_score > best_score:
            best_score = final_score
            best_k = k

    return best_k, results
```

---

### 完整流程伪代码

```python
def auto_select_k_for_gcd(X, labeled_mask, targets,
                         k_range=(3, 20),
                         use_coarse_selection=True):
    """
    GCD任务的k值自适应选择完整流程
    """
    print(f"🔍 开始k值自适应选择 (范围: {k_range})")

    if use_coarse_selection:
        # 阶段1：粗筛选
        print(f"📊 阶段1：粗筛选（密度稳定性）")
        candidate_k = coarse_selection_by_density(
            X, k_range=k_range, n_candidates=5
        )
        print(f"   候选k值: {candidate_k}")
    else:
        # 跳过粗筛选，使用全范围
        candidate_k = list(range(k_range[0], k_range[1] + 1))

    # 阶段2：精筛选
    print(f"🎯 阶段2：精筛选（混合指标）")
    best_k, results = fine_selection_by_hybrid_metrics(
        X, candidate_k, labeled_mask, targets
    )

    # 输出详细结果
    print(f"\n📈 评估结果:")
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"   k={r['k']:2d}: score={r['score']:.3f} "
              f"(sil={r['silhouette']:.3f}, db={r['db_score']:.3f}, "
              f"pur={r['purity']:.3f}, bonus={r['bonus']:.3f})")

    print(f"\n✅ 最优k值: {best_k}")

    return best_k, results
```

---

## 🛠️ 实现路线图

### 阶段1：基础实现

**文件结构**：
```
clustering/
├── k_selection/
│   ├── __init__.py
│   ├── metrics.py          # 评估指标实现
│   ├── coarse_selection.py # 粗筛选实现
│   ├── fine_selection.py   # 精筛选实现
│   └── main.py            # 主函数入口
```

**任务清单**：
- [ ] 实现轮廓系数计算
- [ ] 实现Davies-Bouldin指数计算
- [ ] 实现簇纯度计算
- [ ] 实现小簇保护机制
- [ ] 实现密度稳定性分析
- [ ] 实现拐点检测算法
- [ ] 集成到adaptive_clustering.py

---

### 阶段2：参数优化

**需要调优的参数**：
- 各指标权重：`alpha_silhouette`, `alpha_db`, `alpha_purity`, `alpha_bonus`
- 小簇阈值：`size_threshold` (默认10)
- 纯度阈值：`purity_threshold` (默认0.8)
- 候选k数量：`n_candidates` (默认5)

**方法**：
- 在不同超类数据集上测试
- 调整权重以获得最佳ACC/NMI/ARI
- 记录最优参数配置

---

### 阶段3：可视化

**实现功能**：
- [ ] k vs 各指标曲线图
- [ ] k vs 综合评分曲线图
- [ ] 最优k的聚类结果可视化
- [ ] 小簇分布可视化

**工具**：matplotlib

---

### 阶段4：命令行集成

**添加参数**：
```python
parser.add_argument('--auto_k', type=str2bool, default=False,
                    help='是否启用k值自适应选择')
parser.add_argument('--k_range_min', type=int, default=3,
                    help='k值搜索范围最小值')
parser.add_argument('--k_range_max', type=int, default=20,
                    help='k值搜索范围最大值')
parser.add_argument('--k_selection_method', type=str,
                    default='hybrid', choices=['hybrid', 'silhouette', 'density'],
                    help='k值选择方法')
```

**使用示例**：
```bash
python -m clustering.testing.main \
    --superclass_name trees \
    --auto_k True \
    --k_range_min 3 \
    --k_range_max 20 \
    --k_selection_method hybrid
```

---

## 📈 预期效果

### 优势
1. **自动化**：无需手动调参k值
2. **鲁棒性**：适应不同数据分布
3. **保护小簇**：避免过度合并
4. **利用已知信息**：充分利用GCD任务的已知类标签

### 可能的问题
1. **计算成本**：需要多次运行聚类
   - 解决：使用粗筛选减少候选数量
2. **参数敏感性**：权重设置影响结果
   - 解决：在多个数据集上验证默认参数
3. **小簇判断**：size_threshold设置可能不准确
   - 解决：根据数据集大小自适应调整

---

## 📚 参考资料

### 评估指标
1. **Silhouette Coefficient**: Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.
2. **Davies-Bouldin Index**: Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure.
3. **Elbow Method**: Thorndike, R. L. (1953). Who belongs in the family?

### 相关工作
1. **GCD任务**: Vaze et al. (2022). Generalized Category Discovery
2. **密度聚类**: DBSCAN, HDBSCAN
3. **半监督聚类**: SS-DDBC (本项目)

---

## 🔄 更新日志

### v1.0 (2025-01-15)
- 初始文档创建
- 定义4个备选方案
- 详细设计混合指标方案
- 制定实现路线图

---

## 📝 待讨论问题

1. **权重设置**：
   - 当前：40% sil + 30% db + 20% pur + 10% bonus
   - 是否需要根据数据集动态调整？

2. **小簇定义**：
   - 当前：size < 10
   - 是否应该基于总样本数的百分比？

3. **粗筛选必要性**：
   - 对于小规模数据集（<5000样本），是否可以跳过粗筛选？

4. **计算成本优化**：
   - 是否可以缓存中间结果？
   - 是否可以并行计算不同k值？

---

**文档维护者**：Claude
**最后更新**：2025-01-15
**状态**：设计阶段，未实现
