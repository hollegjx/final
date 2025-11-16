# 伪标签引导对比学习 - 三阶段独立进程训练方案

> **核心思想**: 训练与聚类完全解耦，通过"训练进程→聚类进程→训练进程"的串行自动化实现
> **更新日期**: 2025-11-16

---

## 一、整体自动化流程（三阶段独立进程）

```
主调度脚本（orchestrator.py）
    ↓
[阶段1] 启动训练进程 → 训练到指定epoch → 保存ckpt+特征 → 进程退出
    ↓
[阶段2] 启动聚类进程 → 加载特征 → 并行SSDDBC网格搜索 → 保存伪标签 → 进程退出
    ↓
[阶段3] 启动训练进程 → 加载ckpt+伪标签 → 继续训练 → 进程退出
```

**核心优势**：
- ✅ 进程隔离：训练GPU/线程资源随进程结束自动释放
- ✅ 资源独占：聚类阶段可以放心用多进程，不受训练框架约束
- ✅ 调试友好：每个阶段可以单独运行和测试
- ✅ 代码解耦：训练代码和聚类代码完全分离

---

## 二、三阶段详细描述

### 阶段1：预热训练 [Epoch 0-49]

**执行方式**：
```bash
python train.py --config config.yaml --stop_at_epoch 50 --save_features_and_exit
```

**训练目标**：让特征空间达到"SSDDBC能找到高质量聚类"的状态

**损失函数**：
```python
L_total = L_self_supervised + L_semi_supervised
```
- `L_self_supervised`：所有样本基于数据增强形成正对
- `L_semi_supervised`：有标注样本（旧类）使用真实标签形成正对

**结束时保存**：
- Checkpoint文件（包含model、optimizer、scheduler状态）
- 特征文件（用于后续聚类）

---

### 阶段2：离线聚类（独立进程，可多进程并行）

**执行方式**：
```bash
python offline_clustering.py \
  --superclass aquatic_mammals \
  --ckpt checkpoints/epoch_50.pth \
  --workers 8 \
  --output pseudo_labels/epoch_50.npz
```

**执行内容**：
1. 加载阶段1保存的特征文件
2. 多进程并行执行SSDDBC网格搜索（k_range × density_range）
3. 根据评价指标选择最佳聚类结果
4. 保存伪标签文件

**输出文件格式**（.npz）：
```python
{
    'indices': np.ndarray,       # 核心点样本索引 (N_core,)
    'labels': np.ndarray,        # 核心点伪标签 (N_core,)
    'core_mask': np.ndarray,     # 全体样本核心点掩码 (N_total,)
    'best_params': dict,         # 最佳超参 {k, density_percentile}
}
```

---

### 阶段3：伪标签引导训练 [Epoch 50-200]

**执行方式**：
```bash
python train.py \
  --config config.yaml \
  --resume checkpoints/epoch_50.pth \
  --pseudo_labels pseudo_labels/epoch_50.npz \
  --epochs 200
```

**损失函数**：
```python
L_total = L_self_supervised + L_semi_supervised + γ(epoch) · L_pseudo
```

**L_pseudo计算**（仅对核心点）：
```python
# 方案A（主）：监督对比损失
L_pseudo = SupConLoss(features[core_mask], pseudo_labels[core_mask])

# 方案B（备选）：原型对比损失
L_pseudo = PrototypicalConLoss(features[core_mask], pseudo_labels[core_mask])
```

**γ权重调度**（线性增长）：
```python
def get_gamma(epoch):
    if epoch < 50:
        return 0.0
    return (epoch - 50) / 150  # 50→200: 0.0→1.0
```

---

## 三、主调度脚本实现（orchestrator.py）

```python
import subprocess
import sys

def run_stage1_training(config):
    """阶段1: 训练到epoch 50"""
    cmd = [sys.executable, "train.py",
           "--config", config,
           "--stop_at_epoch", "50",
           "--save_features_and_exit"]
    subprocess.run(cmd, check=True)

def run_stage2_clustering(superclass, ckpt):
    """阶段2: 离线聚类"""
    cmd = [sys.executable, "offline_clustering.py",
           "--superclass", superclass,
           "--ckpt", ckpt,
           "--workers", "8",
           "--output", "pseudo_labels.npz"]
    subprocess.run(cmd, check=True)

def run_stage3_resume(config, ckpt, pseudo):
    """阶段3: 加载伪标签继续训练"""
    cmd = [sys.executable, "train.py",
           "--config", config,
           "--resume", ckpt,
           "--pseudo_labels", pseudo,
           "--epochs", "200"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_stage1_training("config.yaml")
    run_stage2_clustering("aquatic_mammals", "checkpoints/epoch_50.pth")
    run_stage3_resume("config.yaml", "checkpoints/epoch_50.pth", "pseudo_labels.npz")
```

---

## 四、关键设计点

| 设计点 | 选择 | 理由 |
|--------|------|------|
| **预热轮数** | 50 | 特征空间足够成熟 |
| **核心点筛选** | SSDDBC骨干簇 | 高密度、高质量 |
| **聚类方式** | 独立进程多进程并行 | 避免与训练框架冲突 |
| **伪标签损失** | SupCon（主）/Prototypical（备选） | 尺度一致/原型解释 |
| **权重调度** | 0.0→1.0线性 | 平滑无跳变 |

---

## 五、实现状态

| 任务 | 状态 | 文件 |
|------|------|------|
| 训练脚本支持stop_at_epoch | ⏳ 待实现 | train_superclass.py |
| 训练脚本保存特征 | ⏳ 待实现 | train_superclass.py |
| 离线聚类脚本 | ✅ 已完成 | offline_ssddbc_superclass.py |
| 训练脚本加载伪标签 | ⏳ 待实现 | train_superclass.py |
| L_pseudo损失计算 | ⏳ 待实现 | train_superclass.py |
| γ权重调度 | ⚠️ 已实现但未启用 | train_superclass.py |
| 主调度脚本 | ⏳ 待实现 | orchestrator.py（新建） |

---

## 六、监控指标

训练过程记录：
- 损失：`L_total`, `L_contrast`, `L_pseudo`, `gamma`
- 伪标签：簇数量、核心点比例、标签变化率
- 性能：`val_acc_old`, `val_acc_new`, `val_acc_all`

---

**最后更新**: 2025-11-16
**状态**: 核心设计确定，待实现训练脚本改造与主调度脚本
