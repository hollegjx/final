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
[阶段2] 启动聚类进程 → 加载特征 → SSDDBC 网格搜索 → 保存伪标签 → 进程退出
    ↓
[阶段3] 启动训练进程 → 加载ckpt+伪标签 → 继续训练 → 进程退出
```

**核心优势**：
- ✅ 进程隔离：训练GPU/线程资源随进程结束自动释放
- ✅ 资源独占：聚类阶段可以放心用多进程，不受训练框架约束
- ✅ 调试友好：每个阶段可以单独运行和测试
- ✅ 代码解耦：训练代码和聚类代码完全分离

---

## 二、三阶段概述（简版）

- **阶段1：预热训练**  
  训练到 `stop_at_epoch`（如 50），保存完整 ckpt（含优化器/调度器状态）并导出特征缓存，日志写入统一 log_dir。

- **阶段2：离线聚类**  
  读取最新特征缓存和 ckpt，执行 SSDDBC 网格搜索，输出标准 `.npz`（indices/labels/core_mask/best_params）。缺失伪标签时用对应 epoch 的 ckpt 重新聚类，已有则复用。

- **阶段3：伪标签续训**  
  恢复 ckpt + 伪标签继续训练，损失 = `(1-sup_con_weight)*L_contrast + sup_con_weight*L_supcon + γ·L_pseudo`，γ 按全局 total_epochs 线性增长。Stage3 分段由 orchestrator 控制，用 `--stop_at_epoch` 截断。

## 三、关键设计点

| 设计点 | 选择 | 理由 |
|--------|------|------|
| **预热轮数** | 50 | 特征空间足够成熟 |
| **核心点筛选** | SSDDBC骨干簇 | 高密度、高质量 |
| **聚类方式** | 独立进程多进程并行 | 避免与训练框架冲突 |
| **伪标签损失** | SupCon（主）/Prototypical（备选） | 尺度一致/原型解释 |
| **权重调度** | 0.0→1.0线性 | 平滑无跳变 |

---

## 四、实现状态

| 任务 | 状态 | 文件 | 备注 |
|------|------|------|------|
| **阶段1**：离线伪标签生成闭环 | ✅ 已完成 | `scripts/offline_ssddbc_superclass.py`, `scripts/cache_features.py`, `utils/pseudo_labels.py` | - |
| **阶段2**：训练端消费伪标签 | ✅ 已完成 | `scripts/train_superclass.py`, `utils/pseudo_labels.py` | - |
| **阶段3**：伪标签损失与 γ 调度 | ✅ 已完成 | `scripts/train_superclass.py` | - |
| **阶段4**：调度脚本自动化 | ✅ 已完成 | `scripts/pseudo_pipeline.py` | 已支持循环、断点恢复、伪标签复用、加权模式 |
| 训练脚本支持 stop_at_epoch / 保存特征 | ✅ 已完成 | `scripts/train_superclass.py`, `scripts/_feature_cache_runner.py` | - |
| 主调度脚本验证 | ⏳ 待验证 | `scripts/pseudo_pipeline.py` | 需实际运行确认输出正确 |

### 运行中间产物存放与复用关系

1. **阶段1（训练进程）**
   - 生成 ckpt：`<exp_root>/checkpoints/<superclass>/ckpt_epoch_XXX.pt`（供 Stage2/Stage3 使用）；
   - 写入 TensorBoard：`<exp_root>/superclass_train/log/(timestamp)/`（Stage3 通过 `--reuse_log_dir` 复用）；
   - 若启用 `--save_features_and_exit`，自动运行 `scripts/cache_features.py` 在 `<feature_cache_dir>/<superclass>/features.pkl` 写入最新特征；Stage2 读取该缓存。
2. **阶段2（离线 SSDDBC）**
   - 读取 Stage1 的 feature cache；
   - 输出伪标签 `.npz`：默认路径 `feature_cache_dir/<superclass>/pseudo_labels/*.npz` 或 orchestrator 指定的 `runs/<superclass>/<run_id>/pseudo_labels`；Stage3 用 `--pseudo_labels_path` 读取，缺失则用对应 epoch ckpt 重新聚类，已有则复用。
3. **阶段3（伪标签续训）**
   - `--resume_from_ckpt` 指向 Stage1 保存的 ckpt；
   - `--pseudo_labels_path` 指向 Stage2 输出的 `.npz`；
   - `--reuse_log_dir` 复用 Stage1 的日志目录，TensorBoard 曲线连续呈现。
4. **orchestrator (`scripts/pseudo_pipeline.py`)**
   - 创建统一的 `runs_root/<superclass>/<run_id>/`（可用 `--resume_run_dir` 断点续跑），将 `exp_root` 指向该目录；
   - Stage1/Stage3 的 log/ckpt/pseudo 均落在同一 run 下，伪标签缺失时按刷新基点对应的 ckpt 重新聚类，已有则复用。

---

## 五、监控指标

训练过程记录：
- 损失：`L_total`, `L_contrast`, `L_pseudo`, `gamma`
- 伪标签：簇数量、核心点比例、标签变化率
- 性能：`val_acc_old`, `val_acc_new`, `val_acc_all`

---

**最后更新**: 2025-11-20
**状态**: 核心功能已实现，调度脚本待实跑验证
