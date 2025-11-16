## cache_features.py
- 功能：定位最佳超类模型、加载对应数据集，提取训练/测试特征并写入`features.pkl`缓存，供聚类与评估复用。
- 主要参数：
  - `--superclass_name` / `--all_superclasses`：选择单个或全部15个超类。
  - `--model_path` / `--auto_find_best` / `--checkpoint_root`：手动或自动定位`model_best_acc*.pt`等权重。
  - `--cache_dir` / `--use_l2` / `--overwrite`：控制缓存存储位置、是否L2归一化及是否覆盖。
  - `--batch_size` / `--num_workers` / `--gpu` / `--prop_train_labels` / `--seed`：特征提取与数据划分相关超参数。
- Help：`python scripts/cache_features.py --help`

## train_superclass.py
- 功能：在指定CIFAR-100超类上执行对比学习训练，可单独训练或批量训练全部超类，并自动保存最佳模型。
- 主要参数（部分节选）：
  - 数据/调度：`--superclass_name`、`--train_all_superclasses`、`--batch_size`、`--num_workers`、`--epochs`、`--seed`、`--gpu`。
  - 模型/优化：`--warmup_model_dir`、`--grad_from_block`、`--lr`、`--momentum`、`--weight_decay`、`--temperature`。
  - 对比学习：`--sup_con_weight`、`--contrast_unlabel_only`、`--n_views`、`--prop_train_labels`。
  - 记录：`--exp_root`、`--model_name`、`--eval_funcs`、`--save_best_thresh`。
- Help：`python scripts/train_superclass.py --help`

## grid_search_superclass.py
- 功能：基于`train_superclass.py`封装的网格搜索器，对`lr`与`sup_con_weight`组合遍历，自动记录最佳指标准备后续精训。
- 主要参数：
  - 继承训练脚本全部CLI参数（通过`build_superclass_train_parser`）。
  - 额外增加`--superclasses`用于指定需要搜索的超类列表，默认覆盖全部15类。
  - 内置`DEFAULT_LR_GRID=[0.1,0.05,0.01,0.001]`与`DEFAULT_SUP_CON_GRID=0.2~0.8`步长0.05，可通过修改源码自定义。
- Help：`python scripts/grid_search_superclass.py --help`

## data_split_generator.py
- 功能：读取原始CIFAR-100数据，按15个超类生成GCD兼容的训练/验证/测试划分，并输出JSON摘要与索引文件。
- 主要参数：
  - `--output_dir`：划分结果与摘要文件的输出目录。
  - `--cifar100_root`：CIFAR-100数据根目录（若不存在可自动下载）。
- Help：`python scripts/data_split_generator.py --help`

## offline_ssddbc_superclass.py
- 功能：离线 SSDDBC 管线封装脚本（单超类），完成「特征提取 → 网格搜索 → 伪标签生成」全流程。
- 工作流程：
  1. **Stage 1/2**：调用 `cache_features.py` 使用指定 ckpt 提取并缓存特征。
  2. **Stage 2/2**：在缓存特征上执行 SSDDBC 网格搜索，生成伪标签文件。
- 主要参数：
  - **必要参数**：
    - `--superclass_name`：要处理的超类名称（如 `trees`）。
    - `--ckpt_path`：训练阶段保存的完整 ckpt 路径。
  - **特征缓存参数**：
    - `--feature_cache_dir`：特征缓存根目录（默认使用 `config.py` 中的配置）。
    - `--batch_size` / `--num_workers` / `--gpu`：特征提取相关参数。
    - `--prop_train_labels` / `--seed`：数据划分参数。
  - **SSDDBC 网格搜索参数**：
    - `--k_min` / `--k_max`：KNN k 搜索范围（默认 3-21）。
    - `--density_min` / `--density_max` / `--density_step`：密度百分位搜索范围（默认 40-100，步长 5）。
    - `--max_workers`：并行进程数上限（默认使用 CPU 一半核心）。
    - `--pseudo_output_dir`：伪标签输出目录（默认为 `feature_cache_dir/<superclass_name>/pseudo_labels`）。
- **输出**：
  - 伪标签文件：`<superclass_name>_<ckpt_base>_k<k>_dp<density_percentile>.npz`
  - 包含：labels、core_mask、best_params、metadata（ACC、核心点数等）。
- **静默模式**：当前配置为 `silent=True`（网格搜索不输出详细聚类分析，只显示最终结果）。
- Help：`python scripts/offline_ssddbc_superclass.py --help`

## pseudo_pipeline.py
- 功能：三阶段自动化训练管线（单超类），实现「预热训练 → 循环（离线聚类 → 伪标签续训）」的完整流程。
- 工作流程：
  1. **Stage 1（预热训练）**：调用 `train_superclass.py` 训练到指定 epoch，保存 ckpt 并导出特征缓存。
  2. **循环阶段（Stage 2 → Stage 3）**：
     - **Stage 2（离线聚类）**：调用 `offline_ssddbc_superclass.py` 对当前 ckpt 进行聚类，生成伪标签。
     - **Stage 3（伪标签续训）**：调用 `train_superclass.py --resume_from_ckpt` 使用伪标签继续训练 N 轮。
     - 重复 Stage 2 → Stage 3，直到达到总训练轮数。
- 主要参数：
  - **必要参数**：
    - `--superclass_name`：要训练的超类名称（如 `trees`）。
  - **训练控制参数**：
    - `--stage1_epochs`：预热训练轮数（默认 50）。
    - `--update_interval`：伪标签更新间隔（每 N 轮重新聚类，默认 5）。
    - `--total_epochs`：总训练轮数（默认 200）。
  - **训练配置参数**：
    - `--batch_size` / `--num_workers` / `--gpu`：训练相关参数（默认 128 / 8 / 0）。
    - `--prop_train_labels` / `--seed`：数据划分参数（默认 0.8 / 1）。
  - **路径参数**：
    - `--feature_cache_dir`：特征缓存目录（默认 `/data/gjx/checkpoints/features1`）。
    - `--runs_root`：Pipeline 运行目录根路径（默认 `runs_pipeline`）。
- **输出结构**（保存在 `{runs_root}/{superclass_name}/{timestamp}/`）：
  - `log/`：TensorBoard 日志（所有阶段共用）。
  - `checkpoints/{superclass_name}/`：训练检查点（`ckpt_epoch_XXX.pt`）。
  - `pseudo_labels/`：伪标签文件（`*.npz`）。
- **使用示例**：
  ```bash
  # 完整训练（预热50轮 → 每5轮更新伪标签 → 总共200轮）
  python scripts/pseudo_pipeline.py \
    --superclass_name trees \
    --stage1_epochs 50 \
    --update_interval 5 \
    --total_epochs 200 \
    --batch_size 128 \
    --num_workers 16 \
    --gpu 0 \
    --feature_cache_dir /data/gjx/checkpoints/features1 \
    --runs_root /data/gjx/pipeline_runs

  # 快速测试（预热5轮 → 每3轮更新 → 总共15轮）
  python scripts/pseudo_pipeline.py \
    --superclass_name trees \
    --stage1_epochs 5 \
    --update_interval 3 \
    --total_epochs 15 \
    --batch_size 128 \
    --num_workers 16 \
    --gpu 0
  ```
- **执行流程示例**（`--stage1_epochs 50 --update_interval 5 --total_epochs 60`）：
  ```
  Epoch 0-50:   Stage1 预热训练（无伪标签）
  Epoch 50:     Stage2 首次聚类 → Stage3 续训（50-55）
  Epoch 55:     Stage2 更新聚类 → Stage3 续训（55-60）
  ```
- **注意事项**：
  - 最佳模型（`allacc_*.pt`）保存在全局目录 `/data/gjx/checkpoints/gcdsuperclass1/{superclass_name}/`。
  - 完整检查点（`ckpt_epoch_*.pt`）保存在 Pipeline 运行目录中。
  - 每次 Stage2 会覆盖特征缓存（`--overwrite`），确保使用最新 ckpt 的特征。
- Help：`python scripts/pseudo_pipeline.py --help`
