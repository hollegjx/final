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
