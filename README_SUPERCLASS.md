# CIFAR-100 超类训练 - 快速开始

本项目为GCD添加了15个超类的训练支持。

## 🚀 快速开始

### 1. 生成数据划分
```bash
python scripts/data_split_generator.py --output_dir ./data_splits
```

### 2. 训练单个超类
```bash
# 训练trees超类
python scripts/train_superclass.py --superclass_name trees --epochs 20

# 训练mammals超类
python scripts/train_superclass.py --superclass_name mammals --epochs 20
```

### 3. 查看演示
```bash
# 查看超类功能演示
python examples/example_superclass_usage.py

# 查看增强训练功能演示
python examples/demo_enhanced_training.py --demo early_stopping
```

## 📁 新的文件结构

```
generalized-category-discovery-main/
├── scripts/                            # 训练和工具脚本
│   ├── train_superclass.py             # 超类训练脚本
│   └── data_split_generator.py         # 数据划分生成器
├── examples/                           # 示例和演示脚本
│   ├── example_superclass_usage.py     # 超类功能演示
│   └── demo_enhanced_training.py       # 增强功能演示
├── docs/                              # 文档
│   └── SUPERCLASS_README.md           # 详细说明文档
├── data/                              # 数据处理
│   ├── cifar100_superclass.py         # 超类数据集
│   └── get_datasets.py                # 数据集获取(已更新)
├── utils/                             # 工具模块
│   └── training_utils.py              # 增强训练功能
└── methods/contrastive_training/      # 核心训练(已增强)
    └── contrastive_training.py
```

## ✨ 新功能

- **15个超类支持**: 从trees到weather_phenomena
- **增强训练显示**: 轮次分割、时间显示、性能差距
- **智能早停**: 29轮无改善自动停止
- **GCD完全兼容**: 保持原版训练逻辑100%一致

## 📊 推荐训练超类

| 超类名 | 已知类数 | 未知类数 | 推荐度 |
|-------|---------|---------|--------|
| mammals | 20 | 3 | ⭐⭐⭐⭐⭐ |
| vehicles | 6 | 3 | ⭐⭐⭐⭐ |
| insects_arthropods | 6 | 2 | ⭐⭐⭐⭐ |
| trees | 4 | 1 | ⭐⭐⭐ |

查看完整文档: `docs/SUPERCLASS_README.md`