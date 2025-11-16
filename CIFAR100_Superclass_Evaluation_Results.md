# CIFAR-100 超类全局验证结果

## 概述

本文档记录了在CIFAR-100数据集上对15个超类进行全局验证的结果。使用原版GCD的test_kmeans方法，在完整CIFAR-100数据集上训练，然后在每个超类上进行评估。

## 评估方法

- **训练数据**: 完整CIFAR-100数据集 (100个类)
- **评估方法**: 原版GCD test_kmeans (v2方法)
- **特征提取**: ViT-DINO base model (768维特征)
- **聚类方法**: K-means (聚类数=超类包含的类别数)
- **准确率计算**: 使用匈牙利算法进行聚类到类别的最优映射

## 15个超类定义

| 超类名称 | 包含类别 | 类别数量 |
|---------|---------|----------|
| trees | [47, 52, 56, 59, 96] | 5 |
| flowers | [54, 62, 70, 82, 92] | 5 |
| fruits_vegetables | [0, 51, 53, 57, 83] | 5 |
| mammals | [3, 4, 21, 31, 34, 36, 38, 43, 50, 55, 63, 64, 65, 66, 72, 74, 75, 15, 19, 42, 80, 88, 97] | 23 |
| marine_animals | [1, 26, 27, 30, 32, 45, 67, 73, 95, 91] | 10 |
| insects_arthropods | [6, 7, 14, 24, 77, 18, 79, 99] | 8 |
| reptiles | [44, 78, 29, 93] | 4 |
| humans | [11, 35, 46, 2, 98] | 5 |
| furniture | [5, 20, 25, 84, 94] | 5 |
| containers | [16, 9, 10, 28, 61] | 5 |
| vehicles | [8, 13, 48, 58, 69, 81, 85, 89, 90] | 9 |
| electronic_devices | [22, 39, 40, 41, 86, 87] | 6 |
| buildings | [12, 17, 37, 76] | 4 |
| terrain | [33, 49, 60, 68, 71] | 5 |
| weather_phenomena | [23] | 1 |

## 评估结果

### 运行命令模板

```bash
# 超类评估命令
python eval_original_gcd.py \
    --model_path "/path/to/your/model.pt" \
    --eval_mode superclass \
    --superclass_name <SUPERCLASS_NAME> \
    --gpu <GPU_ID>

# 示例：评估trees超类
python eval_original_gcd.py \
    --model_path "/data1/jiangzhen/gjx/exp/newgpc/final/metric_learn_gcd/log/(14.09.2025_|_56.443)/checkpoints/model.pt" \
    --eval_mode superclass \
    --superclass_name trees \
    --gpu 3
```

### 结果表格

| 超类名称 | All ACC | Old ACC | New ACC | 备注 |
|---------|---------|---------|---------|------|
| trees | - | - | - | 待评估 |
| flowers | - | - | - | 待评估 |
| fruits_vegetables | - | - | - | 待评估 |
| mammals | - | - | - | 待评估 |
| marine_animals | - | - | - | 待评估 |
| insects_arthropods | - | - | - | 待评估 |
| reptiles | - | - | - | 待评估 |
| humans | - | - | - | 待评估 |
| furniture | - | - | - | 待评估 |
| containers | - | - | - | 待评估 |
| vehicles | - | - | - | 待评估 |
| electronic_devices | - | - | - | 待评估 |
| buildings | - | - | - | 待评估 |
| terrain | - | - | - | 待评估 |
| weather_phenomena | - | - | - | 待评估 |

## 批量评估脚本

为方便批量评估所有超类，可以使用以下bash脚本：

```bash
#!/bin/bash

MODEL_PATH="/data1/jiangzhen/gjx/exp/newgpc/final/metric_learn_gcd/log/(14.09.2025_|_56.443)/checkpoints/model.pt"
GPU_ID=3

# 15个超类名称
SUPERCLASSES=(
    "trees"
    "flowers"
    "fruits_vegetables"
    "mammals"
    "marine_animals"
    "insects_arthropods"
    "reptiles"
    "humans"
    "furniture"
    "containers"
    "vehicles"
    "electronic_devices"
    "buildings"
    "terrain"
    "weather_phenomena"
)

# 循环评估每个超类
for superclass in "${SUPERCLASSES[@]}"; do
    echo "=== 评估超类: $superclass ==="
    python eval_original_gcd.py \
        --model_path "$MODEL_PATH" \
        --eval_mode superclass \
        --superclass_name "$superclass" \
        --gpu "$GPU_ID"
    echo ""
done
```

## 分析指标说明

### ACC计算方法 (v2方法)
1. **All ACC**: 在该超类包含的所有类别上的整体聚类准确率
2. **Old ACC**: 已知类别（训练时见过的类）的聚类准确率
3. **New ACC**: 未知类别（训练时未见过的类）的聚类准确率

### 评估特点
- 使用在完整CIFAR-100上训练的模型
- 每个超类独立进行K-means聚类
- 聚类数等于该超类包含的类别数
- 使用匈牙利算法进行最优映射
- 反映模型在细粒度任务上的泛化能力

## 预期分析

根据超类的特点，预期结果：

1. **高准确率超类**: trees, flowers, vehicles (视觉特征明显)
2. **中等准确率超类**: furniture, containers, buildings (结构相似但可区分)
3. **低准确率超类**: mammals (类内差异大), insects_arthropods (细节复杂)
4. **特殊情况**: weather_phenomena (只有1个类，准确率取决于是否被正确聚类)

## 更新说明

请在完成每个超类的评估后，更新上述结果表格，并添加相应的分析和观察。

---

*生成时间: 2025年1月*
*评估脚本: eval_original_gcd.py*
*数据集: CIFAR-100*
*模型: ViT-DINO base + GCD训练*