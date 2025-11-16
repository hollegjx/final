# 全数据集训练 + 超类评估功能

## 🎯 **功能概述**

实现了**全CIFAR-100数据集训练 + 指定超类评估**的功能，可以：
1. 使用完整100类数据进行训练（更丰富的特征学习）
2. 在指定超类上进行GCD评估（灵活的评估维度）
3. 比较不同超类的GCD任务难度

## 🚀 **使用方法**

### **1. 单个超类评估**

```bash
# 训练100类，在trees超类上评估
python methods/contrastive_training/contrastive_training.py \
    --dataset_name 'cifar100' \
    --eval_superclass 'trees' \
    --epochs 200 \
    --batch_size 128

# 训练100类，在flowers超类上评估
python methods/contrastive_training/contrastive_training.py \
    --dataset_name 'cifar100' \
    --eval_superclass 'flowers' \
    --epochs 200 \
    --batch_size 128

# 传统全数据集评估（不指定超类）
python methods/contrastive_training/contrastive_training.py \
    --dataset_name 'cifar100' \
    --epochs 200 \
    --batch_size 128
```

### **2. 批量超类评估**

```bash
# 交互式批量评估
python batch_superclass_eval.py
```

## 📊 **可用超类列表**

项目支持以下15个CIFAR-100超类：

| 超类名称 | 包含类别数 | 类别示例 |
|----------|------------|----------|
| trees | 5 | maple_tree, oak_tree, palm_tree, pine_tree, willow_tree |
| flowers | 5 | orchid, poppy, rose, sunflower, tulip |
| fruits_vegetables | 5 | apple, mushroom, orange, pear, sweet_pepper |
| mammals | 23 | beaver, dolphin, elephant, seal, whale 等 |
| marine_animals | 10 | aquarium_fish, flatfish, ray, shark, trout 等 |
| insects_arthropods | 8 | bee, beetle, butterfly, caterpillar, spider 等 |
| reptiles | 4 | crocodile, dinosaur, lizard, snake |
| humans | 5 | baby, boy, girl, man, woman |
| furniture | 5 | bed, chair, couch, table, wardrobe |
| containers | 5 | bottle, bowl, can, cup, plate |
| vehicles | 9 | bicycle, bus, motorcycle, pickup_truck, train 等 |
| electronic_devices | 6 | clock, computer_keyboard, lamp, telephone, television, camera |
| buildings | 4 | castle, house, bridge, road |
| terrain | 5 | cloud, forest, mountain, plain, sea |
| weather_phenomena | 1 | cloud |

## 🔧 **核心功能实现**

### **新增参数**：
- `--eval_superclass`: 指定评估的超类名称
- 如果不指定则进行全数据集评估

### **智能标签映射**：
- 自动过滤到指定超类的样本
- 重新映射标签到连续的0-n范围
- 正确处理已知/未知类别划分

### **评估输出示例**：
```
🎯 开始超类 "trees" 评估...
   超类包含类别: [47, 52, 56, 59, 96]
   过滤后样本数: 5000
   实际类别数: 5
   已知类别数: 4
   未知类别数: 1

📊 超类 'trees' 评估结果:
   All ACC: 0.8150
   Old ACC: 0.8900
   New ACC: 0.6200
```

## 💡 **设计优势**

### **1. 更丰富的特征学习**
- 100类训练 vs 5类训练
- 模型见过更多样的视觉模式
- 特征表示更加鲁棒

### **2. 公平的跨超类比较**
- 所有超类使用相同的预训练模型
- 消除训练差异带来的影响
- 真实反映不同超类的GCD难度

### **3. 灵活的评估维度**
- 可以专注于特定领域（如动物、植物等）
- 便于分析哪些超类更适合GCD
- 支持细粒度的性能分析

## 🧪 **实验建议**

### **1. 超类难度分析**
```bash
# 比较不同超类的GCD难度
python batch_superclass_eval.py
# 选择"评估所有超类"
```

### **2. 参数敏感性分析**
```bash
# 测试不同参数对特定超类的影响
python methods/contrastive_training/contrastive_training.py \
    --dataset_name 'cifar100' \
    --eval_superclass 'trees' \
    --sup_con_weight 0.3 \
    --temperature 0.7
```

### **3. 训练策略对比**
```bash
# 对比全数据集训练 vs 超类训练
# 1. 全数据集训练+超类评估
python methods/contrastive_training/contrastive_training.py \
    --dataset_name 'cifar100' \
    --eval_superclass 'trees'

# 2. 超类训练（仅供对比）
python scripts/train_superclass.py \
    --dataset_name 'cifar100_superclass' \
    --superclass_name 'trees'
```

## 📈 **结果分析**

评估完成后会生成：
1. **控制台输出**：实时显示各超类的评估结果
2. **结果文件**：`superclass_eval_results_YYYYMMDD_HHMMSS.txt`
3. **TensorBoard日志**：可视化训练和评估过程

## ⚠️ **使用注意事项**

1. **数据集要求**：确保使用完整的CIFAR-100数据集
2. **计算资源**：全数据集训练需要更多GPU内存和时间
3. **超类名称**：必须使用预定义的超类名称（见上表）
4. **结果解释**：超类评估的ACC计算基于该超类内部的聚类准确率

## 🔍 **故障排除**

### **找不到超类**：
```
⚠️ 警告: 测试集中没有找到超类 'xxx' 的样本
```
**解决**：检查超类名称是否正确，参考可用超类列表。

### **内存不足**：
**解决**：减小batch_size或使用gradient checkpointing。

### **训练时间过长**：
**解决**：减少epochs数量或使用更少的超类进行测试。

这个功能为GCD研究提供了更灵活和全面的评估工具！