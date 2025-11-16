# ssddbc/data - 独立数据读取模块

## 📋 概述

这是ssddbc模块的独立数据读取系统，提供特征加载和提取功能，**最小化对外部训练代码的依赖**。

### 设计目标

1. **独立性**: ssddbc模块应该能够独立运行，不过度依赖训练代码
2. **复用性**: 优先使用缓存特征，避免重复的特征提取
3. **灵活性**: 支持缓存加载和实时提取两种模式
4. **清晰性**: 职责分离，每个模块功能单一明确

---

## 📁 模块结构

```
ssddbc/data/
├── __init__.py              # 模块导出
├── dataset_config.py        # 数据集配置（独立定义CIFAR100超类）
├── feature_loader.py        # 特征缓存加载器
├── feature_extractor.py     # 模型特征提取器
├── data_provider.py         # 统一数据提供接口
├── test_data_modules.py     # 模块测试文件
├── example_usage.py         # 使用示例
└── README.md               # 本文档
```

---

## 🔧 核心模块说明

### 1. `dataset_config.py` - 数据集配置

**功能**: 独立定义数据集相关配置，不依赖外部训练代码

**主要内容**:
- `CIFAR100_SUPERCLASS_CONFIG`: CIFAR-100超类配置字典
- `get_superclass_info(superclass_name)`: 获取超类详细信息
- `get_all_superclass_names()`: 获取所有超类名称
- `validate_superclass_name(superclass_name)`: 验证超类名称

**使用示例**:
```python
from ssddbc.data.dataset_config import get_superclass_info

info = get_superclass_info('trees')
print(info['known_classes'])      # 原始已知类别ID
print(info['unknown_classes'])    # 原始未知类别ID
print(info['known_classes_mapped'])   # 映射后的已知类ID
print(info['label_mapping'])      # 标签映射字典
```

---

### 2. `feature_loader.py` - 特征缓存加载器

**功能**: 从磁盘缓存文件中加载预提取的特征

**核心类**: `FeatureLoader`

**主要方法**:
- `load(dataset_name, use_l2=True, silent=False)`: 加载特征缓存
- `check_cache_exists(dataset_name, use_l2=True)`: 检查缓存是否存在
- `get_cache_path(dataset_name, use_l2=True)`: 获取缓存路径

**缓存路径**:
- L2归一化: `/data/gjx/checkpoints/features/{dataset_name}/features.pkl`
- 无L2归一化: `/data/gjx/checkpoints/features_nol2/{dataset_name}/features.pkl`

**返回数据格式**:
```python
{
    'all_features': np.ndarray,      # (n_samples, feat_dim)
    'all_targets': np.ndarray,       # (n_samples,)
    'all_known_mask': np.ndarray,    # (n_samples,) bool
    'all_labeled_mask': np.ndarray,  # (n_samples,) bool
    'train_features': np.ndarray,    # 训练集特征
    'train_targets': np.ndarray,
    'train_known_mask': np.ndarray,
    'train_labeled_mask': np.ndarray,
    'test_features': np.ndarray,     # 测试集特征
    'test_targets': np.ndarray,
    'test_known_mask': np.ndarray,
    'test_labeled_mask': np.ndarray
}
```

**使用示例**:
```python
from ssddbc.data import FeatureLoader

loader = FeatureLoader(cache_base_dir='/data/gjx/checkpoints/features')

# 检查缓存是否存在
if loader.check_cache_exists('trees', use_l2=True):
    # 加载缓存
    feature_dict = loader.load('trees', use_l2=True, silent=False)
    all_feats = feature_dict['all_features']
```

---

### 3. `feature_extractor.py` - 模型特征提取器

**功能**: 使用给定的模型从数据加载器中提取特征

**核心类**: `FeatureExtractor`

**主要方法**:
- `extract_from_loader(data_loader, known_classes, silent)`: 从单个数据加载器提取
- `extract_train_test(train_loader, test_loader, known_classes, silent)`: 提取训练+测试集
- `extract_single_dataset(data_loader, known_classes, silent)`: 提取单个数据集

**使用示例**:
```python
from ssddbc.data import FeatureExtractor
import torch

# 假设已有模型和数据加载器
model = ...  # 加载好的PyTorch模型
device = torch.device('cuda')

extractor = FeatureExtractor(model=model, device=device, use_l2=True)

# 提取训练+测试集特征
feature_dict = extractor.extract_train_test(
    train_loader, test_loader,
    known_classes={0, 1, 2, 3},  # 已知类ID集合
    silent=False
)
```

**注意**:
- 模型会自动设置为eval模式
- 支持L2归一化
- 自动清理GPU内存
- 返回格式与FeatureLoader一致

---

### 4. `data_provider.py` - 统一数据提供接口 (推荐使用)

**功能**: 统一管理特征数据获取，自动选择缓存或实时提取

**核心类**: `DataProvider`

**主要方法**:
- `get_features(dataset_name, model, data_loaders, use_l2, use_train_and_test, silent)`: 获取特征数据
- `check_cache_available(dataset_name, use_l2)`: 检查缓存是否可用
- `get_cache_info(dataset_name, use_l2)`: 获取缓存信息

**工作流程**:
1. 优先尝试加载缓存
2. 缓存不存在时，使用模型实时提取
3. 返回特征数据和数据来源标识

**使用示例**:
```python
from ssddbc.data import DataProvider

provider = DataProvider(cache_base_dir='/data/gjx/checkpoints/features')

# 获取特征（自动处理缓存/实时提取）
feature_dict, source = provider.get_features(
    dataset_name='trees',
    model=model,                      # 缓存不存在时需要
    data_loaders=(train_loader, test_loader),  # 缓存不存在时需要
    use_l2=True,
    use_train_and_test=True,
    silent=False
)

print(f"数据来源: {source}")  # 'cache' or 'extraction'
all_feats = feature_dict['all_features']
```

---

## 🚀 快速开始

### 测试模块功能

运行测试文件验证模块是否正常工作:

```bash
# 测试所有模块
python -m ssddbc.data.test_data_modules

# 只测试数据集配置模块
python -m ssddbc.data.test_data_modules --test config

# 只测试缓存加载器
python -m ssddbc.data.test_data_modules --test loader

# 只测试数据提供器
python -m ssddbc.data.test_data_modules --test provider

# 只测试特征提取器结构
python -m ssddbc.data.test_data_modules --test extractor
```

### 查看使用示例

```bash
python -m ssddbc.data.example_usage
```

---

## 📖 典型使用场景

### 场景1: 只使用缓存（最常见）

```python
from ssddbc.data import DataProvider

provider = DataProvider()
feature_dict, source = provider.get_features(
    dataset_name='trees',
    use_l2=True
)

# 直接使用特征进行聚类
all_feats = feature_dict['all_features']
all_targets = feature_dict['all_targets']
all_known_mask = feature_dict['all_known_mask']
all_labeled_mask = feature_dict['all_labeled_mask']
```

### 场景2: 带回退的使用（缓存可能不存在）

```python
from ssddbc.data import DataProvider

provider = DataProvider()

# 先检查缓存
cache_info = provider.get_cache_info('trees', use_l2=True)

if cache_info['exists']:
    # 使用缓存
    feature_dict, source = provider.get_features(
        dataset_name='trees',
        use_l2=True
    )
else:
    # 需要实时提取，准备模型和数据加载器
    # (这里仍需要依赖外部的模型加载和数据集获取)
    model = load_model(...)
    train_loader, test_loader = get_data_loaders(...)

    feature_dict, source = provider.get_features(
        dataset_name='trees',
        model=model,
        data_loaders=(train_loader, test_loader),
        use_l2=True
    )
```

### 场景3: 获取超类配置信息

```python
from ssddbc.data.dataset_config import get_superclass_info

info = get_superclass_info('trees')

# 用于创建known_mask
known_classes_mapped = info['known_classes_mapped']

# 用于数据集划分
known_classes_orig = info['known_classes']
unknown_classes_orig = info['unknown_classes']

# 用于标签映射
label_mapping = info['label_mapping']
```

---

## 🔄 如何迁移现有代码

### 旧代码 (test_superclass.py):

```python
from ..utils.model_utils import load_model, extract_features
from ..utils.cache_utils import try_load_cached_features

cached_features = try_load_cached_features(superclass_name, use_l2=use_l2)

if cached_features is not None:
    all_feats = cached_features['all_features']
    # ...
else:
    model = load_model(args, device)
    train_feats, train_targets, ... = extract_features(train_loader, ...)
    # ...
```

### 新代码 (使用ssddbc/data):

```python
from ssddbc.data import DataProvider, get_superclass_info

provider = DataProvider()

# 获取特征（自动处理缓存/实时提取）
feature_dict, source = provider.get_features(
    dataset_name=superclass_name,
    model=model if needed else None,
    data_loaders=(train_loader, test_loader) if needed else None,
    use_l2=use_l2,
    silent=silent
)

all_feats = feature_dict['all_features']
all_targets = feature_dict['all_targets']
# ...

print(f"数据来源: {source}")
```

**优势**:
- ✅ 代码更简洁
- ✅ 减少对`../utils`的依赖
- ✅ 逻辑更清晰
- ✅ 易于测试和维护

---

## ⚠️ 注意事项

### 1. 外部依赖的最小化

虽然我们尽量减少外部依赖，但在某些情况下仍需要:

**完全独立的部分**:
- `dataset_config.py`: 完全独立，无外部依赖
- `feature_loader.py`: 完全独立，只依赖标准库

**需要外部依赖的部分**:
- `feature_extractor.py`: 需要PyTorch和tqdm
- `data_provider.py`: 整合其他模块

**实时提取时仍需外部帮助**:
- 模型加载: 需要models模块和config（提供DINO路径）
- 数据集获取: 需要data/目录下的数据集定义

### 2. 缓存文件格式

缓存文件必须是pickle格式，包含所有必需字段。如果缓存文件格式不正确，会抛出异常。

### 3. 路径配置

默认缓存路径是 `/data/gjx/checkpoints/features`，可以通过构造函数参数修改:

```python
provider = DataProvider(cache_base_dir='/your/custom/path')
```

---

## 🧪 测试

### 运行单元测试

```bash
# 在项目根目录下运行
cd E:\PythonProjects\fuxian\generalized-category-discovery-main

# 测试所有功能
python -m ssddbc.data.test_data_modules

# 测试特定模块
python -m ssddbc.data.test_data_modules --test config
```

### 测试内容

1. **数据集配置测试**: 验证超类配置的正确性
2. **缓存加载测试**: 验证缓存文件的读取和验证
3. **数据提供器测试**: 验证统一接口的功能
4. **特征提取器结构测试**: 验证类的初始化和配置

---

## 📝 更新日志

- **2025-01-20**: 初始版本，创建独立数据模块
  - 实现`dataset_config.py`（数据集配置）
  - 实现`feature_loader.py`（缓存加载）
  - 实现`feature_extractor.py`（特征提取）
  - 实现`data_provider.py`（统一接口）
  - 添加测试文件和使用示例

---

## 🔗 相关文档

- 主聚类算法: `ssddbc/ssddbc/`
- 损失函数: `ssddbc/evaluation/LOSS_FUNCTION.md`
- 参数指南: `ssddbc/testing/PARAMETERS_GUIDE.md`
