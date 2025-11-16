#!/bin/bash

# 网格搜索脚本 - 顺序运行14个超类
# 创建时间: 2025-10-12
# 说明: 基于ssddbc模块进行网格搜索

echo "=================================================="
echo "开始网格搜索 - 所有14个超类（顺序执行）"

# 参数范围设置（可根据需要修改）
K_MIN=3
K_MAX=21
DP_MIN=20
DP_MAX=100
DP_STEP=5

# KNN_co 截止距离（如果不需要截止距离，设置为空或注释掉）
CO_VALUE=0.8

echo "参数范围: k=${K_MIN}-$((K_MAX-1)), density_percentile=${DP_MIN}-$((DP_MAX-DP_STEP)) (步长${DP_STEP})"
if [ -n "$CO_VALUE" ]; then
    echo "KNN_co截止距离: ${CO_VALUE}"
else
    echo "KNN_co截止距离: 不使用"
fi
k_count=$((K_MAX - K_MIN))
dp_count=$(((DP_MAX - DP_MIN) / DP_STEP))
total_combinations=$((k_count * dp_count))
echo "总参数组合数: ${k_count} × ${dp_count} = ${total_combinations}组/超类"
echo "预计总组合数: $((total_combinations * 14)) 组"
echo "执行方式: 顺序执行（非并行），避免CPU负载过高"
echo "=================================================="

# 定义14个超类名称（移除只有1个类的weather_phenomena）
superclasses=(
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
)

# 模型路径（使用确实存在的模型路径，特征提取后模型路径不影响聚类结果）
MODEL_PATH="/data/gjx/checkpoints/gcdsuperclass/vehicles/allacc_97_date_2025_9_22_12_30.pt"

# 评估版本
EVAL_VERSION="v2"

# 记录开始时间
start_time=$(date)
echo "开始时间: $start_time"
echo ""

# 创建日志文件
log_file="grid_search_all_$(date +%m_%d_%H_%M).log"
echo "日志文件: $log_file"

# 遍历所有超类
total_superclasses=${#superclasses[@]}
current_count=0

for superclass in "${superclasses[@]}"; do
    current_count=$((current_count + 1))

    echo "=================================================="
    echo "[$current_count/$total_superclasses] 开始处理超类: $superclass"
    echo "=================================================="

    # 记录当前超类开始时间
    superclass_start_time=$(date)
    echo "超类开始时间: $superclass_start_time"

    # 运行网格搜索（使用ssddbc模块）
    echo "使用可配置参数范围..."

    # 构建基础命令
    cmd="python -m ssddbc.grid_search.grid_search \
        --superclass_name \"$superclass\" \
        --model_path \"$MODEL_PATH\" \
        --eval_version \"$EVAL_VERSION\" \
        --merge_clusters true \
        --k_min \"$K_MIN\" \
        --k_max \"$K_MAX\" \
        --dp_min \"$DP_MIN\" \
        --dp_max \"$DP_MAX\" \
        --dp_step \"$DP_STEP\""

    # 如果设置了CO_VALUE，添加--co参数
    if [ -n "$CO_VALUE" ]; then
        cmd="$cmd --co \"$CO_VALUE\""
    fi

    # 执行命令
    eval "$cmd" 2>&1 | tee -a "$log_file"

    # 检查网格搜索结果
    if [ $? -eq 0 ]; then
        echo "✅ 超类 $superclass 网格搜索完成"
    else
        echo "❌ 超类 $superclass 网格搜索失败"
    fi

    # 记录当前超类结束时间
    superclass_end_time=$(date)
    echo "超类结束时间: $superclass_end_time"
    echo ""

    # 添加分隔符到日志
    echo "==============================" >> "$log_file"
    echo "超类 $superclass 完成: $superclass_end_time" >> "$log_file"
    echo "==============================" >> "$log_file"
    echo "" >> "$log_file"
done

# 记录总结束时间
end_time=$(date)
echo "=================================================="
echo "所有超类网格搜索完成!"
echo "开始时间: $start_time"
echo "结束时间: $end_time"
echo "搜索结果保存在: /data/gjx/checkpoints/search/"
echo "日志保存在: $log_file"
echo ""
echo "💡 要生成热力图，请运行: bash ssddbc/grid_search/generate_all_heatmaps.sh"
echo "=================================================="

# 统计搜索结果文件
echo ""
echo "搜索结果统计:"
search_dir="/data/gjx/checkpoints/search"

for superclass in "${superclasses[@]}"; do
    superclass_search_dir="$search_dir/$superclass"
    if [ -d "$superclass_search_dir" ]; then
        search_file_count=$(ls -1 "$superclass_search_dir"/*.txt 2>/dev/null | wc -l)
        echo "  $superclass: $search_file_count 个搜索结果文件"
    else
        echo "  $superclass: 目录不存在"
    fi
done
