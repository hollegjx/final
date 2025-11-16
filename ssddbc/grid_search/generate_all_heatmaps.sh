#!/bin/bash

# 一次性生成所有已有超类的热力图脚本
# 创建时间: 2025-10-12
# 说明: 基于ssddbc模块生成热力图

echo "=================================================="
echo "批量生成热力图工具 - 基于已有搜索结果"
echo "扫描目录: /data/gjx/checkpoints/search/"
echo "输出目录: /data/gjx/checkpoints/heatmaps/"
echo "=================================================="

# 设置路径
SEARCH_DIR="/data/gjx/checkpoints/search"
HEATMAP_DIR="/data/gjx/checkpoints/heatmaps"
MODEL_PATH="placeholder"  # 占位符，不会实际使用

# 记录开始时间
start_time=$(date)
echo "开始时间: $start_time"
echo ""

# 创建日志文件
log_file="generate_heatmaps_$(date +%m_%d_%H_%M).log"
echo "日志文件: $log_file"

# 检查搜索结果目录是否存在
if [ ! -d "$SEARCH_DIR" ]; then
    echo "❌ 搜索结果目录不存在: $SEARCH_DIR"
    exit 1
fi

# 扫描所有有结果文件的超类
echo "🔍 扫描已有搜索结果..."
superclasses_with_results=()

for superclass_dir in "$SEARCH_DIR"/*; do
    if [ -d "$superclass_dir" ]; then
        superclass_name=$(basename "$superclass_dir")

        # 检查是否有txt文件
        txt_files=$(ls -1 "$superclass_dir"/*.txt 2>/dev/null | wc -l)
        if [ "$txt_files" -gt 0 ]; then
            latest_file=$(ls -t "$superclass_dir"/*.txt 2>/dev/null | head -1)
            file_size=$(stat -c%s "$latest_file" 2>/dev/null || stat -f%z "$latest_file" 2>/dev/null || echo "0")

            # 检查文件是否非空（大于1KB说明有实际内容）
            if [ "$file_size" -gt 1024 ]; then
                superclasses_with_results+=("$superclass_name")
                echo "  ✅ $superclass_name: $txt_files 个结果文件 (最新: $(basename "$latest_file"), ${file_size}字节)"
            else
                echo "  ⚠️  $superclass_name: 结果文件过小，可能为空"
            fi
        else
            echo "  ❌ $superclass_name: 无结果文件"
        fi
    fi
done

# 检查是否找到有效的超类
total_superclasses=${#superclasses_with_results[@]}
if [ "$total_superclasses" -eq 0 ]; then
    echo ""
    echo "❌ 未找到任何有效的搜索结果文件"
    echo "请先运行网格搜索：bash ssddbc/grid_search/run_grid_search_all.sh"
    exit 1
fi

echo ""
echo "📊 找到 $total_superclasses 个有搜索结果的超类:"
printf '%s\n' "${superclasses_with_results[@]}" | sed 's/^/  - /'
echo ""

# 询问用户确认
read -p "是否继续生成这些超类的热力图？[Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "用户取消操作"
    exit 0
fi

echo "🎨 开始批量生成热力图..."
echo "=================================================="

# 遍历所有有结果的超类
current_count=0
success_count=0
failed_superclasses=()

for superclass in "${superclasses_with_results[@]}"; do
    current_count=$((current_count + 1))

    echo ""
    echo "[$current_count/$total_superclasses] 生成热力图: $superclass"
    echo "----------------------------------------"

    # 记录当前超类开始时间
    superclass_start_time=$(date)
    echo "开始时间: $superclass_start_time"

    # 生成热力图（使用ssddbc模块，基于已有搜索结果）
    python -m ssddbc.grid_search.heatmap \
        --superclass_name "$superclass" \
        --model_path "$MODEL_PATH" \
        --use_existing_results true \
        --eval_version "v2" \
        --merge_clusters true \
        --k_min 3 \
        --k_max 21 \
        --dp_min 20 \
        --dp_max 100 \
        --dp_step 5 \
        2>&1 | tee -a "$log_file"

    # 检查运行结果
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        # 验证热力图文件是否生成
        superclass_heatmap_dir="$HEATMAP_DIR/$superclass"
        if [ -d "$superclass_heatmap_dir" ]; then
            heatmap_count=$(ls -1 "$superclass_heatmap_dir"/*.png 2>/dev/null | wc -l)
            if [ "$heatmap_count" -gt 0 ]; then
                echo "✅ $superclass: 热力图生成成功 ($heatmap_count 个文件)"
                success_count=$((success_count + 1))
            else
                echo "❌ $superclass: 热力图文件未找到"
                failed_superclasses+=("$superclass (文件未生成)")
            fi
        else
            echo "❌ $superclass: 输出目录未创建"
            failed_superclasses+=("$superclass (目录未创建)")
        fi
    else
        echo "❌ $superclass: 热力图生成失败 (退出码: $exit_code)"
        failed_superclasses+=("$superclass (进程失败)")
    fi

    # 记录当前超类结束时间
    superclass_end_time=$(date)
    echo "结束时间: $superclass_end_time"

    # 添加分隔符到日志
    echo "==============================" >> "$log_file"
    echo "超类 $superclass 完成: $superclass_end_time" >> "$log_file"
    echo "==============================" >> "$log_file"
    echo "" >> "$log_file"
done

# 记录总结束时间
end_time=$(date)
echo ""
echo "=================================================="
echo "批量热力图生成完成!"
echo "开始时间: $start_time"
echo "结束时间: $end_time"
echo "成功: $success_count/$total_superclasses"

if [ ${#failed_superclasses[@]} -gt 0 ]; then
    echo "失败的超类:"
    printf '%s\n' "${failed_superclasses[@]}" | sed 's/^/  - /'
fi

echo "热力图保存在: $HEATMAP_DIR"
echo "日志保存在: $log_file"
echo "=================================================="

# 统计最终结果
echo ""
echo "📊 最终结果统计:"
echo "超类名称           搜索结果    热力图"
echo "----------------------------------------"

for superclass in "${superclasses_with_results[@]}"; do
    # 统计搜索结果
    superclass_search_dir="$SEARCH_DIR/$superclass"
    search_file_count=$(ls -1 "$superclass_search_dir"/*.txt 2>/dev/null | wc -l)

    # 统计热力图
    superclass_heatmap_dir="$HEATMAP_DIR/$superclass"
    if [ -d "$superclass_heatmap_dir" ]; then
        heatmap_file_count=$(ls -1 "$superclass_heatmap_dir"/*.png 2>/dev/null | wc -l)
    else
        heatmap_file_count=0
    fi

    printf "%-18s %8s %10s\n" "$superclass" "$search_file_count" "$heatmap_file_count"
done

echo ""
echo "🎉 批量热力图生成任务完成!"

# 如果全部成功，给出使用建议
if [ "$success_count" -eq "$total_superclasses" ]; then
    echo ""
    echo "💡 所有热力图已生成，你可以:"
    echo "  1. 查看热力图: ls -la $HEATMAP_DIR/*/  "
    echo "  2. 打开具体图片查看参数优化结果"
    echo "  3. 根据热力图选择最佳参数组合"
    echo ""
    echo "注意: 现在每个指标会生成两张图:"
    echo "  - {metric}_heatmap_*.png: 显示ACC值"
    echo "  - {metric}_clusters_heatmap_*.png: 显示聚类数（用ACC着色）"
fi
