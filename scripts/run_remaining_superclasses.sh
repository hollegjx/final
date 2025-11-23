#!/usr/bin/env bash
# 临时脚本：运行剩余的若干超类（排除已完成的 flowers, fruits_vegetables, furniture, humans, insects_arthropods, mammals, marine_animals, reptiles, trees）
# 用于从中断的训练恢复
# 用法示例：
#   bash scripts/run_remaining_superclasses.sh \
#     --stage1_epochs 10 --update_interval 5 --total_epochs 50 \
#     --batch_size 128 --num_workers 16 --prop_train_labels 0.8 \
#     --lr 0.03 --grad_from_block 9 --sup_con_weight 0.25 \
#     --gpu 0 --pseudo_weight_mode none \
#     --runs_root /home/jz/temp/gjx/exp/final/runs_pipeline \
#     --pseudo_loss_weight 0.25

set -uo pipefail

# 剩余超类（已排除: flowers, fruits_vegetables, furniture, humans, insects_arthropods, mammals, marine_animals, reptiles, trees）
SUPERCLASSES=(
  containers
  vehicles
  electronic_devices
  buildings
  terrain
  weather_phenomena
)

if [[ "${#@}" -eq 0 ]]; then
    echo "用法: $0 [通用参数，会全部传给 scripts/pseudo_pipeline.py]" >&2
    echo "" >&2
    echo "⚠️  临时脚本：排除已完成的 flowers, fruits_vegetables, mammals, trees" >&2
    exit 1
fi

success=()
failed=()

echo "🔄 恢复训练：运行剩余 ${#SUPERCLASSES[@]} 个超类"
echo "   已排除: flowers, fruits_vegetables, furniture, humans, insects_arthropods, mammals, marine_animals, reptiles, trees"
echo ""

for sc in "${SUPERCLASSES[@]}"; do
    echo "=== 开始超类: ${sc} ==="
    cmd=(python3 scripts/pseudo_pipeline.py --superclass_name "${sc}")
    cmd+=("$@")
    echo "CMD: ${cmd[*]}"
    if "${cmd[@]}"; then
        echo "✅ 完成超类: ${sc}"
        success+=("${sc}")
    else
        echo "❌ 失败超类: ${sc}（继续下一个）"
        failed+=("${sc}")
    fi
done

echo ""
echo "🎉 批处理完成"
echo "   成功: ${#success[@]} -> ${success[*]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "   失败: ${#failed[@]} -> ${failed[*]}"
fi
echo ""
echo "📝 提醒: 这是临时脚本，完整的超类列表请使用 run_all_superclasses.sh"
