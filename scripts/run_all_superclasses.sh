#!/usr/bin/env bash
# 串行运行 CIFAR-100 全部 15 个超类的 three-stage pipeline。
# 所有传入参数都会原样转发给 scripts/pseudo_pipeline.py。
# 用法示例：
#   bash scripts/run_all_superclasses.sh \
#     --stage1_epochs 10 --update_interval 5 --total_epochs 50 \
#     --batch_size 128 --num_workers 16 --prop_train_labels 0.8 \
#     --lr 0.03 --grad_from_block 9 --sup_con_weight 0.25 \
#     --gpu 0 --pseudo_weight_mode none \
#     --runs_root /home/jz/temp/gjx/exp/final/runs_pipeline \
#     --pseudo_loss_weight 0.25
#
# 说明：
# - 不需要指定 --superclass_name，脚本会依次跑完全部超类。
# - 每个超类的 run_dir 在 runs_root/<superclass>/<timestamp>/ 下互不干扰。
# - 若某个超类失败，继续下一个并在末尾汇总（调试友好，不阻塞批处理）。

set -uo pipefail

# 15 个超类列表（与 data/cifar100_superclass.py 的 SUPERCLASS_NAMES 保持一致）
SUPERCLASSES=(
  trees
  flowers
  fruits_vegetables
  mammals
  marine_animals
  insects_arthropods
  reptiles
  humans
  furniture
  containers
  vehicles
  electronic_devices
  buildings
  terrain
  weather_phenomena
)

if [[ "${#@}" -eq 0 ]]; then
    echo "用法: $0 [通用参数，会全部传给 scripts/pseudo_pipeline.py]" >&2
    exit 1
fi

success=()
failed=()

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

echo "🎉 批处理完成"
echo "   成功: ${#success[@]} -> ${success[*]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "   失败: ${#failed[@]} -> ${failed[*]}"
fi
