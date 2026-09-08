#!/bin/bash
# Run WBC-0830 experiments 00 through 13 sequentially on physical GPUs 4 and 5.
# Individual failures are recorded and, by default, do not prevent later jobs.
set -uo pipefail

DEFAULT_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_SCRIPT_DIR}"
SCRIPT_DIR=$(cd "$SCRIPT_DIR" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
# 留空时保留每个子实验自身的 LR（该组包含专门的 LR 消融）；设为非空值时
# 统一覆盖全部子实验。
export LR="${LR:-}"
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
export TRAIN_ROOT="${TRAIN_ROOT:-data/2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_multicandidate_manualoverride_filtered_minmax_20260830_finetune_train_p049_20260901}"
export VAL_ROOT="${VAL_ROOT:-data/2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_multicandidate_manualoverride_filtered_minmax_20260830_finetune_val_p011_20260901}"
export TEST_ROOT="${TEST_ROOT:-data/2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_multicandidate_manualoverride_filtered_minmax_20260830_finetune_test_p011_20260901}"

# Stage-dependent scripts use these defaults unless the caller exports different
# values before launching this runner. These values match the current planned
# combined-augmentation, two-copy, H2 final candidate.
export SELECTED_AUGMENTATION_POLICY="${SELECTED_AUGMENTATION_POLICY:-dihedral_perspective}"
export SELECTED_PERSPECTIVE_PROBABILITY="${SELECTED_PERSPECTIVE_PROBABILITY:-0.5}"
export SELECTED_PERSPECTIVE_SCALE="${SELECTED_PERSPECTIVE_SCALE:-0.05}"
export SELECTED_PERSPECTIVE_PADDING_MODE="${SELECTED_PERSPECTIVE_PADDING_MODE:-reflection}"
export SELECTED_AUGMENTATION_COPIES="${SELECTED_AUGMENTATION_COPIES:-2}"
export SELECTED_CLASSIFICATION_HEAD="${SELECTED_CLASSIFICATION_HEAD:-h2_dual_scale}"
export SELECTED_BALANCE_MODE="${SELECTED_BALANCE_MODE:-none}"
export SELECTED_LR="${SELECTED_LR:-8e-4}"

# Set STOP_ON_ERROR=true to terminate at the first failed experiment. The default
# is more suitable for unattended execution: record the failure and continue.
STOP_ON_ERROR="${STOP_ON_ERROR:-false}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-13}"

if ! [[ "$START_INDEX" =~ ^([0-9]|1[0-3])$ ]]; then
    echo "START_INDEX must be an integer in [0,13], got: $START_INDEX"
    exit 2
fi
if ! [[ "$END_INDEX" =~ ^([0-9]|1[0-3])$ ]]; then
    echo "END_INDEX must be an integer in [0,13], got: $END_INDEX"
    exit 2
fi
if [ "$START_INDEX" -gt "$END_INDEX" ]; then
    echo "START_INDEX must not exceed END_INDEX."
    exit 2
fi
case "$STOP_ON_ERROR" in
    true|false) ;;
    *)
        echo "STOP_ON_ERROR must be true or false, got: $STOP_ON_ERROR"
        exit 2
        ;;
esac

EXPERIMENT_SCRIPTS=(
    "$SCRIPT_DIR/00_w2_a0_cellcrop_noaug_h0.sh"
    "$SCRIPT_DIR/01_w2_a1_dihedral_c1_h0.sh"
    "$SCRIPT_DIR/02_w2_a2_perspective_c1_h0.sh"
    "$SCRIPT_DIR/03_w2_a3_combined_c1_h0.sh"
    "$SCRIPT_DIR/04_w2_c2_bestpolicy_c2_h0.sh"
    "$SCRIPT_DIR/05_w2_c4_bestpolicy_c4_h0.sh"
    "$SCRIPT_DIR/06_w2_h1_bestaug.sh"
    "$SCRIPT_DIR/07_w2_h2_bestaug.sh"
    "$SCRIPT_DIR/08_w2_h3_bestaug.sh"
    "$SCRIPT_DIR/09_w2_lower_lr.sh"
    "$SCRIPT_DIR/10_w2_balanced_sampling.sh"
    "$SCRIPT_DIR/11_w2_final_seed43.sh"
    "$SCRIPT_DIR/12_w2_final_seed44.sh"
    "$SCRIPT_DIR/13_w2_final_seed45.sh"
)

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    if [ ! -f "$script_path" ]; then
        echo "Missing experiment script: $script_path"
        exit 2
    fi
done

QUEUE_TIME=$(date +%Y%m%d_%H%M%S)
QUEUE_ROOT="${QUEUE_ROOT:-records/test_cls_wbc_0830}"
QUEUE_DIR="$QUEUE_ROOT/sequential_${QUEUE_TIME}"
QUEUE_LOG="$QUEUE_DIR/runner.log"
SUMMARY_FILE="$QUEUE_DIR/summary.tsv"
mkdir -p "$QUEUE_DIR"
printf 'index\tscript\tstart_time\tend_time\texit_code\tstatus\n' > "$SUMMARY_FILE"

{
    echo "============================================================================"
    echo "WBC-0830 sequential experiment runner"
    echo "Project root:             $PROJECT_ROOT"
    echo "Physical GPUs:            $CUDA_VISIBLE_DEVICES"
    echo "Distributed processes:    $NUM_GPUS"
    echo "Batch size per GPU:       $BATCH_SIZE_PER_GPU"
    echo "Learning rate override:  ${LR:-<per-script default>}"
    echo "Pretrained checkpoint:   $PRETRAIN_CKPT"
    echo "Train root:              $TRAIN_ROOT"
    echo "Val root:                $VAL_ROOT"
    echo "Test root:               $TEST_ROOT"
    echo "Experiment range:         $START_INDEX..$END_INDEX"
    echo "Continue after failure:   $([ "$STOP_ON_ERROR" = "false" ] && echo true || echo false)"
    echo "Selected policy/copies:   $SELECTED_AUGMENTATION_POLICY / $SELECTED_AUGMENTATION_COPIES"
    echo "Selected head:            $SELECTED_CLASSIFICATION_HEAD"
    echo "Selected balance/LR:      $SELECTED_BALANCE_MODE / $SELECTED_LR"
    echo "Queue record directory:   $QUEUE_DIR"
    echo "Started at:               $(date --iso-8601=seconds)"
    echo "============================================================================"
} | tee "$QUEUE_LOG"

failure_count=0
completed_count=0

for index in "${!EXPERIMENT_SCRIPTS[@]}"; do
    if [ "$index" -lt "$START_INDEX" ] || [ "$index" -gt "$END_INDEX" ]; then
        continue
    fi

    script_path="${EXPERIMENT_SCRIPTS[$index]}"
    script_name=$(basename "$script_path")
    start_time=$(date --iso-8601=seconds)

    {
        echo
        echo "============================================================================"
        echo "[$index/13] START $script_name"
        echo "Start time: $start_time"
        echo "============================================================================"
    } | tee -a "$QUEUE_LOG"

    bash "$script_path" 2>&1 | tee -a "$QUEUE_LOG"
    exit_code=${PIPESTATUS[0]}
    end_time=$(date --iso-8601=seconds)
    completed_count=$((completed_count + 1))

    if [ "$exit_code" -eq 0 ]; then
        status="success"
    else
        status="failed"
        failure_count=$((failure_count + 1))
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$index" "$script_name" "$start_time" "$end_time" "$exit_code" "$status" \
        >> "$SUMMARY_FILE"

    {
        echo "----------------------------------------------------------------------------"
        echo "[$index/13] END $script_name"
        echo "End time:  $end_time"
        echo "Exit code: $exit_code ($status)"
        echo "----------------------------------------------------------------------------"
    } | tee -a "$QUEUE_LOG"

    if [ "$exit_code" -ne 0 ] && [ "$STOP_ON_ERROR" = "true" ]; then
        echo "STOP_ON_ERROR=true: aborting the remaining queue." | tee -a "$QUEUE_LOG"
        break
    fi
done

{
    echo
    echo "============================================================================"
    echo "Sequential queue finished at: $(date --iso-8601=seconds)"
    echo "Experiments attempted:         $completed_count"
    echo "Failed experiments:            $failure_count"
    echo "Summary:                       $SUMMARY_FILE"
    echo "============================================================================"
} | tee -a "$QUEUE_LOG"

if [ "$failure_count" -ne 0 ]; then
    exit 1
fi
