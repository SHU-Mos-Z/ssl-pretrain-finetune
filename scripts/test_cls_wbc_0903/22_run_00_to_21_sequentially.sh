#!/bin/bash
# Run all WBC-0903 union experiments sequentially on one fixed GPU set and split.
# Individual failures are recorded and, by default, do not prevent later jobs.
set -uo pipefail

DEFAULT_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_DIR="${SCRIPT_DIR:-$DEFAULT_SCRIPT_DIR}"
SCRIPT_DIR=$(cd "$SCRIPT_DIR" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

# Change these defaults here (or export them before launch) to move the whole queue.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"

# 留空时保留每个子实验自身的 LR（其中包含 4e-4 对照）；非空时统一覆盖。
export LR="${LR:-}"
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-records/pretrain_conditioned/20260817_005610/ckpt_last.pth}"
export TRAIN_ROOT="${TRAIN_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455_finetune_train_p070_20260903}"
export VAL_ROOT="${VAL_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455_finetune_val_p015_20260903}"
export TEST_ROOT="${TEST_ROOT:-data/2018WBC_cellcrop_512x512_to_256x256_first50bands_multicandidate_manualoverride_filtered_minmax_20260903_1455_finetune_test_p015_20260903}"

for root in "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
    if [ ! -d "$root" ]; then
        echo "Missing WBC classification split: $root"
        exit 2
    fi
done

STOP_ON_ERROR="${STOP_ON_ERROR:-false}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-21}"

if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]] || [ "$START_INDEX" -gt 21 ]; then
    echo "START_INDEX must be an integer in [0,21], got: $START_INDEX"
    exit 2
fi
if ! [[ "$END_INDEX" =~ ^[0-9]+$ ]] || [ "$END_INDEX" -gt 21 ]; then
    echo "END_INDEX must be an integer in [0,21], got: $END_INDEX"
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
    "$SCRIPT_DIR/00_noaug_c1_standard_h0_lr8e4.sh"
    "$SCRIPT_DIR/01_dihedral_c1_standard_h0_lr8e4.sh"
    "$SCRIPT_DIR/02_dihedral_c2_standard_h0_lr8e4.sh"
    "$SCRIPT_DIR/03_dihedral_c4_standard_h0_lr8e4.sh"
    "$SCRIPT_DIR/04_dihedral_c2_balanced_h0_lr8e4.sh"
    "$SCRIPT_DIR/05_dihedral_c2_balanced_h1_lr8e4.sh"
    "$SCRIPT_DIR/06_dihedral_c2_balanced_h2_lr8e4.sh"
    "$SCRIPT_DIR/07_dihedral_c2_balanced_h3_lr8e4.sh"
    "$SCRIPT_DIR/08_perspective_c1_standard_h0_lr8e4.sh"
    "$SCRIPT_DIR/09_combined_c1_standard_h0_lr8e4.sh"
    "$SCRIPT_DIR/10_combined_c2_standard_h0_lr8e4.sh"
    "$SCRIPT_DIR/11_combined_c4_standard_h0_lr8e4.sh"
    "$SCRIPT_DIR/12_combined_c2_standard_h1_lr8e4.sh"
    "$SCRIPT_DIR/13_combined_c2_standard_h2_lr8e4.sh"
    "$SCRIPT_DIR/14_combined_c2_standard_h3_lr8e4.sh"
    "$SCRIPT_DIR/15_combined_c2_standard_h2_lr4e4.sh"
    "$SCRIPT_DIR/16_combined_c2_balanced_h2_lr8e4.sh"
    "$SCRIPT_DIR/17_dihedral_c2_balanced_h2_lr8e4_seed43.sh"
    "$SCRIPT_DIR/18_dihedral_c2_balanced_h2_lr8e4_seed44.sh"
    "$SCRIPT_DIR/19_combined_c2_standard_h2_lr8e4_seed43.sh"
    "$SCRIPT_DIR/20_combined_c2_standard_h2_lr8e4_seed44.sh"
    "$SCRIPT_DIR/21_combined_c2_standard_h2_lr8e4_seed45.sh"
)

for script_path in "${EXPERIMENT_SCRIPTS[@]}"; do
    if [ ! -f "$script_path" ]; then
        echo "Missing experiment script: $script_path"
        exit 2
    fi
done

QUEUE_TIME=$(date +%Y%m%d_%H%M%S)
QUEUE_ROOT="${QUEUE_ROOT:-records/test_cls_wbc_0903}"
QUEUE_DIR="$QUEUE_ROOT/sequential_${QUEUE_TIME}"
QUEUE_LOG="$QUEUE_DIR/runner.log"
SUMMARY_FILE="$QUEUE_DIR/summary.tsv"
mkdir -p "$QUEUE_DIR"
printf 'index\tscript\tstart_time\tend_time\texit_code\tstatus\n' > "$SUMMARY_FILE"

{
    echo "============================================================================"
    echo "WBC-0903 sequential experiment runner"
    echo "Project root:             $PROJECT_ROOT"
    echo "Train root:               $TRAIN_ROOT"
    echo "Val root:                 $VAL_ROOT"
    echo "Test root:                $TEST_ROOT"
    echo "Physical GPUs:            $CUDA_VISIBLE_DEVICES"
    echo "Distributed processes:    $NUM_GPUS"
    echo "Batch size per GPU:       $BATCH_SIZE_PER_GPU"
    echo "Learning rate override:  ${LR:-<per-script default>}"
    echo "Pretrained checkpoint:   $PRETRAIN_CKPT"
    echo "Experiment range:         $START_INDEX..$END_INDEX"
    echo "Continue after failure:   $([ "$STOP_ON_ERROR" = "false" ] && echo true || echo false)"
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
        echo "[$index/21] START $script_name"
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
        echo "[$index/21] END $script_name"
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
