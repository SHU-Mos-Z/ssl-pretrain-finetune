#!/usr/bin/env bash
# Reduced PUAD background-aware pipeline.
#
# Training patches:
#   foreground = 2071 (class 1/2/3 -> 691/690/690)
#   background = 1035
# Validation patches are reduced by the same approximately 0.66 factor:
#   foreground = 345 (115/115/115)
#   background = 172
# Complete val_scenes and test scenes remain unchanged.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DATASET_NAME="${DATASET_NAME:-LUAD_PUAD_official224_bgaware_fg2071_bg1035_fullsceneval}"
export FOREGROUND_TRAIN_PATCHES="${FOREGROUND_TRAIN_PATCHES:-2071}"
export BACKGROUND_TRAIN_PATCHES="${BACKGROUND_TRAIN_PATCHES:-1035}"
export FOREGROUND_VAL_PATCHES="${FOREGROUND_VAL_PATCHES:-345}"
export BACKGROUND_VAL_PATCHES="${BACKGROUND_VAL_PATCHES:-172}"

exec bash "${SCRIPT_DIR}/prepare_puad_bgaware_nmf_and_upload.sh"
