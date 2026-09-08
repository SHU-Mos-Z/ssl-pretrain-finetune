#!/bin/bash
# This file is meant to be sourced by training launch scripts.

# Convert one materialized dataset root into a compact, filesystem-safe tag.
# Examples:
#   MDC_..._to_256x256_..._finetune_train_p049_20260816 -> MDC-256x256
#   2018WBC_..._to_256x256_..._bands50_...             -> 2018WBC-256x256-b50
#   LUAD_.../Training_pretrain_p030_...                 -> LUAD-256x256-Training
dataset_info_from_root() {
    local root="${1%/}"
    local leaf="${root##*/}"
    local core="$leaf"
    local subset=""
    local parent=""
    local dataset=""
    local spatial_size=""
    local bands=""
    local tag=""

    core=$(printf '%s' "$core" | sed -E \
        's/_(pretrain|finetune_(train|val|test))_p[0-9]{3}_[0-9]{8}$//')

    # Some roots end in .../Training_pretrain_...; retain the parent dataset name.
    case "$core" in
        Training|Validation|Testing)
            subset="$core"
            parent=$(basename "$(dirname "$root")")
            parent=$(printf '%s' "$parent" | sed -E \
                's/_(pretrain|finetune_(train|val|test))_p[0-9]{3}_[0-9]{8}$//')
            core="${parent}_${core}"
            ;;
    esac

    dataset="${core%%_*}"
    if [[ "$core" =~ to_([0-9]+)(x|_)([0-9]+) ]]; then
        spatial_size="${BASH_REMATCH[1]}x${BASH_REMATCH[3]}"
    elif [[ "$core" =~ patch_([0-9]+)(x|_)([0-9]+) ]]; then
        spatial_size="${BASH_REMATCH[1]}x${BASH_REMATCH[3]}"
    fi
    if [[ "$core" =~ bands([0-9]+) ]]; then
        bands="${BASH_REMATCH[1]}"
    fi

    tag="$dataset"
    if [ -n "$spatial_size" ]; then tag+="-${spatial_size}"; fi
    if [ -n "$bands" ]; then tag+="-b${bands}"; fi
    if [ -n "$subset" ]; then tag+="-${subset}"; fi
    printf '%s' "$tag" | sed -E 's/[^A-Za-z0-9._+-]+/-/g'
}


# Build one stable tag from any number of roots while preserving input order and
# removing duplicates (normally train/val/test reduce to one dataset tag).
dataset_info_from_roots() {
    local root=""
    local tag=""
    local joined=""
    local seen="+"
    for root in "$@"; do
        tag=$(dataset_info_from_root "$root")
        case "$seen" in
            *"+${tag}+"*) continue ;;
        esac
        seen+="${tag}+"
        if [ -n "$joined" ]; then joined+="+"; fi
        joined+="$tag"
    done
    if [ -z "$joined" ]; then
        joined="unknown-dataset"
    fi
    printf '%s' "$joined"
}
