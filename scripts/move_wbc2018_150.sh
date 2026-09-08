#!/usr/bin/env bash
# 将 detection images 按文件名首字母复制到 B/E/L/M/N 五个目录
set -euo pipefail

SRC="/home/zsq/214DataA/zsq/DFS3R-main/data/2018WBC_detection_scene_1300x1800_noresize_contiguous20_b28to47_refcropminmax_20260902_1725/images"
DEST_ROOT="${HOME}/150t/personal_data/zsq/2018WBC_detection_scene_1300x1800_noresize_contiguous20_b28to47_refcropminmax_20260902_1725"

LETTERS=(B E L M N)

if [[ ! -d "$SRC" ]]; then
  echo "错误: 源目录不存在: $SRC" >&2
  exit 1
fi

for letter in "${LETTERS[@]}"; do
  mkdir -p "${DEST_ROOT}/${letter}"
done

echo "源目录: $SRC"
echo "目标根: $DEST_ROOT"
echo "开始复制..."

copied_total=0
unknown=0

shopt -s nullglob
for src_file in "$SRC"/*; do
  [[ -f "$src_file" ]] || continue
  fname="$(basename "$src_file")"
  first="${fname:0:1}"
  letter="${first^^}"   # 小写 l -> L

  dest_dir="${DEST_ROOT}/${letter}"
  if [[ ! -d "$dest_dir" ]]; then
    echo "警告: 未知首字母 '${first}'，跳过: $fname" >&2
    unknown=$((unknown + 1))
    continue
  fi

  cp -n -- "$src_file" "${dest_dir}/"
  copied_total=$((copied_total + 1))
  if (( copied_total % 20 == 0 )); then
    echo "  已复制 ${copied_total} 个文件..."
  fi
done

echo
echo "========== 复制完成 =========="
echo "源文件数: $(find "$SRC" -maxdepth 1 -type f | wc -l)"
echo "本次复制: ${copied_total}"
echo "未知首字母跳过: ${unknown}"
echo
echo "各目录文件数:"
for letter in "${LETTERS[@]}"; do
  n=$(find "${DEST_ROOT}/${letter}" -maxdepth 1 -type f | wc -l)
  printf "  %s: %s\n" "$letter" "$n"
done