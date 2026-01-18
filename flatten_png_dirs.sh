#!/usr/bin/env bash
set -euo pipefail

for d in */; do
  outer="${d%/}"
  inner="$d$outer"

  # 只处理 X/X 这种结构
  if [[ -d "$inner" ]]; then
    echo "Processing $outer"

    # 逐个移动 png（不会触发 argument list too long）
    find "$inner" -maxdepth 1 -type f -name '*.png' -print0 |
      while IFS= read -r -d '' f; do
        mv -n "$f" "$d"
      done

    # 如果 inner 目录空了，才删除
    rmdir "$inner" 2>/dev/null || true
  fi
done

echo "Done."

