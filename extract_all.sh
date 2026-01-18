#!/usr/bin/env bash
set -euo pipefail

# 在包含这些 .tar.bz2 的目录里运行
shopt -s nullglob

for tarball in ./*.tar.bz2; do
  base="$(basename "$tarball" .tar.bz2)"
  outdir="./$base"

  echo "==> Extracting: $tarball  ->  $outdir"

  # 如果目标文件夹不存在就创建；存在就继续往里解（你也可以改成跳过）
  mkdir -p "$outdir"

  # 解压（-j = bzip2, -f = file, -C = 输出目录）
  # 注意：tar 失败会直接退出脚本（set -e），因此不会删包
  tar -xjf "$tarball" -C "$outdir"

  # 解压成功才删除
  rm -f "$tarball"
  echo "    Done. Removed archive: $tarball"
done

echo "All done."

