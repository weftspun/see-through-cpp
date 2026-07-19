#!/usr/bin/env bash
# Packages GGUF weights for a GitHub release: files at or under ~1.9GiB
# upload raw; larger ones get zstd-compressed and (if still too big) split
# into <2GiB parts, matching the convention documented in README.md
# ("files >2GB are zstd-compressed split parts").
#
# Usage: scripts/package_release.sh <src_dir> <out_dir> <file...>
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <src_dir> <out_dir> <file...>" >&2
    exit 1
fi

SRC="$1"; OUT="$2"; shift 2
mkdir -p "$OUT"

THRESHOLD=$((1900 * 1024 * 1024))
PART_SIZE="1900M"

filesize() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

for f in "$@"; do
    src="$SRC/$f"
    if [ ! -f "$src" ]; then
        echo "skip (missing): $f" >&2
        continue
    fi
    size=$(filesize "$src")
    if [ "$size" -le "$THRESHOLD" ]; then
        cp "$src" "$OUT/$f"
        echo "raw:        $f ($size bytes)"
    else
        echo "compressing: $f ($size bytes)..."
        zstd -9 -T0 -q -f -o "$OUT/$f.zst" "$src"
        zsize=$(filesize "$OUT/$f.zst")
        if [ "$zsize" -le "$THRESHOLD" ]; then
            mv "$OUT/$f.zst" "$OUT/$f.zst.part00"
            echo "  single part: $f.zst.part00 ($zsize bytes)"
        else
            split -b "$PART_SIZE" -d -a 2 "$OUT/$f.zst" "$OUT/$f.zst.part"
            rm "$OUT/$f.zst"
            n=$(ls "$OUT/$f".zst.part* | wc -l)
            echo "  split into $n parts"
        fi
    fi
done
