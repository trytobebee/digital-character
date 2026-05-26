#!/bin/bash
# 从已有 HF cache 拷贝 metadata + 直接 curl 4 个 safetensors（不跟随重定向）
set -e

REPO="mlx-community/Qwen3.6-35B-A3B-4bit"
MIRROR="https://hf-mirror.com"
TARGET_DIR="${HOME}/code/digital_character/local-llm/models/Qwen3.6-35B-A3B-4bit"
CACHE_SNAPSHOT="${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# 1. 从 cache 把 metadata 拷过来（沿符号链接 -L）
SNAP=$(ls "$CACHE_SNAPSHOT" 2>/dev/null | head -1)
if [ -n "$SNAP" ]; then
  echo "=== 复制已有 metadata: $SNAP ==="
  for f in "$CACHE_SNAPSHOT/$SNAP"/*; do
    name=$(basename "$f")
    if [ ! -f "$name" ]; then
      cp -L "$f" "$name" && echo "  [copy] $name ($(du -h "$name" | awk '{print $1}'))"
    fi
  done
fi

# 2. 直接下载 4 个 safetensors（hf-mirror 直接 serve，不重定向）
SHARDS=(
  "model-00001-of-00004.safetensors"
  "model-00002-of-00004.safetensors"
  "model-00003-of-00004.safetensors"
  "model-00004-of-00004.safetensors"
)

echo ""
echo "=== 下载 safetensors shards ==="
for f in "${SHARDS[@]}"; do
  if [ -f "$f" ] && [ -s "$f" ]; then
    SZ=$(du -h "$f" | awk '{print $1}')
    echo "[skip] $f ($SZ 已存在)"
    continue
  fi
  echo "[get ] $f"
  # 不用 -L，直连 hf-mirror（hf-mirror 直接 serve LFS）
  curl --fail --retry 5 --retry-delay 5 --continue-at - \
    -o "$f" "${MIRROR}/${REPO}/resolve/main/${f}"
done

echo ""
echo "=== 完成 ==="
ls -lh "$TARGET_DIR"
echo "总大小: $(du -sh "$TARGET_DIR" | awk '{print $1}')"
