#!/bin/bash
# 启动本地 Qwen3.6 MLX-VLM OpenAI 兼容 server (支持文本 + 图像 + 视频)
# 用法: ./start_server.sh
set -e

cd "$(dirname "$0")"
source .venv/bin/activate

MODEL_PATH="$(pwd)/models/mlx-community/Qwen3.6-35B-A3B-4bit"

if [ ! -d "$MODEL_PATH" ]; then
  echo "模型不存在: $MODEL_PATH"
  exit 1
fi

# 如果端口被占用，先停掉旧 server
if lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[!] 端口 8080 已占用，先杀掉旧进程"
  pkill -f "mlx_lm.server" || true
  pkill -f "mlx_vlm.server" || true
  sleep 2
fi

# 写出 model id 到固定位置,供其他项目自动发现
# (避免它们硬编码或走 /v1/models 拿到的短 id 触发 reload)
DISCOVERY_FILE="/tmp/qwen-local-model-id"
echo "$MODEL_PATH" > "$DISCOVERY_FILE"

echo "[*] 启动 mlx_vlm.server (多模态) ..."
echo "    模型:    $MODEL_PATH"
echo "    监听:    http://127.0.0.1:8080"
echo "    发现:    $DISCOVERY_FILE  (其他项目读这个文件拿 model id)"
echo "    thinking 模式默认关闭 (适合 RAG/Agent)"
echo "    按 Ctrl-C 停止"
echo ""

export HF_HUB_OFFLINE=1
exec python -m mlx_vlm.server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8080 \
  --log-level INFO
