#!/usr/bin/env bash
# 启动 bot 后端 (复用 ../local-llm/.venv 里的 Python 环境)
set -e
cd "$(dirname "$0")"

VENV_PY="../local-llm/.venv/bin/python"

if ! "$VENV_PY" -c "import fastapi, uvicorn, dotenv, multipart, fitz" 2>/dev/null; then
  echo "[setup] 安装 fastapi + uvicorn + python-dotenv + python-multipart + pymupdf 到 ../local-llm/.venv"
  "$VENV_PY" -m pip install -q "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-dotenv>=1.0" "python-multipart>=0.0.9" "pymupdf>=1.24"
fi

HOST="${BOT_HOST:-127.0.0.1}"
PORT="${BOT_PORT:-8090}"
echo "[bot] http://${HOST}:${PORT}"
exec "$VENV_PY" -m uvicorn server:app --host "$HOST" --port "$PORT"
