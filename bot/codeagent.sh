#!/usr/bin/env bash
# 启动本地 Code Agent 终端(类 Claude Code,基于本地 Qwen)。
# 需先 ../local-llm/start_server.sh 把模型服务起在 :8080。
#
# 用法:
#   ./codeagent.sh                      # workdir = 当前目录
#   ./codeagent.sh --workdir ~/proj     # 指定项目目录
set -e
cd "$(dirname "$0")"
VENV_PY="../local-llm/.venv/bin/python"

if ! curl -s -m 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
  echo "[!] 本地模型服务 :8080 没在跑,请先 cd ../local-llm && ./start_server.sh"
  exit 1
fi

exec "$VENV_PY" -m code_agent "$@"
