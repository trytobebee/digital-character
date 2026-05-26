#!/bin/bash
cd /Users/taifeng/code/digital_character/local-llm
source .venv/bin/activate
MODEL=$(find models -type d -name "Qwen3.6-35B-A3B-4bit" | head -1)
echo "model path: $MODEL"
echo ""
HF_HUB_OFFLINE=1 mlx_lm.generate \
  --model "$MODEL" \
  --prompt "用一句话介绍一下黄山。" \
  --max-tokens 200 \
  --temp 0.3
