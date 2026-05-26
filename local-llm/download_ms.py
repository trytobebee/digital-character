"""从 ModelScope (阿里魔搭) 下载 mlx-community/Qwen3.6-35B-A3B-4bit。
ModelScope 国内可达，速度通常 10-30 MB/s。
"""
from modelscope import snapshot_download

target = "/Users/taifeng/code/digital_character/local-llm/models"
path = snapshot_download(
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    cache_dir=target,
)
print(f"\n模型路径: {path}")
