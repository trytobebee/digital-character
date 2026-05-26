"""强制走 hf-mirror.com LFS 下载，禁用 Xet 协议。"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="mlx-community/Qwen3.6-35B-A3B-4bit",
    max_workers=4,
)
print(f"\n下载完成: {path}")
