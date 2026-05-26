"""
最小 RAG demo：本地 bge-m3 做 embedding + 本地 Qwen3.6 (MLX) 通过 OpenAI 兼容 API 生成回答。
运行前先启动 server：
    mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080
然后：
    python rag_demo.py
"""

import os
# 用本地缓存即可，不要联网 huggingface.co（被墙时会报 RuntimeError）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------- 1. 示例知识库（智慧旅游场景）----------
DOCS = [
    "黄山景区开放时间为每日 6:30-17:30，旺季（4-11月）门票 190 元，淡季 150 元。65 岁以上老人免票。",
    "黄山三大主峰：莲花峰（1864m，景区最高峰）、光明顶（1860m，看日出最佳点）、天都峰（1810m，登顶险峻）。",
    "黄山索道有云谷、玉屏、太平、西海大峡谷四条，单程票价均为 80-100 元，建议旺季提前线上预约。",
    "推荐两日游路线：D1 云谷索道上 → 始信峰 → 北海宾馆住宿；D2 看日出 → 西海大峡谷 → 玉屏索道下。",
    "景区内住宿紧张，旺季需提前一个月预订。北海宾馆、白云宾馆是山上主要酒店，价格 800-1500 元/晚。",
    "黄山天气多变，山上常年比山下低 8-10℃，建议带防风外套、雨衣、保暖衣物。",
    "智慧旅游平台已接入景区实时人流监控、电子门票、AI 导览、紧急救援呼叫等功能。",
]

# ---------- 2. 加载本地 bge-m3 做 embedding ----------
print("[1/4] 加载 bge-m3 embedding 模型...")
embedder = SentenceTransformer("BAAI/bge-m3")

print("[2/4] 编码知识库...")
doc_embs = embedder.encode(DOCS, normalize_embeddings=True)
doc_embs = np.array(doc_embs)


def retrieve(query: str, k: int = 3):
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    scores = doc_embs @ q_emb
    top_idx = np.argsort(-scores)[:k]
    return [(DOCS[i], float(scores[i])) for i in top_idx]


# ---------- 3. 用 OpenAI 兼容客户端调本地 mlx_lm.server ----------
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
# server 端 --model 参数填什么这里就填什么；mlx_lm.server 不强校验模型名
MODEL_NAME = os.environ.get("MLX_MODEL_NAME", "qwen3.6-35b-a3b")

SYSTEM_PROMPT = (
    "你是黄山智慧旅游助手。严格基于提供的【参考资料】回答用户问题；"
    "若资料中没有答案，明确说『暂无相关信息』，不要编造。"
)


def ask(question: str):
    hits = retrieve(question, k=3)
    context = "\n".join(f"[{i+1}] {doc}" for i, (doc, _) in enumerate(hits))
    user_msg = f"【参考资料】\n{context}\n\n【问题】{question}"

    print(f"\n--- 问题 ---\n{question}")
    print(f"\n--- 检索到的 top-3 ---")
    for i, (doc, score) in enumerate(hits):
        print(f"  [{i+1}] (sim={score:.3f}) {doc}")

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    print(f"\n--- 回答 ---\n{resp.choices[0].message.content}\n")


# ---------- 4. 演示几个问题 ----------
print("[3/4] 启动问答...")
ask("黄山看日出去哪个峰最好？大概什么海拔？")
ask("我七月份去黄山，门票多少钱？需要带什么衣服？")
ask("智慧旅游平台具体提供哪些功能？")
ask("黄山有什么好吃的本地特色菜？")  # 故意问知识库外的问题，看模型是否会说"暂无信息"
print("[4/4] 完成。")
