# digital-character

本地 Qwen3.6 对话 bot,带联网搜索(博查 API),面向智慧旅游 SaaS 数字人场景的原型。

## 架构

```
浏览器 (index.html, 文本+图片)
    │  SSE / OpenAI content blocks
    ▼
bot/server.py  (FastAPI, :8090)
    │
    ├──► mlx_vlm.server (:8080)  ─ 本地 Qwen3.6-35B-A3B-4bit (文本 + 视觉)
    │
    └──► api.bochaai.com/v1/web-search  (function calling 触发)
```

- **本地推理**:Mac Apple Silicon + MLX-VLM,模型常驻 ~21GB 统一内存
- **联网检索**:模型自主判断是否调用 `web_search` tool,最多 3 次/请求
- **多模态**:文本 + 图像理解可用;视频/音频接口由 mlx-vlm 提供,bot 前端暂未对接

## 快速开始

### 1. 下载模型

```bash
cd local-llm/
./download_direct.sh   # 或 download_ms.py / download_model.py
```

模型会落到 `local-llm/models/mlx-community/Qwen3___6-35B-A3B-4bit/`(~20GB,**不入 git**)。

### 2. 启动本地推理 server

```bash
cd local-llm/
./start_server.sh      # 默认 http://127.0.0.1:8080
```

可单独跑 `python test_client.py` 跑 5 项自检(list / chat / stream / multi-turn / function calling,带 tok/s 速度统计)。

### 3. 配置博查 API Key

去 [open.bochaai.com](https://open.bochaai.com) 注册拿 key,然后:

```bash
cd bot/
cp .env.example .env
# 编辑 .env,填入 BOCHA_API_KEY
```

### 4. 启动 bot

```bash
cd bot/
./start.sh             # 默认 http://127.0.0.1:8090
```

首次会自动装 `fastapi`、`uvicorn`、`python-dotenv` 到 `../local-llm/.venv`。
浏览器打开 http://127.0.0.1:8090 即可对话。

## 目录结构

```
.
├── bot/                          FastAPI + 单页 HTML 的聊天界面
│   ├── server.py                 SSE 流式接口,工具循环 + Bocha 集成
│   ├── index.html                聊天 UI,对话列表 + localStorage 持久化
│   ├── start.sh                  启动脚本(复用 local-llm 的 venv)
│   └── .env.example              环境变量模板
└── local-llm/                    本地 Qwen 推理
    ├── start_server.sh           mlx_lm.server 启动脚本
    ├── test_client.py            5 项 OpenAI 兼容 API 自检
    ├── download_*.py / *.sh      模型下载脚本(三种来源)
    └── rag_demo.py               早期 RAG 实验脚本
```

## Bot 已实现的功能

- **流式对话** — SSE 逐 token 推送,光标动画
- **图像输入** — 📎 按钮 / Cmd+V 粘贴截图 / 拖拽图片到窗口,支持多图;OpenAI content blocks 透传给 mlx-vlm,模型直接看图作答
- **多对话管理** — 左侧对话列表,新建/切换/删除,localStorage 持久化跨刷新(含图片消息)
- **角色预设** — 通用助手 / 导游 / 客服 / 代码 四个 system prompt 模板
- **联网搜索(博查)** — function calling 自主触发,带 freshness 时间过滤;气泡下方展示引用源
- **健康监控** — 右上角实时显示模型与博查可用性
- **性能指标** — 每条回答下显示 TTFT / tokens / decode 速度

## 性能参考(M-series Mac,Qwen3.6-35B-A3B-4bit,mlx-vlm runtime)

| 场景 | 速度 |
|---|---|
| 文本 decode(多轮)| ~50 tok/s |
| 单轮 TTFT(热)| ~200–500ms |
| 单轮 TTFT(联网,含 1 次 Bocha + 第二轮 prefill)| ~2–4s |
| 图像理解(64x64 → 简单回答)| 单次 ~5s,峰值内存 21GB |

> **Streaming 注意**:mlx-vlm 当前实现是"伪流式" — 中间发若干空 delta,最后一个 chunk 一次性塞入完整 content。逐 token 视觉效果丢失,但内容正常。前端可做"假流式重放"补救(TODO)。

## 路线图

- [x] ~~切到 `mlx_vlm.server` 启用图像理解~~ ✅
- [x] ~~Bot 前端添加图片上传(点击/粘贴/拖拽)~~ ✅
- [ ] 前端"假流式"重放,弥补 mlx-vlm 伪流式的视觉缺失
- [ ] PDF 文本提取(PyMuPDF)+ 扫描型 PDF 走 VL
- [ ] 历史污染应对:长对话拒答模式滚雪球时的提示/降级策略
- [ ] 关键词硬路由:`查/搜/股价/天气` 命中即强制 `tool_choice=web_search`,绕过模型 RLHF 反射
- [ ] 工具扩展:天气 / 票务 / 航班(智慧旅游场景特化)
- [ ] 上下文窗口管理(滑窗 + 历史摘要)
- [ ] 部署:把 bot + 模型打包成 SaaS 多租户后端

## License

私人项目,暂无授权。
