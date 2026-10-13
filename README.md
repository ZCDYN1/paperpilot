# PaperPilot

> 基于 **LangGraph 多 Agent** 的 arXiv 论文深度研究工具。
>
> 输入一个研究问题 → 自动拆解 → 并行检索 arXiv → 摘要级抽取 → **逐条校验引用出处** →
> 输出**中文、带可溯源引用**的文献综述 + 结构化对比表。

它解决的不是「读不懂论文」，而是**「不知道哪 5 篇值得读」**和**「读完记不住、引用还引错」**。
最关键的设计是：**每一条结论都必须能点回原文片段，找不到支撑的论断会被系统自己标出来并剔除**
—— 因为在学术场景里，编出来的引用比"我不知道"更有害。

---

## 特性

- **混合检索**：FAISS 向量 + BM25 词面双路召回，RRF 倒数秩融合，BGE 交叉编码器重排
  （消融实验会量化每一档的效果 / 耗时权衡）
- **多 Agent 编排**：Manager–Worker 结构，子问题并行检索，引用校验 Agent 做自我反思
- **引用可溯源**：每条引用落在具体 chunk 上，点开可定位到章节 / 原文片段
- **主动暴露不确定性**：报告中单列「未找到原文支撑的结论」，不假装自己都对
- **实时过程可见**：SSE 流式推送节点 / 工具调用 / 召回片段，等待期间不是黑盒
- **状态可续跑**：LangGraph SqliteSaver 持久化，中断后可恢复，不重复烧 token
- **工程化**：tenacity 重试 / 超时 / 熔断、工具入参强校验、Prompt 注入防护、Token 用量统计
- **可观测与评测**：LangSmith 全链路追踪 + RAGAS 自动评测（均为可选，不配也能跑）
- **一键部署**：Docker Compose 多服务编排，数据持久化到 volume

---

## 快速开始

### 方式一：Docker（推荐）

```bash
cp .env.example .env        # 按需修改；不建也能起（走 mock 模式跑通）
docker compose up --build
```

- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/healthz
- 交互面板：http://localhost:8501

### 方式二：本地运行

需要 **Python 3.12**。

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -e ".[dev]"          # 核心依赖
pip install -e ".[local]"        # 可选：本地 BGE（含 torch，较大）

cp .env.example .env             # 然后编辑 .env
uvicorn app.main:app --reload    # 起后端 :8000
streamlit run ui/streamlit_app.py  # 另开一个终端起面板 :8501
```

---

## 最小配置

只想先跑起来看看，改这两行就够：

```dotenv
LLM_PROVIDER=openai                      # 或 ollama / mock
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-xxxx
LLM_MODEL=deepseek-chat
```

**完全不想配任何 key？** 把两个 provider 都设成 `mock`，全链路可离线跑通：

```dotenv
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=mock
```

> ⚠️ **合规提示**：arXiv 要求请求头带声明式 User-Agent 且包含联系方式，
> 请把 `.env` 里 `ARXIV_USER_AGENT` 的邮箱换成你自己的。

embedding 有三种模式，按机器条件选：

| `EMBEDDING_PROVIDER` | 说明 |
|---|---|
| `local`（默认） | 本地 BGE 模型，检索质量最好；需 `pip install -e ".[local]"`，首次自动下载权重（约 2~3GB） |
| `api` | 走 OpenAI 兼容的 `/v1/embeddings`，无需下模型 |
| `mock` | 确定性伪向量，仅用于跑通流程与演示；**不含语义**，不要用它评估检索效果 |

**默认走 API，开箱即用** —— 不占磁盘、不依赖本机算力。LLM 与 embedding 是**互相独立**的配置，
可以来自不同厂商（例如 LLM 用 DeepSeek，embedding 用阿里云百炼）：

```dotenv
# LLM
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-xxxx
LLM_MODEL=deepseek-chat

# Embedding（需选择提供 embeddings 接口的服务商）
EMBEDDING_PROVIDER=api
EMBEDDING_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=sk-xxxx
EMBEDDING_API_MODEL=text-embedding-v3
EMBEDDING_DIM=1024
```

> ⚠️ **更换 embedding 模型后必须重建索引**（不同模型的向量空间不通用）：
> `python scripts/build_index.py`

---

## 评测结果

检索策略消融（架构文档 §5.3 那张表）。**40 条人工标注查询 × 24 篇真实 arXiv 论文（1210 chunk）**，
同一份索引、同一批查询，**只变检索配置**。embedding 为**本地 BGE-M3**（CPU，1024 维）。

| 配置 | Recall@5 | Precision@5 | nDCG@10 | MRR | 耗时 / 查询 |
|---|---|---|---|---|---|
| ① 纯向量（BGE-M3） | 0.8542 | 0.2100 | 0.8418 | 0.8833 | 125.5 ms |
| ② 纯 BM25 | 0.7333 | 0.1850 | 0.7053 | 0.7438 | **1.2 ms** |
| ③ 向量 + BM25 加权求和 | 0.9042 | 0.2200 | 0.8417 | 0.8675 | 123.8 ms |
| ④ 向量 + BM25 + **RRF** | **0.9250** | **0.2300** | **0.8777** | **0.9062** | 124.0 ms |
| ⑤ ④ + Reranker | 0.9250 | 0.2300 | 0.7851 | 0.7612 | 123.8 ms |
| ⑥ ⑤ + metadata 过滤 | 0.9250 | 0.2300 | 0.8436 | 0.8354 | 125.4 ms |

三条结论都有数据支撑，不是断言：

1. **RRF 确实优于加权求和**：④ 相对 ③ 的 nDCG@10 高 **0.0361**、Recall@5 高 **0.0208**。
   加权求和必须先归一化（BM25 分数无上界、余弦相似度落在 [-1,1]），而归一化本身引入分布假设 ——
   实测印证了这个设计取舍。
2. **纯词面召回不够用，但便宜得惊人**：② 的 Recall@5 比 ① 低 0.1208、nDCG@10 低 0.1365，
   却**快约 100 倍**（1.2 ms vs 125.5 ms）。这正是混合检索存在的理由 —— 两路各自都不足。
3. **重排收益为负（⑤ 相对 ④，nDCG −0.0927）** —— ⚠️ **这不等于"Reranker 没用"**。
   本次跑的是 `lexical` 降级重排器（Jaccard 词面重叠），**不是** BGE 交叉编码器。
   换成真 Reranker 后才算数，因此当前配置里 Reranker 默认不参与结论。

复现：
```bash
python -m app.eval.ablation            # 或 make eval-ablation
```

> ✅ **本次 embedding 为本地 BGE-M3，向量路数字具备参考价值**（报告顶部会动态标注可信度）。
> 换用其它 embedding 模型后数值会变 —— 必须**重建索引并重跑**；
> 索引自带「指纹」，配置不匹配时检索会直接报错并提示 `python scripts/build_index.py`。

> ⚠️ **指标口径与局限**：`Recall@5` / `nDCG@10` 均为 **paper 级**（chunk 检索 → paper 聚合 → 再算指标）。
> ④⑤⑥ 的 Recall@5 完全相同，说明在当前规模下命中率已接近饱和，
> **配置差异主要体现在排序质量（nDCG / MRR）上** —— 只报 Recall 会看不出差别。
> 毫秒级耗时差异不宜直接外推；⑥ 的收益部分依赖「分类约束已知且正确」这一前提，
> 真实场景分类由 LLM 推断，可能出错。

### 语义检索抽样

中文提问 → 英文论文索引，是产品的核心用法。抽样用于**肉眼确认语义方向**是否正确：

| 查询 | 金标准论文 | 命中 | 召回路径 |
|---|---|---|---|
| RAG 系统里的融合机制（fusion）有哪些类型？ | 2601.05264（RAG Stack 综述） | **第 1 名** | bm25 + vector |
| 面向多跳查询的 RAG 框架如何系统性地补全证据缺口？ | 2510.22344（FAIR-RAG） | **第 2 名** | vector |
| 用 NLP 技术自动生成文献综述有哪些方法？ | 2411.18583 | **第 2 名** | vector |
| How to evaluate a RAG system? datasets and metrics | 2601.05264 / 2503.16581 | **第 1、4 名** | bm25 + vector |

中文问「多跳补证据缺口」命中 FAIR-RAG、问「融合机制」命中综述的架构章节 —— 方向正确。
（抽样集随语料扩充可重新生成：`python scripts/demo_search.py --query "..." --keywords "..."`）

### 引用准确率（自动口径）

从 `citations` 表直接统计 `verified=true` 的占比 —— **不需要任何 API key**：

```
run verify-058fc534 → {"total": 3, "verified": 3, "accuracy": 1.0, "method": "auto"}
```

⚠️ 这是**自动口径**（自己批改自己的作业）。架构文档 §12 要求另有人工抽检口径用于校准，目前尚未执行。

---

## 项目结构

```
arXiv论文研究Agent/
├── app/
│   ├── main.py              # FastAPI 入口 + lifespan
│   ├── core/                # 配置 / 日志 / 安全 / 容错
│   ├── db/                  # SQLAlchemy 模型与会话（SQLite + WAL）
│   ├── retrieval/           # 分块 / embedding / FAISS / BM25 / RRF / 重排
│   ├── agents/              # LangGraph state / graph / nodes / tools / prompts
│   ├── ingest/              # arXiv / OpenAlex 客户端与入库流水线
│   ├── memory/              # 短期记忆 + 向量长期记忆
│   ├── observability/       # 追踪 / Token 计量 / SSE 事件发射
│   └── eval/                # RAGAS 评测与消融实验
├── ui/streamlit_app.py      # 交互面板
├── tests/                   # 单元测试
├── sample_data/             # 样例论文数据，保证离线可跑
├── Dockerfile / Dockerfile.ui / docker-compose.yml / Makefile
└── 设计文档：产品定义 + 架构设计（见仓库内 .md 文件）
```

---

## 技术栈

| 层 | 选型 |
|---|---|
| 后端 | FastAPI · LangChain · LangGraph（SqliteSaver 状态持久化） |
| 流式 | SSE + `StreamingResponse` |
| 检索 | FAISS · BM25 · RRF 融合 · BGE Embedding / Reranker |
| 数据 | SQLAlchemy 2.0 · SQLite（WAL） |
| 容错 | tenacity 重试 / 超时 / 熔断 |
| 观测 | LangSmith（可选）· RAGAS（可选）· Token 统计 |
| 演示 | Streamlit |
| 部署 | Docker + Docker Compose |

---

## 开发

```bash
make help          # 查看所有命令
make test          # 单元测试
make lint          # 代码检查
make fmt           # 格式化
make ablate        # 检索策略消融实验
```

---

## 许可证

[MIT](LICENSE)
