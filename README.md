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

### 方式一：Docker（推荐，零 key）

```bash
docker compose up --build
```

**不建 `.env`、不填任何 key 也能起**：compose 会把 `LLM_PROVIDER` / `EMBEDDING_PROVIDER` 退回 `mock`、
`RERANK_ENABLED` 退回 `false`，并在启动时自动导入 `sample_data/corpus.json`（24 篇真实 arXiv 论文的摘要级语料），
所以第一次请求不会去联网抓 arXiv。默认镜像**不含 torch**（约 2GB+），跑的是 API/mock 路线；
要本地 BGE 的检索质量：

```bash
docker compose build --build-arg INSTALL_LOCAL=true && docker compose up
```

> ⚠️ 建好 `.env` 并把 `EMBEDDING_PROVIDER` 设成 `local` 时，必须用上面这条带
> `INSTALL_LOCAL=true` 的构建，否则容器里没有 FlagEmbedding，检索会明确报错
> （报错信息里就写着该跑哪条命令，不会静默变差）。

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
python scripts/load_sample_data.py   # 导入离线样例语料（24 篇，零联网）
uvicorn app.main:app --reload    # 起后端 :8000
streamlit run ui/streamlit_app.py  # 另开一个终端起面板 :8501
```

**只想先确认链路是通的**（不填 key、不联网、不下模型）：

```bash
LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock INGEST_ENABLED=false \
  python scripts/load_sample_data.py && uvicorn app.main:app
```

`INGEST_ENABLED=false` 会关掉联网抓取分支，只用库里已有语料并如实声明覆盖度。
mock 只用于跑通流程，**不要用它评估任何检索或引用质量**（见「评测结果」里的口径说明）。

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

### 主表：真 BGE 交叉编码器重排（`bge-reranker-v2-m3`）

同轮可比 —— 这四行是同一次运行产出的（报告：`eval_results/ablation_24papers_bge_rerank.md`）：

| 配置 | Recall@5 | Precision@5 | nDCG@10 | MRR | 每条查询耗时（中位数） |
|---|---|---|---|---|---|
| ① 纯向量（BGE-M3 基线） | 0.8542 | 0.2100 | 0.8418 | 0.8833 | 137 ms |
| ④ ① + BM25 + **RRF** | 0.9250 | 0.2300 | 0.8777 | 0.9062 | 131 ms |
| ⑤ ④ + **BGE 交叉编码器重排** | **0.9333** | **0.2350** | **0.9222** | **0.9625** | **5734 ms** |
| ⑥ ⑤ + metadata 过滤 | 0.9333 | 0.2350 | **0.9314** | **0.9750** | 3114 ms |

### 参考行：词面 / 融合策略（来自 `eval_results/ablation_24papers.md`，同一份索引、`lexical` 重排器那轮）

②③ 不使用重排器，因此不受重排器更换影响；但它们的**耗时与上表不是同一轮，不可直接比**：

| 配置 | Recall@5 | Precision@5 | nDCG@10 | MRR | 耗时 / 查询 |
|---|---|---|---|---|---|
| ② 纯 BM25 | 0.7333 | 0.1850 | 0.7053 | 0.7438 | **1.2 ms** |
| ③ 向量 + BM25 加权求和 | 0.9042 | 0.2200 | 0.8417 | 0.8675 | 123.8 ms |

四条结论都有数据支撑，不是断言：

1. **RRF 优于加权求和**：④ 相对 ③ 的 nDCG@10 高 **0.0360**、Recall@5 高 **0.0208**。
   加权求和必须先归一化（BM25 分数无上界、余弦相似度落在 [-1,1]），而归一化本身引入分布假设 ——
   实测印证了这个设计取舍。④ 的质量指标在两轮运行里**逐位相同**（检索路径是确定性的），
   所以跨轮的 ④−③ 比较成立；跨轮**唯一不可比的是耗时**。
2. **纯词面召回不够用，但便宜得惊人**：② 的 Recall@5 比 ① 低 0.1208、nDCG@10 低 0.1365，
   却**快约 100 倍**（1.2 ms vs 137 ms）。这正是混合检索存在的理由 —— 两路各自都不足。
3. **重排的真正结论是"用延迟换排序质量"，方向与我之前写过的结论相反。**
   换成真 `bge-reranker-v2-m3` 后，⑤ 相对 ④：nDCG@10 **+0.0445**、MRR **+0.0563**、Recall@5 +0.0083 ——
   收益为正；代价是每条查询从 131 ms 涨到 **5734 ms**（CPU fp32，约 44 倍）。
   ⚠️ 这一版 README 曾写过"重排收益为负（nDCG −0.0927）"，那是 `lexical` 降级重排器
   （Jaccard 词面重叠）产出的**假结论**：同一份索引、同一批查询，只把重排器换掉，结论方向就反了。
   这就是"降级路径必须在报告里显式标注"这条设计的由来（`app/eval/ablation.py` 里名字不含
   `bge-reranker` 就自动打 ⚠️）。
   **实际部署含义**：一次研究会并行跑 3~5 个子问题，默认开重排意味着光重排就要多花 15~30 秒。
4. **过滤与重排互补，而且过滤反而更快**：⑥ 相对 ⑤ 再 +0.0092 nDCG@10，耗时却从 5734 ms 降到
   3114 ms —— metadata 先缩小候选集，进重排器的配对数变少。但 ⑥ 的收益依赖
   「分类约束已知且正确」，真实场景分类由 Planner 推断，推错就会把金标准挡在召回之外。

复现：
```bash
python -m app.eval.ablation           # 或 make eval-ablation；默认写到 eval_results/ablation_latest.md
```

> 仓库里只保留两份报告：`ablation_24papers_bge_rerank.md`（主表，真重排器）与
> `ablation_24papers.md`（参考行来源）。上面那条命令的默认产物 `ablation_latest.md` 已被 `.gitignore` 排除 ——
> 它与那两份不是同一轮，留着只会被误当证据引用。24 篇语料之前那三轮（5 篇 / 373 chunk，含一份 mock embedding
> 产出的）已删除 —— 它们的数字与当前口径不同轮、不同规模，留着只会被误引。

> ✅ **本次 embedding 为本地 BGE-M3，向量路数字具备参考价值**（报告顶部会动态标注可信度）。
> 换用其它 embedding 模型后数值会变 —— 必须**重建索引并重跑**；
> 索引自带「指纹」，配置不匹配时检索会直接报错并提示 `python scripts/build_index.py`。

> ⚠️ **指标口径与局限**：`Recall@5` / `nDCG@10` 均为 **paper 级**（chunk 检索 → paper 聚合 → 再算指标）。
> ④ 之后 Recall@5 已接近饱和（0.925~0.933），
> **配置差异主要体现在排序质量（nDCG / MRR）上** —— 只报 Recall 会看不出差别。
> ④ 在两轮运行中质量指标一致、耗时却相差约 7 ms，说明**毫秒级差异不宜直接外推**；
> 重排那 5.6 秒是在 24 篇 / 1210 chunk 的极小语料、CPU fp32 单一部署形态下测得的，
> 换 GPU 或换语料规模都要重测才能引用。
> 换 embedding 模型后所有数字都会变，必须重建索引并重跑；换重排器**不需要**重建索引。

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

### 容错与输入闸门：怎么证的、证到哪一步

多 Agent 系统最容易自欺的地方，是**用"会抛异常的假客户端"去证明熔断生效** ——
真实 provider 的失败未必以异常形态出现，那样测出来的绿是假的。
所以这里刻意从**生产入口 + 传输层**两个方向打：

```bash
python scripts/verify_wiring.py        # 离线确定性，约 12 秒；--only A|B|C 可分段跑
```

实测到的事实：

- **熔断真的会打开**：把 `OPENAI_BASE_URL` 指向不可达地址后连续调用 `structured_invoke`，
  第 1、2 次各抛 `OpenAIConnectionError`（每次内部按 `RESILIENCE_ATTEMPTS=2` 重试），
  第 2 次后 `CircuitBreaker.state=open`；此后调用 **0.00 s** 返回 `CircuitOpenError`，不再打下游。
- **三个 provider 路径都会正常抛异常**（OpenAI 兼容 / `langchain_ollama` / community Ollama）：
  覆盖 401、429、500、连接失败、模型不存在（404），也覆盖"200 但响应体畸形"这类会静默退化成空结果的情况。
- **输入闸门不误伤**：`reject` 下注入文本 → 400 + `{code:"prompt_injection_detected", matched_patterns, hint}`；
  空白/超长 → 422；四条真实风格的中文研究问题全部正常出首帧事件。
  `degrade` 模式下注入文本会放行并打 `security.input_degraded` 告警（用清洗后的文本）。
- **已知边界（不是 bug，但会误导使用者）**：`INPUT_MAX_CHARS=4000` 对 `question` 够不到 ——
  请求体 schema 先按 `max_length=2000` 拦下并返回 422，这道 400 的闸门只对 schema 没限死的入口生效
  （例如 resume 时用户编辑过的子问题文本）。两处错误的响应结构不同，前端需要分开处理。

### 引用准确率（自动口径 + 人工抽检口径）

架构文档 §12 要求**两个口径一起报**，因为只报自动口径等于自己批改自己的作业：

```bash
python scripts/spotcheck_citations.py --n 20 --seed 7              # ① 导出待判定 CSV（零 key）
#    在 eval_results/citation_spotcheck.csv 的 human_verdict 列逐条填
#    supported / unsupported / ambiguous，然后：
python scripts/spotcheck_citations.py --from-csv eval_results/citation_spotcheck.csv   # ② 人工口径
python scripts/run_eval.py --suite ragas                           # 自动口径（引用 verified 占比）
# 等价快捷命令：make spotcheck / make spotcheck-stats
```

判定依据也一并写进 CSV（`hard_check_passed` = 「引用片段能否在原文里逐字找到」这个程序事实），
人只需要判断「这条 quote 是否真的支撑这条 claim」。

**两个口径实测出来的数字差得很远，这正是它值得做的理由：**

| 口径 | 数值 | 说明 |
|---|---|---|
| 自动（mock provider 产出的历史 run） | 1.0 | ⚠️ **不是能力证据**：mock 的校验分支恒返回「有支撑」，`verified` 必然为 true |
| 自动（真实 LLM：本地 ollama `qwen2.5:7b`，2 条真实问题 / 18 条引用） | **0.61** | 7 条未通过：6 条硬校验（引文不是逐字摘录）、1 条软校验（论断超出片段）；原因逐条落进 `citations.note` |
| 人工抽检 | 待填 | 抽样脚本已就绪，需要对真实 run 逐条判定 |

也就是说：Verifier 在真实模型下**确实会拒引用**（不是橡皮图章），但离验收线 90% 还有距离。

**这条差距现在定位清楚了，而且已经不在"引用定位"上。** 同一份代码在修定位根因前后的对比：

| 观测点 | 修根因前（4 条问题 / 53 条引用） | 修根因后（2 条问题 / 18 条引用） |
|---|---|---|
| read 侧因定位失败剔除的引用 | 每轮 0~4 条 | **0 条** |
| 需要靠重锚救回的引用（`reanchored`） | 每轮 2~11 条，`swapped_chunk` 反复触发 | **0 条**（全部精确命中） |
| verify 未通过的原因分布 | 定位失败与语义判定混在一起，分不清谁该负责 | 6 硬校验（引文非逐字）/ 1 软校验（语义夸大） |

根因是**渲染格式与字段名不对齐**，不是模型乱写：文档块当时渲染 `id="370"`，而模型要填的字段叫
`paper_id` / `chunk_id` —— 块里唯一可见的纯数字只有一个，带 "id" 的字段名却有两个，
小模型填反是必然结果。把属性改名为 `chunk_id`、并把该字段从可空改成必填之后，
模型产出的 `(paper_id, chunk_id)` 直接精确命中，`swapped_chunk` / `quote` 两条兜底路径**一次都没再触发**
（保留它们是为了吃掉残差：实测必填之后模型在没把握时会吐 `-1` 这类占位值）。

剩下的 0.61 是**另一件事，而且不是链路缺陷**：7B 级模型会把摘要里的内容改写几个词再"引用"，
被硬校验逐字比对拒掉 —— 这正是这条校验存在的意义（宁缺毋滥）。
它靠换更大的模型或接 API provider 来解决，而不是靠放松校验；把阈值调松等于把
"每条引用都能回到原文"这条产品承诺关掉。样本只有 18 条，真实 provider 下的数字还要重测。

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
├── sample_data/corpus.json  # 24 篇真实 arXiv 论文的摘要级语料（83KB），零 key 零联网起步
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
make test          # 单元测试（全离线：mock provider，不联网、不下模型）
make lint          # 代码检查
make fmt           # 格式化
make sample        # 导入离线样例语料（新 clone 第一次跑之前）
make zerokey       # 零 key 冷启动：样例语料 + mock provider 直接起服务
make ablate        # 检索策略消融实验
make spotcheck     # 导出引用人工抽检 CSV（人工口径）
```

---

## 许可证

[MIT](LICENSE)
