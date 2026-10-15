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
- **三类语料过滤**：分类 / 年份 / 是否有开源仓库（`has_code` 从 arXiv 官方 `comment`、`abstract`
  里检测仓库链接得到，检测不到就留空，不参与"有代码"这一侧的判定）；
  也可以直接点名 2~3 篇论文只做这几篇的对比（`arxiv_ids`）
- **多 Agent 编排**：Manager–Worker 结构，子问题并行检索，引用校验 Agent 做自我反思
- **引用可溯源**：每条引用落在具体 chunk 上，点开可定位到章节 / 原文片段
- **主动暴露不确定性**：报告中单列「未找到原文支撑的结论」，并且这些论断**不会进入正文与参考文献**
  —— 标记而不剔除等于把编造的引用换个位置放，本产品的承诺是"剔除 + 承认"
- **实时过程可见**：SSE 流式推送节点 / 工具调用 / 召回片段，等待期间不是黑盒
- **一键导出**：`GET /api/v1/research/{run_id}/export.md` 在服务端渲染完整报告
  （对比表 + 正文 + 带章节定位的参考文献 + 薄弱结论 + 成本耗时 + 免责声明），下载即是用 markdown 写好的材料；
  另有 `related-work.md` 输出按方法分组的「相关工作」章节草稿
- **单篇深挖与自有文献**：`/papers/{id}/ask` 只在某一篇的内容上作答，取不到支撑就明说；
  `POST /papers/upload` 能把自己的 PDF / TXT / MD 切块入库，之后同样可被检索与溯源
- **引用关系与被引趋势**：从 OpenAlex 的 `referenced_works` 落引用边（谁引用了谁），
  被引数按天存快照、够两个点才谈趋势
- **状态可续跑**：LangGraph SqliteSaver 持久化，中断后可恢复，不重复烧 token
- **长期记忆**：记住关注方向 / 已读论文 / 输出偏好，**跨会话生效**。已读过的论文在下次研究时
  让出精读候选位（检索候选一篇不删，未读不足时按重排序补位），对应产品定义 F10「下次不重复推荐」
- **工程化**：tenacity 重试 / 超时 / 熔断、工具入参强校验、Prompt 注入防护、Token 用量统计
- **可观测与评测**：LangSmith 全链路追踪 + RAGAS 自动评测（均为可选，不配也能跑）
- **界面内配置**：系统状态页的「配置中心」可直接改 LLM / Embedding / Rerank 的 provider、模型、地址、密钥
  并写回 `.env`（按行更新、保留注释、自动备份）；填完可先点「测试连通性」再保存，不必去找编辑器改 key
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

- **React 控制台：http://localhost:8000** （`web/dist` 在镜像构建阶段产出、由后端同源挂载，
  所以只有一个端口、不需要 CORS、也不需要第二个容器）
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/healthz

旧的 Streamlit 面板（:8501）现在挂在 compose 的 `legacy-ui` profile 下，**默认不启动**：

```bash
docker compose --profile legacy-ui up -d ui     # 或 make docker-up-legacy
```

它是控制台的先行版本，留作对照原始 agent 输出、排查后端问题时用。
停止时也要带上 profile，否则 compose 不会停 profile 里的容器：`make docker-down` 已经带了。

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

# 前端控制台：构建一次，之后由后端同源提供（不需要第二个端口，也不需要 CORS）
cd web && npm ci && npm run build && cd ..

uvicorn app.main:app --reload    # 打开 http://127.0.0.1:8000 就是 React 控制台
```

> ⚠️ 端口的真相源有两条，别混：`.env` 里的 `API_PORT` 只有**启动器**（双击 `启动PaperPilot.bat` /
> `python scripts/launch.py`）会读；手敲 `uvicorn app.main:app` 时命令行赢，默认就是 8000。
> 要换端口就走启动器，或显式 `uvicorn app.main:app --port 9000`（`make dev-api` 支持
> `API_PORT=9000 make dev-api`）。

开发前端时不必每次重新构建：`cd web && npm run dev` 起 :5173（Vite 代理到 :8000）。
`web/.env.development` 默认 `VITE_USE_MOCK=1`，即**没有 Python 后端也能浏览全部界面**；
连真后端调试时在 `web/.env.local` 里写 `VITE_USE_MOCK=0`。
生产构建读 `web/.env.production`（`VITE_USE_MOCK=0`），而代码级默认同样是"走真实后端"——
mock 只能被显式打开，否则一次读不到 env 文件的构建就会静默产出一份全是演示数据的前端。

> ⚠️ **改了 `web/src/` 之后必须重新 `npm run build` 才会生效**：后端挂的是构建产物 `web/dist`。
> 静态文件是**每次请求现读盘**的，所以重建后不必重启进程、刷新浏览器即可；但改了 `app/`（Python 侧）才需要重启。
> `scripts/launch.py` 启动时会比对 `web/dist` 与 `web/src` 的时间戳，产物比源码旧就打警告并给出
> `cd web && npm install && npm run build` —— 双击启动却看到旧界面时，先看这条。

> **用 uv 也行**（仓库里的 `uv.lock` 锁定了这套解析结果）：
> `uv venv --python 3.12 && uv sync --extra dev`（要本地 BGE 再加 `--extra local`）。
> 注意 `dev` 定义在 `[project.optional-dependencies]` 而不是 PEP 735 的
> `dependency-groups`，所以**裸 `uv sync` 不会装 pytest / ruff**，必须显式 `--extra dev`。
> README、Makefile 与 CI 仍以 `pip install -e` 为准；改了 `pyproject.toml` 后
> 用 `uv lock` 刷新、`uv lock --check` 校验一致性。

**只想先确认链路是通的**（不填 key、不联网、不下模型）：

```bash
LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock RERANK_ENABLED=false INGEST_ENABLED=false \
  python scripts/load_sample_data.py && uvicorn app.main:app
```

（与 `Makefile` 的 `make zerokey` 是同一组变量。）
`INGEST_ENABLED=false` 会关掉联网抓取分支，只用库里已有语料并如实声明覆盖度。
`RERANK_ENABLED=false` 是必需的：`RERANK_ENABLED` 默认为 `true`，一旦本机恰好缓存着
reranker 权重（`app/retrieval/reranker.py:228` 的预检命中本地目录或 HF 缓存就不会降级 lexical），
这条"零模型下载"的演示会去加载 torch 和 2.3GB 权重 —— mock 链路根本不需要重排质量。
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

`local` 的两个权重（BGE-M3 约 2.1GB、bge-reranker-v2-m3 约 2.3GB）默认从 Hugging Face 拉取，
国内直连常常拿不到 —— 这是新 clone 者最容易卡死的一步，而且卡点藏在第一次**检索**里
（启动不报错，检索才报错）。两条出路：

```bash
# ① 走镜像再建索引（检索失败时的日志里也会原样打出这条命令）
HF_ENDPOINT=https://hf-mirror.com python scripts/build_index.py
# ② 或手动把权重下到 models/ 目录，并把 EMBEDDING_MODEL / RERANK_MODEL 指到那些路径（全程离线）
```

权重缺失时后端不会崩：重排预检（`app/retrieval/reranker.py`）发现本地没有权重就退回词面重排
并打印获取命令，向量检索按 `MIN_CORPUS` 门槛走降级路径。界面照常可用，只是检索质量下降 ——
当前用的是哪一档，`/healthz` 的 `embedding.identity` 与 `index.hint` 会如实写出来。

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

### 界面内配置中心（不想手动改 `.env`）

系统状态页的「**配置中心**」把上面这一小组字段（provider / 模型名 / 接口地址 / 密钥 / `MIN_CORPUS`）
搬到了界面上，填完可以先点「测试连通性」再保存 —— 它写回的就是根目录的 `.env` 本身
（按行更新、保留你的注释，改前自动备份成 `.env.bak-<时间戳>`），不引入第二份配置。

| 接口 | 行为 |
|---|---|
| `GET /api/v1/config` | 回当前生效值 + 每个字段的来源（`env` / `dotenv` / `default`）。**密钥只回 `{set, masked}`，绝不下发明文** |
| `PUT /api/v1/config` | 写回 `.env`；改完 `provider` / `model` / `rerank` / `min_corpus` 需要**重启进程**才生效，响应里的 `restart_required` 会把这几项点出来 |
| `POST /api/v1/config/test` | 探测 LLM 服务连不连得上；地址与密钥可以随请求带上、不必先保存 |

**安全边界（刻意写在这里，因为这是一个默认无鉴权、且会把密钥明文落盘的写接口）**：

- 写接口只接受本机 Host（`localhost` / `127.0.0.1` / `::1`），并拒绝跨站 `Origin` / `Referer`；
  确实要从别的机器改配置，用 `CONFIG_ALLOWED_HOSTS=<主机名>` 追加（逗号分隔）。
- 总开关 `CONFIG_WRITABLE`（默认 `true`）。设成 `false` 时 `PUT` 一律 403、`GET` 回 `writable: false`，
  界面自动变只读；但 `POST /test` **仍可用** —— 它不落盘、只发一次只读探测，只读模式下反而是唯一能验证配置的入口。
- ⚠️ `API_HOST=0.0.0.0`（默认）+ `CONFIG_WRITABLE=true` 等于"同网段任何人都能改你的配置"。
  要那样部署，请至少把 `CONFIG_WRITABLE` 设成 `false`。

---

## 接口一览

React 控制台与 Streamlit 面板都只是这些接口的客户端，下面这些才是对外契约（`http://localhost:8000/docs` 有 OpenAPI）：

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/v1/research/stream` | 发起一次研究，`text/event-stream` 流式推进程；跑到拆解后停在中断点 A |
| POST | `/api/v1/research/{run_id}/resume` | 从中断点 A 继续（可带被用户改过的子问题）。只有停在 A 的 run 能续：非 `awaiting_input` 状态回 `409 {code:"not_resumable", status}`，run 不存在回 `404 {code:"run_not_found"}` —— 否则任意 run_id 都能被反复触发续跑、白烧配额 |
| GET | `/api/v1/research?limit=&offset=&status=` | 最近的运行列表（带每次运行的引用数与支撑率；界面上"看哪一次报告"靠它选） |
| GET | `/api/v1/research/{run_id}` | run 状态、报告正文、对比表、引用、token 用量 |
| GET | `/api/v1/research/{run_id}/citations` | 引用证据链（含章节定位与证据强度） |
| GET | `/api/v1/research/{run_id}/export.md` | 导出 Markdown 报告（F8） |
| GET | `/api/v1/research/{run_id}/related-work.md` | 导出「相关工作」章节草稿，按方法分组（F18） |
| GET | `/api/v1/research/{run_id}/timeline` | 方法演进时间线（F12；没建引用边时 `addresses_gap` 恒为空） |
| POST | `/api/v1/research/{run_id}/followup` | **基于这次研究已读且已校验的证据**追问，不重新检索（F9） |
| GET | `/api/v1/papers?limit=&offset=&q=&category=` | 语料分页查询（标题/摘要 LIKE 过滤；每行带 `chunk_count` 与 `venue_tier`） |
| GET | `/api/v1/papers/{arxiv_id}` | 单篇详情：元数据 + 章节级 chunk + 该篇历史引用（F13） |
| POST | `/api/v1/papers/{arxiv_id}/ask` | 只在这一篇的内容上问答，取不到支撑就明说（F13） |
| POST | `/api/v1/papers/upload` | 上传自己的 PDF / TXT / MD 入库（F15；PDF 需 `pip install -e ".[pdf]"`） |
| POST | `/api/v1/papers/references/build` | 抓 OpenAlex 的 `referenced_works` 落引用边（F16） |
| GET | `/api/v1/papers/references/graph?ids=a,b` | 取一层引用关系图（F16） |
| POST | `/api/v1/papers/citations/snapshot` | 记录今天的被引数（F19 趋势的数据源，需定期调用） |
| GET | `/api/v1/papers/{arxiv_id}/citation-trend` | 被引趋势；采样点 < 2 时 `verdict=insufficient_data`（F19） |
| GET | `/api/v1/papers/venue-tier?venue=ACL 2024` | 会议/期刊分档，独立字段、不进 9 列对比表（F19） |
| POST | `/api/v1/papers/ingest` | 按 `arxiv_ids` / `query` 批量抓取入库并刷新索引。**同步返回 200**：arXiv 侧限速约 1 req/3s、单次 50 篇封顶，入库幂等，客户端超时后原样重试就是最优恢复 |
| GET | `/api/v1/terms?text=…` | 标出文本里的术语及位置（F11） |
| GET | `/api/v1/terms/{term}` | 查单个术语；未命中是 `200 + known=false`，不是 404（F11） |
| POST | `/api/v1/terms` | 与 GET 同结果，但正文走请求体（整篇报告 4700+ 字符放不进 query，F11） |
| POST | `/api/v1/terms/unknown` | 批量问"哪些术语词表里没有"（F11） |
| GET | `/api/v1/stats/overview` | 全局聚合指标（论文/分块/运行数、平均支撑率、按天 runs 与 tokens），首页卡片与系统页的数据源 |
| POST | `/api/v1/eval/run` | 触发一次评测（`mode=ablation\|metrics\|ragas`），**202 立刻返回 `eval_run_id`**，评测在后台跑 |
| GET | `/api/v1/eval/runs/{eval_run_id}` | 查评测状态与结果（`status` + `is_running` + 汇总 + 逐条样本） |
| GET | `/api/v1/config` | 读取当前生效配置：密钥字段只回 `{set, masked}`、绝不下发明文，并附每字段来源（`env` / `dotenv` / `default`）与 `writable` 开关 |
| PUT | `/api/v1/config` | 写回根目录 `.env`（按行更新、保留注释、自动备份 `.env.bak-<时间戳>`）；Host 白名单 + 同源校验，`CONFIG_WRITABLE=false` 时 403 |
| POST | `/api/v1/config/test` | 探测 LLM provider 连通性（地址 / 密钥可随请求带、不必先保存）；**不受 `CONFIG_WRITABLE` 限制** —— 只读模式下它是唯一能验证配置对不对的入口 |
| GET | `/healthz` | 健康检查 + 索引状态（`index.compatible` 会告诉你换模型后该重建） |
| GET | `/resilience` | 熔断器实时状态 |

请求体里几个"会改变行为"的字段（`ResearchRequest` 是 `extra="forbid"`，拼错字段名会直接 422，
不会静默忽略）：

```jsonc
{
  "question": "近两年 RAG 的重排技术有哪些进展，代价是什么",
  "session_id": "上一轮 run 所在会话的 id",   // 传了 = 追问（能读到本会话此前的问答）；不传 = 新会话
  "language": "zh",                          // zh | en，报告与对比表字段值的输出语言
  "top_k": 8,                                // 每个子问题**最终**保留几篇论文（不是每路召回量）
  "rerank": true,                            // 请求级关掉交叉编码器重排（换延迟）
  "arxiv_ids": ["2305.15334", "2401.00001"], // 只在这几篇里做对比，并且跳过按关键词抓新语料
  "filters": {
    "categories": ["cs.CL", "cs.IR"],        // 多分类走全局通道，不再被静默丢弃
    "year_gte": 2024, "year_lte": 2026,
    "has_code": true                         // true=元数据里确认有仓库链接；false=未见链接
  }
}
```

```bash
# 跑完一次研究之后，把报告取成文件（未知 run_id 是 404，不会给你一份空报告）
curl -OJ "http://localhost:8000/api/v1/research/<run_id>/export.md"
```

> ⚠️ **`Send` 扇出的分支只看得到 payload 里的键**：加新的请求级配置时，
> 除了 `_initial_state` 要转发，还要把它放进 `app/agents/graph.py::_worker_payload`，
> 否则它会一路传到检索节点才凭空消失（这条坑的回归钉子在 `tests/test_api_feature_switches.py`）。

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
- **对外错误只有码 + 人话**：SSE `error` 事件的 `message` 与论文详情/单篇问答的 404 `detail`
  都不带异常原文 —— 底层异常可能含 SQL 片段、连接串、含用户名的本机路径，而这是开源仓库，
  每个 clone 者的部署环境不该从错误响应里被推断出来。完整异常在服务端日志里，
  排查路径是 `code` + `run_id` → 日志。`code` 与 `recoverable` 是对外契约，不会因脱敏而变动。

### 用 HTTP 触发评测（`POST /api/v1/eval/run`）

`scripts/run_eval.py` 与这个接口共用同一套服务层（`app/eval/service.py`），
所以命令行能做到的事接口也能做到，报告落库口径完全一致。评测是分钟级的，
因此受理即 `202`、结果靠轮询，而不是把 HTTP 连接挂住：

```bash
curl -sS -X POST localhost:8000/api/v1/eval/run -H 'Content-Type: application/json' \
  -d '{"mode":"ablation","configs":"rrf,rerank"}'
# → {"eval_run_id":1,"status":"running","mode":"ablation","dataset":"golden_40","poll":"/api/v1/eval/runs/1"}

curl -sS localhost:8000/api/v1/eval/runs/1        # status=completed 时 result.configs 就是消融表
```

| `mode` | 需要什么 | 拿不到时怎样 |
|---|---|---|
| `ablation` | 只要检索索引，**不需要 LLM**（六种检索配置的 nDCG@10 / MRR / Recall@K 对照） | 索引为空 → `status=unavailable` + 原因 |
| `metrics` | 只读数据库（产品定义 §8 的成功指标） | 无数据 → 对应指标 `status` 明确标注，不填占位数 |
| `ragas` | `pip install -e ".[eval]"` + 真实 LLM 裁判 | 缺依赖 → **501** 并附可照抄的安装命令；`LLM_PROVIDER=mock` 或未配 key → 报告 `available=false`，**绝不产出 mock 分数** |

三条接口层面的取舍，都是为了不让"没跑成"看起来像"跑过了"：

- **同一时刻只允许一条评测**（`409`，响应体带 `running_eval_run_id` 与轮询地址）。
  并发跑会污染消融表的耗时列、并让两个任务抢同一份索引与模型；不做排队是因为
  队列需要一份能活过进程重启的任务状态设施，而那超出这个项目现阶段要引入依赖的范围。
- **依赖缺失在受理时就探测**（同步 `import` 判定，放在后台任务之外）。让后台任务去发现，
  调用方只能先拿一个 `202`、很久之后读到 `failed` —— 而那件缺的依赖在提交那一刻就已确定。
- `golden_path` 解析后必须仍在仓库内（先 `resolve()` 再比目录，符号链接绕不过去），
  且只接受 `.jsonl`；越界与拼错字段都是 `422` 并带 `hint`。请求模型是 `extra="forbid"`。

进程重启后，库里遗留的 `running` 行不会再被更新，所以 `GET /eval/runs/{id}` 同时返回
`status` 与 `is_running`（后者看的是当前进程的闸门状态）—— 看到 `status=running` 但
`is_running=false` 就知道这条已经不会被更新了，重跑一次即可。

### 引用准确率（自动口径 + 抽检口径）

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

**抽样默认只取真实 LLM 的 run**（`--provider real`，可用 `mock` / `unknown` / `all` 显式取回）。
provider 由 `token_usage.model` 反查（mock 模式下 `llm_identity()` 返回 `model="mock"`），
每行的来源写在 CSV 的 `run_provider` 列里。之所以要过滤：mock 的校验分支按设计永远放行，
那些行的 `verified=true` 是**结构恒真**，让人核对它们既浪费人力、又会把人工口径系统性拉高。
没跑到 `finalize` 的 run 没有 `token_usage` 行，来源确实不可考 —— 如实标 `unknown` 并同样不抽，
不猜测也不静默丢弃。

**自动口径与独立裁判口径差 40%，这正是它值得做的理由：**

| 口径 | 数值 | 说明 |
|---|---|---|
| 自动（mock provider 产出的历史 run） | 1.0 | ⚠️ **不是能力证据**：mock 的校验分支恒返回「有支撑」，`verified` 必然为 true |
| 自动（真实 LLM：本地 ollama `qwen2.5:7b`，2 条真实问题 / 18 条引用） | **0.61** | 7 条未通过：6 条硬校验（引文不是逐字摘录）、1 条软校验（论断超出片段）；原因逐条落进 `citations.note` |
| 人工抽检 | 未采人工，**已做模型裁判抽检**（见下） | 20 条抽样的逐条判定由模型裁判给出，并在每条 `human_note` 里标明"模型裁判判定，非人工" —— 它不是人工口径，不能当人工口径引用 |

**模型裁判抽检（20 条，`eval_results/citation_spotcheck.csv`）**：判定为
`supported=20 / unsupported=0 / ambiguous=0`，与自动口径的一致率 0.60 —— 也就是
自动口径把 8 条内容上成立的引用判成了"未通过"。逐条查完，这 8 条的成因只有两种，
且都能复现：（该 CSV 在 `.gitignore` 里：判定明细跟着机器走，任何人跑
`make spotcheck` 都会得到同名文件重新判定，仓库里不固化一份别人的答案。）

| 条数 | 现象 | 归因 |
|---|---|---|
| 5 | 引文**逐字就在它所指向的那个 chunk 里**，但 `verified=false` | 软校验（让模型判断"这段是否支撑这句论断"）在本地小模型上误判 —— 是裁判模型能力，不是定位或抽取问题 |
| 3 | `note=引文未能在原文中逐字匹配`，实际差异只在排版 | 当时的逐字比对没做归一化（现已实现，见下）：原文带 PDF 抽取残留的引用号 `[58]`、以及 `matches) .` 这种句号前多一个空格，模型摘录时写成 `matches).` |

第二类的后果是**自动口径系统性偏低**：这些引用会被列进「未找到原文支撑的结论」，
读者看到的是"这条结论没证据"，而事实是"证据在、只是比对方式太严"。

**这条归因已经落地成实现（2026-10-15）**：`app/agents/nodes/_common.py::quote_match` 把逐字比对
改成**两段式** —— 先按原口径（折叠空白 + 小写）精确比较，**只有精确失败**才退到"容忍 PDF/HTML
抽取残留"的容忍层，覆盖三类残留：内联引用标号（`[58]`）、标点前多余空白（`matches) .` → `matches).`）、
弯引号 vs 直引号。`read` 侧同篇重锚（`evidence.py`）与 `verify` 硬校验共用这一份定义，
避免"read 刚按宽口径救回来、verify 又按严口径二次杀掉"。
两条通过路径**必须可区分**：精确通过保持空（落库 NULL），容忍通过把痕迹
（`residual:citation_markers` / `residual:punct_spacing` / `residual:quotes` 经 `verify_note`）落进
`citations.note`，所以"这轮有多少条引用是靠归一化救回来的"可以对表统计，不必 grep 服务端日志。
宽容的方向意味着放宽，代价必须讲清：两侧同剥标号之后，引文写 `[57]` 而原文是 `[58]` 也会放行 ——
放宽的只有"方括号里数字标号的正确性"，句子其余文字仍须逐字相同；这条取舍钉在
`tests/test_quote_matching.py::test_marker_number_mismatch_is_a_documented_cost`。

**仍未做（这才是真实遗留）**：归一化会改变自动口径的数值，所以**接上真实 LLM 后需要重跑一次评测
拿到新基线**，否则新旧数字无法对比。在重跑之前，上面那个 0.61 仍是"归一化前"的成绩，
不该被当成归一化后的结果引用；把它改写成"已归一化"只是把 3 条的成因讲清楚了，数字本身没动。

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

**这张表现在可以自查，不必信我们的日志**：救回统计已经从后端日志搬到对外契约上 ——
SSE 第 16 种事件 `citation_rescue`（read / verify 各发一条），以及 `citations` 表新增的
`resolution_how` / `verify_reanchored` 两列。跑完一次研究后直接查分布：

```bash
python -c "import sqlite3;print(sqlite3.connect('data/sqlite/paperpilot.db').execute(\
'select resolution_how, verify_reanchored, count(*) from citations group by 1,2').fetchall())"
# 形如 [(None, None, 152), ('exact', 0, 19)] —— NULL 是加列之前的历史行（当时没记录）
```

⚠️ `rescued=0` 也**照发照存**：只在非零时出现的指标无法区分"根因确实修好了"与"统计断线了"，
而这条统计存在的意义正是随时验证修复没有回退。

剩下的 0.61 是**另一件事，而且不是链路缺陷**：7B 级模型会把摘要里的内容改写几个词再"引用"，
被硬校验逐字比对拒掉 —— 这正是这条校验存在的意义（宁缺毋滥）。
它靠换更大的模型或接 API provider 来解决，而不是靠放松校验；把阈值调松等于把
"每条引用都能回到原文"这条产品承诺关掉。样本只有 18 条，真实 provider 下的数字还要重测。

---

### 已知边界与换真实 API 时的排查点

这一节把"当前测出来是什么样、以及换 provider 之后该先看什么"写清楚，避免下一次排查从零猜起。

**1. 换 provider 时，provider / model / base_url / key 四项必须一起换**
`.env.example` 出厂是一套自洽的 API 配置（`LLM_PROVIDER=openai` + `OPENAI_BASE_URL=https://api.deepseek.com/v1`
+ `LLM_MODEL=deepseek-chat`，只需要补 key）。本机开发时把它改成了 `LLM_PROVIDER=ollama`
却沿用了 `deepseek-chat`，而 ollama 里实际只有 `bge-m3` / `nomic-embed-text` / `qwen2.5:7b` ——
于是第一次 LLM 调用就失败。"只改 provider 不改模型名"是这类故障最常见的形态，
而 `verify_wiring.py` 已覆盖"模型不存在（404）"会以异常抛出、不会静默返回空结果。
本文所有真实数字都是在覆盖 `LLM_MODEL=qwen2.5:7b` 之后测得的。

**2. `0.61` 是"当前模型的上限"，不是"链路的上限"**
一次完整的真实端到端（26 条引用）里，只有 9 条通过校验（**0.35**），而**全部 26 条都走 `exact` 路径、
0 条需要重锚、0 条被定位失败剔除**。这两个数放在一起才是结论：
失败与引用定位无关，纯粹是这个量级模型的引文保真度不够。

换 provider 后用这条查询就能自己分辨回归属于哪一类：

```bash
python -c "import sqlite3;print(sqlite3.connect('data/sqlite/paperpilot.db').execute(\
'select resolution_how, verify_reanchored, count(*) from citations group by 1,2').fetchall())"
```

- `resolution_how` 里**非 `exact` 的比例上升** → 是**定位**出问题了（渲染契约或 schema 被改动过）；
- **仍全是 `exact` 但 `verified` 偏低** → 是**模型**引文保真度问题，不要动校验阈值。

**3. 端点不支持结构化输出时，会自动降级到可用方式**
结构化输出在不同 OpenAI 兼容端点上的可用组合不一样，而且**只有真发一次请求才知道**
（400 是端点回来的，本地无从推断）。`app/core/llm.py::ainvoke_structured_raw` 按顺序试三种组合
（provider 默认 → `function_calling` + 关思考 → `function_calling`），把第一个成功的记在进程里、
之后所有节点复用，避免每条问句都先浪费一次 400。换方案只在"请求形状被端点拒绝"时发生
（状态码 400 且命中 `response_format` / `tool_choice` / `json_schema` / `thinking mode` 标记）；
401 / 429 / 5xx 一律原样上抛 —— 否则一次真故障会被放大成三次，熔断计数与 token 账单跟着失真。
降级时会打 `llm.structured_recipe note=端点不支持默认方式，已自动改用这一种`。
`plan` / `read` / `verify` / `revise` 四个节点都依赖结构化输出，少了这条适配会出现
"四个节点里只有一个能跑通"的假象。

**4. 长期记忆的「避免重复精读」在 24 篇语料上没有发挥空间**
实测日志 `read.prioritized_unread known_read=19 reused_read=8 selected=8`：
8 个精读位全部由已读论文补位，功能正确执行却带不来任何新论文。
原因是 sample_data 只有 24 篇，其中 18 篇已被真实运行精读过（清理脚本污染后仍未读只有 6 篇）。
这条机制要产生可见价值需要把语料扩到产品定义 §8 的 20+ 篇覆盖 / 数百篇入库规模，**目前刻意未做**。
判断它是否在工作，看 `known_read>0` —— 若为 0 且整行日志不出现，说明已读清单没读到东西，
优先怀疑下一项。

**5. 会话内追问可用，但"追问"目前不等于"便宜"**
`session_id` 是可选的会话加入凭证：客户端传它，本次提问就挂进那个会话，`plan` 节点会读回
该会话此前的问答（`short_term.load_context`）注入拆解 prompt，指代（"那方法 B"）因此能被补全。
不传则 `session_id = run_id`，一次提问即一个全新会话（老行为不变）。
注意两条边界：① `thread_id` **永远是 run 级**，会话靠 `session_id` 串多个 run，
共用 checkpoint thread 会让新提问接着上一轮已跑完的 state 走（不报错，只是被污染）；
② F9 有**两条不同的路径**，成本完全不同：`POST /research/stream` 带 `session_id` 走完整链路
（拆解时能看到本会话此前的问答，指代能被补全，但仍会重新检索）；
`POST /research/{run_id}/followup` 才是"不重新检索"那条 —— 它只在上一轮**已校验通过**的引用上作答，
一次 LLM 调用，不碰索引也不联网，复用不了时如实返回 `status="no_evidence"` 而不是偷偷退化成全量研究。
长期记忆则是另一套口径：它属于用户、跨会话生效，读取与淘汰都不按会话过滤。
⚠️ 若将来接多用户或部署成公共服务，这里**必须**换成显式的 user id，否则记忆会跨用户串。

**6. 评测样本量都很小，不要把小数当稳定值**
自动口径的 0.61 来自 2 条问题 / 18 条引用，0.35 来自 1 条问题 / 26 条引用。
两者都远不足以支撑"引用准确率是 X"这种表述，只能说明量级和失败归因。
人工口径的 20 条 `human_verdict` **尚未填写**（抽样已默认只取真实 LLM 的 run），
所以两个口径目前只有一个能报。

**7. 验证脚本不会污染数据**
`scripts/verify_agents.py` 跑完整图（含 `finalize` 写记忆），因此它把库**复制到临时目录**再跑，
结束后删除副本。历史版本直接写真实库，把合成 run 的论文标记成"已读"——
在长期记忆变成跨会话生效之后，这种污染会让排除机制逐步失效且不报错。

**8. 功能覆盖到哪一步**
产品定义 §5 的 F1~F19 **后端全部有实现**（`tests/test_api_feature_switches.py` 有一条从 HTTP
打到节点的真实链路冒烟，其余按功能分文件）。剩下的边界是"每条功能各自不做什么"：

| 功能 | 边界（不是 bug，是刻意范围） |
|---|---|
| F11 术语解释 | 词表是**人工维护**的几十条，未命中返回 `known=false` 而不让模型现场编一个定义。覆盖面换可信度 |
| F12 演进时间线 | `addresses_gap`（谁补了谁的坑）只在 `citation_edges` 里有引用边证据时才非空。**不用年份先后推断演进关系** |
| F13 单篇深挖 | 只在该篇 chunks 上作答；取不到支撑就回答"内容不足以回答"，实验细节层面的问题仍不答（§7 边界） |
| F15 上传入库 | **只抽文本，不解析版面/表格/公式**；PDF 解析器是可选 extra，没装时上传 .pdf 返回 501 并给出照抄命令 |
| F16 引用关系 | 边只来自 OpenAlex `referenced_works`；解析不出 arXiv id 的参考文献不建边。需要显式调 `references/build`，不会自动抓 |
| F19 质量信号 | 分档是人工白名单（判不准就 `preprint`/`unknown`）；趋势需要**多天**采样，只调一次接口拿不到趋势 |
| F9 追问 | 复用不了已校验证据时返回 `no_evidence`，不会悄悄退化成一次全量研究 |

产品定义 §5 的 F1~F19 与架构 §7 列出的路由**已全部有 HTTP 接口**：评测触发
（`POST /api/v1/eval/run` + `GET /api/v1/eval/runs/{id}`）与批量入库
（`POST /api/v1/papers/ingest`）是最后补上的三个，原先由 `scripts/run_eval.py` 与
`ensure_corpus` 节点承担的部分现在是可选快捷路径、不再是唯一入口。
（更早列在"待实现"里的"语料分页查询"与"独立的成本统计接口"分别是
`GET /api/v1/papers` 与 `GET /api/v1/stats/overview`。）
另外 **React 控制台（`web/`）已覆盖 F1~F19 的界面**；`ui/` 下的 Streamlit 面板是它的先行版本，
保留作轻量调试入口。

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
│   ├── eval/                # RAGAS 评测与消融实验
│   ├── export/              # 服务端渲染 Markdown 报告与「相关工作」草稿
│   ├── papers/              # 单篇详情与单篇深挖问答
│   ├── references/          # 引用边（OpenAlex referenced_works）与被引趋势
│   ├── analysis/            # 方法演进时间线、会议/期刊分档
│   ├── terms/               # 术语词表与文本标注
│   ├── followup/            # 基于已校验证据的追问（不重新检索）
│   └── api/v1/              # research / papers / terms / stats / eval / config / health
├── web/                     # React 控制台（Vite + TS，主界面；构建后由 :8000 同源提供）
│   ├── src/api/             # 后端契约的 TS 镜像 + 契约测试（打真实后端）
│   └── src/pages/           # 控制台 / 报告 / 会话 / 论文库 / 分析 / 系统 / 设置（配置中心）
├── ui/streamlit_app.py      # Streamlit 面板（控制台的先行版本，保留作调试入口）
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
| 前端 | React · Vite · TypeScript（`web/`，主界面，构建后同源挂载）；Streamlit（`ui/`，先行版本，可选调试入口） |
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
