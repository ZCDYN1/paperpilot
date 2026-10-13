"""全局配置。

设计原则：**所有可调行为集中在此，且默认值就能跑起来。**

- 默认 LLM 走 OpenAI 兼容接口 —— 使用者只需填 `OPENAI_BASE_URL` + `OPENAI_API_KEY`
  （OpenAI / DeepSeek / 通义 / 智谱 / 本地 vLLM / Ollama 的兼容端点都可以）
- 默认 embedding 走本地 BGE，但可一键切到 API 或 mock
- 不想配任何 key 时，把 `LLM_PROVIDER` 与 `EMBEDDING_PROVIDER` 都设为 `mock`
  仍可跑通全链路（用于离线评估与演示）
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["openai", "ollama", "mock"]
EmbeddingProvider = Literal["local", "api", "mock"]


class Settings(BaseSettings):
    """从环境变量 / `.env` 读取的全局配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------ LLM ------------------------------
    llm_provider: LLMProvider = "openai"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_timeout: int = 60

    # ----------------------------- Ollama ----------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # --------------------------- Embedding ---------------------------
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16
    # 仅 embedding_provider=api 时生效
    embedding_api_base_url: str = ""
    embedding_api_key: str = ""
    embedding_api_model: str = "text-embedding-3-small"

    # --------------------------- Reranker ----------------------------
    # 关闭可大幅降低延迟（代价是精度下降）；消融实验会量化这个权衡
    rerank_enabled: bool = True
    # local   = 真正的 BGE 交叉编码器（需装 [local] 依赖）
    # lexical = 词面重叠降级（无需模型、确定性，供离线与单测使用）
    rerank_provider: Literal["local", "lexical"] = "local"
    # HF 仓库名（如 BAAI/bge-reranker-v2-m3，首次加载自动下载）**或本地目录**
    # （如 models/bge-reranker-v2-m3）。权重缺失时不会崩：预检后自动降级 lexical，
    # 并在日志里给出可照抄的下载命令（含 HF_ENDPOINT=https://hf-mirror.com 镜像）。
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---------------------------- 检索 -------------------------------
    retrieve_top_k: int = 50
    rrf_k: int = 60
    fuse_top_n: int = 30
    rerank_top_n: int = 8
    # Reader 精读的论文数。默认与 rerank_top_n 一致，但两者语义不同
    # （一个是"召回多少篇"，一个是"精读多少篇"），拆开才能独立调优。
    read_top_k: int = 8
    max_chunks_per_paper: int = 2
    min_corpus: int = 8

    # -------------------- 文档抓取（arXiv 合规项）--------------------
    arxiv_user_agent: str = "PaperPilot/0.1 (mailto:you@example.com)"
    arxiv_request_interval: float = 3.0
    arxiv_max_results: int = 200
    # 优先抓 arXiv 官方 HTML 正文（2023-12 后的论文基本都有），取不到自动降级为摘要。
    # 关掉它就只剩摘要级证据，引用只能定位到 abstract。
    fetch_html: bool = True
    # 用 OpenAlex 补充被引数等**数字字段**（文本字段只信 arXiv 官方）
    openalex_enabled: bool = True
    # 是否允许联网抓取。关掉后只能检索已有索引 ——
    # 供离线演示与 CI 使用（避免单元测试偷偷联网，这是很常见的坑）。
    ingest_enabled: bool = True
    # OpenAlex 的礼节性联系方式（拼进 User-Agent）；留空则从 arxiv_user_agent 里自动取 mailto
    openalex_mailto: str = ""
    # OpenAlex 没有 arXiv 那样严格的限速要求，但保持礼貌调用
    openalex_request_interval: float = 0.2
    # 外部 HTTP 请求超时（秒）—— arXiv HTML 页面较大，给宽一点
    http_timeout: float = 30.0

    # ------------------------ 图 / 反思机制 --------------------------
    max_revision: int = 2
    confirm_outline: bool = False

    # ------------------- 容错：重试 / 超时 / 熔断 -------------------
    # 关掉 = 三层保护全部旁路（LLM 调用直接透传）。
    # 默认开；离线 / mock 场景不需要关，因为 MockChatModel 永不失败、断路器永远闭合，
    # 而"关掉才跑得通"本身就是一个 bug（说明保护和链路互相打架）。
    resilience_enabled: bool = True
    # 一次 LLM 调用（**含内部重试**）的总时间预算；<=0 表示不设超时。
    # 语义见 `app/core/resilience.py` 模块 docstring：预算按整次调用计，不按每次尝试计，
    # 否则最坏延迟 = 尝试数 × 超时，SSE 这种交互式场景受不了。
    # ⚠️ 应 ≥ `LLM_TIMEOUT × RESILIENCE_ATTEMPTS + 退避`，否则时间预算会把还差一次的
    #    调用掐死（表现为"明明第二就该成功了"）。默认 180 = 60s × 2 次 + 退避余量。
    resilience_timeout: float = 180.0
    resilience_attempts: int = 2
    # 指数退避的乘数 / 上限 / 随机抖动上界（秒）。测试把三个都设 0 就不会真的等待；
    # 生产保持默认 —— 惊群与"把刚缓过来的下游再打趴"靠的就是这几秒。
    resilience_backoff: float = 1.0
    resilience_max_backoff: float = 8.0
    resilience_jitter: float = 0.5
    # 连续多少次"整段调用失败"（内部重试若干次仍失败）才打开熔断
    breaker_failure_threshold: int = 5
    # 熔断打开后的冷却时长；冷却结束放行一次半开试探
    breaker_recovery_timeout: float = 60.0
    # 对外 HTTP（arXiv / OpenAlex / embedding API）出口是否也挂熔断
    breaker_for_http: bool = True

    # ------------------------- 输入侧防护 ---------------------------
    # 用户可控文本（研究问题、编辑后的子问题）进图之前过 `app.core.security.validate_input`。
    input_guard_enabled: bool = True
    input_max_chars: int = 4000
    # reject  = 命中注入模式直接 400 拒绝（默认：研究工具没有"降级为纯检索"的必要）
    # degrade = 放行但使用清洗后的文本并打告警日志（用户只是转述论文原句时不误伤）
    input_injection_action: Literal["reject", "degrade"] = "reject"

    # --------------------- 追踪 / 评测（可选）------------------------
    # ★ 这三个值会在 lifespan 里同步进 os.environ（见 `app/observability/tracing.py`），
    #   因此 `.env` 里配了就真的生效；没配则安静跳过（不发网络请求、不影响跑通链路）。
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "paperpilot"
    # LangSmith 端点（私有部署 / 自检时改它）；留空则用 langsmith 的默认端点
    langchain_endpoint: str = ""

    # ---------------------------- 存储 -------------------------------
    data_dir: Path = Path("./data")
    faiss_index_dir: Path = Path("./data/faiss")
    sqlite_dir: Path = Path("./data/sqlite")
    checkpoint_db: Path = Path("./data/sqlite/checkpoints.db")
    sqlite_url: str = "sqlite+aiosqlite:///./data/sqlite/paperpilot.db"

    # ---------------------------- 服务 -------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ---------------------------- 便捷判定 ----------------------------

    @property
    def llm_is_mock(self) -> bool:
        return self.llm_provider == "mock"

    @property
    def embedding_is_mock(self) -> bool:
        return self.embedding_provider == "mock"

    @property
    def mock_mode(self) -> bool:
        """完全离线：不调用任何外部模型。"""
        return self.llm_is_mock and self.embedding_is_mock

    @property
    def llm_configured(self) -> bool:
        """LLM 是否已配置到可用状态（用于健康检查给出明确提示）。"""
        if self.llm_provider == "openai":
            return bool(self.openai_api_key and self.openai_base_url)
        return True  # ollama / mock 无需 key

    # ---------------------------- 副作用 ------------------------------

    def ensure_dirs(self) -> None:
        """创建运行期所需目录（幂等）。"""
        for d in (self.data_dir, self.faiss_index_dir, self.sqlite_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict:
        """脱敏摘要，可安全打进日志或返回给前端 —— 绝不包含任何 key。"""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model if self.llm_provider != "ollama" else self.ollama_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": (
                self.embedding_model
                if self.embedding_provider == "local"
                else self.embedding_api_model
                if self.embedding_provider == "api"
                else "mock"
            ),
            "rerank_enabled": self.rerank_enabled,
            "mock_mode": self.mock_mode,
            "llm_configured": self.llm_configured,
            # 容错与安全的"开关面板"：只报状态，绝不报阈值之外的敏感值
            "resilience_enabled": self.resilience_enabled,
            "input_guard_enabled": self.input_guard_enabled,
            "tracing_enabled": bool(self.langchain_tracing_v2 and self.langchain_api_key),
        }


@lru_cache
def get_settings() -> Settings:
    """进程内单例。测试时可调用 `get_settings.cache_clear()` 重置。"""
    return Settings()
