"""LLM 工厂 —— 全项目**唯一**构造 ChatModel 的地方。

为什么要收敛成一个工厂？因为编排层、节点、工具、评测脚本都要用 LLM，
如果各处 `ChatOpenAI(...)` 满天飞，换 provider（OpenAI 兼容 → Ollama → mock）
就要改十几处，且极易漏改某一处导致"线上跑了 mock"这种灾难。

三种 provider：
- `openai`  任何 OpenAI 兼容端点（OpenAI / DeepSeek / 通义 / 智谱 / 本地 vLLM）
- `ollama`  本机 Ollama
- `mock`    不调用任何模型，**确定性**返回预设结构（离线跑通全流程与单测的基石）

关于 `MockChatModel` 为什么必须"能配合 `with_structured_output()` 工作"：
我们的 plan / read / verify / revise 四个节点**全部**依赖 structured output。
如果 mock 不支持它，那么"无 key 也能跑通全流程"这条产品承诺（产品定义 §8 工程指标）
就只是一句口号 —— 离线只能跑到一半。所以这里把 `with_structured_output` 也实现了：
它从 prompt 里**解析自己渲染进去的 `<untrusted_document>` 块**，据此反向"编"出
与真实文档对齐的结构，从而让 mock 输出与真实链路同构（而不是返回一堆占位符，
那样下游的引用校验必然全线失败，等于没验证）。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any, Literal, get_args, get_origin

import structlog
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.observability.token_meter import get_token_meter, resolve_node

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# 与 prompts/loader 共享的"不可信文档"标记格式。
# 这里**故意**重复定义正则而不是 import，理由：llm.py 属于 core 层，
# 让它反向依赖 agents 层会形成环（agents → core.llm → agents.prompts）。
# 标记格式本身是"契约"，两侧各写一份正则、并用单测锁死一致性，比制造循环依赖划算。
# ---------------------------------------------------------------------------
_DOC_BLOCK_RE = re.compile(
    r"<untrusted_document\s+([^>]*?)>\s*(.*?)\s*</untrusted_document>", re.S
)
_ATTR_RE = re.compile(r"([A-Za-z_]+)\s*=\s*[\"']([^\"']*)[\"']")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

# 关键词里的高频虚词：它们在任何方向都出现，进 BM25 只会稀释信号、引入噪声
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "which", "these", "those", "into", "using", "used", "our", "their", "its",
        "can", "has", "have", "not", "but", "such", "than", "then", "been", "more",
        "also", "how", "what", "when", "where", "why", "all", "any", "may", "one",
        "two", "new", "基于", "paper", "papers", "we", "it", "in", "on", "of", "to",
        # 提示模板自身的高频词：即便 _focus_question_text 没截干净，也不该漏进关键词
        "planner", "reader", "verifier", "writer", "untrusted", "document", "keywords",
        "chunk", "claim", "quote", "supported", "source", "section", "locator",
    }
)


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算：中英混排下 ~4 字符 / token。

    只用于 mock 模式的成本展示。真实模式一律取服务端返回的 `usage`，
    因为自估必然失真（不同分词器差异可达 ±30%），拿它算钱会算错。
    """
    return max(1, len(text) // 4)


def parse_untrusted_documents(text: str) -> list[dict[str, str]]:
    """从 prompt 文本里解析出 `<untrusted_document ...>` 块。

    返回每块的属性（chunk_id / arxiv_id / section / source）+ 正文。
    **为什么放这里而不是 prompts 层**：mock 模型需要"读懂"自己被喂了什么文档，
    才能产出与真实文档对齐的结构；这是 mock 能力的实现细节，不属于 prompt 模板。
    """
    docs: list[dict[str, str]] = []
    for attrs_raw, body in _DOC_BLOCK_RE.findall(text):
        attrs = {k: v for k, v in _ATTR_RE.findall(attrs_raw)}
        attrs["text"] = body.strip()
        docs.append(attrs)
    return docs


def _focus_question_text(prompt: str) -> str:
    """截出"用户提问 + 文档"这一段，甩掉系统提示。

    为什么较真：系统提示（planner.md / reader.md）里满是 `BM25`、`keywords`、`untrusted`
    这类英文词。若直接对整段 prompt 抽词，抽出来的"关键词"全是提示模板里的词，
    BM25 侧等于喂了一堆与研究方向无关的噪声。截到提问处即可根治。
    """
    marker = "研究问题："
    idx = prompt.rfind(marker)
    return prompt[idx:] if idx >= 0 else prompt


def _english_keywords(text: str, limit: int = 6) -> list[str]:
    """从任意文本里抽英文术语（去重、去虚词）。

    存在的唯一理由：`SubQuestion.keywords` 必须是英文（架构 §4.1 注①），
    而 mock 拿到的 prompt 是中文系统提示 + 英文文档，必须保证产出**过得了校验**。

    不足 3 个时用一个极短的通用术语表补齐：
    mock 的真正目标是"让检索路径被走通且不产生非法参数"，不是模拟检索质量。
    """
    seen: list[str] = []
    for w in _ASCII_WORD_RE.findall(text):
        lw = w.lower()
        if lw in _STOPWORDS or lw in seen:
            continue
        seen.append(lw)
        if len(seen) >= limit:
            break
    for filler in ("retrieval", "reranking", "cross-encoder", "benchmark", "language-model"):
        if len(seen) >= 3:
            break
        if filler not in seen:
            seen.append(filler)
    return seen[:limit]


class MockChatModel(BaseChatModel):
    """确定性 ChatModel：不联网、不加载权重，用于离线跑通全流程与单测。

    设计上的三个硬约束：
    1. **确定性**：同一 prompt 永远得到同一输出（缓存/幂等/断言全靠它）；
    2. **支持 structured output**：否则 plan/read/verify/revise 四个节点离线跑不了；
    3. **输出与真实文档对齐**：从 prompt 里解析出被喂进来的文档块，据此产出
       `chunk_id / paper_id / quote`，使下游的"引用必须能在原文定位"校验可以真正跑起来
       —— 返回占位符会让校验必然失败，那就等于没验证链路。
    """

    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    # ------------------------------------------------------------------ 自由文本
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        text = self._render_free_text(prompt)
        prompt_tokens = sum(_estimate_tokens(str(m.content)) for m in messages)
        completion_tokens = _estimate_tokens(text)
        message = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            response_metadata={"model_name": "mock"},
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "model_name": "mock",
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    def _render_free_text(self, prompt: str) -> str:
        """按"最近一个 human 消息里的文档"生成一段确定性的中文文本。

        有文档时输出带 `[n]` 标记的中文综述（writer 节点走这条分支），
        没有文档时输出确定性短句（其它零散调用走这条，保证不返回空串 ——
        空串会让"报告非空"这类断言变得毫无意义）。
        """
        docs = parse_untrusted_documents(prompt)
        if not docs:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
            return f"（mock 输出 #{digest}）"

        lines: list[str] = ["## 研究综述（mock 生成）", ""]
        lines.append("本次综述基于本地语料库中检索到的文献，逐条标注引用来源：")
        lines.append("")
        for idx, doc in enumerate(docs, start=1):
            title = doc.get("title") or doc.get("arxiv_id", "unknown")
            snippet = re.sub(r"\s+", " ", doc.get("text", ""))[:120]
            lines.append(f"- 文献 [{idx}]（{title}）指出：{snippet} …")
        lines.append("")
        lines.append("> 本文由 AI 生成，引用请以原文为准。")
        return "\n".join(lines)

    # ------------------------------------------------------- structured output
    def with_structured_output(  # type: ignore[override]
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """覆盖基类实现，走"schema 感知的确定性填充"。

        基类默认实现要求模型支持 tool calling（bind_tools → 解析 tool_calls），
        而 mock 没有"真的想调用工具"这一层。强行伪造 tool_calls 会让每个 schema
        都要维护一份报文模板；直接按 schema 生成合法实例更简单、也更不容易随
        schema 演进而腐烂。**代价**：mock 不覆盖"模型选错工具名"这类失败模式 ——
        那属于集成测试（用真模型）的职责，不是离线单测的职责。
        """
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return super().with_structured_output(schema, include_raw=include_raw, **kwargs)
        model_cls: type[BaseModel] = schema

        def _invoke(input: Any, config: Any = None, **_: Any) -> Any:  # noqa: ANN401
            prompt = _messages_to_text(input)
            parsed = _fill_model(model_cls, prompt)
            usage = {
                "prompt_tokens": _estimate_tokens(prompt),
                "completion_tokens": _estimate_tokens(parsed.model_dump_json()),
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            raw = AIMessage(
                content=parsed.model_dump_json(),
                usage_metadata={
                    "input_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                },
            )
            # 为什么在这里手动记账：mock 的结构化输出**不经过 `_generate`**，
            # 因此不会触发任何 LLM 回调 —— 若不管，离线模式下 plan/read/verify/revise
            # 四个节点在 `token_by_node` 里会整条缺失，只剩 write 一条，
            # 那"按节点细分"的验证就成了假绿。节点名从 config.metadata 取（LangGraph 注入），
            # 与真实链路走的是同一个取值口径。
            metadata = config.get("metadata") if isinstance(config, dict) else None
            get_token_meter().record(resolve_node(metadata), usage)
            if include_raw:
                return {
                    "raw": raw,
                    "parsed": parsed,
                    "parsing_error": None,
                }
            return parsed

        return RunnableLambda(_invoke)


def _messages_to_text(input: Any) -> str:  # noqa: ANN401
    """把 Runnable 的入参（messages / PromptValue / str）统一压成纯文本。"""
    if isinstance(input, str):
        return input
    if isinstance(input, BaseMessage):
        return str(input.content)
    if isinstance(input, dict):
        return json.dumps(input, ensure_ascii=False)
    try:
        messages = convert_to_messages(input)
    except Exception:  # noqa: BLE001 - 非消息类型时退化为 str()
        return str(input)
    return "\n".join(str(m.content) for m in messages)


# ---------------------------------------------------------------------------
# schema 感知的确定性填充
# ---------------------------------------------------------------------------
def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """拆掉 `X | None`，返回 (X, 是否可空)。"""
    origin = get_origin(annotation)
    if origin is None:
        return annotation, False
    args = get_args(annotation)
    if type(None) in args:
        rest = [a for a in args if a is not type(None)]
        return (rest[0] if len(rest) == 1 else Any), True
    # Annotated[X, ...] / list[X] 等：这里只关心 Union 形式的可选
    return annotation, False


def _fill_model(model_cls: type[BaseModel], prompt: str, index: int = 1) -> BaseModel:
    values: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        values[name] = _fill_field(name, field.annotation, field.default, prompt, index)
    return model_cls(**values)


def _grounded_value(name: str, prompt: str, index: int) -> Any:
    """对"必须落回原文"的字段，直接取 prompt 里真实文档的属性。

    这些字段是产品核心承诺（引用可溯源）的载体：如果 mock 给的是占位符，
    verifier 的"引用片段必须能在原文找到"这条校验会**必然失败**，
    那么离线跑通的只是"图不报错"，而不是"链路正确"。所以宁可在这里多花 20 行。
    """
    docs = parse_untrusted_documents(prompt)
    if not docs:
        return None
    doc = docs[(index - 1) % len(docs)]
    if name in ("paper_id", "arxiv_id"):
        return doc.get("arxiv_id") or None
    if name == "chunk_id":
        # 渲染属性与 schema 字段同名（prompts.render_document 刻意如此，见其 docstring），
        # 所以这里不再需要 id→chunk_id 的名称映射。
        raw = doc.get("chunk_id")
        return int(raw) if raw and str(raw).isdigit() else None
    if name in ("quote", "evidence", "snippet") or name.endswith("_evidence"):
        # `*_evidence`（对比表里"中文值 + 英文原文摘录"的摘录侧）走同一条路：
        # mock 若给空串，新的摘录校验分支在离线测试里就永远走不到，
        # 表现为"单测全绿、换真模型才发现整列被清空"。
        body = re.sub(r"\s+", " ", doc.get("text", "")).strip()
        return body[:80] if body else None
    if name in ("locator", "section"):
        return doc.get("section") or None
    if name == "source":
        src = doc.get("source")
        return src if src in ("html", "abstract") else None
    return None


def _fill_field(name: str, annotation: Any, default: Any, prompt: str, index: int) -> Any:
    # 0) 关键词：必须是**英文列表**（架构 §4.1 注①），单独处理，不能走通用容器分支
    if name == "keywords":
        return _english_keywords(_focus_question_text(prompt), limit=5)

    # 1) 需要落回原文的字段优先（类型可能是 str / int / Literal，统一在这里兜住）
    grounded = _grounded_value(name, prompt, index)
    if grounded is not None:
        return grounded

    base, optional = _unwrap_optional(annotation)
    origin = get_origin(base)
    args = get_args(base)

    # 可空字段且默认就是 None（如 SubQuestion.category）：留空比硬编造更安全
    if optional and default is None and (origin is None) and base is str:
        return None
    if base is type(None):
        return None

    # ---- 标量 ----
    if isinstance(base, type) and issubclass(base, bool):
        return True
    if isinstance(base, type) and issubclass(base, int):
        return index
    if isinstance(base, type) and issubclass(base, float):
        return 0.9 if "score" in name else 0.5
    if origin is Literal:
        choices = args
        # source 必须落在 html|abstract 上，取第一个即可（abstract 是安全侧的降级值）
        return choices[0] if choices else None
    if isinstance(base, type) and issubclass(base, str):
        return _fill_str(name, prompt, index)

    # ---- 容器 ----
    if origin in (list, tuple, set):
        item_type = args[0] if args else str
        n = _list_size(name, item_type, prompt)
        return [_fill_item(name, item_type, prompt, i + 1) for i in range(n)]
    if isinstance(base, type) and issubclass(base, BaseModel):
        return _fill_model(base, prompt, index)
    if origin is dict:
        return {}

    # 兜底：给一个过得了校验的最小值
    return _fill_str(name, prompt, index)


def _list_size(name: str, item_type: Any, prompt: str) -> int:
    """列表字段要生成几个元素。

    这几个数字不是随手写的：
    - `sub_questions` = 3：**这是 Send 扇出验证的基准**（架构 §4.3-④ 要求
      "子问题数 = 3"才能暴露并行汇合把下游执行 N 次的问题），mock 必须能稳定产出 3；
    - `claims` 跟随 prompt 里的文档数：断言"引用来自真实文档"，多篇才有意义；
    """
    if "sub_question" in name:
        return 3
    if name in ("unsupported", "failed", "failed_papers", "errors"):
        # "未通过的论断"列表：mock 下应为空 —— 让 happy path 干净地通过校验，
        # 想验证 revise 分支应当直接构造 state 调路由函数，而不是靠在 mock 里埋开关。
        return 0
    if name in ("claims", "evidence", "items", "citations"):
        docs = parse_untrusted_documents(prompt)
        return max(1, min(3, len(docs))) if docs else 2
    if isinstance(item_type, type) and issubclass(item_type, BaseModel):
        return 1
    return 1


def _fill_item(container_name: str, item_type: Any, prompt: str, index: int) -> Any:
    if isinstance(item_type, type) and issubclass(item_type, BaseModel):
        return _fill_model(item_type, prompt, index)
    if item_type is str:
        return _fill_str(container_name, prompt, index)
    return _fill_field(container_name, item_type, None, prompt, index)


def _fill_str(name: str, prompt: str, index: int) -> str:
    """按字段名给出确定性的字符串值。

    `quote / paper_id / chunk_id / locator / source` 这几个字段刻意要从 prompt 里
    **解析真实文档**来填 —— 因为下游 verifier 会校验"引用片段是否真在原文里"，
    用占位符会让这条最核心的链路在离线模式下永远校验失败。
    """
    docs = parse_untrusted_documents(prompt)
    doc = docs[(index - 1) % len(docs)] if docs else {}

    if name in ("id", "sub_question_id"):
        return f"sq-{index}"
    if name in ("text", "question", "sub_question", "title"):
        return f"子问题{index}：该方向的核心方法与代价是什么？"
    if name in ("claim", "statement"):
        return f"论断{index}：检索到的文献报告了可复现的方法与结果。"
    if name in ("quote", "evidence", "snippet"):
        # 取原文前若干字符：保证它是原文的**真子串**，才能通过引用校验
        body = re.sub(r"\s+", " ", doc.get("text", "")).strip()
        return body[:80] if body else "（mock 原文片段）"
    if name in ("paper_id", "arxiv_id"):
        return doc.get("arxiv_id", "0000.00000") or "0000.00000"
    if name in ("chunk_id",):
        return doc.get("chunk_id", "1") or "1"
    if name in ("locator", "section"):
        return doc.get("section", "abstract") or "abstract"
    if name == "source":
        return doc.get("source", "abstract") or "abstract"
    if name in ("reason", "note", "explanation"):
        return "mock 校验：引用片段与原文一致。"
    if name in ("rationale", "summary", "outline", "draft", "markdown"):
        return "mock 文本"
    return f"mock-{name}-{index}"


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def get_llm(temperature: float | None = None) -> BaseChatModel:
    """按 `settings.llm_provider` 构造 ChatModel。

    Args:
        temperature: 不传则用 `settings.llm_temperature`。evaluator 需要"冻结温度"
                  才能做消融归因（架构 §12），所以这里显式留出覆盖口。

    ★ 签名刻意只有一个参数：全项目有大量测试与调用点按 `get_llm(temperature)` 打桩，
    给它加关键字参数会让那些桩当场失配（症状是容错测试报"没有抛出预期异常"，
    看着像容错坏了，其实是签名漂移）。端点差异的处置放在 `_with_thinking_off`，
    它复制实例、不改这个入口。

    ★ 为什么在这里统一挂 token 计量回调：
    `get_llm` 是全项目**唯一**构造 ChatModel 的地方，把计量器挂在这一处，
    就保证"任何一次 LLM 调用都会被记账"，不需要每个节点自觉上报，
    也不会因为将来新增节点忘了上报而静默漏计。回调是同一个进程内单例，
    因此各节点的用量能聚合成一张完整的账（见 `app/observability/token_meter.py`）。
    """
    s = get_settings()
    t = s.llm_temperature if temperature is None else temperature
    meter = get_token_meter()

    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.llm_model,
            base_url=s.openai_base_url,
            api_key=s.openai_api_key or "not-needed",
            temperature=t,
            timeout=s.llm_timeout,
            callbacks=[meter],
        )

    if s.llm_provider == "ollama":
        # 优先官方 langchain-ollama：社区版 ChatOllama 已在 LangChain 0.3 废弃，
        # 且它的 with_structured_output 直接抛 NotImplementedError ——
        # 而 plan/read/verify/revise 四个节点全部依赖 structured output，
        # 选了社区版就等于"ollama 只能聊天、跑不了图"。社区版仅作兜底。
        try:
            from langchain_ollama import ChatOllama
        except ImportError:  # pragma: no cover - 取决于安装形态
            from langchain_community.chat_models import ChatOllama  # type: ignore[no-redef]

        return ChatOllama(
            model=s.ollama_model, base_url=s.ollama_base_url, temperature=t, callbacks=[meter]
        )

    return MockChatModel(temperature=t, callbacks=[meter])


# 结构化输出在不同 OpenAI 兼容端点上的可用组合不一样，而且**只有真发一次请求才知道**
# —— 400 是端点回来的，本地无从推断（实测：DeepSeek 的思考模式模型对三种方式分别回
# "This response_format type is unavailable now" / "Prompt must contain the word 'json'"
# / "Thinking mode does not support this tool_choice"）。所以这里按顺序试，而不是要求
# 用户先去配一个开关 —— 他配的是模型，不该再来背端点差异。
# 顺序把"什么都不传"放第一位：那是各 provider 的现行默认，ollama 与 mock 走的正是它，
# 一上来就改写等于给一条没有故障的链路换实现。
_STRUCTURED_RECIPES: tuple[dict[str, Any], ...] = (
    {},  # provider 默认实现
    {"method": "function_calling", "thinking_off": True},
    {"method": "function_calling"},
)

# 第一个成功的组合记在进程里，之后所有节点都用它（否则每条问句都要先浪费一次 400）。
_working_recipe: int | None = None

# 只在"请求形状被端点拒绝"时才换下一种组合。判窄有两条：
# 1. 状态码存在且不是 400 就**一定不算** —— 401/403/429/5xx 是真故障，重试另一种写法
#    只会把一次失败变成三次失败（熔断器的计数、耗时、token 账单都跟着失真）；
# 2. 其余情况必须命中形状标记。刻意不收 `invalid_request_error` 这种通用类型名：
#    DeepSeek 的"Authentication error"也带它，收进来就等于把鉴权失败当成写法问题。
_SHAPE_MARKERS = ("response_format", "tool_choice", "json_schema", "thinking mode")


def _is_request_shape_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None and status != 400:
        return False
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _SHAPE_MARKERS)


def _with_thinking_off(model: BaseChatModel) -> BaseChatModel:
    """返回一个"这次请求关掉思考模式"的模型副本。

    为什么复制实例而不是给 `get_llm` 加参数：思考模式只与**结构化调用**这一次请求冲突
    （正文生成没这个限制），而 `get_llm(temperature)` 是全项目、包括大量测试桩共用的
    签名 —— 为端点兼容去改它，代价会以"容错测试莫名变红"的形式在别处付。

    没有 `extra_body` 字段的模型（ollama / mock）原样返回：它们本来就没有这个问题。
    """
    if "extra_body" not in getattr(type(model), "model_fields", {}):
        return model
    return model.model_copy(update={"extra_body": {"thinking": {"type": "disabled"}}})


async def ainvoke_structured_raw(
    schema: type[BaseModel],
    messages: list[BaseMessage],
    temperature: float | None = None,
    *,
    make_llm: Callable[[], BaseChatModel] | None = None,
) -> dict[str, Any]:
    """结构化调用，自动适配端点支持哪一种实现方式；返回 `{"parsed":…, "raw":…}`。

    返回原始 dict 而不是拆好的 `(parsed, usage)`：`usage_of()` 住在节点层，而节点层
    已经 import 了本模块 —— 在这里拆用量会把依赖方向绕成环。

    `make_llm` 是**给调用方留的注入口**，节点层必须传它：容错与熔断的测试是把
    `_common.get_llm` 换成假模型来验证"失败会记进熔断器"的。如果这里自己调
    `get_llm`，那种 patch 就落不到实处 —— 症状不是测试变红，而是**测试直接去打真实 API**
    （已经真发生过一次：断言"抛 TransientError"的用例拿到了真模型的正常返回，
    报的是 DID NOT RAISE，看着像容错坏了）。

    `parsed` 可能是 None：`include_raw=True` 下端点返回了不合规内容时**不抛异常**，
    只给 `parsed=None`。是否降级由各节点自己决定（见 planner / reader 的处置）。

    ⚠️ 不要把构造好的 Runnable 缓存起来：不同节点 schema 不同，缓存 key 会退化成
    `id(schema)`（不可靠），而 ChatModel 本身很轻。这里只缓存"哪一种组合可用"。
    """
    global _working_recipe  # noqa: PLW0603 - 进程内的能力探测结果，见上面注释

    factory = make_llm or (lambda: get_llm(temperature))
    # 已探测出可用组合时先试它（一次调用就够）；但它也会失效 —— 运行中换模型/换端点
    # 之后原来的组合可能又被拒了，所以形状类 400 一律清掉缓存、回退到全量探测。
    order = list(range(len(_STRUCTURED_RECIPES)))
    if _working_recipe is not None:
        order.remove(_working_recipe)
        order.insert(0, _working_recipe)
    last_error: BaseException | None = None

    for idx in order:
        recipe = _STRUCTURED_RECIPES[idx]
        kwargs = {"method": recipe["method"]} if "method" in recipe else {}
        try:
            llm = factory()
            if recipe.get("thinking_off"):
                llm = _with_thinking_off(llm)
            result = await llm.with_structured_output(schema, include_raw=True, **kwargs).ainvoke(
                messages
            )
        except Exception as exc:  # noqa: BLE001 - 只在"形状被拒"时换方案，其余原样上抛
            if not _is_request_shape_error(exc):
                raise
            last_error = exc
            _working_recipe = None
            logger.warning(
                "llm.structured_recipe_rejected",
                recipe=recipe or "provider-default",
                error=str(exc)[:200],
            )
            continue

        _working_recipe = idx
        if idx:  # 只在真的偏离默认实现时说一声，免得日志里全是噪音
            logger.info(
                "llm.structured_recipe",
                recipe=recipe,
                note="端点不支持默认方式，已自动改用这一种",
            )
        return result

    raise RuntimeError(
        "这个端点不接受任何已知的结构化输出方式"
        f"（试过 {len(_STRUCTURED_RECIPES)} 种，最后一次的错误：{str(last_error)[:200]}）。"
        "plan/read/verify/revise 四个节点都依赖结构化输出，缺它整条研究链路跑不起来。"
    )


def llm_identity() -> dict[str, str]:
    """当前 LLM 的可展示身份（用于 SSE `run_started` 与日志，不含任何密钥）。"""
    s: Settings = get_settings()
    if s.llm_provider == "ollama":
        return {"provider": "ollama", "model": s.ollama_model}
    if s.llm_provider == "mock":
        return {"provider": "mock", "model": "mock"}
    return {"provider": "openai", "model": s.llm_model}


__all__ = [
    "MockChatModel",
    "get_llm",
    "ainvoke_structured_raw",
    "llm_identity",
    "parse_untrusted_documents",
]
