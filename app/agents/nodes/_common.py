r"""节点共用的小工具：LLM 调用 + token 记账 + 检索结果整形。

## 为什么要把 token 记账放在这里
`finalize` 要产出"本次研究消耗 42,180 tokens / 约 ¥0.63"（产品定义 §4 产出物 E），
这要求**每个节点的消耗都能被归集**。做法是让统一的调用封装顺手把
`AIMessage.usage_metadata` 抠出来返回，节点把它并进 `state["token_usage"]`
（该 channel 的 reducer 按 key 求和）。

为什么不自己按字符数估：不同分词器差异可达 ±30%，拿它算钱会算错，
而"成本可量化"是硬指标。真实 provider 一律以服务端返回的 usage 为准。

## 为什么用 `include_raw=True`
`with_structured_output` 默认只返回解析后的对象，**把原始 AIMessage 丢了** ——
丢掉的正是 usage。打开 `include_raw` 才能既拿到结构化结果、又拿到 token 计数。

## ★ 为什么"重试 → 超时 → 熔断"接在这里（以及只接在这里）
README 承诺的是"tenacity 重试 / 超时 / 熔断"一条完整链路，而不是三个各自为政的工具函数。
接线的落点选择依据只有一条：**必须是所有 LLM 调用都要经过的公共入口**。
本模块的 `structured_invoke` / `text_invoke` 正是这个入口
（plan / read / verify / revise / compare / write 六个节点全部走它们，
见 `grep -n "structured_invoke\|text_invoke" app/agents/nodes/*.py`），
所以在这里包一层 = 全项目的所有 LLM 调用都被保护，
而**不需要**改任何节点（节点里也没有重试参数可传，容易漏接）。

顺序由 `app/core/resilience.resilient()` 保证：重试在最内、超时在中间、熔断在最外
（一次调用只记一次失败，详见该模块 docstring 与 `tests/test_resilience.py` 的顺序用例）。

## ★ 流式入口为什么只有 `text_invoke` 一个（而不是每个节点都能流）
SSE 的 `token` 事件（架构 §8「LLM 输出增量，前端打字机」）要求**真实生成过程中产生的增量**，
而增量只有 `llm.astream` 给得出来。但它不能变成"人人可用的第二调用入口"：
plan / read / verify / compare 走的是 `with_structured_output`，增量是 JSON 碎片，
吐给用户只会看到 `{"rows": [{"claim` 这种半截语法 —— 那些节点**没有**合法的 token 流可言。
所以这里只给自由文本那条路开流式口子，且当前唯一的使用者是 `write` 节点
（综述正文是这篇产品唯一"值得逐字看"的产出物）。

## 三条"不许把链路堵死"的护栏
1. `RESILIENCE_ENABLED=false` 时 `guarded()` 直接透传 —— 保护逻辑自己出问题时一条环境变量就能退场；
2. mock provider 永不失败 → 断路器永远停在 closed，离线链路行为与接线前完全一致
   （`tests/test_resilience_wiring.py` 用「mock 下连打 N 次、断言熔断计数为零」把这条钉住）；
3. 熔断打开时**快速失败并携带明确 code**（`llm_circuit_open`），由 API 层转成 SSE `error` 事件 ——
   对用户是"立刻知道下游挂了"，而不是"卡 3 分钟后超时"。LLM 没有合理的降级产物
   （伪造一份报告会违反反幻觉承诺），所以这里的"降级"= 快速失败 + 如实上报。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from app.core.llm import ainvoke_structured_raw, get_llm
from app.core.resilience import BREAKER_LLM, CircuitOpenError, guarded
from app.observability.events import emit_out_of_band
from app.observability.resilience_events import RESILIENCE_NODE, sse_observer

logger = structlog.get_logger(__name__)

# 引用校验时用的空白归一化：PDF/HTML 抽取出来的文本里有大量换行与多余空格，
# 直接做子串匹配会因为"少一个空格"而误判为不通过 —— 那会把好引用错杀。
_WS_RE = re.compile(r"\s+")

# 熔断打开时上报给前端的错误码：与 API 层的异常→错误码映射同一口径，
# 前端按 code 分支（`llm_circuit_open` 可提示"稍后重试"，`internal_error` 只报错）。
CIRCUIT_OPEN_CODE = "llm_circuit_open"

# 流式生成中途失败、且**已经往用户面前吐过字**时用的错误码（`TokenSink.abort`）。
# 不复用 `llm_circuit_open`：那条链路最后成功了，要说的不是"下游挂了"，
# 而是"刚才那段流作废了" —— 前端只有按 code 分支才能把两者区分开。
STREAM_ABORT_CODE = "token_stream_aborted"


def normalize_text(text: str) -> str:
    """折叠空白并统一小写，供"引文是否真在原文里"的包含判定使用。"""
    return _WS_RE.sub(" ", (text or "")).strip().lower()


# ---------------------------------------------------------------------------
# 「引文是否逐字在这一块里」的两段式判定 —— verify 硬校验的两条路径与
# evidence 的同篇重锚共用这一份定义（判定口径分叉过一次实测代价：
# read 刚按宽口径救回来的引文，verify 又会按严口径二次杀掉，等于没修）。
#
# 背景（真实证据，不是假想敌）：对 20 条抽样引用做独立裁判复核，"折叠空白 + 小写"
# 的精确比较产生 3/20 假阴性，成因只有两种 PDF/HTML 抽取残留：
#   1. 原文带着内联引用标号（`... versions [58]. These state-of-the-art ...`），
#      模型摘录时写成了干净文本（把 `[58]` 去掉了）；
#   2. 原文标点前多一个空格（`... matches) . The results ...`），模型写成 `matches). The`。
# 这类引用会被列进导出报告的「未找到原文支撑的结论」，读者以为"这条结论没有证据"，
# 而事实是"证据在、只是比对方式过严" —— 自动口径因此系统性偏低。
#
# 做法：两段式，先按现有口径精确比较，**只有精确失败**才退到"容忍抽取残留"；
# 且两条通过路径必须可区分（返回值携带路径码，verifier 据此写进
# verify_note → citations.note，事后能审计"这轮有多少引用是靠归一化救回来的"）。
#
# ★ 真实的取舍（别假装没有代价）：两侧同剥内联引用标号之后，引文写 `[57]` 而原文是
#   `[58]` 也会放行 —— 放宽的只有"方括号里数字标号的正确性"，没放宽的部分是
#   句子其余文字的逐字相同。接受这个代价的理由：引用标号不是论断的实质内容；
#   "引对了没编号"实质由定位层（chunk_id 锚定）与人工抽检负责，不归这一层判。
#   其余情形同理都是"更容易判通过"的方向，所以痕迹必须留（见 quote_match 的返回值）。
# ---------------------------------------------------------------------------

# 内联引用标号：`[58]` / `[12, 13]` / `[3-5]`（normalize_text 之后只剩单空格，`\s*` 是余量）。
_INLINE_REF_RE = re.compile(r"\[\d+(?:\s*[,–—-]\s*\d+)*\]")
# 标点前的多余空白：只在标点前收（`) .` → `).`、` .` → `.`），
# 正常词间空格不在匹配范围内 —— 否则就不是"容忍残留"而是"改写原文"了。
_PUNCT_SPACE_RE = re.compile(r"\s+([.,;:!?)\]}])")
# 弯/直引号混用：PDF 正文是弯引号，模型摘录时一律回吐 ASCII 直引号。
_QUOTE_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

MATCH_EXACT = "exact"
_MATCH_RESIDUAL_PREFIX = "residual:"


def _tol_close_punct_space(text: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_SPACE_RE.sub(r"\1", text)).strip()


def _tol_strip_inline_refs(text: str) -> str:
    return _WS_RE.sub(" ", _INLINE_REF_RE.sub("", text)).strip()


def _tol_straighten_quotes(text: str) -> str:
    return text.translate(_QUOTE_TRANS)


def _tol_markers(text: str) -> str:
    # 剥标号会留下" ."残渣（标号本来就贴在标点前），折叠标点后空白必须同批做掉 ——
    # 这是两类残留的机械耦合，不是顺手放宽。
    return _tol_close_punct_space(_tol_strip_inline_refs(text))


def _tol_combined(text: str) -> str:
    return _tol_straighten_quotes(_tol_markers(text))


# 按"解释力从小到大"排列：小规则单独能对上就记在小规则头上，不让组合规则抢功 ——
# 审计统计要回答的是"到底放宽了什么救回来的"。
_RESIDUAL_KINDS: tuple[tuple[str, str, Callable[[str], str]], ...] = (
    ("punct_spacing", "折叠标点前空白", _tol_close_punct_space),
    ("quotes", "统一弯/直引号", _tol_straighten_quotes),
    ("citation_markers", "剥离内联引用标号", _tol_markers),
    ("combined", "组合应用多条残留规则", _tol_combined),
)
_RESIDUAL_LABELS: dict[str, str] = {kind: label for kind, label, _fn in _RESIDUAL_KINDS}


def quote_match(quote: str, chunk_text: str) -> str:
    """判"引文逐字在不在这块里"，返回通过路径；空串 = 两层都救不回来 = 真找不到。

    - `"exact"`：现有口径（折叠空白 + 小写）直接命中，什么都没放宽；
    - `"residual:<类>"`：精确比较失败后、靠某类抽取残留容忍才命中（类码进库当审计痕迹）；
    - `""`：引文为空，或两段都判不出 —— 后者仍按"未找到原文支撑"处理，闸门不松。
    """
    q0 = normalize_text(quote)
    if not q0:
        return ""
    c0 = normalize_text(chunk_text)
    if q0 in c0:
        return MATCH_EXACT
    for kind, _label, fn in _RESIDUAL_KINDS:
        # ★ 归一化后引文变空串时**必须停下**，不许再判包含：`"" in 任意串` 恒为真，
        # 于是"整条引文就是一个内联标号"（模型填占位值时的真实形态，如 `[58]`、`[1][2]`）
        # 会匹配上**任意**一块原文 —— 方向恰好是最危险的那种假阳性：编造/占位引文被判通过，
        # 重锚还会把它挂到该篇第一个片段上。这一道闸门与"精确比较"的 `if not q0` 同源。
        normalized_quote = fn(q0)
        if normalized_quote and normalized_quote in fn(c0):
            return _MATCH_RESIDUAL_PREFIX + kind
    return ""


def residual_kind(match: str) -> str:
    """`"residual:citation_markers"` → `"citation_markers"`；非容忍路径返回空串。"""
    head, sep, tail = match.partition(":")
    return tail if sep and head == _MATCH_RESIDUAL_PREFIX.rstrip(":") else ""


def residual_note(match: str) -> str:
    """容忍路径通过时写进 verify_note（→ citations.note）的审计痕迹，口径只在这一处定义。"""
    kind = residual_kind(match)
    label = _RESIDUAL_LABELS.get(kind, kind)
    return f"引文在容忍 PDF/HTML 抽取残留后逐字命中（{label}）"


def node_event(event_type: str, **data: Any) -> dict[str, Any]:
    """构造一条放进 `state["events"]` 的事件（由 SSEEmitter 转成 SSE 帧）。"""
    return {"type": event_type, **data}


def usage_of(message: Any) -> dict[str, int]:
    """从 AIMessage 里抠出 token 用量，归一成 `prompt_tokens` / `completion_tokens`。

    兼容两条来源：`usage_metadata`（langchain-core 1.x 标准位）与
    `response_metadata.token_usage`（部分 provider 只给这个）。
    """
    um = getattr(message, "usage_metadata", None) or {}
    if um:
        return {
            "prompt_tokens": int(um.get("input_tokens", 0) or 0),
            "completion_tokens": int(um.get("output_tokens", 0) or 0),
            "total_tokens": int(um.get("total_tokens", 0) or 0),
        }
    rm = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
    if rm:
        return {
            "prompt_tokens": int(rm.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(rm.get("completion_tokens", 0) or 0),
            "total_tokens": int(rm.get("total_tokens", 0) or 0),
        }
    return {}


def merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    """把若干次调用的用量并成一个（同 key 相加）。"""
    out: dict[str, int] = {}
    for u in usages:
        for k, v in (u or {}).items():
            out[k] = out.get(k, 0) + int(v or 0)
    return out


async def _structured_call(
    schema: type[BaseModel],
    messages: list[BaseMessage],
    temperature: float | None,
) -> tuple[Any, dict[str, int]]:
    """**裸**的结构化调用（不带任何容错）。抽出来只为让"被保护的东西"边界清楚。

    走 `ainvoke_structured_raw` 而不是在这里直接 `get_llm(...).with_structured_output(...)`：
    端点支持哪一种结构化实现是**llm 层要自己探明的事**（实测 DeepSeek 思考模式模型对默认实现
    回 400），放在节点这一层就等于让每个调用点各自决定用哪种方式 ——
    漏一处的症状是"四个节点里只有一个能跑通"，而其余三个每次提问都失败。

    `make_llm` 必须显式传：容错/熔断的用例是靠 patch **本模块的 `get_llm`** 来注入失败的。
    少了这一句，那些 patch 落不到实处，用例不会变红而是**真的去打远端 API**
    （断言"抛 TransientError"却拿到模型的正常返回，报 DID NOT RAISE，看着像容错坏了）。
    """
    result = await ainvoke_structured_raw(
        schema, messages, temperature, make_llm=lambda: get_llm(temperature)
    )
    parsed = result.get("parsed")
    raw = result.get("raw")
    return parsed, usage_of(raw)


async def _text_call(
    messages: list[BaseMessage],
    temperature: float | None,
    on_delta: Callable[[str], None] | None = None,
    on_abort: Callable[[BaseException], None] | None = None,
) -> tuple[str, dict[str, int]]:
    """**裸**的自由文本调用；给了 `on_delta` 就走真流式（逐增量回调）。

    两条路径的**返回值口径完全一致**（`(全文, usage)`），所以调用方不需要区分：
    流式只是顺带把增量送出去，不改变返回形状、也不改变失败语义 —— 异常照原样往上抛，
    外层的重试 / 超时 / 熔断继续生效（`guarded` 包的正是这个函数本身）。

    `on_abort` 只在"**已经吐过增量、这次尝试却又失败**"时调用：这是流式与重试唯一
    真正打架的地方（重试会从头重新生成一遍），处置口径见 `TokenSink` 的 docstring。
    超时是 `asyncio.wait_for` 取消这次调用，异常同样在这个函数里被感知到。
    """
    if on_delta is None:
        message: AIMessage = await get_llm(temperature).ainvoke(messages)
        content = message.content
        text = content if isinstance(content, str) else str(content)
        return text, usage_of(message)

    parts: list[str] = []
    streamed_chars = 0
    merged: Any = None
    try:
        async for chunk in get_llm(temperature).astream(messages):
            # 用 `+` 归并而不是自己拼字符串：langchain-core 的 chunk 相加会把
            # `usage_metadata` / `response_metadata` 一起并进来，usage 才是拿得到的前提。
            merged = chunk if merged is None else merged + chunk
            delta = chunk_text(chunk)
            if not delta:
                continue
            parts.append(delta)
            streamed_chars += len(delta)
            on_delta(delta)
    except BaseException as exc:  # noqa: BLE001 - 原样上抛，只在"已经吐过字"时补一次作废通知
        if streamed_chars and on_abort is not None:
            on_abort(exc)
        raise
    return "".join(parts) or chunk_text(merged), usage_of(merged)


def _on_circuit_open(breaker_name: str, exc: CircuitOpenError) -> None:
    """熔断拒绝请求时：给前端一条明确的 SSE `error`（带着可识别的 code），然后照原样抛。

    为什么"抛"也算降级：LLM 是**必需**依赖，没有合法的降级产物
    （伪造一份报告会直接违反"每条引用都能回到原文"的产品承诺）。
    这里降的是**失败的速度与可解释性** —— 用户立刻看到"下游已被熔断"，
    而不是等三个 60s 超时后收到一句含糊的 internal_error。
    """
    emit_out_of_band(
        "error",
        node=RESILIENCE_NODE,
        code=CIRCUIT_OPEN_CODE,
        message=str(exc)[:300],
        recoverable=True,  # 冷却结束后半开试探可能成功，稍后重试是有意义的
        breaker=breaker_name,
    )


async def structured_invoke(
    schema: type[BaseModel],
    messages: list[BaseMessage],
    *,
    temperature: float | None = None,
) -> tuple[Any, dict[str, int]]:
    """调 LLM 并返回 `(结构化结果, token 用量)`，全程走「重试 → 超时 → 熔断」。"""
    try:
        return await guarded(
            _structured_call,
            schema,
            messages,
            temperature,
            breaker_name=BREAKER_LLM,
            observer=sse_observer,
        )
    except CircuitOpenError as exc:
        _on_circuit_open(BREAKER_LLM, exc)
        raise


async def text_invoke(
    messages: list[BaseMessage],
    *,
    temperature: float | None = None,
    on_delta: Callable[[str], None] | None = None,
    on_abort: Callable[[BaseException], None] | None = None,
) -> tuple[str, dict[str, int]]:
    """调 LLM 并返回 `(文本, token 用量)`，保护口径与 `structured_invoke` 一致。

    传 `on_delta` 时走**真流式**（底层 `llm.astream`，每来一个增量回调一次），
    返回值与不传时逐字同口径。要清楚它换来的是什么、代价是什么：

    - 换到的：`token` 事件带的是**生成过程中真实产生的文本片段**，不是事后切片假流；
    - 代价：**重试会从头重新生成**。第一次尝试已经吐过字又失败时，
      同一段内容会第二次进入用户的预览（`token` 契约只有 `delta` 拼接语义，
      没有"重置"字段，也不该为了它再改一轮前端）。
      所以这里的原则是"**只让没失败过的那次尝试往外吐字**"：
      作废由 `on_abort` 明确告知前端，之后的尝试不再流（正文仍完整返回，
      最终报告的正确性由 `final_report` 事件兜住）。实现见 `TokenSink`。
    """
    try:
        return await guarded(
            _text_call,
            messages,
            temperature,
            on_delta,
            on_abort,
            breaker_name=BREAKER_LLM,
            observer=sse_observer,
        )
    except CircuitOpenError as exc:
        _on_circuit_open(BREAKER_LLM, exc)
        raise


# ---------------------------------------------------------------------------
# 真流式的落地：把节点内部的生成增量送上 SSE 流
# ---------------------------------------------------------------------------
def chunk_text(chunk: Any) -> str:
    """从一次（可能是流式）返回里取出**文本增量**。

    三种形态都要照顾到：纯 `str`、内容块列表（`[{"type": "text", ...}]`，
    langchain-core 1.x 的标准表示）、以及非消息对象。`.text` 属性已经把前两种归一好了，
    拿不到时才退回 `content` —— 顺序很关键，直接用 `str(content)` 会把
    `[{'type': 'text', 'text': '你好'}]` 这种 Python 字面量吐给用户。
    """
    if chunk is None:
        return ""
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return "" if content is None else str(content)


class TokenSink:
    """把一次 LLM 生成的**真实增量**送进当前这条 SSE 流。

    ## 为什么走 LangGraph 的 `custom` 通道，而不是现成的 `emit_out_of_band`
    侧信道队列是 `stream()` **每收到一个 chunk 之后**才 drain 一次的，
    而 `stream_mode="updates"` 只有等整个节点跑完才给 chunk ——
    于是 write 期间攒下的 token 会全部憋到生成结束才一次性落地，
    用户看到的是"卡半分钟然后整篇出现"，等于没流式。
    `custom` 通道（`get_stream_writer()`）是 LangGraph 专为"节点内部往上冒增量"留的口子，
    **写入即到达**；`app/observability/events.py` 的 `stream()` 负责把 payload 翻成 SSE 帧，
    所以协议仍然只在 events.py 一处（节点侧照旧只写 `{"type": ..., **data}` 裸 dict）。

    ## ★ 与"失败重试"的冲突怎么解（这个类一半的存在理由）
    `guarded()` 的重试包在整个生成外面，一次尝试失败会**从头重新生成**。
    若第一次已经吐了 N 字、第二次再吐一遍，前端 `streamed += delta` 拼出来的就是重复内容
    —— 而它没有任何"清空预览"的入口。取舍如下（宁少流、不重流）：
    1. 没吐过字的失败尝试**不作废**，下一次尝试继续正常流（最常见的失败发生在这之前）；
    2. 已经吐过字又失败 → 补发一条 `error`（`code=token_stream_aborted`,
       `recoverable=True`）明确告知"已流出的内容作废，请以最终报告为准"，
       并把这条 sink **关闭**：后续尝试不再往外吐字；
    3. 重试语义**完全不动** —— 该重试还重试、该超时还超时、该熔断还熔断，
       变的只是"要不要把这段进程实时演给用户看"。
    """

    def __init__(self, node: str, writer: Callable[[Any], Any] | None = None) -> None:
        self.node = node
        self._writer = writer
        self.chunks_sent = 0
        self.chars_sent = 0
        self.closed = False

    @property
    def streaming(self) -> bool:
        """现在还能不能往外吐字（没有通道、或已经作废 → 都不能）。"""
        return self._writer is not None and not self.closed

    def feed(self, delta: str) -> None:
        """送出一个真实增量。作废之后静默吞掉（这就是"不重流"的执行点）。"""
        if not self.streaming or not delta:
            return
        self._push(node_event("token", node=self.node, delta=delta))
        self.chunks_sent += 1
        self.chars_sent += len(delta)

    def abort(self, exc: BaseException) -> None:
        """本次尝试已吐过字却又失败（马上要被重试从头再生）→ 声明作废并关流。

        一个字都没吐过时**不作废也不关流**：没有内容需要丢弃，
        而"让下一次尝试继续流"是这次运行能保住打字机效果的唯一机会。
        """
        if self._writer is None or self.closed or not self.chunks_sent:
            return
        self.closed = True
        self._push(
            node_event(
                "error",
                node=self.node,
                code=STREAM_ABORT_CODE,
                message=(
                    f"生成中断，正在重试：已流出的 {self.chars_sent} 字作废，"
                    f"最终内容以报告为准（{str(exc)[:160]}）"
                ),
                recoverable=True,
            )
        )

    def _push(self, event: dict[str, Any]) -> None:
        if self._writer is None:
            return
        try:
            self._writer(event)
        except Exception as exc:  # noqa: BLE001 - 事件发不出去绝不能毁掉一次生成
            logger.debug("sse.token_push_failed", node=self.node, error=str(exc)[:120])


def make_token_sink(node: str) -> TokenSink:
    """取"当前这次图执行"的 custom 流通道并包成 `TokenSink`。

    拿不到通道时（单测直接调节点函数、脚本跑 `ainvoke`、图没开 custom 流）
    返回的 sink 全程静默 no-op —— 与 `emit_out_of_band` "不在流里就只打日志"
    同一个降级口径：**发不出事件绝不许让节点失败**。
    """
    try:
        # 延迟 import：只有流式入口需要图运行时，core/observability 层都不该依赖它。
        from langgraph.config import get_stream_writer

        writer: Callable[[Any], Any] | None = get_stream_writer()
    except Exception as exc:  # noqa: BLE001 - 不在图运行上下文里（含 RuntimeError: 无 RunnableConfig）
        logger.debug("sse.no_stream_writer", node=node, error=str(exc)[:120])
        writer = None
    return TokenSink(node, writer)


# ---------------------------------------------------------------------------
# 检索结果整形
# ---------------------------------------------------------------------------
def dedup_papers(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """跨子问题去重（paper 级，口径见架构 §5.2 的"3.5 步"）。

    同一篇论文会被多个子问题命中，直接喂给 Reader 会让它的 token 消耗
    随子问题数线性膨胀，而信息量几乎不增。去重时**保留最高分**，
    并把各分支命中的 chunk 合并（不同子问题命中的 chunk 往往互补）。
    """
    buckets: dict[str, dict[str, Any]] = {}
    for paper in retrieved or []:
        aid = paper.get("arxiv_id")
        if not aid:
            continue
        cur = buckets.get(aid)
        if cur is None:
            buckets[aid] = {
                "arxiv_id": aid,
                "title": paper.get("title", ""),
                "score": float(paper.get("score", 0.0)),
                "chunks": list(paper.get("chunks") or []),
            }
            continue
        if float(paper.get("score", 0.0)) > cur["score"]:
            cur["score"] = float(paper.get("score", 0.0))
        seen = {c.get("chunk_id") for c in cur["chunks"]}
        for chunk in paper.get("chunks") or []:
            if chunk.get("chunk_id") not in seen:
                cur["chunks"].append(chunk)
                seen.add(chunk.get("chunk_id"))
    return sorted(buckets.values(), key=lambda p: (-p["score"], p["arxiv_id"]))


def prioritize_unread(
    papers: list[dict[str, Any]], read_ids: set[str] | None
) -> list[dict[str, Any]]:
    """把长期记忆里"已精读过"的论文排到队尾（产品定义 F10「下次不重复推荐」）。

    **稳定分区**：组内保持 `dedup_papers` 给出的重排序不变，只是未读整体前移。
    不重排是因为重排分是唯一有数据支撑的顺序（架构 §5.3 的消融表），
    在这里引入任何新的打分口径都会让"为什么这篇排第 6"变得无法解释。

    **不剔除**：已读论文只是让出候选位，未读不足 `read_top_k`（精读篇数）时仍会补位进来。
    "读过"不等于"与新问题无关"，把它彻底踢出去等于主动丢掉合法证据。
    """
    if not read_ids:
        return list(papers)
    unread: list[dict[str, Any]] = []
    already: list[dict[str, Any]] = []
    for paper in papers:
        target = already if str(paper.get("arxiv_id")) in read_ids else unread
        target.append(paper)
    return unread + already


def chunk_index(retrieved: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """建 `(arxiv_id, chunk_id) → chunk` 索引。

    这是**反幻觉的落点之一**（架构 §10 输出侧）：校验与写作时，只有能在
    本次检索结果里找到对应 chunk 的引用才被允许保留。
    """
    idx: dict[tuple[str, int], dict[str, Any]] = {}
    for paper in retrieved or []:
        aid = paper.get("arxiv_id")
        for chunk in paper.get("chunks") or []:
            cid = chunk.get("chunk_id")
            if aid is None or cid is None:
                continue
            idx[(aid, int(cid))] = chunk
    return idx


__all__ = [
    "CIRCUIT_OPEN_CODE",
    "MATCH_EXACT",
    "STREAM_ABORT_CODE",
    "TokenSink",
    "chunk_index",
    "chunk_text",
    "dedup_papers",
    "make_token_sink",
    "merge_usage",
    "node_event",
    "normalize_text",
    "prioritize_unread",
    "quote_match",
    "residual_kind",
    "residual_note",
    "structured_invoke",
    "text_invoke",
    "usage_of",
]
