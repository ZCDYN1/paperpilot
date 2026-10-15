"""结构化输出的端点自适应（接真实 API 时实测出来的问题）。

真实故障：`LLM_PROVIDER=openai` 指向 api.deepseek.com 的 `deepseek-flash` 之后，
每次提问都在 `plan` 节点当场 400，一个 run 都跑不出来。这个模型是**思考模式**模型，
而 plan/read/verify/revise 四个节点全靠结构化输出：

    默认实现（json_schema）  → 400 "This response_format type is unavailable now"
    json_mode               → 400 "Prompt must contain the word 'json'…"（提示词里没有 json 就整体失败）
    function_calling        → 400 "Thinking mode does not support this tool_choice"
    function_calling + 关思考模式 → 正常

用户配的是**模型**，不该再来背端点差异，所以由代码按顺序探测，而不是加配置开关。
本文件钉住探测策略的四条底线：默认实现优先、只把"形状被拒"当线索、真故障不被掩盖、
以及**调用点确实经过这里**（逻辑写对但没人调，等于没修）。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.core import llm as llm_mod


class _Schema(BaseModel):
    answer: str


class _ShapeError(Exception):
    """假装是端点回的 400：状态码 400 + 消息里带 response_format。"""

    status_code = 400


_MSGS = [HumanMessage(content="拆一下")]


class _Model:
    """假 ChatModel：按本次构造参数决定成功还是回 400。"""

    def __init__(self, accept, calls: list[dict[str, Any]]) -> None:
        self.accept = accept
        self.calls = calls

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001, ARG002
        outer = self

        class _R:
            async def ainvoke(self, messages):  # noqa: ANN001
                seen = dict(kwargs)
                outer.calls.append(seen)
                if not outer.accept(seen):
                    raise _ShapeError("Error code: 400 - response_format type unavailable")
                return {"parsed": _Schema(answer="ok"), "raw": None}

        return _R()


@pytest.fixture(autouse=True)
def _reset_recipe_cache():
    llm_mod._working_recipe = None
    yield
    llm_mod._working_recipe = None


@pytest.fixture
def patch_get_llm(monkeypatch):
    """把 `get_llm` 换成假模型，并记录每次结构化调用实际用到的参数组合。"""

    def _patch(accept):
        calls: list[dict[str, Any]] = []

        def fake_get_llm(temperature=None, **kwargs):  # noqa: ANN001, ARG001
            return _Model(accept, calls)

        monkeypatch.setattr(llm_mod, "get_llm", fake_get_llm)
        return calls

    return _patch


async def test_default_tried_first_and_no_fallback_when_it_works(patch_get_llm):
    """端点接受默认实现时，一次调用都不多花，请求形状也不变。"""
    calls = patch_get_llm(lambda record: True)
    result = await llm_mod.ainvoke_structured_raw(_Schema, _MSGS)
    assert result["parsed"].answer == "ok"
    assert len(calls) == 1
    assert "method" not in calls[0], "ollama / mock 走的这条路必须一个字都不变"


async def test_falls_back_to_function_calling(patch_get_llm):
    """默认被拒时退到 function_calling —— DeepSeek 实测里只有它可用（配合关思考模式）。"""
    calls = patch_get_llm(lambda r: r.get("method") == "function_calling")
    result = await llm_mod.ainvoke_structured_raw(_Schema, _MSGS)
    assert result["parsed"].answer == "ok"
    assert len(calls) == 2, "第一次失败、第二次成功"


async def test_working_recipe_is_cached(patch_get_llm):
    """探明的组合记进进程：否则每个节点、每条问句都要先浪费一次 400。"""
    calls = patch_get_llm(lambda r: r.get("method") == "function_calling")
    await llm_mod.ainvoke_structured_raw(_Schema, _MSGS)
    after_probe = len(calls)
    await llm_mod.ainvoke_structured_raw(_Schema, _MSGS)
    assert len(calls) == after_probe + 1, "第二次应当直接命中已探明的组合"


async def test_cached_recipe_failure_reopens_probe(patch_get_llm):
    """运行中换端点会让原本可用的组合重新被拒 —— 这时必须清缓存、回到全量探测。"""
    state = {"ok": False}
    patch_get_llm(lambda r: r.get("method") == "function_calling" and state["ok"])

    with pytest.raises(RuntimeError):
        await llm_mod.ainvoke_structured_raw(_Schema, _MSGS)
    assert llm_mod._working_recipe is None, "全部失败后不能留下一个不可用的缓存"

    state["ok"] = True
    result = await llm_mod.ainvoke_structured_raw(_Schema, _MSGS)
    assert result["parsed"].answer == "ok"


async def test_non_shape_error_is_not_masked(patch_get_llm, monkeypatch):
    """★ 超时/鉴权这类真故障不许被"再试一种写法"掩盖。

    掩盖的后果：每次调用都要把三种组合试完才报错（慢三倍、烧三次 token），
    而用户看到的是一个和真正原因无关的消息。
    """
    calls = patch_get_llm(lambda r: False)
    monkeypatch.setattr(llm_mod, "_is_request_shape_error", lambda exc: False)
    with pytest.raises(_ShapeError):
        await llm_mod.ainvoke_structured_raw(_Schema, _MSGS)
    assert len(calls) == 1, "非形状错误应当第一次就抛出，不再试其余组合"


async def test_all_recipes_failing_says_what_to_do(patch_get_llm):
    """三种都不被接受时，报错要带上最后一次真实错误，并说清为什么这是硬依赖。"""
    patch_get_llm(lambda r: False)
    with pytest.raises(RuntimeError) as exc:
        await llm_mod.ainvoke_structured_raw(_Schema, _MSGS)
    text = str(exc.value)
    assert "response_format" in text
    assert "plan/read/verify/revise" in text


def test_shape_error_detection_is_narrow():
    """判据：**状态码不是 400 就一定不算**，是 400 还要命中形状标记。

    这条是被真实事故逼出来的：鉴权失败（401）的响应体里也带 `invalid_request_error`
    这个通用类型名，早期把它当标记 → 一次 401 被当成"写法不对"，于是三种组合全试一遍，
    下游被打三次、熔断计数失真，而用户看到的错误和真正原因无关。
    """

    def _with_status(code: int, message: str) -> Exception:
        return type("E", (Exception,), {"status_code": code})(message)

    assert llm_mod._is_request_shape_error(_ShapeError("This response_format type is unavailable"))
    assert llm_mod._is_request_shape_error(_with_status(400, "does not support this tool_choice"))
    # 非 400 一律不算，哪怕消息里全是标记词
    assert not llm_mod._is_request_shape_error(_with_status(401, "invalid_request_error: response_format"))
    assert not llm_mod._is_request_shape_error(_with_status(429, "tool_choice rate limited"))
    # 400 但没命中标记（另一种坏请求）也不算
    assert not llm_mod._is_request_shape_error(_with_status(400, "temperature out of range"))
    assert not llm_mod._is_request_shape_error(TimeoutError("read timeout"))
    assert not llm_mod._is_request_shape_error(RuntimeError("401 unauthorized"))


async def test_auth_failure_is_not_mistaken_for_a_recipe_problem(monkeypatch):
    """401 必须**第一次就抛出**：把它当写法问题会让一次鉴权失败打出三次请求。"""
    calls: list[dict[str, Any]] = []

    class _Unauthorized(Exception):
        status_code = 401

    class _Denied:
        def with_structured_output(self, schema, **kwargs):  # noqa: ANN001, ARG002
            class _R:
                async def ainvoke(self, messages):  # noqa: ANN001
                    calls.append(dict(kwargs))
                    raise _Unauthorized("Authentication error: invalid_request_error")

            return _R()

    monkeypatch.setattr(llm_mod, "get_llm", lambda temperature=None, **kw: _Denied())
    with pytest.raises(_Unauthorized):
        await llm_mod.ainvoke_structured_raw(_Schema, _MSGS)
    assert len(calls) == 1, "鉴权失败不该试第二种写法"


async def test_nodes_go_through_the_probe_and_keep_the_injection_point(monkeypatch):
    """★ 光测探测逻辑不够 —— 真实事故有过两次，都出在"调用点"上。

    第一次：`_structured_call` 直接 `get_llm(...).with_structured_output(...)`，
    于是端点自适应全项目没人执行（接 DeepSeek 时每个节点照样 400），
    而本文件上面那几条用例全是绿的 —— 它们证明"逻辑正确"，没证明"有人用它"。

    第二次：改成走 llm 层之后，容错用例 patch 的 `_common.get_llm` 落不到实处，
    于是**测试直接去打真实 API**：断言抛 TransientError，拿到的却是真模型的正常返回，
    报 DID NOT RAISE，看着像容错坏了。所以这里不仅断言"经过探测"，
    还断言 `make_llm` 这个注入口由节点层交出、并被真正调用。
    """
    from app.agents.nodes import _common as common

    seen: dict[str, Any] = {}

    class _Raw:
        usage_metadata = None

    async def fake_probe(schema, messages, temperature=None, *, make_llm=None):
        seen["args"] = (schema.__name__, temperature)
        seen["model"] = make_llm() if make_llm is not None else None
        return {"parsed": _Schema(answer="ok"), "raw": _Raw()}

    marker = object()
    monkeypatch.setattr(common, "ainvoke_structured_raw", fake_probe)
    monkeypatch.setattr(common, "get_llm", lambda temperature=None: marker)

    parsed, usage = await common._structured_call(_Schema, ["msg"], None)
    assert seen["args"] == ("_Schema", None)
    assert seen["model"] is marker, "取模型必须经过节点层的 get_llm，否则测试桩与注入全部失效"
    assert parsed.answer == "ok"
    assert usage == {}


def test_thinking_off_copies_the_instance_and_leaves_the_original(monkeypatch):
    """关思考模式是**复制实例**，不是改 `get_llm` 的签名、也不是改全局配置。

    三条都要钉住：
    - 只有支持 `extra_body` 的模型（OpenAI 兼容那类）会被改；
    - 原实例不受影响 —— 正文生成共用它，被顺手改掉就是静默降低报告质量；
    - ollama / mock 原样返回，它们没有这个问题，不该被塞未知字段。
    """

    class _OpenAIish(BaseModel):
        extra_body: dict[str, Any] | None = None

    original = _OpenAIish()
    copy = llm_mod._with_thinking_off(original)
    assert copy is not original
    assert copy.extra_body == {"thinking": {"type": "disabled"}}
    assert original.extra_body is None, "原实例必须保持不动"

    from app.core.llm import MockChatModel

    plain = MockChatModel(temperature=0.1)
    assert llm_mod._with_thinking_off(plain) is plain, "ollama/mock 不该被带上未知字段"
