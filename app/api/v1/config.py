"""配置读写接口：让使用者在界面上填模型 / 接口地址 / 密钥，而不必去编辑 `.env`。

## 为什么直接写 `.env`，不加一层数据库覆盖

`Settings` 是 `pydantic-settings` + `@lru_cache`，`.env` 就是唯一真相源。
再做一层"DB 里的覆盖值"就会造出第二个真相源，而这个项目已经因为
"两处配置对不上、而且不报错"吃过一次亏（见 `app/main.py` 里 `configure_tracing()`
那段注释：`.env` 读到的 `LANGCHAIN_*` 不会自动进 `os.environ`，于是
"我填了 key"和"真的有 trace"之间静默断链）。所以这里只写 `.env` 本身。

## 安全：这是一个会把密钥明文落盘、且默认无鉴权的写接口

`api_host` 默认是 `0.0.0.0`，所以"只有本机能碰到"并不成立。三层缓解：

1. **同源 + Host 白名单**（见 `_check_origin`）：挡跨站 POST 与 DNS rebinding。
   前端与 API 同源部署（`app/main.py` 的 `_mount_web_ui`），所以正常请求一定通过。
2. **密钥永不下发明文**：`GET` 只给 `{set, masked}`。表单里留空提交 = 不修改。
3. **总开关** `CONFIG_WRITABLE`（默认 true）。设成 false 时 `PUT` 一律 403，
   `GET` 回 `writable: false`，前端自动变只读。
   它是 `Settings` 上的一个字段（`app/core/config.py`），所以 `.env` 与进程环境变量
   两种设法都算数 —— 早期这里读的是 `os.environ`，而 `.env` 的值并不会进进程环境，
   结果是"照文档在 `.env` 里关掉"根本不生效。
   `POST /test` **不受这个开关限制** —— 它不落盘、只发一次只读探测，而只读模式下
   它反而是使用者唯一能验证自己配置对不对的入口。

默认开着是权衡不是疏忽：这是本地优先的工具，"clone 就能配"优先于
"默认锁死"。要彻底关掉就设 `CONFIG_WRITABLE=false`。

## 为什么 `options` 要从 Settings 的 Literal 反射

`llm_provider` 之类可选值定义在 `app/core/config.py` 的 `Literal[...]` 里。
接口里再手抄一份，等于以后加一个 provider 就有两处要改，而前端下拉会静默漏项。
mock 数据就犯过一次：给 `rerank_provider` 多写了 `"api"`，后端其实只有 `local|lexical`。
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from typing import Annotated, Any, get_args
from urllib.parse import urlsplit

import structlog
from fastapi import Body, HTTPException, Request
from pydantic import ConfigDict, create_model
from pydantic.fields import FieldInfo

from app.core.config import REPO_ROOT, LLMProvider, Settings, get_settings

logger = structlog.get_logger(__name__)

ENV_PATH = REPO_ROOT / ".env"

#: 界面上允许改的字段。不在这里的键一律拒绝，避免这个接口变成"任意写 .env"。
EDITABLE_ALL: tuple[str, ...] = (
    "llm_provider",
    "llm_model",
    "llm_temperature",
    "openai_base_url",
    "openai_api_key",
    "ollama_base_url",
    "ollama_model",
    "embedding_provider",
    "embedding_model",
    "embedding_api_base_url",
    "embedding_api_key",
    "rerank_provider",
    "rerank_model",
    "min_corpus",
)


def _in_settings(name: str) -> bool:
    return name in Settings.model_fields


#: 真正生效的白名单 = 上面那份 ∩ `Settings` 里确实存在的字段。
#:
#: 刻意**不在启动时抛错**：这个模块只是配置表单，为一个字段没跟上就让整个 API
#: 起不来（检索、研究、评测全跟着停）是不成比例的代价 —— 而这种情况真实发生过：
#: 并发改动 `app/core/config.py` 时把两个字段带掉了，导入期的断言当场把产品打死。
#: 缺的字段现在只是不出现在表单里，并在启动日志里留一行。
EDITABLE: tuple[str, ...] = tuple(n for n in EDITABLE_ALL if _in_settings(n))

_MISSING_FIELDS = sorted(set(EDITABLE_ALL) - set(EDITABLE))
if _MISSING_FIELDS:  # pragma: no cover - 只在配置字段漂移时走到
    logger.warning("config.editable_fields_missing", fields=_MISSING_FIELDS)


#: 密钥字段：空串表示"保持原值"，不是"清空"。清空要显式传这个哨兵。
SECRET_FIELDS = ("openai_api_key", "embedding_api_key")
CLEAR_SECRET = "__CLEAR__"

#: 改了这些要重启进程才真正生效。`GET /config` 会回这份，前端必须显示出来 ——
#: "点了保存、界面也变新了、实际还在用旧 provider"是最容易被当成 bug 的静默失效。
RESTART_REQUIRED = (
    "llm_provider",
    "embedding_provider",
    "embedding_model",
    "rerank_provider",
    "rerank_model",
    "min_corpus",
)

#: 允许访问写接口的 Host（本机场景）。额外值用 `CONFIG_ALLOWED_HOSTS` 逗号分隔追加。
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

#: 连通性探测只对 LLM 有意义（embedding/rerank 没有对应的探测端点）。
#: 取 `LLMProvider` 的 Literal 而不是再手抄一份 —— 加 provider 时只有一处要改。
_TESTABLE_PROVIDERS = get_args(LLMProvider)


# --------------------------------------------------------------------- 模型构造
def _field(name: str) -> FieldInfo:
    f = Settings.model_fields.get(name)
    if f is None:  # 配置字段被改名而这里没跟上 —— 宁可启动就炸，不要静默少一个字段
        raise RuntimeError(f"Settings 里没有字段 {name!r}，请同步 app/api/v1/config.py")
    return f


def _patch_model():
    """从 `Settings.model_fields` 派生校验模型。

    这样 provider 的合法取值、数值范围都只有 `config.py` 一处定义。
    全部字段可选（`None` = 本次不改），`extra="forbid"` 拒掉白名单外的键。
    """
    defs: dict[str, Any] = {}
    for name in EDITABLE:
        # 刻意不取 Settings 的真实默认值：patch 是"部分更新"，
        # 拿默认值当缺省会让人无法把字段改回默认值。
        defs[name] = (Annotated[_field(name).annotation, None], None)
    return create_model("ConfigPatch", **defs, __config__=ConfigDict(extra="forbid"))


PatchModel = _patch_model()


# --------------------------------------------------------------------- .env 读写
def _dotenv_keys() -> set[str]:
    if not ENV_PATH.is_file():
        return set()
    keys = set()
    for line in ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if m:
            keys.add(m.group(1).lower())
    return keys


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:3]}…{value[-4:]}"


def _sources() -> dict[str, str]:
    """每个字段当前值从哪来。

    优先级是 pydantic-settings 的真实顺序：进程环境变量 > .env > 代码默认值。
    把这一层摊开是给使用者看的 —— "我明明改了 .env 怎么没生效"的答案通常就是
    这里冒出来的 `env`。
    """
    dotenv = _dotenv_keys()
    env_upper = {k.upper() for k in os.environ}
    out: dict[str, str] = {}
    for name in EDITABLE:
        if name.upper() in env_upper:
            out[name] = "env"
        elif name in dotenv:
            out[name] = "dotenv"
        else:
            out[name] = "default"
    return out


def _write_env(updates: dict[str, str]) -> str | None:
    """按行更新 `.env`：只替换目标键，保留注释、空行与所有无关条目。

    不整体重写是刻意的 —— 这个文件里有大量带说明注释的行，
    `"\n".join(f"{k}={v}")` 那种写法会把用户的注释和手动加的键全冲掉。
    返回备份文件名；没有旧文件时返回 None。
    """
    original = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.is_file() else ""
    lines = original.splitlines()

    backup_name: str | None = None
    if lines:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_name = f".env.bak-{stamp}"
        (REPO_ROOT / backup_name).write_text(original, encoding="utf-8")

    remaining = dict(updates)
    pattern = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*)=")
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if not m:
            continue
        key = m.group(2)
        if key.upper() in remaining:
            lines[i] = f"{m.group(1)}{key}{m.group(3)}={remaining.pop(key.upper())}"

    if remaining:
        if lines and lines[-1].strip():
            lines.append("")
        for key, value in remaining.items():
            lines.append(f"{key}={value}")

    # 临时文件 + os.replace：中途崩了也不会把用户的 .env 写成半截
    fd, tmp = tempfile.mkstemp(dir=str(REPO_ROOT), prefix=".env.tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, ENV_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return backup_name


# --------------------------------------------------------------------- 安全闸门
def _writable() -> bool:
    return get_settings().config_writable


def _check_origin(request: Request) -> None:
    """同源 + Host 白名单。

    这个接口会把密钥写进磁盘，而 `api_host` 默认 `0.0.0.0`：没有这道闸门时，
    任意网页都能对一个跑在本机 8000 上的服务发 POST 来塞配置或换走 provider。
    """
    host = (request.headers.get("host") or "").split(":")[0].strip("[]").lower()
    extra = get_settings().config_allowed_hosts
    allowed = {h.strip().lower() for h in extra.split(",") if h.strip()}
    if host not in _LOCAL_HOSTS | allowed:
        raise HTTPException(status_code=403, detail="只允许本机访问配置写接口")

    for header in ("origin", "referer"):
        value = request.headers.get(header)
        if not value:
            continue
        origin_host = (urlsplit(value).netloc or "").split(":")[0].strip("[]").lower()
        if origin_host and origin_host != host:
            raise HTTPException(status_code=403, detail="跨站请求被拒绝")


# --------------------------------------------------------------------- 路由
from fastapi import APIRouter  # noqa: E402  （放在常量之后，避免 router 定义早于上面的辅助函数）

router = APIRouter(prefix="/config", tags=["config"])


@router.get("", summary="读取生效配置（密钥只回掩码）")
async def read_config() -> dict[str, Any]:
    settings = get_settings()
    values: dict[str, Any] = {}
    secrets: dict[str, dict[str, Any]] = {}

    for name in EDITABLE:
        raw = getattr(settings, name)
        if name in SECRET_FIELDS:
            text = str(raw or "")
            secrets[name] = {"set": bool(text), "masked": _mask(text)}
            values[name] = ""  # 绝不下发明文
        else:
            values[name] = raw

    return {
        "values": values,
        "secrets": secrets,
        "sources": _sources(),
        "writable": _writable(),
        "restart_required": list(RESTART_REQUIRED),
        "options": _options(),
    }


def _options() -> dict[str, list[str]]:
    """从 Settings 的 Literal 反射可选值，不在这里手抄。"""
    return {
        name: [str(v) for v in get_args(_field(name).annotation)]
        for name in ("llm_provider", "llm_structured_method", "embedding_provider", "rerank_provider")
        if _in_settings(name)  # 字段不在 Settings 里就整个不报，别给前端一个空列表
    }


@router.put("", summary="写入配置（落到 .env，保留注释与无关键）")
async def write_config(request: Request, body: Annotated[dict, Body()]) -> dict[str, Any]:
    _check_origin(request)
    if not _writable():
        raise HTTPException(status_code=403, detail="CONFIG_WRITABLE=false，配置为只读")

    patch = body.get("values") if isinstance(body, dict) else None
    if not isinstance(patch, dict) or not patch:
        raise HTTPException(status_code=422, detail={"code": "empty_patch", "reason": "values 不能为空"})

    unknown = sorted(set(patch) - set(EDITABLE))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"code": "unknown_field", "fields": unknown, "allowed": list(EDITABLE)},
        )

    # 密钥留空 = 不改；哨兵 = 清空
    updates: dict[str, str] = {}
    rejected: list[dict[str, str]] = []
    for name, value in patch.items():
        if name in SECRET_FIELDS:
            if value is None or value == "":
                continue
            updates[name.upper()] = "" if value == CLEAR_SECRET else str(value)
            continue
        if value is None:
            continue
        # 布尔要写成 `true`/`false` 而不是 Python 的 `True`/`False`：
        # 这个文件是人也会打开看的，而且小写是这个项目其余布尔配置的统一写法
        updates[name.upper()] = ("true" if value else "false") if isinstance(value, bool) else str(value)

    # 用派生模型校验取值（provider 白名单、数值范围都来自 config.py 那一处定义）
    try:
        PatchModel(**{k.lower(): v for k, v in updates.items()})
    except Exception as exc:  # noqa: BLE001 - 逐字段回给前端，而不是整批 422
        for err in getattr(exc, "errors", lambda: [])():
            loc = ".".join(str(p) for p in err.get("loc", ())) or "values"
            rejected.append({"field": str(loc), "reason": str(err.get("msg", "取值不合法"))})
        if rejected:
            return {
                "ok": False,
                "updated": [],
                "restart_required": [],
                "backup": None,
                "rejected": rejected,
                "shadowed": [],
            }
        raise

    backup = _write_env(updates)
    get_settings.cache_clear()

    changed = sorted(k.lower() for k in updates)
    # ★ 只记字段名，绝不记值：这些值里可能有 API key，进日志就等于明文外泄
    logger.info("config.updated", fields=changed, backup=backup)

    return {
        "ok": True,
        "updated": changed,
        "restart_required": [f for f in changed if f in RESTART_REQUIRED],
        "backup": backup,
        "rejected": [],
        # 写进去了，但进程环境变量优先级更高，所以**当前生效值不是刚写的这个**，
        # 而且重启也改不过来（除非 unset）。不回报的话，用户看到的就是一张
        # "保存成功"的绿卡和一个仍在用旧值的系统 —— 这是最难查的那种静默失效。
        "shadowed": [f for f in changed if f.upper() in os.environ],
    }


@router.post("/test", summary="探测 provider 连通性（保存前先确认填对了）")
async def test_connectivity(request: Request, body: Annotated[dict, Body()]) -> dict[str, Any]:
    """探测 LLM 服务是否连得上 —— 地址与密钥**可以随请求带上、不必先保存**。

    这条接口的价值是"填完立刻知道对不对"：错配置的代价在别处是几分钟
    （跑一次研究才发现连不上），在这里是一次 HTTP。
    """
    _check_origin(request)

    provider = str((body or {}).get("provider") or "")
    if provider not in _TESTABLE_PROVIDERS:
        raise HTTPException(
            status_code=422,
            detail={"code": "unknown_provider", "provider": provider, "allowed": list(_TESTABLE_PROVIDERS)},
        )
    if provider == "mock":
        return {"ok": True, "provider": provider, "detail": "mock provider 不需要连通性检查。"}

    settings = get_settings()
    model = str((body or {}).get("model") or "").strip()

    if provider == "ollama":
        base = _or_default(body.get("ollama_base_url"), settings.ollama_base_url).rstrip("/")
        result = await _probe(f"{base}/api/tags", None, "ollama")
        if result.get("ok") and model:
            # Ollama 的 /api/tags 返回的**就是调用名**，所以这里比对列表是可靠的
            result["model_check"] = _check_by_list(model, result.get("models") or [])
        return result

    base = _or_default(body.get("openai_base_url"), settings.openai_base_url).rstrip("/")
    key, why_missing = _resolve_probe_key(settings, base, str(body.get("openai_api_key") or ""))
    if not key:
        return {"ok": False, "provider": provider, "detail": why_missing}
    result = await _probe(f"{base}/models", key, "openai")
    if result.get("ok") and model:
        result["model_check"] = await _check_by_call(base, key, model)
    return result


def _check_by_list(model: str, listed: list[str]) -> dict[str, Any]:
    if model in listed:
        return {"checked": True, "ok": True, "detail": f"模型 {model} 在这个服务的列表里。"}
    return {
        "checked": True,
        "ok": False,
        "detail": f"模型 {model} 不在这个服务的列表里，先 ollama pull 一个或者换成列表里的名字。",
    }


async def _check_by_call(base: str, api_key: str, model: str) -> dict[str, Any]:
    """用一次 `max_tokens=1` 的真实请求判定模型名能不能用。

    ★ 为什么不能靠比对 `/models`：OpenAI 兼容网关（DeepSeek 就是）在那里返回的是
    **展示名**（`DeepSeek-V4.1-Flash`），而调用要用的 id 是另一个写法（`deepseek-flash`）。
    拿列表当权威的结果是：配置明明能用，界面却说"你的模型名不对"——
    一个刚上线的按钮第一分钟就在骗第一个使用者，这比没有这个按钮更糟。
    代价是每次探测多一个 token 的补全请求，换来的是"能"和"不能"这两个字有依据。
    """
    import httpx

    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{base}/chat/completions", json=payload, headers=headers)
    except Exception as exc:  # noqa: BLE001 - 异常原文可能带 URL/凭据，不外传
        logger.warning("config.model_check_failed", provider="openai", error=str(exc)[:200])
        return {"checked": False, "ok": False, "detail": f"没能完成模型名校验（{type(exc).__name__}）。"}

    if resp.status_code == 200:
        return {"checked": True, "ok": True, "detail": f"模型 {model} 可以正常调用（已用一次真实请求验证）。"}
    if resp.status_code in {400, 404}:
        return {
            "checked": True,
            "ok": False,
            "detail": f"这个服务不认识模型名 {model}（返回 {resp.status_code}）。换成它支持的模型名再试。",
        }
    return {
        "checked": False,
        "ok": False,
        "detail": f"校验请求返回 {resp.status_code}，无法据此判断模型名对不对（更可能是地址或密钥问题）。",
    }


def _or_default(value: object, fallback: str) -> str:
    """表单里"留空 = 不改"，所以空值要退回后端当前生效的那个值。"""
    text = str(value or "").strip()
    return text or fallback


def _host_of(url: str) -> str:
    return (urlsplit(url).netloc or "").split(":")[0].strip("[]").lower()


def _resolve_probe_key(settings: Settings, base_url: str, typed_key: str) -> tuple[str, str]:
    """决定这次探测带哪把 key。

    ★ 这里必须做一次同源检查，否则接口就是一个**密钥外送器**：调用方能传一个
    `openai_base_url=http:// attacker`，而服务端会兴冲冲地把 `.env` 里那把真 key
    当 Bearer 发过去。跨站已经被 `_check_origin` 挡住，但"同源页面能任意指定探测
    目标"这条一直开着，所以规则是：只把已存的 key 发往**已存的那个网关**。
    要试别的地址，就自己把那个地址的 key 一起传过来（用一次即丢，不落盘）。
    """
    typed = typed_key.strip()
    if typed and typed != CLEAR_SECRET:
        return typed, ""

    stored = settings.openai_api_key
    if not stored:
        return "", "还没有填 API Key（填了但没保存也算未填）。"

    if _host_of(base_url) != _host_of(settings.openai_base_url):
        return "", (
            "已经存过 API Key，但这次要探测的地址和你配置的网关不是同一个主机。"
            "为避免把你的密钥发往别处，这里不会带着它去探测 —— "
            "要测那个服务，请在密钥框里填它自己的 key。"
        )
    return stored, ""


async def _probe(url: str, api_key: str | None, provider: str) -> dict[str, Any]:
    """打一次只读探测。失败时只回状态码与一句人话，**不回显 key 或完整 URL 查询串**。"""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(url, headers=headers)
    except Exception as exc:  # noqa: BLE001 - 网络异常的原文可能带 URL/凭据，不外传
        logger.warning("config.probe_failed", provider=provider, error=str(exc)[:200])
        return {
            "ok": False,
            "provider": provider,
            "detail": f"连不上目标服务（{type(exc).__name__}）。检查地址、服务是否在跑、以及网络可达性。",
        }

    if resp.status_code != 200:
        return {
            "ok": False,
            "provider": provider,
            "detail": f"服务返回 {resp.status_code}。地址对但凭据或路径可能不对。",
        }

    names: list[str] = []
    try:
        data = resp.json()
        for item in (data.get("models") or data.get("data") or [])[:50]:
            if not isinstance(item, dict):
                continue
            # Ollama 用 name，OpenAI 及其兼容网关用 id —— 只认 name 时 DeepSeek 之类
            # 每次都回一个空列表，前端就会显示"没有模型可核对"，等于这条接口白做。
            name = item.get("name") or item.get("id")
            if name:
                names.append(str(name))
    except Exception:  # noqa: BLE001 - 解析失败不影响"能连上"这个结论
        pass

    return {"ok": True, "provider": provider, "detail": "连接成功。", "models": names}
