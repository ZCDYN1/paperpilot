"""`GET/PUT /api/v1/config` 与连通性探测接口的对外契约。

★ 这一整份测试都跑在 `tmp_path` 里造的假 `.env` 上：`ENV_PATH` 与 `REPO_ROOT`
被 monkeypatch 掉，所以**绝不会碰到开发者机器上真实的 `.env`**。
写接口的测试如果不这样隔离，就是在拿用户配置当试验田。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

import app.db.models  # noqa: F401 - 注册表到 metadata
from app.api.v1 import config as config_module
from tests.api_support import api_client

FAKE_ENV = """# PaperPilot 配置（测试用）
# 这行注释必须活下来

LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b

# 不属于本表单的键，一个字都不许动
LANGCHAIN_PROJECT=paperpilot
OPENAI_API_KEY=sk-real-secret-value-1234
"""


@asynccontextmanager
async def _client(monkeypatch, tmp_path, **extra_env):
    env = tmp_path / ".env"
    env.write_text(FAKE_ENV, encoding="utf-8")
    # ENV_PATH / REPO_ROOT 只在这个模块里用作路径常量，patch 它们不碰 app 的加载逻辑
    monkeypatch.setattr(config_module, "ENV_PATH", env)
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    async with api_client(monkeypatch, tmp_path, **extra_env) as c:
        yield c


@pytest.fixture
async def client(monkeypatch, tmp_path):
    async with _client(monkeypatch, tmp_path) as c:
        yield c


@pytest.fixture
async def client_with_secret(monkeypatch, tmp_path):
    """进程里真的有一个可用 key 的客户端 —— 用来验证"生效值"不会被下发。"""
    async with _client(monkeypatch, tmp_path, OPENAI_API_KEY="sk-real-secret-value-1234") as c:
        yield c


def _same_origin() -> dict[str, str]:
    # httpx 的 ASGI 传输默认 Host 是 testserver，放进白名单等价于"本机"
    return {"Host": "127.0.0.1:8000", "Origin": "http://127.0.0.1:8000"}


# ------------------------------------------------------------------ GET
async def test_get_never_leaks_plaintext_secret(client_with_secret):
    """进程里真的有 key 时，接口也只回掩码。

    ★ 用环境变量而不是假 `.env` 注入：`set` 反映的是**生效值**（来自 Settings），
    而 pydantic-settings 读的是仓库那个真 `.env`，测试里 patch 掉的 ENV_PATH 不会
    进入它的加载路径。拿 tmp 文件断言"生效"会永远得到空值 —— 那不是接口的问题，
    是断言问错了对象。
    """
    r = await client_with_secret.get("/api/v1/config")
    assert r.status_code == 200
    body = r.json()
    assert body["values"]["openai_api_key"] == ""
    assert "sk-real-secret-value-1234" not in r.text
    assert body["secrets"]["openai_api_key"]["set"] is True
    assert "…" in body["secrets"]["openai_api_key"]["masked"]
    # 被进程环境变量压住的密钥，来源必须标成 env —— 这就是"改了 .env 没生效"的答案
    assert body["sources"]["openai_api_key"] == "env"
    # 掩码只留头 3 位 + 尾 4 位，中间必须断掉
    masked = body["secrets"]["openai_api_key"]["masked"]
    assert "real" not in masked and len(masked) < len("sk-real-secret-value-1234")


async def test_get_options_are_reflected_from_settings_literals(client):
    """provider 可选值必须来自 `config.py` 的 Literal，不能是接口里手抄的第二份。"""
    from app.core.config import Settings

    body = (await client.get("/api/v1/config")).json()
    for name in ("llm_provider", "embedding_provider", "rerank_provider"):
        from typing import get_args

        assert set(body["options"][name]) == set(get_args(Settings.model_fields[name].annotation))
    # 回归：mock 曾给 rerank 多写一个 "api"，后端其实只有 local|lexical
    assert "api" not in body["options"]["rerank_provider"]


async def test_get_reports_where_each_value_comes_from(client):
    body = (await client.get("/api/v1/config")).json()
    # api_client 会往环境里塞 LLM_PROVIDER，所以它是 env 而不是 dotenv —— 这正是
    # "改了 .env 却没生效"的成因，界面必须能把它摊开
    assert body["sources"]["llm_provider"] == "env"


# ------------------------------------------------------------------ PUT
async def test_put_preserves_comments_and_unrelated_keys(client):
    r = await client.put("/api/v1/config", json={"values": {"llm_model": "qwen2.5:14b"}}, headers=_same_origin())
    assert r.status_code == 200
    assert r.json()["ok"] is True

    text = config_module.ENV_PATH.read_text(encoding="utf-8")
    assert "这行注释必须活下来" in text
    assert "LANGCHAIN_PROJECT=paperpilot" in text
    assert "OPENAI_API_KEY=sk-real-secret-value-1234" in text
    assert "LLM_MODEL=qwen2.5:14b" in text


async def test_put_rejects_fields_outside_the_whitelist(client):
    r = await client.put("/api/v1/config", json={"values": {"api_host": "0.0.0.0"}}, headers=_same_origin())
    assert r.status_code == 422
    assert "api_host" in r.text


async def test_put_rejects_invalid_provider_without_writing(client):
    before = config_module.ENV_PATH.read_text(encoding="utf-8")
    r = await client.put("/api/v1/config", json={"values": {"rerank_provider": "api"}}, headers=_same_origin())
    assert r.status_code == 200  # 逐字段回报，而不是整批 500
    payload = r.json()
    assert payload["ok"] is False
    assert payload["rejected"]
    assert config_module.ENV_PATH.read_text(encoding="utf-8") == before


async def test_empty_secret_means_unchanged_not_cleared(client):
    await client.put("/api/v1/config", json={"values": {"openai_api_key": ""}}, headers=_same_origin())
    assert "OPENAI_API_KEY=sk-real-secret-value-1234" in config_module.ENV_PATH.read_text(encoding="utf-8")


async def test_backup_is_written_and_reported(client):
    r = await client.put("/api/v1/config", json={"values": {"min_corpus": 5}}, headers=_same_origin())
    backup = r.json()["backup"]
    assert backup and backup.startswith(".env.bak-")
    assert (config_module.REPO_ROOT / backup).read_text(encoding="utf-8") == FAKE_ENV


async def test_restart_required_is_returned_for_the_changed_fields(client):
    r = await client.put("/api/v1/config", json={"values": {"llm_provider": "mock"}}, headers=_same_origin())
    assert "llm_provider" in r.json()["restart_required"]
    # api_client 用 monkeypatch.setenv 把 LLM_PROVIDER 设成了 mock，所以这次写 openai
    # 虽然落进了 .env，**当前进程生效的仍是 mock** —— 这正是 shadowed 要暴露的东西
    assert r.json()["shadowed"] == ["llm_provider"]


async def test_put_reports_fields_shadowed_by_process_env_vars(client, monkeypatch):
    """写进 `.env` ≠ 生效：进程环境变量优先级更高时必须在**写响应**里说出来。

    只有 `sources`（GET）不够 —— 保存那一下界面展示的是 PUT 的返回体，
    "点了保存、绿条闪过、值其实没变"在这个项目里已经出现过一次（LANGCHAIN_* 那回）。
    这里 BASE_ENV 就把 LLM_PROVIDER 设成了 mock，所以改 llm_provider 必然被压住。
    """
    monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
    r = await client.put(
        "/api/v1/config",
        json={"values": {"llm_provider": "openai", "llm_temperature": 0.1}},
        headers=_same_origin(),
    )
    body = r.json()
    assert body["ok"] is True
    assert set(body["shadowed"]) == {"llm_provider", "llm_temperature"}
    # 写了确实写了（备份与内容都在），问题只在"当前不生效"
    assert "LLM_TEMPERATURE=0.1" in config_module.ENV_PATH.read_text(encoding="utf-8")


# ------------------------------------------------------------------ 安全闸门
async def test_cross_origin_write_is_refused(client):
    r = await client.put(
        "/api/v1/config",
        json={"values": {"llm_model": "evil"}},
        headers={"Host": "127.0.0.1:8000", "Origin": "http://attacker.example"},
    )
    assert r.status_code == 403


async def test_foreign_host_is_refused(client):
    r = await client.put("/api/v1/config", json={"values": {"llm_model": "evil"}}, headers={"Host": "attacker.example"})
    assert r.status_code == 403


async def test_readonly_switch_blocks_write(client, monkeypatch):
    monkeypatch.setenv("CONFIG_WRITABLE", "false")
    assert (await client.get("/api/v1/config")).json()["writable"] is False
    assert (await client.put("/api/v1/config", json={"values": {"llm_model": "x"}}, headers=_same_origin())).status_code == 403


async def test_secret_never_reaches_the_logs(client, capfd):
    """密钥只记字段名。日志里出现值就等于把明文搬到另一个地方去了。"""
    secret = "sk-abcdef1234567890"
    await client.put("/api/v1/config", json={"values": {"openai_api_key": secret}}, headers=_same_origin())
    await client.get("/api/v1/config")
    out = capfd.readouterr()
    assert secret not in out.out + out.err


# ------------------------------------------------------------------ 连通性探测
async def test_probe_mock_provider_needs_no_network(client):
    r = await client.post("/api/v1/config/test", json={"provider": "mock"}, headers=_same_origin())
    assert r.status_code == 200 and r.json()["ok"] is True


async def test_probe_rejects_embedding_provider_names_with_the_allowed_list(client):
    """`_TESTABLE_PROVIDERS` 只有 LLM 那一组值。

    前端确实该防，但**接口不能只靠前端防**：422 里带 allowed，
    使用者（和以后的第三方客户端）才知道能传什么，而不是看到一个裸错误。
    """
    r = await client.post("/api/v1/config/test", json={"provider": "local"}, headers=_same_origin())
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "unknown_provider" and set(detail["allowed"]) == {"openai", "ollama", "mock"}


async def test_probe_refuses_to_reuse_stored_key_on_another_host(client, monkeypatch):
    """★ 这是这个接口最要紧的一条性质。

    接口允许调用方随便传 `openai_base_url`。如果此时服务端把 `.env` 里那把真 key
    当 Bearer 发过去，这个"帮你测连通性"的接口就成了一台**密钥外送机**：
    谁能够到 8000，谁就能把我配置的 key 打到自己的服务器上。
    所以规则是：已存的 key 只发往已存的那个网关；要试别的地址就自己带 key。
    """
    sent: list[tuple[str, dict]] = []

    class FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "some-model"}]}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            sent.append((url, dict(headers or {})))
            return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    # 后端已存一把真 key，且配置的网关是 api.openai.com
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stored-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    from app.core.config import get_settings as gs

    gs.cache_clear()

    r = await client.post(
        "/api/v1/config/test",
        json={"provider": "openai", "openai_base_url": "https://attacker.example/v1"},
        headers=_same_origin(),
    )
    assert r.json()["ok"] is False
    assert sent == [], "带着已存密钥去打调用方指定的地址 = 密钥外送"
    assert "sk-stored-real-key" not in r.text

    # 同一个网关时照旧可用（否则这条闸门会把正常路径也堵死）
    r2 = await client.post(
        "/api/v1/config/test",
        json={"provider": "openai", "openai_base_url": "https://api.openai.com/v1"},
        headers=_same_origin(),
    )
    assert r2.json()["ok"] is True
    assert sent and "sk-stored-real-key" in sent[-1][1].get("Authorization", "")

    # 显式传 key 时，用传来的那把，不碰已存的
    sent.clear()
    r3 = await client.post(
        "/api/v1/config/test",
        json={
            "provider": "openai",
            "openai_base_url": "https://other-gw.example/v1",
            "openai_api_key": "sk-throwaway",
        },
        headers=_same_origin(),
    )
    assert r3.json()["ok"] is True
    assert sent[-1][1]["Authorization"].endswith("sk-throwaway")
    gs.cache_clear()


async def test_probe_reads_model_names_from_both_field_spellings(client, monkeypatch):
    """Ollama 的 `/api/tags` 用 `name`，OpenAI 兼容网关用 `id`。

    只认 `name` 时，DeepSeek / OpenAI 的用户每次都会看到"没有可核对的模型列表"，
    而这个列表正是"我填的模型名到底存不存在"唯一的确认途径。
    """

    class FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}, {"name": "legacy"}]}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    r = await client.post(
        "/api/v1/config/test",
        json={"provider": "openai", "openai_base_url": "https://x.example/v1", "openai_api_key": "sk-any"},
        headers=_same_origin(),
    )
    body = r.json()
    assert body["ok"] is True
    assert set(body["models"]) == {"deepseek-chat", "deepseek-reasoner", "legacy"}


async def test_probe_validates_model_name_by_a_real_call(client, monkeypatch):
    """★ 模型名对不对**不能靠比对 /models**，必须打一次真实调用。

    DeepSeek 这类 OpenAI 兼容网关的 `/models` 返回展示名（`DeepSeek-V4.1-Flash`），
    而调用用的 id 是 `deepseek-flash`。只比对列表的话，界面会对一个
    **完全能用的配置**说"你的模型名不对" —— 一个刚上线的按钮第一分钟就在骗人。
    """
    calls: list[str] = []

    class FakeResp:
        def __init__(self, status: int, payload: dict):
            self.status_code = status
            self._payload = payload

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            calls.append(f"GET {url}")
            return FakeResp(200, {"data": [{"id": "DeepSeek-V4.1-Flash"}]})

        async def post(self, url, json=None, headers=None):
            calls.append(f"POST {url}")
            # 展示名和调用 id 不一样，但调用是成功的 —— 这正是那条误报的场景
            return FakeResp(200, {"choices": [{"message": {"content": "p"}}]})

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stored")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    from app.core.config import get_settings as gs

    gs.cache_clear()

    r = await client.post(
        "/api/v1/config/test",
        json={"provider": "openai", "model": "deepseek-flash"},
        headers=_same_origin(),
    )
    body = r.json()
    assert body["ok"] is True
    assert body["model_check"] == {
        "checked": True,
        "ok": True,
        "detail": "模型 deepseek-flash 可以正常调用（已用一次真实请求验证）。",
    }
    # 列表比对会被骗，所以这里断言的是"确实发了那次补全请求"
    assert any(c.endswith("/chat/completions") for c in calls)
    gs.cache_clear()


async def test_probe_model_check_reports_unknown_model_and_stays_silent_when_undecidable(client, monkeypatch):
    """400/404 = 判定为"这个模型名不存在"；其它错误码 = **没能判定**，不能算模型的错。"""
    post_status = {"code": 404}

    class FakeResp:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return FakeResp(200, {"data": [{"id": "a-model"}]})

        async def post(self, url, json=None, headers=None):
            return FakeResp(post_status["code"], {"error": {"message": "The model `x` does not exist"}})

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stored")
    from app.core.config import get_settings as gs

    gs.cache_clear()

    r = await client.post("/api/v1/config/test", json={"provider": "openai", "model": "nope"}, headers=_same_origin())
    check = r.json()["model_check"]
    assert check["checked"] is True and check["ok"] is False
    assert "nope" in check["detail"]
    # 上游错误原文可能带别的信息，只回我们自己的话
    assert "does not exist" not in r.text

    post_status["code"] = 503
    r2 = await client.post("/api/v1/config/test", json={"provider": "openai", "model": "nope"}, headers=_same_origin())
    check2 = r2.json()["model_check"]
    assert check2["checked"] is False and check2["ok"] is False
    assert "无法" in check2["detail"]
    gs.cache_clear()


async def test_probe_ollama_model_check_uses_the_tag_list(client, monkeypatch):
    """Ollama 的 `/api/tags` 返回的就是调用名，所以那边比对列表是可靠的。"""

    class FakeResp:
        status_code = 200

        def json(self):
            return {"models": [{"name": "qwen2.5:7b"}]}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return FakeResp()

        async def post(self, url, json=None, headers=None):  # 不该被调用
            raise AssertionError("ollama 分支不该发补全请求")

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    r = await client.post(
        "/api/v1/config/test",
        json={"provider": "ollama", "model": "llama3:latest"},
        headers=_same_origin(),
    )
    check = r.json()["model_check"]
    assert check["checked"] is True and check["ok"] is False
    assert "llama3:latest" in check["detail"]


async def test_probe_reports_failure_without_leaking_target_details(client, monkeypatch):
    """探测失败时不能把 URL / 凭据原样吐回前端。"""

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise OSError("connect refused to http://secret-host.internal/v1 with key sk-leakme123")

        async def __aexit__(self, *a):
            return False

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: Boom())
    r = await client.post(
        "/api/v1/config/test",
        json={"provider": "openai", "openai_base_url": "http://secret-host.internal/v1", "openai_api_key": "sk-leakme123"},
        headers=_same_origin(),
    )
    body = r.json()
    assert body["ok"] is False
    assert "secret-host.internal" not in r.text
    assert "sk-leakme123" not in r.text
