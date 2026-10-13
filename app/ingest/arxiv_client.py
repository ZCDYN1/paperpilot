"""arXiv API 客户端。

三条不可让步的约束，都是被 arXiv 官方规则和实测坑逼出来的：

1. **声明式 User-Agent（含 mailto）**：arXiv 靠它在出问题时联系你。不带联系方式
   的高频请求会被直接限流，且没有任何提示 —— 表现为"偶尔 503、重试就好"，
   排查起来毫无头绪。
2. **串行限速 ≥ 3 秒/请求**：arXiv 明确要求约 1 请求 / 3 秒。这里用一把锁把
   整个客户端串行化，**刻意不提供任何并发入口** —— 因为限速是客户端实例级的，
   只要允许并发，调用方就会绕过去，最后变成"本地只有一个调用方所以没事"，
   一上量就被封。宁可让调用方多等，也不能让这个约束形同虚设。
3. **只对网络错误 / 5xx 重试**：4xx 说明请求本身有问题（参数错、id 不存在），
   重试只会白等再加上一次限速配额。所以重试谓词里把 4xx 显式排除。

P.S. 端点写的是 `http://export.arxiv.org`，但它现在会 301 跳 https ——
所以 httpx 必须开 `follow_redirects`，否则拿到一个空的 301 响应，feedparser
解析出 0 条数据而不报错，是最隐蔽的一种失败。
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import feedparser
import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

ARXIV_API_ENDPOINT = "http://export.arxiv.org/api/query"
ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

# 新式 id：2401.12345；旧式 id：cs/0701001、math.GT/0309136
_NEW_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_OLD_ID_RE = re.compile(r"([a-zA-Z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?")
_WS_RE = re.compile(r"\s+")


class ArxivError(RuntimeError):
    """arXiv 请求失败的基类。"""


class ArxivRequestError(ArxivError):
    """4xx：请求本身有问题，**不可重试**。"""


class ArxivServerError(ArxivError):
    """5xx / 网关类瞬时故障，值得重试。"""


def parse_arxiv_id(raw: str) -> str:
    """从 abs 链接或带版本号的 id 中提取**短 id**（去掉 `v2` 这类版本后缀）。

    为什么统一成短 id：同一个 `arxiv_id` 是 `papers` 表的主键，若把 `2401.12345v1`
    和 `2401.12345v2` 当成两篇，会出现重复行 → 缓存判断失效、引用对不上、
    FAISS 里同一篇论文的 chunk 分成两组。版本差异用 `updated` 字段表达即可。
    """
    s = (raw or "").strip()
    if not s:
        return ""
    m = _NEW_ID_RE.search(s)
    if m:
        return m.group(1)
    m = _OLD_ID_RE.search(s)
    if m:
        return m.group(1)
    # 兜底：截最后一段再去版本号，保证"永远返回一个可用字符串"而不是抛异常
    tail = s.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", tail)


def _clean_text(value: str) -> str:
    """归一化空白：Atom 里的 title/abstract 常有换行与多余空格，直接入库会污染分块。"""
    return _WS_RE.sub(" ", value or "").strip()


def _to_datetime(parsed) -> datetime | None:
    """feedparser 的 `*_parsed` 是 UTC 的 struct_time；转成 tz-aware datetime。

    不用 `published` 字符串自己 parse：feedparser 已经处理了格式变体，
    自己再 parse 一遍等于把已经解决的兼容性问题重新引入。
    """
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=UTC)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True, frozen=True)
class ArxivPaper:
    """一篇 arXiv 论文的**权威元数据**（文本字段的唯一可信来源）。

    刻意用 dataclass 而不是裸 dict：下游（pipeline / Reader / 对比表）靠字段名
    访问，拼错立刻报错；dict 拼错只会静默丢字段，等发现时数据已经入库了。
    """

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime | None
    updated: datetime | None
    primary_category: str | None
    categories: list[str]
    comment: str | None
    journal_ref: str | None
    pdf_url: str | None
    html_url: str

    @property
    def year(self) -> int | None:
        """检索层按年份过滤需要整数年份，这里统一算好，避免各处重复推导。"""
        return self.published.year if self.published else None


def _is_retryable(exc: BaseException) -> bool:
    """重试谓词：**只有网络类错误与 5xx 才重试**。

    把 4xx（ArxivRequestError）排除在外是关键——否则一个写错的 `cat:cs.XX`
    会被重试 3 次，每次还要等 3 秒限速，白占 9 秒配额。
    """
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException, ArxivServerError))


class ArxivClient:
    """arXiv API 异步客户端（内部串行限速，**禁止并发**）。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        endpoint: str = ARXIV_API_ENDPOINT,
        page_size: int = 100,
    ) -> None:
        self.settings = settings or get_settings()
        self._endpoint = endpoint
        # arXiv 单页上限 100（调大不会更快，只会被限流）
        self.page_size = max(1, min(int(page_size), 100))
        self._client = client
        self._owns_client = client is None
        # 用锁而不是"记录上次时间戳"：锁能保证"检查-等待-发出"三步原子，
        # 单纯记时间戳在并发下会两个协程同时判断"还没到点"然后一起发出去。
        self._throttle_lock = asyncio.Lock()
        self._last_request_at: float | None = None

    # ------------------------------------------------------------------ 生命周期
    async def __aenter__(self) -> ArxivClient:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": self.settings.arxiv_user_agent,  # 合规项：含 mailto
                    "Accept": "application/atom+xml",
                },
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,  # http 端点会 301 到 https，不开就静默拿到空数据
            )
        return self._client

    # ------------------------------------------------------------------ 限速
    async def _respect_rate_limit(self) -> None:
        """把两次请求的**实际发出时间**间隔压到 ≥ `arxiv_request_interval`。"""
        interval = float(self.settings.arxiv_request_interval or 0.0)
        async with self._throttle_lock:
            if self._last_request_at is not None and interval > 0:
                elapsed = time.monotonic() - self._last_request_at
                wait = interval - elapsed
                if wait > 0:
                    logger.debug("arxiv.throttle", wait=round(wait, 2))
                    await asyncio.sleep(wait)
            # 在「真正要发请求」的这一刻打点，而不是在调用开始时，
            # 否则等待时间会被算进间隔里，实际节奏会快于 3s/请求
            self._last_request_at = time.monotonic()

    # ------------------------------------------------------------------ 请求
    async def _fetch_page(self, params: dict) -> dict:
        """发一次请求并解析 Atom。限速 + 状态码分级都在这里。"""
        await self._respect_rate_limit()
        client = self._ensure_client()
        resp = await client.get(self._endpoint, params=params)

        if resp.status_code >= 500:
            raise ArxivServerError(f"arxiv http {resp.status_code}")
        if resp.status_code >= 400:
            # 4xx 不重试：把状态码带进异常，让调用方一眼看出是参数问题
            raise ArxivRequestError(f"arxiv http {resp.status_code}: {resp.text[:200]}")

        feed = feedparser.parse(resp.content)
        # bozo 只说明 XML 有瑕疵，不代表没数据；真正没数据要看 entries
        if getattr(feed, "bozo", 0) and not getattr(feed, "entries", None):
            logger.warning("arxiv.malformed_feed", bozo=str(getattr(feed, "bozo_exception", ""))[:200])
        return feed

    async def _request_with_retry(self, params: dict) -> dict:
        """`stop_after_attempt(3)` + 指数退避 + 随机抖动，**外面再套一层熔断**。

        抖动是必要的：若多个进程同时被限流，固定退避会让它们在同一个时刻
        一起重试，形成整齐的"重试风暴"。

        ★ 为什么这里也要接熔断（README 的"重试 / 超时 / 熔断"指的是 arXiv 这个对外 HTTP 出口）：
        arXiv 5xx / 连不上时，重试救不了"整个服务端挂了"，而一次研究可能连着打几十次请求，
        每次都白等 3 秒限速 + 3 次退避 —— 既把自己的时间烧光，也是在给下游施压。
        接上熔断后，连续若干**整段调用**失败就会在本地快速失败，
        `ensure_corpus` 的既有降级分支（`corpus_ready=False` → 直接写作）立刻生效。

        `attempts=1` 是刻意的：**重试这一层已经由下面的 tenacity 循环做了**，
        `guarded` 只补"总时间预算 + 熔断"，绝不能再叠一层重试（否则 3×N 次请求，
        既超 arXiv 限速又烧配额）。顺序仍是「重试在内、熔断在外」，与 `resilient()` 一致。
        """
        settings = self.settings
        if not (settings.resilience_enabled and settings.breaker_for_http):
            return await self._retry_loop(params)

        from app.core.resilience import BREAKER_HTTP, guarded

        return await guarded(
            self._retry_loop,
            params,
            breaker_name=BREAKER_HTTP,
            attempts=1,
            # 整段（含内部 3 次重试）的预算：比 3 次 30s 超时短一档，卡死时快速失败
            seconds=float(settings.http_timeout) * 3 + 15.0,
        )

    async def _retry_loop(self, params: dict) -> dict:
        """原有的 tenacity 重试循环（被 `_request_with_retry` 套在熔断**里面**）。"""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, max=8) + wait_random(0, 1),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                n = attempt.retry_state.attempt_number
                if n > 1:
                    logger.warning("arxiv.retry", attempt=n)
                return await self._fetch_page(params)
        raise ArxivError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ 查询构造
    @staticmethod
    def _build_query(
        query: str | None,
        categories: list[str] | None,
        year_from: int | None,
        year_to: int | None,
    ) -> str:
        parts: list[str] = []

        q = _clean_text(query or "").replace('"', " ")
        if q:
            # 多词必须加引号成短语。裸空格会被 arXiv 当成意外的分隔符，
            # 结果从"精确短语匹配"退化成松散匹配，题干越具体召回越不准。
            parts.append(f'all:"{q}"' if " " in q else f"all:{q}")

        cats = [c.strip() for c in (categories or []) if c and c.strip()]
        if cats:
            parts.append(
                f"cat:{cats[0]}" if len(cats) == 1 else "(" + " OR ".join(f"cat:{c}" for c in cats) + ")"
            )

        if year_from or year_to:
            # arXiv 的 submittedDate 区间需要 YYYYMMDDHHMM 这种 12 位格式
            start = f"{int(year_from or 1991)}01010000"
            end = f"{int(year_to or 2999)}12312359"
            parts.append(f"submittedDate:[{start} TO {end}]")

        if not parts:
            raise ValueError("search() 至少要有一个检索条件（query / categories / 年份）")
        return " AND ".join(parts)

    # ------------------------------------------------------------------ 解析
    def _parse_entry(self, entry) -> ArxivPaper | None:
        """把单条 Atom entry 转成 ArxivPaper；字段异常时返回 None 而非抛异常。

        一篇解析失败不该毁掉整次检索 —— 这跟"单篇论文失败不中断全流程"
        是同一条原则，在客户端这一层就要守住。
        """
        try:
            arxiv_id = parse_arxiv_id(entry.get("id", ""))
            if not arxiv_id:
                return None

            primary = (entry.get("arxiv_primary_category") or {}).get("term")
            categories = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
            # links 里同时有 abs 与 pdf：pdf 的标记是 title=="pdf"（rel=="related"）
            pdf_url = next(
                (link.get("href") for link in entry.get("links", []) if link.get("title") == "pdf"),
                None,
            ) or ARXIV_PDF_URL.format(arxiv_id=arxiv_id)

            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=_clean_text(entry.get("title", "")),
                abstract=_clean_text(entry.get("summary", "")),
                authors=[_clean_text(a.get("name", "")) for a in entry.get("authors", []) if a.get("name")],
                published=_to_datetime(entry.get("published_parsed")),
                updated=_to_datetime(entry.get("updated_parsed")),
                primary_category=primary,
                categories=categories,
                comment=_clean_text(entry.get("arxiv_comment", "")) or None,
                journal_ref=_clean_text(entry.get("arxiv_journal_ref", "")) or None,
                pdf_url=pdf_url,
                html_url=ARXIV_HTML_URL.format(arxiv_id=arxiv_id),
            )
        except Exception as exc:  # noqa: BLE001 - 故意兜底
            logger.warning("arxiv.parse_entry_failed", error=str(exc)[:200])
            return None

    # ------------------------------------------------------------------ 公开 API
    async def search(
        self,
        query: str | None = None,
        *,
        categories: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        max_results: int | None = None,
        sort_by: str = "relevance",
    ) -> list[ArxivPaper]:
        """按关键词 / 分类 / 年份检索，自动翻页直到取满 `max_results`。

        翻页是**逐页串行**的（每页一次请求，各自受 3 秒限速约束）——
        这正是架构文档 §2.2 里"抓取阶段串行、检索阶段并行"的那条边界。
        """
        limit = int(max_results if max_results is not None else self.settings.arxiv_max_results)
        if limit <= 0:
            return []

        search_query = self._build_query(query, categories, year_from, year_to)
        collected: dict[str, ArxivPaper] = {}
        start = 0
        pages = 0

        while len(collected) < limit:
            want = min(self.page_size, limit - len(collected))
            params = {
                "search_query": search_query,
                "start": start,
                "max_results": want,
                "sortBy": sort_by,
                "sortOrder": "descending",
            }
            feed = await self._request_with_retry(params)
            pages += 1
            entries = getattr(feed, "entries", None) or []
            if not entries:
                break

            for entry in entries:
                paper = self._parse_entry(entry)
                if paper:
                    collected.setdefault(paper.arxiv_id, paper)

            # 返回条数少于请求数 = 已经到末页，继续翻只会拿到空页
            if len(entries) < want:
                break
            start += len(entries)

        results = list(collected.values())[:limit]
        logger.info(
            "arxiv.search_done",
            query=(query or "")[:60],
            categories=categories or [],
            pages=pages,
            papers=len(results),
        )
        return results

    async def get_paper(self, arxiv_id: str) -> ArxivPaper | None:
        """按 id 取单篇（`id_list` 查询）。用于补抓或校验。"""
        aid = parse_arxiv_id(arxiv_id)
        if not aid:
            return None
        feed = await self._request_with_retry({"id_list": aid, "max_results": 1})
        entries = getattr(feed, "entries", None) or []
        if not entries:
            return None
        return self._parse_entry(entries[0])


__all__ = [
    "ARXIV_API_ENDPOINT",
    "ARXIV_HTML_URL",
    "ARXIV_PDF_URL",
    "ArxivClient",
    "ArxivError",
    "ArxivPaper",
    "ArxivRequestError",
    "ArxivServerError",
    "parse_arxiv_id",
]
