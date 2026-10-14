"""OpenAlex 补充源：**只补数字字段，且必须交叉校验**（架构文档 §2.1 / ADR D7）。

为什么不能直接采信 OpenAlex 的返回：
实测中 `arXiv:2005.11401`（经典 RAG 论文）在 OpenAlex 上搜标题时，返回的候选里
会出现**标题属于另一篇论文、作者却对得上**的记录。第三方聚合源存在元数据污染，
如果不做校验就直接取第一个结果，会把别的论文的被引数写到这篇论文头上 ——
而且**不会报错**，只会静默产出错误的对比表。学术场景里这比"没有数据"危害大得多。

因此这里的硬规则是：
1. 搜索回来后逐个候选做 **标题 + 作者双重交叉校验**，任一不通过就丢弃并记 warning；
2. 全部候选都不通过 → 返回 `None`（宁缺毋滥，对比表里留空，绝不编造）；
3. 被引数**必须带 `cited_at`**（截至日期）。最新论文被引数普遍是 0，
   一个孤零零的 "0" 会被误读成"没人引用"，带上日期才是可解释的事实。

文本字段（标题/作者/摘要）一律不回写 —— 那些只信 arXiv 官方。
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx
import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"

# 标题比对时忽略的虚词：它们不携带区分度，留着只会抬高"假匹配"的相似度
_TITLE_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "on", "in", "with", "to", "via",
    "using", "towards", "toward", "from", "at", "by", "as", "is", "are",
}

# 通过阈值。0.7 是实测折中：同一篇论文因副标题/大小写/连字符差异通常在 0.8 以上，
# 而"标题串成另一篇论文"的相似度普遍在 0.3 以下，中间有一段很宽的安全区。
_TITLE_SIM_THRESHOLD = 0.70
# 允许"一个标题是另一个的子集"（如 arXiv 版无副标题），但要求交集足够多才有意义
# arXiv 前缀形状（`arxiv.org/abs/2305.15334v2`、`10.48550/arXiv.2305.15334`）里抠短 id 用
_ARXIV_REF_RE = re.compile(
    r"arxiv\.org/abs/(\d{4}\.\d{4,5})|arxiv\.(\d{4}\.\d{4,5})|arXiv:(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)
# DOI 里用的短 id：新式一律是 `NNNN.NNNNN`，版本号后缀（v3）不属于 DOI 的一部分
_ARXIV_NEW_ID_RE = re.compile(r"\d{4}\.\d{4,5}")
# 一次 `filter=openalex_id:W1|W2|…` 最多带多少个 id。OpenAlex 允许上百个值，
# 但 URL 长度与响应体积在 50 这一档最稳（参考文献常有 150+ 条，必须分批）。
_RESOLVE_BATCH = 50


def _doi_bare(arxiv_id: str) -> str:
    """把任意形状的 arXiv id 归一成 DOI 里用的那种短 id。

    先按 `NNNN.NNNNN` 抓，是因为调用方传来的形状太杂（`arxiv/2305.15334v3`、
    `ARXIV:2305.15334`、`2305.15334v10`）；抓不到就退回原样，
    让老式 id（`cs.CL/0201001`）仍有机会命中它自己的 DOI。
    """
    text = (arxiv_id or "").strip()
    m = _ARXIV_NEW_ID_RE.search(text)
    return m.group(0) if m else text


def arxiv_id_from_work(work: dict) -> str | None:
    """从一条 OpenAlex work 记录里抠出 arXiv 短 id，抠不到就返回 None。

    三条来源按可信度排序：`ids.arxiv`（官方登记）→ `doi`（arXiv 的 DOI 前缀固定是
    `10.48550/arXiv.`）→ `primary_location` 里的 URL。
    ⚠️ 不做"标题像就是一篇"的模糊匹配：引用边会画进关系图里，
    一条错边比少一条边糟得多（产品定义 §7：宁可留空也不编造）。
    """
    if not isinstance(work, dict):
        return None
    ids = work.get("ids") or {}
    for value in (ids.get("arxiv"), work.get("doi"), (work.get("primary_location") or {}).get("url")):
        if not value:
            continue
        m = _ARXIV_REF_RE.search(str(value))
        if m:
            return next((g for g in m.groups() if g), None)
    return None
_CONTAINMENT_THRESHOLD = 0.90
_CONTAINMENT_MIN_TOKENS = 6

_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_MAILTO_RE = re.compile(r"mailto:\s*([^)\s,;]+)", re.I)


def _normalize(text: str) -> str:
    """去掉变音符号 / 标点，转小写，折叠空白。

    标题里 `Retrieval-Augmented` 与 `Retrieval Augmented` 必须视为同一串，
    否则会被自己的校验规则误杀。
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _WS_RE.sub(" ", _NON_WORD_RE.sub(" ", text.lower())).strip()


def _title_tokens(text: str) -> set[str]:
    return {t for t in _normalize(text).split() if t and t not in _TITLE_STOPWORDS}


def title_similarity(a: str, b: str) -> float:
    """标题相似度 = max(词集 Jaccard, 子集包含率)。

    用词集而不是编辑距离：论文标题里的词序差异很常见，而"串篇"是整体换词，
    词集重叠率对这种污染最敏感。包含率这一项是为了放过"arXiv 版缺副标题"的情况。
    """
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    jaccard = inter / len(ta | tb)
    containment = inter / min(len(ta), len(tb))
    if containment >= _CONTAINMENT_THRESHOLD and min(len(ta), len(tb)) >= _CONTAINMENT_MIN_TOKENS:
        return max(jaccard, containment)
    return jaccard


def _surnames(names) -> set[str]:
    """取姓氏集合（取最后一个词）。中英混排时这个近似足够用。"""
    out: set[str] = set()
    for name in names or []:
        parts = _normalize(name).split()
        if parts:
            out.add(parts[-1])
    return out


def validate_match(
    candidate_title: str,
    candidate_authors,
    arxiv_title: str,
    arxiv_authors,
) -> tuple[bool, float, str]:
    """标题 + 作者交叉校验。返回 `(是否通过, 标题相似度, 原因)`。

    为什么作者也一定要校验：OpenAlex 里同一篇论文可能有多条记录，
    **标题对、作者不对**（或相反）都出现过，单一字段校验不足以挡住污染。
    作者只在双方都有数据时才参与判定 —— 缺作者就只信标题，避免误杀。
    """
    sim = title_similarity(arxiv_title, candidate_title)
    if sim < _TITLE_SIM_THRESHOLD:
        return False, sim, "title_mismatch"

    cand_surnames = _surnames(candidate_authors)
    arxiv_surnames = _surnames(arxiv_authors)
    if cand_surnames and arxiv_surnames and not (cand_surnames & arxiv_surnames):
        return False, sim, "author_mismatch"
    return True, sim, "ok"


@dataclass(slots=True, frozen=True)
class OpenAlexMetrics:
    """补充到的**数字字段**（外加候选标题，便于人工复盘校验过程）。"""

    openalex_id: str
    title: str
    cited_by_count: int
    cited_at: date                    # ⚠️ 被引数的截至日期，必须与数字一起存
    referenced_works_count: int | None = None
    is_oa: bool | None = None
    oa_status: str | None = None
    oa_url: str | None = None
    title_similarity: float = 0.0
    matched_authors: tuple[str, ...] = ()


class OpenAlexClient:
    """OpenAlex `/works` 免 key 客户端（含限速礼貌头 + 交叉校验）。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        endpoint: str = OPENALEX_WORKS_ENDPOINT,
        timeout: float = 20.0,
    ) -> None:
        self.settings = settings or get_settings()
        self._endpoint = endpoint
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        # OpenAlex 比 arXiv 宽松得多，不需要 3s；但完全不打间隔容易被共享池限流。
        # ⚠️ 读 `OPENALEX_REQUEST_INTERVAL`：这里曾经硬编码 0.2，
        # 于是配置项存在但改了没任何效果（又一种"字段存在却没接通"）。
        self._min_interval = float(getattr(self.settings, "openalex_request_interval", 0.2) or 0.2)
        self._last_request_at: float | None = None
        # 礼貌性联系方式：优先用专门的 `OPENALEX_MAILTO`，没填才从 arXiv 的 UA 里抠。
        # 走"专门配置优先、否则回退"而不是只回退：被限流时用户需要一个能单独调的旋钮。
        mailto = (getattr(self.settings, "openalex_mailto", "") or "").strip()
        if not mailto:
            m = _MAILTO_RE.search(self.settings.arxiv_user_agent or "")
            mailto = m.group(1) if m else ""
        self._mailto = mailto or None

    # ------------------------------------------------------------------ 生命周期
    async def __aenter__(self) -> OpenAlexClient:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            ua = self.settings.arxiv_user_agent or "PaperPilot/0.1"
            self._client = httpx.AsyncClient(
                headers={"User-Agent": ua, "Accept": "application/json"},
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    async def _throttle(self) -> None:
        if self._last_request_at is not None:
            wait = self._min_interval - (time.monotonic() - self._last_request_at)
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_request_at = time.monotonic()

    # ------------------------------------------------------------------ 查询
    async def enrich(
        self,
        title: str,
        authors,
        *,
        arxiv_id: str | None = None,
        per_page: int = 5,
    ) -> OpenAlexMetrics | None:
        """按标题搜索并用**交叉校验**挑出可信候选。

        **永不抛异常**：补充源挂了不影响主流程，只是这次对比表少一列数字。
        """
        if not title or not title.strip():
            return None

        await self._throttle()
        params: dict = {
            "search": title,
            "per-page": per_page,
            # 只要用到的字段，响应体积小一半以上（被引数用不上 abstract 全文）
            "select": "id,title,display_name,cited_by_count,referenced_works_count,"
            "open_access,authorships,ids",
        }
        if self._mailto:
            params["mailto"] = self._mailto

        try:
            resp = await self._ensure_client().get(self._endpoint, params=params)
            if resp.status_code != 200:
                logger.warning("openalex.http_error", arxiv_id=arxiv_id, status=resp.status_code)
                return None
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - 补充源一律不致命
            logger.warning("openalex.request_failed", arxiv_id=arxiv_id, error=str(exc)[:200])
            return None

        candidates = payload.get("results") or []
        rejected: list[dict] = []
        for work in candidates:
            cand_title = work.get("title") or work.get("display_name") or ""
            cand_authors = [
                (a.get("author") or {}).get("display_name", "")
                for a in (work.get("authorships") or [])
            ]
            ok, sim, reason = validate_match(cand_title, cand_authors, title, authors)
            if not ok:
                rejected.append(
                    {"title": cand_title[:90], "sim": round(sim, 3), "reason": reason}
                )
                continue

            oa = work.get("open_access") or {}
            metrics = OpenAlexMetrics(
                openalex_id=str(work.get("id", "")).rsplit("/", 1)[-1],
                title=cand_title,
                cited_by_count=int(work.get("cited_by_count") or 0),
                cited_at=date.today(),  # 截至日期 = 本次抓取日
                referenced_works_count=work.get("referenced_works_count"),
                is_oa=oa.get("is_oa"),
                oa_status=oa.get("oa_status"),
                oa_url=oa.get("oa_url"),
                title_similarity=round(sim, 3),
                matched_authors=tuple(a for a in cand_authors if a),
            )
            logger.info(
                "openalex.matched",
                arxiv_id=arxiv_id,
                cited=metrics.cited_by_count,
                sim=metrics.title_similarity,
            )
            return metrics

        # 走到这里说明有候选但全部被校验挡下 —— 这正是"元数据污染"的现场，
        # 必须留下 warning 供排查（丢弃是正确行为，但不能丢得无声无息）
        logger.warning(
            "openalex.no_valid_match",
            arxiv_id=arxiv_id,
            title=title[:90],
            candidates=len(candidates),
            rejected=rejected[:3],
        )
        return None


    # ------------------------------------------------------------------ 引用关系（F16）
    async def fetch_referenced_works(self, arxiv_id: str) -> tuple[str, list[str]] | None:
        """按 arXiv DOI 精确取这篇论文的 OpenAlex 记录。

        返回 `(openalex_id, [它引用的 work id 列表])`；取不到返回 `None`
        （调用方一律按"没有引用边证据"处理，**不许**退化成"猜一些相近论文当边"）。

        为什么这里用 DOI 精确匹配而不是复用 `enrich()` 的标题搜索：
        `10.48550/arXiv.<id>` 是 arXiv 官方 DOI，一条命中、无歧义 ——
        标题 + 作者交叉校验是给"模糊搜索"兜底的，精确查找不需要它，
        而且引用边是**结构性事实**，错一条比错一个被引数影响大得多（它会画进图里）。
        """
        bare = _doi_bare(arxiv_id)
        if not bare:
            return None

        await self._throttle()
        params: dict = {
            "filter": f"doi:10.48550/arxiv.{bare}",
            "per-page": 1,
            "select": "id,referenced_works",
        }
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            resp = await self._ensure_client().get(self._endpoint, params=params)
            if resp.status_code != 200:
                logger.warning("openalex.refs_http_error", arxiv_id=bare, status=resp.status_code)
                return None
            results = (resp.json() or {}).get("results") or []
        except Exception as exc:  # noqa: BLE001 - 补充源一律不致命
            logger.warning("openalex.refs_request_failed", arxiv_id=bare, error=str(exc)[:200])
            return None

        if not results:
            logger.info("openalex.refs_no_work", arxiv_id=bare)
            return None
        work = results[0]
        wid = str(work.get("id", "")).rsplit("/", 1)[-1]
        refs = [str(r).rsplit("/", 1)[-1] for r in (work.get("referenced_works") or []) if r]
        if not wid or not refs:
            return None
        logger.info("openalex.refs_fetched", arxiv_id=bare, work=wid, refs=len(refs))
        return wid, refs

    async def resolve_arxiv_ids(
        self, work_ids: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        """把 OpenAlex work id 批量翻成 `{work_id: {"arxiv_id": str|None, "title": str}}`。

        分批（每批 50 个）而不是把整串 id 拼进一个 URL：一篇论文的参考文献可以有 150+ 条，
        超长 query 会被 OpenAlex 以 400 拒掉，而 400 里看不出"是太长"还是"格式错"。
        翻不出 arXiv id 的条目**照样返回**（`arxiv_id=None`）—— 调用方需要知道
        "这条参考文献存在、只是不在 arXiv 上"，与"这条没查到"是两件事。
        """
        out: dict[str, dict[str, Any]] = {}
        ids = [str(w).strip() for w in (work_ids or []) if str(w).strip()]
        for start in range(0, len(ids), _RESOLVE_BATCH):
            batch = ids[start : start + _RESOLVE_BATCH]
            await self._throttle()
            params: dict = {
                # OpenAlex 的多值 OR 过滤是 `filter=openalex_id:W1|W2|…`，键名前缀不能漏
                "filter": "openalex_id:" + "|".join(batch),
                "select": "id,doi,title,display_name,ids",
            }
            if self._mailto:
                params["mailto"] = self._mailto
            try:
                resp = await self._ensure_client().get(self._endpoint, params=params)
                if resp.status_code != 200:
                    logger.warning(
                        "openalex.resolve_http_error", batch=len(batch), status=resp.status_code
                    )
                    continue
                for work in (resp.json() or {}).get("results") or []:
                    wid = str(work.get("id", "")).rsplit("/", 1)[-1]
                    if not wid:
                        continue
                    out[wid] = {
                        "arxiv_id": arxiv_id_from_work(work),
                        "title": work.get("title") or work.get("display_name") or "",
                    }
            except Exception as exc:  # noqa: BLE001 - 补边失败不影响已入库的语料
                logger.warning("openalex.resolve_failed", batch=len(batch), error=str(exc)[:200])
        return out


__all__ = [
    "OPENALEX_WORKS_ENDPOINT",
    "OpenAlexClient",
    "OpenAlexMetrics",
    "arxiv_id_from_work",
    "title_similarity",
    "validate_match",
]
