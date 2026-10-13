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
from dataclasses import dataclass
from datetime import date

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
        # OpenAlex 比 arXiv 宽松得多，不需要 3s；但完全不打间隔容易被共享池限流
        self._min_interval = 0.2
        self._last_request_at: float | None = None
        # 从 arXiv 的 User-Agent 里顺手抠出 mailto —— OpenAlex 的"礼貌池"靠它给更快的配额，
        # 而 Settings 里没有专门的 openalex 配置项（见交付报告里的配置缺口说明）
        m = _MAILTO_RE.search(self.settings.arxiv_user_agent or "")
        self._mailto = m.group(1) if m else None

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


__all__ = [
    "OPENALEX_WORKS_ENDPOINT",
    "OpenAlexClient",
    "OpenAlexMetrics",
    "title_similarity",
    "validate_match",
]
