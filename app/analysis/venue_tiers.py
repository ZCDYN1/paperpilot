"""F19 —— 会议/期刊分档：**纯查表**模块（产品定义 §5「论文质量信号」）。

## 定位
把 `papers.venue`（入库时已从 arXiv comment 抽取，ADR D9）映射成三档质量信号：

- `top`       —— 公认顶会/顶刊（ACL / EMNLP / NeurIPS / ICML / ICLR / CVPR 等白名单）；
- `core`      —— 其它明确给出了会议/期刊名的发表（命中次级会议白名单，或匹配"会议/期刊"通用形状）；
- `preprint`  —— 只有 arXiv、没抽到 venue（venue 为空 / 仅含 arXiv 字样）；
- `unknown`   —— venue 非空但**判不准**（一个陌生字符串，不像任何会议名）。

⚠️ 分档方向上的铁律（对应任务"判不准就返回 preprint/unknown，不要猜一个高档位"）：
这个字段会以"论文质量信号"的身份直接展示给用户，**升档误判 = 编造**。
所以本模块只在**白名单点名**时才给 `top`；
`ACL 2024 Findings` / `NeurIPS Workshop 2025` 这类"主会名 + 附属卷"一律降为 `core`
—— Findings / Workshop 不是主会录用，把它们标成 top 会系统性高估质量。

## 与 comparator 的同一条纪律：只消费，不重抽
`lookup` 的输入是**已经躺在 `papers.venue` 里的字符串**。本模块绝不回头去 comment 里
二次抽取 venue（那会造成"入库一套规则、分析另一套规则"，同一篇论文两个会议名，
见 `app/agents/nodes/comparator.py` 中 `load_paper_meta` 的注释）。
`lookup` 的 `comment` 关键字参数**只影响 evidence 的表述**（注明命中片段确实出现在
comment 的哪一段），对档位判定零影响。

## 无状态、无 IO、无新表
纯常量 + 纯函数：不建表、不改表、不碰 DB（表的约束归主控 Agent）。
"""

from __future__ import annotations

import re
from typing import Any

# ---- 档位常量（对外口径就这四个值）----
TIER_TOP = "top"
TIER_CORE = "core"
TIER_PREPRINT = "preprint"
TIER_UNKNOWN = "unknown"

_LABELS: dict[str, str] = {
    TIER_TOP: "顶会/顶刊",
    TIER_CORE: "核心会议/期刊",
    TIER_PREPRINT: "预印本（仅 arXiv，未抽到发表信息）",
    TIER_UNKNOWN: "未识别（保守不判档，勿作质量信号）",
}

# 顶会/顶刊白名单：只收"点名无争议"的。NAACL / EACL / WWW 等虽是好会，
# 但不在任务给的示例名单里 → 刻意的保守选择放 core，宁可低档不可猜高。
_TOP_NAMES: dict[str, str] = {
    "ACL": "Association for Computational Linguistics",
    "EMNLP": "Empirical Methods in Natural Language Processing",
    "NeurIPS": "Conference on Neural Information Processing Systems",
    "NIPS": "NeurIPS 旧缩写",
    "ICML": "International Conference on Machine Learning",
    "ICLR": "International Conference on Learning Representations",
    "CVPR": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    "ICCV": "International Conference on Computer Vision",
    "ECCV": "European Conference on Computer Vision",
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "IJCAI": "International Joint Conference on Artificial Intelligence",
    "SIGIR": "ACM SIGIR Conference on Research and Development in Information Retrieval",
    "KDD": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
    "JMLR": "Journal of Machine Learning Research",
    "TPAMI": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
}

# 明确存在的次级会议/期刊 → core。
_CORE_NAMES: dict[str, str] = {
    "NAACL": "North American Chapter of the ACL",
    "EACL": "European Chapter of the ACL",
    "COLING": "International Conference on Computational Linguistics",
    "TACL": "Transactions of the Association for Computational Linguistics",
    "WSDM": "ACM International Conference on Web Search and Data Mining",
    "CIKM": "ACM International Conference on Information and Knowledge Management",
    "WWW": "The Web Conference",
    "AISTATS": "International Conference on Artificial Intelligence and Statistics",
    "UAI": "Conference on Uncertainty in Artificial Intelligence",
    "RecSys": "ACM Conference on Recommender Systems",
    "SIGDIAL": "Signature Events in Human Language Technology",
    "ICASSP": "IEEE International Conference on Acoustics, Speech and Signal Processing",
    "INTERSPEECH": "Conference of the International Speech Communication Association",
    "TALIP": "ACM Transactions on Asian and Low-Resource Language Information Processing",
}

# "主会名 + 附属卷" → 一律从 top 降到 core：这些不是主会录用。
_DOWNGRADE_RE = re.compile(
    r"\b(findings|workshops?|demos?|tutorials?|companion|doctoral|student|extracts?)\b",
    re.I,
)

# 通用"像会议/期刊"的形状（命中 → core，但 matched 说不清具体会议名）：
# ① 刊名关键词；② 全大写缩写 + 年份（如 "IJCKG 2023"，白名单外的真实会议常见形态）。
_GENERIC_NAME_RE = re.compile(r"\b(journal|transactions|proceedings|conference|symposium|colloquium)\b", re.I)
_ACRONYM_YEAR_RE = re.compile(r"\b[A-Z]{2,10}\s+(?:19|20)\d{2}\b")
_ARXIV_RE = re.compile(r"\barxiv\b", re.I)

_TOP_PATTERNS = {k: re.compile(rf"\b{re.escape(k)}\b", re.I) for k in _TOP_NAMES}
_CORE_PATTERNS = {k: re.compile(rf"\b{re.escape(k)}\b", re.I) for k in _CORE_NAMES}


def _earliest(patterns: dict[str, re.Pattern[str]], text: str) -> tuple[str, int] | None:
    """白名单里在 text 中**最早出现**的会议名 → (名称, 起始位置)；都没命中返回 None。"""
    best: tuple[str, int] | None = None
    for name, rx in patterns.items():
        m = rx.search(text)
        if m and (best is None or m.start() < best[1]):
            best = (name, m.start())
    return best


def _excerpt(comment: str, fragment: str) -> str | None:
    """在 comment 里定位 fragment（大小写不敏感），给出命中处附近的上下文；定位不到返回 None。"""
    if not fragment:
        return None
    idx = comment.lower().find(fragment.lower())
    if idx < 0:
        return None
    lo = max(0, idx - 30)
    hi = min(len(comment), idx + len(fragment) + 30)
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(comment) else ""
    return f"{prefix}{comment[lo:hi]}{suffix}"


def lookup(venue: str | None, *, comment: str | None = None) -> dict[str, Any]:
    """把 venue 字符串映射成分档。**永不抛异常**，判不准就落在 preprint/unknown。

    返回 `{tier, label, evidence}`：
    - `tier`: `top` / `core` / `preprint` / `unknown` 四值之一；
    - `label`: 给人看的中文档名；
    - `evidence`: 命中依据的字典，字段固定为
      `{matched, fragment, modifier, found_in, comment_excerpt, note}`，
      写清命中的是哪个会议名、输入串里哪段命中、是否被附属卷降档、
      以及（传了 comment 时）该片段对应 arXiv comment 的哪一段。
      档位存疑时 `matched` 为 None —— 宁可"说不清命中了什么"，不虚标具体会议。

    ⚠️ `comment` 仅用于**证据表述**（注明 venue 确实来自 comment 的哪段），不参与判档。
    """
    raw = (venue or "").strip()
    text = re.sub(r"\s+", " ", raw)

    evidence: dict[str, Any] = {
        "matched": None,
        "fragment": None,
        "modifier": None,
        # 默认来源口径：papers.venue 本身就是入库时从 arXiv comment 抽出来的（ADR D9）。
        "found_in": "papers.venue（入库时抽取自 arXiv comment）",
        "comment_excerpt": None,
        "note": "",
    }

    if not text:
        evidence["note"] = "venue 字段为空：只有 arXiv 预印本，没有任何发表信息可依据。"
        return {"tier": TIER_PREPRINT, "label": _LABELS[TIER_PREPRINT], "evidence": evidence}

    modifier = _DOWNGRADE_RE.search(text)
    top_hit = _earliest(_TOP_PATTERNS, text)
    core_hit = _earliest(_CORE_PATTERNS, text)

    if top_hit is not None:
        name, start = top_hit
        m = _TOP_PATTERNS[name].search(text)
        fragment = text[start : start + (m.end() - m.start() if m else len(name))]
        evidence["matched"] = name
        evidence["fragment"] = fragment
        if modifier is not None:
            # 顶会名 + Findings/Workshop 等附属卷 → 降为 core。
            tier = TIER_CORE
            evidence["modifier"] = modifier.group(0).lower()
            evidence["note"] = (
                f"命中顶会名「{name}」，但 venue 同时含附属卷标记「{modifier.group(0)}」，"
                f"不是主会录用，保守降为 core。"
            )
        else:
            tier = TIER_TOP
            evidence["note"] = f"命中顶会/顶刊白名单「{name}」。"
    elif core_hit is not None:
        name, start = core_hit
        m = _CORE_PATTERNS[name].search(text)
        fragment = text[start : start + (m.end() - m.start() if m else len(name))]
        tier = TIER_CORE
        evidence["matched"] = name
        evidence["fragment"] = fragment
        if modifier is not None:
            evidence["modifier"] = modifier.group(0).lower()
        evidence["note"] = f"命中会议/期刊白名单「{name}」（非顶档）。"
    elif _GENERIC_NAME_RE.search(text) or _ACRONYM_YEAR_RE.search(text):
        # 形状明确是"某个会议/期刊 + 年份"，但白名单点不出名 —— 给 core，
        # 并把 matched 留 None：说清"没认出具体是谁"，不假称命中了某个名字。
        tier = TIER_CORE
        m = _GENERIC_NAME_RE.search(text) or _ACRONYM_YEAR_RE.search(text)
        assert m is not None  # 上面 or 已保证
        evidence["fragment"] = m.group(0)
        evidence["note"] = "未命中白名单，但 venue 呈明确的会议/期刊名形状，给 core；白名单外的名字不予点名。"
    elif _ARXIV_RE.search(text):
        tier = TIER_PREPRINT
        evidence["note"] = "venue 只含 arXiv 字样，视为预印本。"
    else:
        # 兜底：一个既不像会议名也不像预印本的字符串 —— 判不准就不判（unknown），
        # 绝不因为"看起来像发表"就送一个 core/top。
        tier = TIER_UNKNOWN
        evidence["note"] = f"venue「{text}」无法识别，保守返回 unknown，不作质量信号。"

    # comment 只用来加强证据表述：命中片段确实出现在 comment 里时，注明来自哪一段。
    if comment and evidence["fragment"]:
        excerpt = _excerpt(comment, str(evidence["fragment"]))
        if excerpt is not None:
            evidence["found_in"] = "arXiv comment（venue 与 comment 定位一致）"
            evidence["comment_excerpt"] = excerpt

    return {"tier": tier, "label": _LABELS[tier], "evidence": evidence}


__all__ = [
    "TIER_CORE",
    "TIER_PREPRINT",
    "TIER_TOP",
    "TIER_UNKNOWN",
    "lookup",
]
