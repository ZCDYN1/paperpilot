"""F11「术语解释」的词表与匹配。

## 为什么是一张写死的词表，而不是让模型现场解释
这个产品的立身之本是"没有依据的话就不说"（产品定义 §6 核心差异两条）。
术语解释恰好是最容易编的地方：模型对没听过的缩写会给出**读起来很顺、但是错的**定义，
而读者恰恰因为没有怀疑理由而照抄。所以这里的规则是：

- 词表命中 → 给解释；
- 词表没有 → `known=False`，**不猜**。前端此时应当显示"词表内暂无该词条"，
  而不是退回一个 LLM 生成的"通用解释"。

代价是覆盖面受限于人工维护的条目数。这是刻意换取的那部分可信度。
"""

from __future__ import annotations

import re
from typing import Any

# 规范词条 → (别名, 中文解释, 分类)。别名用于匹配，规范词用于展示。
# 只收"本产品的用户在读 arXiv CS 论文时高频遇到、且中文材料里常被混用"的术语。
_GLOSSARY: dict[str, tuple[tuple[str, ...], str, str]] = {
    "RAG": (
        ("retrieval-augmented generation", "retrieval augmented generation"),
        "先从外部语料库检索相关片段、再把片段连同问题一起交给语言模型生成答案的范式。"
        "目的是让答案可溯源、并缓解模型知识过时。",
        "范式",
    ),
    "hybrid retrieval": (
        ("hybrid search", "hybrid retrieval"),
        "同时使用稠密向量检索与词面（稀疏）检索，再融合两路结果。"
        "两路的失败模式互补：向量擅长同义改写，词面擅长精确术语。",
        "检索",
    ),
    "dense retrieval": (
        ("dense retrieval", "dense passage retrieval"),
        "把查询与文档都编码成稠密向量，用向量相似度做召回；靠模型学到语义相关性。",
        "检索",
    ),
    "sparse retrieval": (
        ("sparse retrieval", "lexical retrieval"),
        "按词面匹配召回（BM25 是代表）。对没见过的术语、代码、型号名更稳，但对同义改写无力。",
        "检索",
    ),
    "BM25": (
        ("bm25", "okapi bm25"),
        "经典的词频-逆文档频率打分函数，衡量查询词与文档的词面匹配程度，分数无上界。",
        "检索",
    ),
    "bi-encoder": (
        ("bi-encoder", "bi encoder", "dual encoder"),
        "查询与文档各自独立编码成向量，可预先算好文档向量、检索时只做近邻查找。"
        "快，但看不到词与词之间的细粒度交互。",
        "模型结构",
    ),
    "cross-encoder": (
        ("cross-encoder", "cross encoder"),
        "把「查询 + 文档」拼成一个输入过一次模型，能建模词间交互因而更准；"
        "代价是每个候选都要过一次前向，无法预计算，通常只用于对少量候选重排。",
        "模型结构",
    ),
    "reranking": (
        ("rerank", "reranking", "re-ranking"),
        "对第一路召回出的少量候选重新打分排序，常用交叉编码器或大模型。"
        "本项目的重排输入必须是 chunk 文本而不是 id。",
        "检索",
    ),
    "late interaction": (
        ("late interaction", "colbert"),
        "介于双塔与交叉编码器之间：文档与查询各自编码成 token 级向量，"
        "打分时才做 token 间的最大相似求和。精度接近交叉编码器，索引可预计算，"
        "代价是索引体积显著膨胀。",
        "模型结构",
    ),
    "RRF": (
        ("reciprocal rank fusion", "reciprocal rank"),
        "倒数秩融合：只用各路结果的**名次**按 1/(k+rank) 累加。"
        "绕开了不同检索器分数量纲不可比的问题（BM25 无上界、余弦在 [-1,1]）。",
        "融合",
    ),
    "nDCG": (
        ("ndcg", "normalized discounted cumulative gain"),
        "归一化折损累计增益：位置越靠后收益折损越多，再除以理想排序做归一，"
        "用于评价「相关文档排得够不够靠前」。",
        "指标",
    ),
    "recall@k": (
        ("recall@5", "recall@10", "recall at k"),
        "前 k 个结果里命中了多少比例的本应命中文档。检索层常用它衡量「漏了多少」。",
        "指标",
    ),
    "precision@k": (
        ("precision@5", "precision at k"),
        "前 k 个结果里有多少是相关的。与 recall@k 一起看，单看一个容易被阈值带偏。",
        "指标",
    ),
    "ablation study": (
        ("ablation", "ablation study"),
        "消融实验：一次只改动一个组件（如关掉重排），量化它对指标的贡献，"
        "用来回答「这个组件值不值这个延迟」。",
        "评测",
    ),
    "embedding": (
        ("embedding", "vector embedding", "embeddings"),
        "把文本映射成固定维度向量的过程或结果；不同模型产出的向量空间互不通用，"
        "换模型必须重建索引。",
        "基础概念",
    ),
    "FAISS": (
        ("faiss",),
        "Facebook 开源的向量近邻检索库。注意它的 search 不支持任意条件过滤，"
        "metadata 过滤要走 IDSelector。",
        "工具",
    ),
    "fine-tuning": (
        ("fine-tune", "fine-tuning", "finetuning"),
        "在预训练模型上用下游数据继续训练。与「提示工程」相对：改权重 vs 改输入。",
        "训练",
    ),
    "LoRA": (
        ("lora", "low-rank adaptation"),
        "低秩适配：冻结原权重，只训练一对小的低秩矩阵，显著降低微调的显存与存储开销。"
        "代价是能力上限通常低于全参数微调。",
        "训练",
    ),
    "distillation": (
        ("knowledge distillation", "distillation"),
        "用大模型的输出（软标签或生成结果）训练小模型，换取推理成本下降，"
        "通常会损失一部分精度与多样性。",
        "训练",
    ),
    "instruction tuning": (
        ("instruction tuning", "instruction-following"),
        "用「指令-回答」对继续训练，让模型学会按人类意图作答，而非只做下一词预测。",
        "训练",
    ),
    "chain-of-thought": (
        ("chain of thought", "cot"),
        "让模型先输出中间推理步骤再给结论。对多步任务常有帮助，"
        "但**不会**让它对事实更诚实，也换不来可溯源的引用。",
        "提示技术",
    ),
    "hallucination": (
        ("hallucination", "hallucinations"),
        "模型输出看起来合理但没有事实依据的内容。学术场景里最有害的形态是编造引用，"
        "因为读者很难在不查原文的情况下发现。",
        "基础概念",
    ),
    "preprint": (
        ("preprint", "pre-print"),
        "尚未经同行评审就公开的论文（arXiv 上的绝大多数属于此类）。"
        "因此本产品的定位是「筛选与整理」，不判断论文结论真伪。",
        "基础概念",
    ),
    "SOTA": (
        ("state of the art", "sota"),
        "当前最优。⚠️ 论文里的 SOTA 声明高度依赖其自选基线与数据集，"
        "跨论文比较时必须注明数据集与指标，否则比的不是同一件事。",
        "基础概念",
    ),
    "benchmark": (
        ("benchmark", "benchmarks"),
        "公开数据集 + 固定评测协议的组合。同一模型在不同 benchmark 上的数字不可直接比较。",
        "评测",
    ),
    "chunking": (
        ("chunking", "chunk", "chunks"),
        "把长文切成可检索的小段。切得细定位准但丢上下文，切得粗反之；"
        "本项目按章节结构切，并保留 chunk 到论文与章节的映射。",
        "检索",
    ),
    "query expansion": (
        ("query expansion", "query rewriting"),
        "改写或扩展用户查询以提高召回（如补同义词、拆子问题）。"
        "本项目的「研究问题拆解」就是一种结构化查询扩展。",
        "检索",
    ),
    "agent": (
        ("agent", "agentic"),
        "让模型在多轮里自主决定调用哪些工具、何时停止的程序结构。"
        "与固定流水线的区别是把控制交给了模型，因此必须配校验与退出条件。",
        "范式",
    ),
}

# 匹配用：按"别名长度降序"扫描，保证 `cross-encoder reranking` 这类长词
# 不会被其中的 `cross-encoder` 抢先占位（否则标注会碎成一堆短词）。
_ALIASES: list[tuple[str, str]] = sorted(
    ((alias, term) for term, (aliases, _, _) in _GLOSSARY.items() for alias in (term, *aliases)),
    key=lambda kv: -len(kv[0]),
)


def _pattern(alias: str) -> re.Pattern[str]:
    """词边界匹配。`\u3000-\u9fff` 那段是中文边界 —— 中文没有空格，
    `\\b` 在「用RAG做检索」这种混排里会判不出边界，于是把 CJK 字符也当边界。
    """
    escaped = re.escape(alias).replace(r"\ ", r"[\s\-]?")
    return re.compile(
        rf"(?<![\w\u3000-\u9fff]){escaped}(?![\w\u3000-\u9fff])",
        re.IGNORECASE,
    )


_PATTERNS: dict[str, re.Pattern[str]] = {a: _pattern(a) for a, _ in _ALIASES}


def lookup(term: str) -> dict[str, Any] | None:
    "「」查一个术语。命中返回 `{term, explanation, category, aliases}`，没命中返回 None。「」"
    needle = (term or "").strip().lower()
    if not needle:
        return None
    for canonical, (aliases, explanation, category) in _GLOSSARY.items():
        candidates = {canonical.lower(), *(a.lower() for a in aliases)}
        if needle in candidates:
            return {
                "term": canonical,
                "explanation": explanation,
                "category": category,
                "aliases": list(aliases),
            }
    return None


def annotate(text: str) -> list[dict[str, Any]]:
    """标出文本里出现的术语及其位置，供前端做悬停/下划线。

    重叠的匹配只保留**更长**的那个（见 `_ALIASES` 的排序理由）。
    返回按出现位置排序的 `{term, matched, start, end, explanation}`。
    """
    body = text or ""
    found: list[dict[str, Any]] = []
    taken: list[tuple[int, int]] = []
    for alias, canonical in _ALIASES:
        for m in _PATTERNS[alias].finditer(body):
            start, end = m.span()
            # 与已接受的匹配重叠就丢掉：短词被长词包含时不该再标一次
            if any(start < e2 and s2 < end for s2, e2 in taken):
                continue
            taken.append((start, end))
            entry = lookup(canonical) or {}
            found.append(
                {
                    "term": canonical,
                    "matched": m.group(0),
                    "start": start,
                    "end": end,
                    "explanation": entry.get("explanation", ""),
                    "category": entry.get("category", ""),
                }
            )
    return sorted(found, key=lambda item: item["start"])


def extract_terms(text: str) -> list[dict[str, Any]]:
    "「」文本里出现的术语（去重后的词条列表），供「这段里有哪些术语」类接口用。「」"
    out: dict[str, dict[str, Any]] = {}
    for hit in annotate(text):
        out.setdefault(hit["term"], {k: hit[k] for k in ("term", "explanation", "category")})
    return list(out.values())


def unknown_terms(terms: list[str]) -> list[str]:
    "「」批量查词时，把词表里没有的挑出来（前端据此显示「暂无词条」而不是空白）。「」"
    return [t for t in (terms or []) if lookup(t) is None]


__all__ = ["annotate", "extract_terms", "lookup", "unknown_terms"]
