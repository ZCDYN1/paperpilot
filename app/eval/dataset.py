"""golden 评测集的加载与校验。

**为什么单独抽出数据集模块，而不是散在脚本里？**

金标准（golden set）是整个评测的地基 —— 消融实验、RAGAS 评测、引用准确率
三处都要读它。把格式、校验、加载集中在一处，才能保证三处用的是**同一份数据、
同一套口径**；否则迟早出现"消融表说 Recall@5 = 0.80，RAGAS 却说 context_recall = 0.50"
这种自相矛盾的结论，而排查成本极高（等于要重做一遍评测）。

加载时**强校验英文关键词**：BM25 是词面匹配，中文关键词打英文索引等于引入噪声。
这条约束在设计文档里被反复强调，所以在数据入口处就用硬错误拦住，
而不是等消融表跑出一堆难看的数字后再回头怀疑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 默认数据集：与模块同级的 datasets/ 目录。
# 用 __file__ 定位而不是相对 CWD，保证从任何工作目录调用都能找到。
#
# `golden_20.jsonl` 是最初 5 篇语料时期的标注（20 条），保留作对照；
# `golden_40.jsonl` 是语料扩到 24 篇后重标的版本（40 条，覆盖全部 24 篇论文），
# 因此作为默认。语料规模变了就要重标 —— 否则新增论文没有金标准，
# 它们对评测就是"死重量"：只会稀释指标，不会提升区分度。
DEFAULT_GOLDEN_PATH = Path(__file__).resolve().parent / "datasets" / "golden_40.jsonl"


@dataclass(frozen=True, slots=True)
class GoldenItem:
    """一条金标准样本。"""

    id: str
    question: str
    keywords: tuple[str, ...]
    gold_paper_ids: tuple[str, ...]
    # 可选：模拟 Planner 会产出的 arXiv 分类约束。
    # 它存在的意义是让消融实验的「⑥ metadata 过滤」有可过滤的东西 ——
    # 若不设，⑥ 会与 ⑤ 完全等价，表格里那一行就是废的（比没有更糟：读者会以为验证过了）。
    # 跨领域的查询刻意留空：单分类过滤会让它们漏掉另一领域的金标准，反而变差。
    categories: tuple[str, ...] = ()
    reference_answer: str = ""
    notes: str = ""

    @property
    def has_ground_truth(self) -> bool:
        """是否有参考答案。

        RAGAS 的 `answer_correctness` / `context_recall` 依赖参考答案，
        缺了就应跳过这两项而不是算成 0 分 —— 否则会把"没标注"误判成"答错了"。
        """
        return bool(self.reference_answer.strip())


class GoldenFormatError(ValueError):
    """golden 集格式错误。单独定义类型，便于上层区分"数据坏了"与"逻辑坏了"。"""


def _parse_line(raw: str, line_no: int) -> GoldenItem:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GoldenFormatError(f"第 {line_no} 行不是合法 JSON：{exc}") from exc

    if not isinstance(obj, dict):
        raise GoldenFormatError(f"第 {line_no} 行应为 JSON 对象，实际是 {type(obj).__name__}")

    missing = [k for k in ("id", "question", "gold_paper_ids") if not obj.get(k)]
    if missing:
        raise GoldenFormatError(f"第 {line_no} 行缺少必填字段：{', '.join(missing)}")

    keywords = tuple(obj.get("keywords") or ())
    for kw in keywords:
        # BM25 只认词面。中文关键词打英文索引会引入噪声且不报错，
        # 属于"结果变差但查不出原因"的典型，所以在入口处直接拦下来。
        if not kw.isascii():
            raise GoldenFormatError(
                f"第 {line_no} 行（id={obj['id']}）的关键词 {kw!r} 含非 ASCII 字符；"
                "BM25 需要英文术语，请改写为英文（可含同义/缩写扩展）"
            )

    return GoldenItem(
        id=str(obj["id"]),
        question=str(obj["question"]),
        keywords=keywords,
        gold_paper_ids=tuple(obj["gold_paper_ids"]),
        categories=tuple(obj.get("categories") or ()),
        reference_answer=str(obj.get("reference_answer") or ""),
        notes=str(obj.get("notes") or ""),
    )


def load_golden(path: Path | str | None = None) -> list[GoldenItem]:
    """读取 golden 集（JSONL：一行一条）。

    空行会被跳过（便于人工增删条目）；id 重复则报错 ——
    重复 id 会让明细表无法定位到具体样本，属于必须当场发现的问题。
    """
    p = Path(path) if path is not None else DEFAULT_GOLDEN_PATH
    if not p.exists():
        raise FileNotFoundError(f"找不到 golden 数据集：{p}")

    items: list[GoldenItem] = []
    seen: set[str] = set()
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            item = _parse_line(raw, line_no)
            if item.id in seen:
                raise GoldenFormatError(f"第 {line_no} 行 id={item.id!r} 重复")
            seen.add(item.id)
            items.append(item)

    return items
