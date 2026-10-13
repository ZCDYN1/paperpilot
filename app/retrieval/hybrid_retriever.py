"""混合检索门面 —— **Agent 层唯一允许接触的检索入口**。

架构文档 §2 的关键设计原则：Agent **不允许**直接碰 FAISS / BM25 / DB。
所有检索能力都收敛到这一个类。好处有三个：
① 检索策略可以独立做消融实验；② 换 Agent 框架时检索层完全复用；③ 单测好写。

完整流水线：

```
1) metadata 过滤        → 得到 chunk 白名单（先过滤再检索）
2) 双路召回             → FAISS(语义) + BM25(词面)
3) RRF 融合             → 仍在 chunk 粒度
4) ★ chunk → paper 聚合 → 这一步最容易漏：融合结果是 chunk，
                          但对比表、去重口径、Recall@K 全是 paper 级
5) 交叉编码器重排        → 对每篇的代表 chunk 打分
6) 返回 paper 级结果     → 每篇带上命中的 chunk，供引用溯源使用
```
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog

from app.core.config import Settings, get_settings
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import BaseEmbedder, get_embedder
from app.retrieval.faiss_store import FaissStore
from app.retrieval.fusion import rrf_fuse, weighted_fuse
from app.retrieval.reranker import BaseReranker, get_reranker

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ChunkRecord:
    """检索层需要的最小 chunk 视图（与 ORM 解耦，便于单测）。"""

    chunk_id: int
    arxiv_id: str
    section: str
    text: str
    source: str = "abstract"      # html | abstract
    chunk_type: str = "text"
    year: int | None = None
    primary_category: str | None = None
    page: int | None = None


@dataclass(slots=True)
class MetadataFilter:
    """metadata 过滤条件。限定分类 / 年份是真实且高频的需求。"""

    categories: Sequence[str] | None = None
    year_gte: int | None = None
    year_lte: int | None = None
    arxiv_ids: Sequence[str] | None = None


@dataclass
class PaperHit:
    """paper 级检索结果。"""

    arxiv_id: str
    title: str
    score: float
    rep_chunk: ChunkRecord                 # 代表 chunk（分数最高，用于重排与展示）
    chunks: list[ChunkRecord] = field(default_factory=list)
    matched_by: list[str] = field(default_factory=list)  # ["vector", "bm25"]，便于调试与前端展示


@dataclass(slots=True)
class SearchConfig:
    """检索流水线的开关面板 —— **消融实验的自变量**。

    默认值必须与线上生产行为**逐位一致**（双路 + RRF + 重排 + 过滤全开）。
    理由：消融表里的"混合检索（④）"必须就是线上真正跑的那条链路，
    否则我们比的是"实验实现 vs 线上实现"，而不是两种检索策略。
    """

    use_vector: bool = True            # 启用 FAISS 语义召回
    use_bm25: bool = True              # 启用 BM25 词面召回
    fusion: str = "rrf"                # rrf | weighted | none
    vector_weight: float = 0.5         # 仅 fusion="weighted" 时生效
    bm25_weight: float = 0.5
    use_rerank: bool = True            # 交叉编码器重排
    use_metadata_filter: bool = True   # 分类 / 年份白名单过滤


# 线上默认配置。之所以导出成模块级常量，是为了让调用方能显式引用
# "生产配置"而不是各写一份，避免默认值悄悄漂移。
DEFAULT_SEARCH_CONFIG = SearchConfig()


class IndexCompatibilityError(RuntimeError):
    """索引与当前 embedding 配置不匹配。

    这是使用者最容易踩、也最难自查的一类问题：**换了 embedding 模型却没重建索引**。

    此时向量空间已经变了，检索结果会变成噪声 —— 但如果维度恰好相同，
    程序不会报任何错，用户看到的只是"结果好像不太相关"。
    单靠一句 RuntimeError 用户根本想不到问题出在索引上，
    所以本异常的消息**必须直接给出该跑哪条命令**。
    """


class HybridRetriever:
    """向量 + BM25 混合检索的编排者。"""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        reranker: BaseReranker | None = None,
        settings: Settings | None = None,
        *,
        index_dir: Path | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.reranker = reranker or get_reranker(self.settings)

        index_dir = Path(index_dir or self.settings.faiss_index_dir)
        self.index_dir = index_dir
        self.faiss_path = index_dir / "paperpilot.faiss"
        self.bm25_path = index_dir / "paperpilot.bm25.pkl"
        self.meta_path = index_dir / "paperpilot.meta.npz"

        self.faiss = FaissStore.load(self.faiss_path, self.embedder.dim) if self.faiss_path.exists() else FaissStore(self.embedder.dim, self.faiss_path)
        self.bm25 = BM25Store.load(self.bm25_path) if self.bm25_path.exists() else BM25Store(self.bm25_path)

        self._chunks: dict[int, ChunkRecord] = {}
        self._titles: dict[str, str] = {}
        # 索引是用哪一版 embedding 建的。None 表示"未知"（老索引没写这个字段），
        # 未知时不阻塞检索，但要提醒 —— 直接报错会让老用户升级后完全用不了。
        self.index_identity: str | None = None
        self.index_built_at: str | None = None
        # 索引文件三件套（faiss / bm25 / meta）必须一起恢复：只 load 向量与 BM25
        # 而不恢复 chunk 元数据，`_chunks` 会是空的，`is_empty` 判定为真，
        # 于是"进程重启后检索永远返回空"—— 这是最隐蔽的一类 bug（不报错，只是没结果）。
        if self.meta_path.exists():
            self.load_meta()

    # ------------------------------------------------------------------ 状态
    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    def check_index_compatibility(self) -> str | None:
        """检查索引与当前 embedding 配置是否匹配。

        Returns:
            `None` 表示可以放心用（匹配、索引为空、或来源未知）；
            否则返回一段**可直接展示给用户**的说明与修复命令。

        为什么返回字符串而不是 bool？
        因为这个检查的全部价值就在于"告诉用户该怎么做"。返回 bool 的话，
        每个调用方都得自己再拼一遍提示语 —— 迟早有调用方漏掉，
        而漏掉的那个往往正是用户唯一会看到的地方。
        """
        # 空索引没必要拦（本来就没东西可检索）；来源未知也不拦（见 __init__ 的说明）
        if self.is_empty or self.index_identity is None:
            return None

        current = self.embedder.identity
        if self.index_identity == current:
            return None

        return (
            "索引与当前 embedding 配置不匹配：\n"
            f"  · 索引是用  {self.index_identity}  建立的\n"
            f"  · 当前配置  {current}\n"
            "不同 embedding 模型的向量空间互不相通 —— 维度即使相同，继续检索也只会得到噪声。\n"
            "修复：python scripts/build_index.py   （语料存在数据库里，重建不需要重新抓 arXiv）"
        )

    # ------------------------------------------------------------------ 建索引
    async def build_index(self, records: Sequence[ChunkRecord], titles: dict[str, str]) -> int:
        """重建全部索引（幂等）。

        BM25 是全局统计量（新增文档会改变 IDF），所以只能整体重建；
        FAISS 侧顺带用上「重建 + 原子切换」，避免读侧看到半截索引。
        """
        records = list(records)
        self._titles = dict(titles)
        self._chunks = {r.chunk_id: r for r in records}

        if not records:
            self.faiss.rebuild_atomic([], np.zeros((0, self.embedder.dim), dtype="float32"))
            self.bm25.build([], [])
            self.bm25.save()
            self._save_meta()
            logger.info("retriever.index_built", chunks=0)
            return 0

        ids = [r.chunk_id for r in records]
        texts = [r.text for r in records]

        vectors = await self.embedder.encode(texts)
        self.faiss.rebuild_atomic(ids, vectors)
        self.bm25.build(ids, texts)
        # BM25 是内存对象（BM25Okapi 无通用交换格式），build 之后**必须显式落盘**。
        # 漏掉这一步的后果很隐蔽：同进程内检索正常，重启后 BM25 静默变空，
        # 混合检索悄悄退化成纯向量检索 —— 结果变差但完全没有任何报错。
        self.bm25.save()
        self._save_meta()

        logger.info("retriever.index_built", chunks=len(records), dim=self.embedder.dim)
        return len(records)

    # ------------------------------------------------------------------ 检索
    async def search(
        self,
        query: str,
        keywords: Sequence[str] | None = None,
        *,
        top_k: int | None = None,
        fuse_top_n: int | None = None,
        rerank_top_n: int | None = None,
        filters: MetadataFilter | None = None,
        config: SearchConfig | None = None,
    ) -> list[PaperHit]:
        """执行完整混合检索。

        Args:
            query: 用户的自然语言查询（中文也可以，向量侧支持跨语言）。
            keywords: **英文**术语列表，供 BM25 使用。不传则退化为用 query 分词 ——
                      但中文 query 打英文索引效果很差，所以强烈建议由 Planner 提供。
            filters: metadata 过滤条件。
            config: 检索配置开关。**默认 None 等价于线上生产配置**（双路 + RRF + 重排 +
                    过滤全开），只有消融实验会传非默认值。做成参数而不是拆成几个函数，
                    是为了让消融实验与线上走**同一条代码路径** —— 否则两边的细微差异
                    会污染对比结论，那样的消融表没有说服力。
        """
        cfg = config or DEFAULT_SEARCH_CONFIG

        if self.is_empty:
            logger.warning("retriever.empty_index")
            return []

        if not cfg.use_vector and not cfg.use_bm25:
            # 两路都关就没有召回来源了。显式报错而非静默返回空 ——
            # 静默返回空会让消融表里出现"某配置效果为 0"的假象。
            raise ValueError("SearchConfig 至少需启用一路召回（use_vector / use_bm25）")

        # 只在真正会用到向量时才拦：纯 BM25 检索完全不依赖 embedding，
        # 索引指纹不匹配也照样能用 —— 拦它等于把一个本来可用的功能一起废掉。
        if cfg.use_vector:
            mismatch = self.check_index_compatibility()
            if mismatch:
                raise IndexCompatibilityError(mismatch)

        s = self.settings
        top_k = top_k or s.retrieve_top_k
        fuse_top_n = fuse_top_n or s.fuse_top_n
        rerank_top_n = rerank_top_n or s.rerank_top_n

        # 1) metadata 过滤（先过滤再检索，顺序很关键）
        allow_ids = self._prefilter(filters) if cfg.use_metadata_filter else None
        if allow_ids is not None and not allow_ids:
            return []

        # 2) 双路召回（按配置裁剪；关掉的那一路连 encode 都不做，省时也免去依赖）
        vec_hits: list[tuple[int, float]] = []
        sparse_hits: list[tuple[int, float]] = []
        if cfg.use_vector:
            query_vec = (await self.embedder.encode([query]))[0]
            vec_hits = self.faiss.search(query_vec, top_k, allow_ids=allow_ids)
        if cfg.use_bm25:
            sparse_hits = self.bm25.search(keywords or query, top_k, allow_ids=allow_ids)

        matched: dict[int, set[str]] = defaultdict(set)
        for cid, _ in vec_hits:
            matched[cid].add("vector")
        for cid, _ in sparse_hits:
            matched[cid].add("bm25")

        # 3) 融合（chunk 粒度）—— 融合算法是消融实验的核心自变量
        fused = self._fuse(vec_hits, sparse_hits, cfg, s.rrf_k, fuse_top_n)
        if not fused:
            return []

        # 4) chunk → paper 聚合（v1 漏掉的一步）
        candidates = self._aggregate_to_papers(fused, matched, s.max_chunks_per_paper)

        # 5) 交叉编码器重排
        if cfg.use_rerank and candidates:
            docs = [hit.rep_chunk.text for hit in candidates]
            ranked = await self.reranker.rerank(query, docs, top_n=rerank_top_n)
            reranked: list[PaperHit] = []
            for idx, score in ranked:
                hit = candidates[idx]
                hit.score = score
                reranked.append(hit)
            candidates = reranked

        logger.info(
            "retriever.search_done",
            query_len=len(query),
            vec_hits=len(vec_hits),
            sparse_hits=len(sparse_hits),
            fused=len(fused),
            papers=len(candidates),
            fusion=cfg.fusion,
        )
        return candidates

    @staticmethod
    def _fuse(
        vec_hits: Sequence[tuple[int, float]],
        sparse_hits: Sequence[tuple[int, float]],
        cfg: SearchConfig,
        rrf_k: int,
        top_n: int,
    ) -> list[tuple[int, float]]:
        """按 `SearchConfig.fusion` 选择融合算法。

        三档意义各不相同：
        `rrf` 是线上默认（只看名次，天然免疫量纲问题）；
        `weighted` 是它的对照组（看归一化后的分数）；
        `none` 只做保序合并 —— 给"纯向量 / 纯 BM25"这类单路配置用，
        避免单路也被 RRF 的 `1/(k+rank)` 重新赋分，从而模糊"基线"的定义。
        """
        if cfg.fusion == "rrf":
            return rrf_fuse(
                [[cid for cid, _ in vec_hits], [cid for cid, _ in sparse_hits]],
                k=rrf_k,
                top_n=top_n,
            )
        if cfg.fusion == "weighted":
            return weighted_fuse(
                [vec_hits, sparse_hits],
                weights=[cfg.vector_weight, cfg.bm25_weight],
                top_n=top_n,
            )
        if cfg.fusion == "none":
            merged: dict[int, float] = {}
            # 保序合并：先出现的名次更靠前。分数用「-当前已有条数」表示，
            # 这样越早出现分数越大，可以直接套用统一的降序排序。
            for lst in (vec_hits, sparse_hits):
                for cid, _ in lst:
                    if cid not in merged:
                        merged[cid] = -float(len(merged))
            return sorted(merged.items(), key=lambda kv: (-kv[1], str(kv[0])))[:top_n]
        raise ValueError(f"未知的融合方式：{cfg.fusion!r}（可选 rrf / weighted / none）")

    # ------------------------------------------------------------ 内部逻辑
    def _prefilter(self, filters: MetadataFilter | None) -> set[int] | None:
        if filters is None:
            return None

        allow: set[int] = set()
        for cid, rec in self._chunks.items():
            if filters.categories and rec.primary_category not in set(filters.categories):
                continue
            if filters.year_gte is not None and (rec.year or 0) < filters.year_gte:
                continue
            if filters.year_lte is not None and (rec.year or 9999) > filters.year_lte:
                continue
            if filters.arxiv_ids and rec.arxiv_id not in set(filters.arxiv_ids):
                continue
            allow.add(cid)
        return allow

    def _aggregate_to_papers(
        self,
        fused: Sequence[tuple[int, float]],
        matched: dict[int, set[str]],
        max_chunks_per_paper: int,
    ) -> list[PaperHit]:
        """chunk 级融合结果 → paper 级候选。

        同一篇论文可能贡献多个 chunk，它们在融合结果里互相竞争名次。
        这里取每篇最高分的 chunk 作为代表分，并最多保留 `max_chunks_per_paper` 个 chunk ——
        既保住论文多样性，又不让某篇长论文霸占全部重排名额。
        """
        buckets: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for chunk_id, score in fused:
            rec = self._chunks.get(chunk_id)
            if rec is None:
                continue
            buckets[rec.arxiv_id].append((chunk_id, score))

        hits: list[PaperHit] = []
        for arxiv_id, items in buckets.items():
            items.sort(key=lambda kv: (-kv[1], kv[0]))
            kept = items[:max_chunks_per_paper]
            rep = self._chunks[kept[0][0]]
            by: set[str] = set()
            for cid, _ in kept:
                by |= matched.get(cid, set())
            hits.append(
                PaperHit(
                    arxiv_id=arxiv_id,
                    title=self._titles.get(arxiv_id, arxiv_id),
                    score=kept[0][1],
                    rep_chunk=rep,
                    chunks=[self._chunks[cid] for cid, _ in kept],
                    matched_by=sorted(by),
                )
            )

        hits.sort(key=lambda h: (-h.score, h.arxiv_id))
        return hits

    # -------------------------------------------------------------- 持久化
    def _save_meta(self) -> None:
        """把 chunk 元数据以 npz（对象数组）落盘，避免依赖额外格式。"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        built_at = datetime.now().isoformat(timespec="seconds")
        ids = np.asarray(sorted(self._chunks), dtype="int64")
        payload = {
            "ids": ids,
            "records": np.asarray(
                [
                    (
                        self._chunks[i].arxiv_id,
                        self._chunks[i].section,
                        self._chunks[i].text,
                        self._chunks[i].source,
                        self._chunks[i].chunk_type,
                        self._chunks[i].year if self._chunks[i].year is not None else -1,
                        self._chunks[i].primary_category or "",
                        self._chunks[i].page if self._chunks[i].page is not None else -1,
                    )
                    for i in ids
                ],
                dtype=object,
            ),
            "title_ids": np.asarray(sorted(self._titles), dtype=object),
            "title_values": np.asarray([self._titles[k] for k in sorted(self._titles)], dtype=object),
            # ★ 索引指纹：记录"这一版向量是用什么模型算出来的"。
            #   没有这一条，用户换 embedding 服务商后无从判断该不该重建索引 ——
            #   而这恰恰是"clone 下来接自己的 API"时最常见的一步操作。
            "embedding_identity": np.asarray([self.embedder.identity], dtype=object),
            "built_at": np.asarray([built_at], dtype=object),
        }
        np.savez_compressed(self.meta_path, **payload)
        # 落盘之后，索引的指纹就是当前 embedder 的指纹（供同进程内后续检查直接使用）
        self.index_identity = self.embedder.identity
        self.index_built_at = built_at

        # 再写一份**极简清单**。存在的理由很实际：健康检查是高频接口，
        # 不能每次都把几百条 chunk 文本从 npz 里反序列化一遍才回答"索引就绪吗"。
        # 顺带它也是纯文本 JSON —— 用户遇到问题时可以直接 cat 出来看。
        manifest = {
            "embedding_identity": self.embedder.identity,
            "built_at": built_at,
            "chunks": len(self._chunks),
            "papers": len(self._titles),
            "reranker": getattr(self.reranker, "name", None),
        }
        (self.index_dir / "paperpilot.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_meta(self) -> None:
        """从磁盘恢复 chunk 元数据映射（索引文件仍需各自 load）。"""
        if not self.meta_path.exists():
            return
        data = np.load(self.meta_path, allow_pickle=True)
        ids = data["ids"]
        records = data["records"]
        self._chunks = {}
        for cid, row in zip(ids, records, strict=True):
            arxiv_id, section, text, source, chunk_type, year, category, page = row
            self._chunks[int(cid)] = ChunkRecord(
                chunk_id=int(cid),
                arxiv_id=arxiv_id,
                section=section,
                text=text,
                source=source,
                chunk_type=chunk_type,
                year=None if year == -1 else int(year),
                primary_category=category or None,
                page=None if page == -1 else int(page),
            )
        self._titles = {
            str(k): str(v) for k, v in zip(data["title_ids"], data["title_values"], strict=True)
        }
        # 老索引没有这两个字段 → 保持 None，交由 check_index_compatibility 按"未知"处理。
        # 这里刻意不报错：否则升级代码后老用户连检索都跑不起来，代价远大于收益。
        if "embedding_identity" in data:
            self.index_identity = str(data["embedding_identity"][0])
        if "built_at" in data:
            self.index_built_at = str(data["built_at"][0])
