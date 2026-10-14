"""F19 `app/analysis/venue_tiers.lookup` 的纯单测（无 DB、无 IO）。

锁死四条：
1. 白名单顶会才给 `top`，且 `evidence` 写清命中的会议名与输入串里的命中片段；
2. "顶会名 + Findings/Workshop 等附属卷"一律降 `core`（升档误判 = 编造质量信号）；
3. venue 为空 / 只有 arXiv → `preprint`；判不准的陌生串 → `unknown`，**绝不给高档位**；
4. 各种脏输入（None、全空格、全角字符、纯数字、超长串）不抛异常。
"""

from __future__ import annotations

from app.analysis.venue_tiers import lookup

_EVIDENCE_KEYS = {"matched", "fragment", "modifier", "found_in", "comment_excerpt", "note"}


def test_known_top_acronyms():
    for v in ("ACL 2024", "EMNLP 2023", "NeurIPS 2025", "ICML", "ICLR 2026", "CVPR 2025",
              "AAAI 2024", "IJCAI", "SIGIR 2022", "KDD", "JMLR", "TPAMI"):
        r = lookup(v)
        assert r["tier"] == "top", v
        assert r["label"] == "顶会/顶刊"
        assert set(r["evidence"]) == _EVIDENCE_KEYS
        assert r["evidence"]["matched"]  # top 必须点得出命中的是哪个会议名


def test_acl_2024_found_in_comment_shapes_evidence():
    """真实任务给的形状：venue 来自 comment 抽取；传 comment 时证据要注明来自哪一段。"""
    r = lookup("ACL 2024", comment="10 pages. Accepted at ACL 2024, main conference.")
    assert r["tier"] == "top"
    ev = r["evidence"]
    assert ev["matched"] == "ACL"
    assert ev["fragment"] == "ACL"
    assert "arXiv comment" in ev["found_in"]
    assert "ACL 2024" in ev["comment_excerpt"]


def test_no_comment_still_top_and_falls_back_to_venue_source():
    r = lookup("ACL 2024")
    assert r["tier"] == "top"
    assert "papers.venue" in r["evidence"]["found_in"]
    assert r["evidence"]["comment_excerpt"] is None


def test_comment_without_fragment_does_not_claim_comment_source():
    """传了 comment 但 venue 片段并不在其中（历史脏数据）：证据不许假称来自 comment。"""
    r = lookup("EMNLP 2024", comment="something unrelated")
    assert r["tier"] == "top"
    assert "papers.venue" in r["evidence"]["found_in"]
    assert r["evidence"]["comment_excerpt"] is None


def test_findings_and_workshop_downgrade_from_top():
    r = lookup("EMNLP 2023 Findings")
    assert r["tier"] == "core"
    assert r["evidence"]["matched"] == "EMNLP"
    assert r["evidence"]["modifier"] == "findings"

    r = lookup("NeurIPS Workshop 2025")
    assert r["tier"] == "core"
    assert r["evidence"]["modifier"] == "workshop"


def test_lowercase_and_punctuated_variants():
    assert lookup("cvpr 2025")["tier"] == "top"
    assert lookup("ACL-2024")["tier"] == "top"
    assert lookup("  EMNLP 2026 (Oral) ")["tier"] == "top"


def test_core_whitelist():
    for v in ("NAACL 2022", "COLING 2020", "WSDM 2021", "CIKM 2019", "TACL", "AISTATS 2023"):
        r = lookup(v)
        assert r["tier"] == "core", v
        assert r["evidence"]["matched"]
    # TACL 里含 "ACL" 字样，但词边界保证它不被误判成顶会 ACL。
    assert lookup("TACL")["evidence"]["matched"] == "TACL"


def test_generic_venue_shapes_get_core_without_naming_a_conference():
    r = lookup("Proceedings of the Third International Conference on XX")
    assert r["tier"] == "core"
    assert r["evidence"]["matched"] is None  # 说不清命中了谁，就不点名
    r = lookup("Journal of Applied Thing Sciences")
    assert r["tier"] == "core"
    assert r["evidence"]["matched"] is None
    r = lookup("IJCKG 2023")  # 全大写缩写 + 年份（白名单外的真实会议常见形状）
    assert r["tier"] == "core"
    assert r["evidence"]["matched"] is None


def test_preprint_cases():
    for v in (None, "", "   ", "arXiv", "arXiv:2401.00001v2", "arXiv preprint only"):
        assert lookup(v)["tier"] == "preprint", repr(v)
    # 空 venue 的证据 matched/fragment 都是 None，note 解释为什么。
    r = lookup(None)
    assert r["evidence"]["matched"] is None and r["evidence"]["note"]


def test_unknown_never_inflates():
    for v in ("some random text", "2024", "会议", "ＡＣＬ ２０２４", "x" * 300, "hello foo bar"):
        r = lookup(v)
        assert r["tier"] in ("unknown", "preprint"), repr(v)
        assert r["tier"] != "top"


def test_lookalike_substrings_do_not_match_top():
    # "CAL 2024" 之类：词边界 + 白名单点名，不给误判成 ACL。
    assert lookup("CAL 2024")["tier"] != "top"


def test_dirty_inputs_never_raise_and_always_return_shape():
    for v in ("|", "\n\t", "【ACL】", "ACL" + " " * 200 + "2024"):
        r = lookup(v)
        assert set(r) == {"tier", "label", "evidence"}
        assert r["tier"] in ("top", "core", "preprint", "unknown")
