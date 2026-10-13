"""端到端 API 验证：SSE 流式研究 + HITL 恢复 + 结果查询。

用法：
    python scripts/verify_api.py                      # 默认打本机 :8000
    python scripts/verify_api.py --base http://127.0.0.1:8000
    python scripts/verify_api.py --question "你的问题"

前置：先把后端跑起来。离线验证（不需要任何 API key）：
    LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock RERANK_PROVIDER=lexical MIN_CORPUS=3 \\
      uvicorn app.main:app --port 8000

**为什么用 Python 而不是 curl**：Windows 上 curl 发中文 JSON 时，
终端编码（GBK）与 HTTP 要求的 UTF-8 不一致，会把中文发成乱码，
服务端校验直接 400。这不是服务端的问题，但很容易被误判成 bug。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import httpx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaperPilot 端到端 API 验证")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--question", default="RAG 的重排技术有哪些进展")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    return p.parse_args()


def read_sse(client: httpx.Client, url: str, payload: dict, timeout: float) -> tuple[int, list[str], dict]:
    """消费一次 SSE 流，返回 (状态码, 事件类型列表, 关键载荷)。"""
    events: list[str] = []
    highlights: dict = {}
    last_event = ""

    with client.stream("POST", url, json=payload, timeout=timeout) as resp:
        if resp.status_code >= 400:
            return resp.status_code, [], {"error": resp.read().decode("utf-8", "replace")[:600]}

        for line in resp.iter_lines():
            if line.startswith("event:"):
                last_event = line.split(":", 1)[1].strip()
                events.append(last_event)
            elif line.startswith("data:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if last_event == "plan_created":
                    highlights["sub_questions"] = data.get("sub_questions") or []
                elif last_event == "interrupted":
                    highlights["interrupted"] = data
                elif last_event == "verification":
                    highlights["verification"] = data
                elif last_event == "final_report":
                    highlights["final_report"] = data
                elif last_event == "token_usage":
                    highlights["token_usage"] = data
                elif last_event == "error":
                    highlights["error"] = data
                elif last_event == "run_started":
                    highlights["run_id"] = data.get("run_id")

    return 200, events, highlights


def main() -> int:
    args = parse_args()
    payload = {"question": args.question, "top_k": args.top_k}

    with httpx.Client() as client:
        # ---------- 1. 发起研究（会在 plan 之后中断，等用户确认子问题）----------
        print(f"▶ POST {args.base}/api/v1/research/stream")
        status, events, hl = read_sse(
            client, f"{args.base}/api/v1/research/stream", payload, args.timeout
        )
        if status >= 400:
            print(f"✗ HTTP {status}\n{hl.get('error')}")
            return 1

        run_id = hl.get("run_id", "")
        print(f"  HTTP {status} · 事件 {len(events)} 条 · run_id={run_id}")
        for name, count in Counter(events).most_common():
            print(f"    {count:>3}  {name}")

        if "error" in hl:
            print(f"✗ 流中出现 error 事件：{hl['error']}")
            return 1

        sub_questions = hl.get("sub_questions") or []
        print(f"\n  拆解出 {len(sub_questions)} 个子问题：")
        for sq in sub_questions:
            print(f"    - {sq.get('text', '')[:70]}")
            print(f"      keywords: {sq.get('keywords')}")

        interrupted = hl.get("interrupted")
        if interrupted:
            print(f"\n⏸ 中断点：{interrupted.get('at')} —— {interrupted.get('reason')}")
        else:
            print("\n（未触发中断）")

        # ---------- 2. HITL 恢复（这里演示"用户改写了第一个子问题"）----------
        if run_id and interrupted:
            edited = list(sub_questions)
            if edited:
                edited[0] = {**edited[0], "text": edited[0].get("text", "") + "（用户补充：关注跨编码器）"}
            print(f"\n▶ POST {args.base}/api/v1/research/{run_id}/resume （改写第 1 个子问题）")
            status2, events2, hl2 = read_sse(
                client,
                f"{args.base}/api/v1/research/{run_id}/resume",
                {"sub_questions": edited},
                args.timeout,
            )
            if status2 >= 400:
                print(f"✗ HTTP {status2}\n{hl2.get('error')}")
                return 1
            print(f"  HTTP {status2} · 事件 {len(events2)} 条")
            for name, count in Counter(events2).most_common():
                print(f"    {count:>3}  {name}")

        # ---------- 3. 查询最终结果 ----------
        print(f"\n▶ GET {args.base}/api/v1/research/{run_id}")
        detail = client.get(f"{args.base}/api/v1/research/{run_id}", timeout=60).json()
        print(f"  status={detail.get('status')} stage={detail.get('stage')} "
              f"revision_count={detail.get('revision_count')}")
        verification = detail.get("verification") or {}
        print(f"  引用校验：passed={verification.get('passed')} score={verification.get('score')} "
              f"unsupported={len(verification.get('unsupported') or [])} 条")
        print(f"  token：{detail.get('token_usage')}")

        draft = detail.get("draft") or ""
        print(f"\n  报告 {len(draft)} 字，开头 300 字：\n")
        print("  " + draft[:300].replace("\n", "\n  "))

        # ---------- 3.5 结构化对比表（产品定义的核心产出物 A）----------
        comparison = detail.get("comparison") or []
        print(f"\n  结构化对比表：{len(comparison)} 行")
        for row in comparison[:3]:
            print(f"    - {(row.get('title') or row.get('arxiv_id') or '')[:60]}")
            for key in ("method_name", "core_idea", "datasets", "improvement", "cost", "limitation", "venue"):
                val = row.get(key) or ""
                print(f"        {key:<12} {'（缺 → 留空）' if not val else str(val)[:64]}")
        print(f"  按节点 token：{detail.get('token_by_node')}")
        print(f"  长期记忆召回：{len(detail.get('memory_context') or [])} 条")

        # ---------- 4. 引用证据链 ----------
        print(f"\n▶ GET {args.base}/api/v1/research/{run_id}/citations")
        cits = client.get(f"{args.base}/api/v1/research/{run_id}/citations", timeout=60).json()
        print(f"  引用 {cits.get('count')} 条 · verified_rate={cits.get('verified_rate')}")
        for c in (cits.get("citations") or [])[:3]:
            print(f"    [{c.get('source')}] {c.get('arxiv_id')} · {c.get('locator')} · "
                  f"verified={c.get('verified')}")
            print(f"      {c.get('claim', '')[:60]}")

    print("\n✓ 端到端通过：流式 → 中断 → 恢复 → 报告 → 引用链")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
