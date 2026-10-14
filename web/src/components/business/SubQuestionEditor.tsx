import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Check, PauseCircle, PencilLine, Plus, RotateCcw, Trash2 } from "lucide-react";
import { cn } from "@/lib/cn";

// 这是 HITL 断点：本地草稿 state 与外部数据分离，"确认继续"才把编辑结果交回（后端会先写回 state 再续跑）。

export interface SubQuestion {
  id?: string;
  text: string;
  keywords?: string[];
  category?: string;
}

/** 后端 SSE/接口给的是自由 JSON（interrupted.payload.sub_questions），这里做防御式解析 */
export type SubQuestionLike = SubQuestion | Record<string, unknown>;

function parseItem(raw: unknown): Draft {
  const o = (raw ?? {}) as Record<string, unknown>;
  const text = typeof o.text === "string" ? o.text : typeof o === "string" ? o : "";
  const keywords = Array.isArray(o.keywords) ? o.keywords.map((k) => String(k)) : [];
  return {
    id: typeof o.id === "string" ? o.id : nextId(),
    text,
    keywordsText: keywords.join(", "),
    category: typeof o.category === "string" ? o.category : undefined,
  };
}

export interface SubQuestionEditorProps {
  /** 来自 interrupted.payload.sub_questions 或 plan_created —— 作为初始草稿 */
  items?: SubQuestionLike[];
  /** 兼容别名（与 items 二选一，items 优先） */
  subQuestions?: SubQuestionLike[];
  onResume: (subQuestions: Array<Record<string, unknown>>) => void;
  /** 传了就多一个"放弃修改"按钮 */
  onCancel?: () => void;
  disabled?: boolean;
  runningHint?: string;
  className?: string;
}

interface Draft extends SubQuestion {
  id: string;
  keywordsText: string;
}

let seq = 0;
const nextId = () => `sq-${Date.now().toString(36)}-${(seq++).toString(36)}`;

const toPayload = (d: Draft): Record<string, unknown> => ({
  id: d.id,
  text: d.text.trim(),
  category: d.category?.trim() ? d.category.trim() : undefined,
  keywords: d.keywordsText
    .split(/[,，;；]+/)
    .map((k) => k.trim())
    .filter(Boolean),
});

export function SubQuestionEditor({
  items,
  subQuestions,
  onResume,
  onCancel,
  disabled,
  runningHint,
  className,
}: SubQuestionEditorProps) {
  const source = items ?? subQuestions ?? [];
  // 用内容签名而非数组引用重置草稿：父组件每次 render 都可能给新数组，按引用会把用户正在打的字冲掉
  const signature = JSON.stringify(source);
  const initial = useMemo(() => source.map(parseItem), [signature]);
  const [draft, setDraft] = useState<Draft[]>(initial);

  useEffect(() => setDraft(initial), [initial]);

  const patch = (id: string, p: Partial<Draft>) =>
    setDraft((ds) => ds.map((d) => (d.id === id ? { ...d, ...p } : d)));
  const remove = (id: string) => setDraft((ds) => ds.filter((d) => d.id !== id));
  const add = () =>
    setDraft((ds) => [...ds, { id: nextId(), text: "", keywordsText: "" }]);

  const valid = draft.filter((d) => d.text.trim().length > 0);
  const dirty =
    valid.length !== initial.length ||
    valid.some((v, i) => !initial[i] || v.text.trim() !== initial[i].text.trim());

  return (
    <section className={cn("glass panel p-4", className)}>
      <header className="flex flex-wrap items-center gap-2">
        <PauseCircle size={16} className="text-neon-purple" />
        <h3 className="text-sm font-semibold text-ink">确认研究拆解</h3>
        <span className="rounded-full bg-[color:var(--accent-purple-dim)] px-2 py-0.5 font-mono text-[10px] text-neon-purple">
          HITL · 流程已暂停
        </span>
        <span className="ml-auto font-mono text-[10px] text-ink-faint">
          {valid.length}/{draft.length} 条有效
        </span>
      </header>

      <p className="mt-2 flex items-start gap-1.5 rounded-md border border-dashed border-neon-purple/40 bg-[color:var(--accent-purple-dim)] px-3 py-2 text-xs leading-relaxed text-ink-soft">
        <PencilLine size={12} className="mt-0.5 shrink-0 text-neon-purple" />
        你现在改的每一条子问题，都会<strong className="mx-0.5 font-semibold text-ink">真实地</strong>
        决定后续的检索关键词、精读对象与最终报告 —— 不是演示，改完点「确认继续」才会恢复执行。
      </p>

      <ul className="mt-3 space-y-2">
        {draft.map((d, i) => (
          <motion.li
            key={d.id}
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-lg bg-white/[0.03] p-2.5"
          >
            <div className="flex items-start gap-2">
              <span className="mt-1.5 shrink-0 font-mono text-[10px] text-neon-purple">Q{i + 1}</span>
              <textarea
                value={d.text}
                rows={2}
                disabled={disabled}
                placeholder="子问题（必填）"
                onChange={(e) => patch(d.id, { text: e.target.value })}
                className={cn(
                  "min-h-0 flex-1 resize-y rounded-md border border-hairline bg-transparent px-2 py-1.5 text-xs leading-relaxed text-ink outline-none placeholder:text-ink-faint/70",
                  "focus:border-neon-cyan/60",
                  !d.text.trim() && "border-neon-amber/50",
                )}
              />
              <button
                type="button"
                disabled={disabled}
                onClick={() => remove(d.id)}
                title="删除这条子问题"
                className="mt-1 rounded-md p-1.5 text-ink-faint transition hover:bg-white/[0.05] hover:text-neon-red disabled:opacity-40"
              >
                <Trash2 size={13} />
              </button>
            </div>
            <div className="mt-1.5 flex flex-wrap items-center gap-2 pl-6">
              <input
                value={d.keywordsText}
                disabled={disabled}
                placeholder="检索关键词，逗号分隔"
                onChange={(e) => patch(d.id, { keywordsText: e.target.value })}
                className="min-w-0 flex-1 rounded-md border border-hairline bg-transparent px-2 py-1 font-mono text-[11px] text-ink-soft outline-none placeholder:text-ink-faint/70 focus:border-neon-cyan/60"
              />
              <input
                value={d.category ?? ""}
                disabled={disabled}
                placeholder="类别（可选）"
                onChange={(e) => patch(d.id, { category: e.target.value })}
                className="w-28 rounded-md border border-hairline bg-transparent px-2 py-1 text-[11px] text-ink-soft outline-none placeholder:text-ink-faint/70 focus:border-neon-cyan/60"
              />
            </div>
          </motion.li>
        ))}
      </ul>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={add}
          className="inline-flex items-center gap-1 rounded-lg border border-hairline px-3 py-1.5 text-xs text-ink-soft transition hover:border-neon-cyan/50 hover:text-neon-cyan disabled:opacity-40"
        >
          <Plus size={13} /> 新增子问题
        </button>
        <button
          type="button"
          disabled={disabled || draft.length === initial.length && !dirty}
          onClick={() => setDraft(initial)}
          title="回退到后端最新一次拆解"
          className="inline-flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs text-ink-faint transition hover:text-ink disabled:opacity-40"
        >
          <RotateCcw size={12} /> 重置
        </button>

        <div className="ml-auto flex items-center gap-2">
          {onCancel ? (
            <button
              type="button"
              disabled={disabled}
              onClick={onCancel}
              className="rounded-lg px-3 py-1.5 text-xs text-ink-faint transition hover:text-ink disabled:opacity-40"
            >
              取消
            </button>
          ) : null}
          <button
            type="button"
            disabled={disabled || valid.length === 0}
            onClick={() => onResume(valid.map(toPayload))}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-lg px-4 py-1.5 text-xs font-semibold transition",
              "bg-[color:var(--accent-cyan-dim)] text-neon-cyan hover:brightness-125",
              "disabled:cursor-not-allowed disabled:opacity-40",
            )}
          >
            <Check size={13} />
            {dirty ? "确认修改并继续" : "沿用当前拆解继续"}
          </button>
        </div>
      </div>

      {runningHint ? <p className="mt-2 text-[11px] text-ink-faint">{runningHint}</p> : null}
      {draft.length > 0 && valid.length === 0 ? (
        <p className="mt-1 text-[11px] text-neon-amber">至少保留一条非空子问题才能继续。</p>
      ) : null}
    </section>
  );
}

export default SubQuestionEditor;
