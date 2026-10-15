import type { SSEEvent, SSEEventType } from "@/api/events";
import {
  MOCK_RUN_ID,
  MOCK_SESSION_ID,
  fixtures,
  mockCitations,
  mockComparison,
  mockGraph,
  mockReport,
  mockTokenByNode,
  mockVerification,
} from "@/mock/fixtures";

/**
 * Mock SSE 流：把一次真实研究跑完整重放一遍，用来在没有后端时验证时间线组件。
 *
 * 事件名与 data 字段严格照 `app/observability/events.py` 的 helper 签名构造，
 * 所以后端改了字段，这里也会第一时间对不上（这是故意的：契约变更必须可见）。
 * `ev()` 的入参加了 `SSEEventType` 约束，事件名拼错会直接编译不过。
 *
 * ★ 关键行为一：`/stream` 会停在 `interrupted` 并正常收尾，剩下的事件由 `/resume` 那条流
 * 继续推 —— 与后端"两段流式"的真实形态一致，而不是把中断当动画糊过去。
 *
 * ★ 关键行为二（架构设计文档 §8 的三条边界，逐条落实在这里）：
 * ① 只流**模型生成的正文**：`final_report.markdown` 里代码追加的
 *    「⚠️ 未找到原文支撑的结论」「## 参考文献」「页脚声明」都**不进流**，
 *    所以 `streamed` 拼起来必然**短于** `final_report`，权威内容永远以 `final_report` 为准；
 * ② 流式与重试冲突按"**宁少流、不重流**"处置：`token_stream_aborted` 之后
 *    `MockTokenSink` 直接关闸，一条 token 都不再往外吐（见 §b 的开关）；
 * ③ `node_started(write)` 排在**所有** token **之后** —— `updates` 通道要等节点跑完才给
 *    chunk，而 token 走 `custom` 通道写入即到达，这是后端的固有顺序，
 *    前端不能把"节点开始"当成流式起点。
 *
 * ★ 事件条数量级：真实模式一次生成实测 800+ 条 token 事件，所以这里也按
 * 3~6 字符的小块切（贴近 BPE 吐字粒度），并把正文扩到 5000 字符上下，
 * 一次生成落 1000 条量级 —— 逐条 setState 会不会把前端压垮，只有这个量级能试出来。
 * 延迟不是一条一条 sleep（1000 × 26ms 要跑半分钟），而是**成组产出**：
 * 每 `TOKEN_EVENT_BATCH` 条才 sleep 一次 `TOKEN_BATCH_DELAY_MS`，
 * 条数是真实的、总时长控制在 1 秒内。
 */

const now = () => Date.now();

function ev(name: SSEEventType, data: Record<string, unknown>): SSEEvent {
  return { type: name, name, data: { schema_version: "1", ...data }, receivedAt: now() };
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("aborted", "AbortError"));
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(t);
      reject(new DOMException("aborted", "AbortError"));
    }
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** 形状严格照 `app/schemas/research.py` 的 SubQuestion：只有 id / text / keywords / category */
const planSubQuestions = [
  { id: "q1", text: "Self-RAG / CRAG 这类方法如何决定检索时机？", keywords: ["self-rag", "adaptive retrieval", "on-demand retrieval"], category: "cs.CL" },
  { id: "q2", text: "各方法相对 baseline 的事实性提升幅度是多少？", keywords: ["factuality", "ASQA", "hallucination reduction"], category: "cs.CL" },
  { id: "q3", text: "额外训练或额外推理调用带来的工程代价？", keywords: ["inference latency", "fine-tuning cost", "retrieval overhead"], category: "cs.LG" },
  { id: "q4", text: "作者在原文里自述的局限有哪些？", keywords: ["limitations", "failure cases", "ablation"], category: "cs.CL" },
];

/* ============================================================ 流式正文的构造
   注意：`fixtures.ts` 不能改，而 `mockReport` 只有 672 字符 —— 按真实吐字粒度切块
   只能凑出 ~150 条事件，压不出批处理问题。所以下面在**本文件内**把正文按段落扩写
   到目标长度。这些扩写段落只服务于"事件条数量级"，不是真实研究内容：
   真实报告有多长由 write 节点决定，与这里的常量无关。 */

/** 与 `app/agents/nodes/writer.py` 的 `_FOOTER` 同字：页脚是代码追加的，不进流 */
const FOOTER = "> 本文由 AI 生成，引用请以原文为准。";

/** 剥掉代码追加的页脚与它前面那条 `---` 分隔线，得到"模型生成的正文" */
const modelBody = (() => {
  const withoutFooter = mockReport.replace(FOOTER, "").trim();
  return withoutFooter.replace(/\n*-{3,}\s*$/, "").trim();
})();

/**
 * 扩写段落：把流式正文垫到真实量级。
 * ⚠️ 这些段落的存在**只为压事件条数**（800+ 条 token），不是因为研究内容真的需要它们 ——
 * 真实报告的字数由 write 节点的生成结果决定，与本文件的常量无关；改长度只看要不要更多事件。
 */
const bodyFillerParagraphs = [
  "### 把「代价」拆成可比较的四项\n\n第一项是训练代价。反思 token 那类路线要把监督信号造出来：先用强模型标注一批「该不该检索、检索到的段落有没有用、这句生成有没有依据」的三元组，再拿这批数据去微调底座，7B 规模通常是一次全参微调加若干轮偏好优化。它的隐性成本不在算力，而在数据流水线 —— 标注口径一改就得重训，而口径这种东西第一次定下来时几乎没人当真，等到第二批标注结果与前一批自相矛盾，返工就已经按周计了。",
  "第二项是推理代价。加一道检索评估器就是每次查询多一跳模型调用；改走联网搜索更是把延迟交给外部服务。量级上，一次评估大约是主生成的十分之一耗时，但用户感知到的 P99 由最慢的那一跳决定，所以尾部延迟的恶化远大于均值。对交互式的问答入口，这条通常比训练代价更容易被否掉，因为它的痛是每天都能感觉到的。",
  "第三项是索引与存储代价。压缩检索把索引压小一个数量级，换来的是长尾领域的召回下降，需要重排补偿；重排又要额外的交叉编码调用。这三项之间不是并列选项，而是互相转嫁：省下存储的那一笔，会以推理延迟的形式重新出现在账单上。选型时该问的不是「哪种方法更省」，而是「我的瓶颈在哪个账单上」，两个答案不同的团队本来就不该抄同一份方案。",
  "第四项是维护代价，也是最少被计入的一项。评估器的阈值、反思 token 的判定边界、重排模型的召回口径，都需要按领域重标；语料一旦换域，前面三项的调参成果大概率一起作废。横评论文从不报这一项，因为没人愿意把「三个月后要重做一遍」写进结论，但它恰恰是决定一套方案能不能活过第一年的变量。",
  "### 检索时机与生成质量是怎么互相拖累的\n\n把「要不要检索」交给模型之后，会出现一类新错误：该检索时没检索，答案全靠参数化记忆，幻觉照旧；不该检索时检索了，噪声段落挤进上下文，反而把本来答得对的句子带偏。两种错误在端到端指标上会互相抵消，所以只看总体准确率会觉得「加不加反思 token 差不多」，必须把「该触发没触发」与「不该触发却触发了」两类分开统计，才能看清代价花在哪一侧。",
  "更麻烦的是阈值不可迁移：同一个反思 token 的判定门限，在短答案任务上合适，到长文综述上就系统性偏保守，因为长文里每一段都在「可能需要证据」的边界上。可行的做法是按输出长度分档设门限，而不是全局调一个标量；分档之后又要盯住各档的样本量，小样本档位的估计本身就不稳定，这已经变成一个运维问题了。",
  "### 「引用准确」与「事实准确」是两件事\n\n引用能把上一句锚到原文的某一段，只说明这句话有出处，不说明出处支持它。抽取式校验能挡住「引文里根本没这句话」，挡不住「引文在，但原文讲的是相反情形」。这两层在指标上要分开报：第一层是可以程序化验证的，第二层需要判定蕴含关系，通常需要另一个模型来打分，而那个模型自己也会错。把两者合并成一个「准确率」报出去，等于把可验证的部分与不可验证的部分混进同一个数字。",
  "这也是为什么被剔除的论断要显式列在报告末尾而不是悄悄删掉：读者需要知道系统在哪些地方没把握。一份看起来完美但没交代边界的报告，比一份带着两条待核结论的报告更危险，因为前者把所有不确定性都藏进了生成过程。",
  "### 评测口径为什么总对不上\n\n同一个方法在三篇论文里的提升幅度可能完全不同，多数时候不是方法不稳，而是可比性断了：baseline 的检索器不同（稀疏与稠密混用）、指标取的位置不同（前十位与前一百度量）、事实性打分用的模型不同。任何一条变化都足以把正的提升变成负的。所以读横评结论时，先确认三件事：谁的检索器、谁的切分、谁的打分器；这三项对不上，表里的数字就没有比较意义，只是三份不同实验的分数被塞进同一列。",
  "### 多跳与数值型问题是共同的盲区\n\n现有方法在单跳事实问答上的收益最稳，一旦问题需要跨两三篇文档拼接证据，召回阶段就开始失守：需要的第二段证据在第一跳的检索里根本没被提及，后续所有环节都只能在残缺证据上推理。数值型问题更麻烦，原文里的数字常常分散在表格、脚注与图里，抽取时就已经丢了结构，模型只能凭语言模式猜一个。这类问题需要的是不同的分解策略，不是同一套检索器调得更细。",
  "### 落地顺序的建议\n\n在没有标注预算、也没有重训权限的团队里，合理的顺序是自下而上：先把召回本身做准（混合检索加排名融合，成本是一次索引重建），再挂一道轻量质量闸门（不换底座，评估器可用现成模型），最后才考虑让模型自己决定检索时机（需要训练预算）。反过来做最常见的失败是：花三个月微调出一个会「自主检索」的模型，却一直在给它喂错的证据 —— 上游的错从来不会被下游的聪明补救。",
  "### 一个容易被忽略的正交问题\n\n上面所有路线都在优化「证据对不对」，但幻觉还有另一半来源：模型拿到了正确证据，却在长文档生成里把它写歪。分段生成时上下文窗口有限，早期段落建立的引用编号到后面会错位，读起来通顺、点进去对不上。工程上唯一的兜底是不信任模型自己编号 —— 引用序号由程序按去重后的顺序插入，模型只负责留下占位符。这一条不显眼，但它决定了报告能不能对外发布。",
  "### 把「无支撑」变成可运营的信号\n\n校验环节剔除了多少条论断，本身就是一个质量指标。剔除率长期接近零，通常说明 Verifier 形同虚设；剔除率忽高忽低，说明提示词或抽取口径在漂。把它和引用准确率一起按周画在一张图上，比盯单份报告更能反映系统在不在退化 —— 前提是这个统计每次都发，哪怕是零。只在出问题时才出现的通知，等于没有通知，因为你无法区分「没问题」与「没人上报」。",
  "### 三种「证据不足」要分开处置\n\n第一条路是原文里根本没有这句话，属于模型自己编，处理方式是直接从正文剔除并列进未支撑清单；第二条路是话有据但被夸大了范围，原文只在部分子集上做了实验却被写成全部，处理方式是改写回原文的口径而不是删掉；第三条路是当时没校验上，既不能算有支撑也不能算被驳回，只能带着「未校验」的标记保留。三条路的用户含义完全不同，把它们并成一类「可疑」是最省事也最容易误导读者的做法。",
  "### 重排补不了召回的洞\n\n把交叉编码的重排挂在检索之后，能在候选池里把对的那条从第十一位置到第一，但候选池里压根没有它时，重排能做的只是把最好的错误答案排到前面。所以召回率与最终准确率之间有一个硬上限，任何下游环节都无法突破。评测时应该同时报检索前的 Recall@K 与重排后的 nDCG@10：前者告诉你索引的天花板在哪，后者告诉你这套流水线离天花板还有多远。",
  "### 成本要折算到同一个单位才可比\n\n「训练一次」与「每次查询多一跳」在预算表上不是同一行。粗略做法是把训练成本按预期查询量摊平，摊完之后再看谁更贵 —— 但这一步的结论高度依赖流量假设，把一个十万次查询的产品和一个日均百次的内部工具放在一起比，会得到两个相反的答案。所以任何跨方法的成本结论都必须带上量级前提，否则它只是一句口号。",
  "### 上线之后真正会退化的是哪一段\n\n三个信号值得按周盯：检索空返率（说明语料或分词在漂）、引用锚定失败率（说明正文结构与索引对不上）、剔除率（说明生成侧的口径与证据脱节）。三者同时抬头通常是同一件事：语料换了但提示词与切分规则没跟上。分开看它们各自都能解释成噪声，合在一张图上才是信号。",
  "### 长上下文会不会让检索失去意义\n\n窗口从八千扩到百万之后，一个直觉结论是「把语料全塞进去就行，不必检索」。但两件事没有被解决：模型在超长上下文里的有效注意力呈中间塌陷，关键片段放在尾部比放在中部更容易被引用；而把证据密度降低一百倍的同时，成本与延迟是按窗口线性上去的。长上下文改变的是切分与排序的策略，不是让检索消失。",
  "### 小参数模型上多出来的坑\n\n如果底座只有七到十四亿参数，会看到一批在论文口径里不存在的问题：结构化输出的 JSON 经常半截、指令跟随在第三段之后开始漂移、自我批判环节会把正确的引用判成错误。这些属于能力边界，不是加一层校验就能抹平的；可行的应对是把任务切得更细、每一步只要求模型做一个判断，而不是指望一次调用同时完成抽取、判定与改写。",
  "### 引用粒度该落在哪一层\n\n锚到整篇论文最省事，但读者点开之后仍然要自己找那句话；锚到章节好一些；锚到具体片段才是可核验的，代价是索引必须持久化片段标识，一旦重新切分，历史报告的引用会全部失效。所以引用粒度本质上是一个「重切分成本」与「可核验性」的交换，把它当成纯质量问题来讨论会漏掉真正的约束。",
  "### 一份可以照着做的取舍\n\n如果要一个不带前置条件的排序：证据正确性优先于延迟，可核验性优先于文采，覆盖率优先于单点指标的漂亮。落到具体动作上，就是先把召回与锚定这两层做实，再谈让模型自主决定检索时机；评测口径在同一份报告里只允许一套，出现第二套就要说明理由。这套排序并不领先，它的价值只是把「谁更省」的争论换成了「瓶颈在哪个账单」的测量。",
];

/** 目标流式字符数：按 3~6 字符/块（均值 4.5）算，4000 字符 ≈ 900 条 token 事件，
 *  够压到真实量级（实测 800+），又不至于让一轮扩写用尽 —— 循环靠扩写段落垫长度，
 *  段落不够时会退回"带轮次标记的复述"，那是量级手段不是内容 */
const STREAM_TARGET_CHARS = 4000;

const streamedBody = (() => {
  let text = modelBody;
  let round = 0;
  while (Array.from(text).length < STREAM_TARGET_CHARS) {
    const base = bodyFillerParagraphs[round % bodyFillerParagraphs.length];
    // 语料就这么多，第二轮起给标题加轮次标记，避免拼接出逐字重复的同一段
    const para = round < bodyFillerParagraphs.length ? base : base.replace("### ", `### 回读 ${Math.floor(round / bodyFillerParagraphs.length) + 1} · `);
    text += `\n\n${para}`;
    round += 1;
  }
  return text;
})();

/** final_report 里代码追加的 ⚠️ 一节（内容直接取自 verification.unsupported） */
const unsupportedSection = (() => {
  const items = mockVerification.unsupported.filter(Boolean);
  if (items.length === 0) return "";
  return [
    "## ⚠️ 未找到原文支撑的结论",
    "",
    "以下结论未在检索到的文献中找到直接原文支撑，请勿直接引用：",
    "",
    ...items.map((c) => `- "${c}"`),
  ].join("\n");
})();

/** final_report 里代码追加的参考文献（按 arxiv_id + chunk_id 去重、剔除 verified=false） */
const referenceSection = (() => {
  const titles = new Map(mockGraph.nodes.map((n) => [n.arxiv_id, n.title] as const));
  const seen = new Set<string>();
  const lines: string[] = [];
  for (const c of mockCitations.citations) {
    if (c.verified === false) continue; // 三态口径：只有 false 才是被剔除
    const key = `${c.arxiv_id}::${c.chunk_id ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const idx = lines.length + 1;
    const title = titles.get(c.arxiv_id) ?? c.claim.slice(0, 24);
    const strength = c.source === "abstract" ? "摘要级证据" : c.source === "pdf" ? "上传件全文" : "正文原文";
    lines.push(`[${idx}] ${title} — arXiv:${c.arxiv_id}（章节：${c.locator ?? "abstract"}｜${strength}）`);
  }
  if (lines.length === 0) return "";
  return ["## 参考文献", "", ...lines].join("\n");
})();

/**
 * `final_report.citations` 的事件侧形状（键名与类型都照 `app/agents/nodes/writer.py`）。
 * fixtures 里的 chunk_id 是 `"2310.11511_p4_c02"` 这种展示用串，后端事件里是**数字**，
 * 所以取尾部的 `_cNN` 还原成整数；还原不了就用序号，保证类型对得上而不是混进字符串。
 */
const eventCitations = mockCitations.citations.map((c, i) => {
  const tail = /_c(\d+)$/.exec(String(c.chunk_id ?? ""));
  return {
    claim: c.claim,
    paper_id: c.arxiv_id,
    chunk_id: tail ? Number(tail[1]) : i + 1,
    quote: c.quote,
    locator: c.locator,
    source: c.source,
    verified: c.verified ?? null,
  };
});

/**
 * ★ 边界①：流出去的是 `streamedBody`，`final_report` 在它后面**追加** ⚠️ 一节 / 参考文献 / 页脚。
 * 两者长度必然不等（脚本会断言这一点），前端要验的就是"预览 ≠ 最终报告"。
 */
const streamedText = streamedBody;
const finalMarkdown = [streamedText, unsupportedSection, referenceSection, FOOTER]
  .filter((s) => s && s.trim())
  .join("\n\n");

/* ============================================================ token 切块 */

const TOKEN_CHUNK_MIN = 3;
const TOKEN_CHUNK_SPAN = 4; // 3~6 字符，模拟 BPE 吐字粒度

/** 成组产出：每这么多条 token 才让出一次事件循环，条数是真的、时长是可控的 */
const TOKEN_EVENT_BATCH = 40;
const TOKEN_BATCH_DELAY_MS = 8;

/** 作废场景：吐到这么多条就宣布这段流作废 */
const ABORT_AFTER_CHUNKS = 300;
const ABORT_REASON = "上游在生成中途返回 5xx，本次尝试已吐出的增量不再续用";

function chunkForStreaming(text: string): string[] {
  const cps = Array.from(text);
  const out: string[] = [];
  let i = 0;
  while (i < cps.length) {
    // 确定性伪随机（Knuth 乘法散列）：让块长有 3~6 的抖动，像真实 token 边界
    const size = TOKEN_CHUNK_MIN + ((i * 2654435761) >>> 0) % TOKEN_CHUNK_SPAN;
    out.push(cps.slice(i, i + size).join(""));
    i += size;
  }
  return out;
}

const tokenChunks = chunkForStreaming(streamedText);

/* ============================================================ 作废开关 */

let streamAbortEnabled = false;

/**
 * 打开后，`/resume` 那条流的 write 节点会走"吐了字又失败"的分支：
 * 吐约 `ABORT_AFTER_CHUNKS` 条 token → `error{code:"token_stream_aborted", recoverable:true}`
 * → 关闸，之后**零条 token** → 重试轮不流，直接 verification / final_report / token_usage / done。
 *
 * 默认关闭（正常流程不受影响）。也可以不碰全局开关，直接在请求体上带
 * `mock_stream_abort: true` —— 真实后端会把这个键当 `extra="forbid"` 拒掉，
 * 但 mock 模式根本不出网，够页面一次性触发这个场景。
 */
export function setMockStreamAbort(on: boolean): void {
  streamAbortEnabled = on;
}

function resolveStreamAbort(body: unknown): boolean {
  const b = body as Record<string, unknown> | null | undefined;
  const flag = b?.mock_stream_abort ?? b?.mockStreamAbort;
  if (typeof flag === "boolean") return flag;
  return streamAbortEnabled;
}

/**
 * `token` 的闸门，语义照抄 `app/agents/nodes/_common.py` 的 `TokenSink`：
 * `feed` 在作废之后静默吞掉（这就是"不重流"的执行点），`abort` 只在**已吐过字**时发作废通知。
 */
class MockTokenSink {
  chunksSent = 0;
  charsSent = 0;
  closed = false;
  private out: SSEEvent[] = [];

  get streaming(): boolean {
    return !this.closed;
  }

  feed(delta: string): SSEEvent[] {
    if (!this.streaming || !delta) return [];
    this.out.push(ev("token", { node: "write", delta }));
    this.chunksSent += 1;
    this.charsSent += delta.length;
    return this.drain();
  }

  abort(reason: string): SSEEvent[] {
    if (!this.streaming || this.chunksSent === 0) return [];
    this.closed = true;
    this.out.push(
      ev("error", {
        node: "write",
        code: "token_stream_aborted",
        message: `生成中断，正在重试：已流出的 ${this.charsSent} 字作废，最终内容以报告为准（${reason}）`,
        recoverable: true,
      }),
    );
    return this.drain();
  }

  private drain(): SSEEvent[] {
    const events = this.out;
    this.out = [];
    return events;
  }
}

/* ============================================================ 两段流 */

/** 第一段：拆解 → HITL 中断 */
async function* phaseOne(signal?: AbortSignal): AsyncGenerator<SSEEvent> {
  yield ev("run_started", { run_id: MOCK_RUN_ID, question: fixtures.question });
  await sleep(320, signal);

  yield ev("node_started", { node: "plan", label: "拆解研究问题", ts: now() });
  await sleep(260, signal);
  yield ev("thought", { node: "plan", delta: "识别到这是一个「方法横评」类问题，需要按机制 / 效果 / 代价 / 局限四个维度切分。" });
  await sleep(420, signal);

  yield ev("plan_created", { sub_questions: planSubQuestions, rationale: "四个维度各自可独立检索，并行召回后在对比表里汇合。", count: 4 });
  await sleep(200, signal);

  // ★ HITL 中断点 A：必须补发 interrupted，否则前端只看到流"卡住不动"
  yield ev("interrupted", {
    at: "plan_confirm",
    reason: "等待用户确认或修改子问题拆解",
    payload: { sub_questions: planSubQuestions },
    resume_token: MOCK_RUN_ID,
  });
  await sleep(160, signal);
  yield ev("done", { run_id: MOCK_RUN_ID, status: "interrupted" });
}

/** 第二段：恢复后一路跑到报告 */
async function* phaseTwo(edited: boolean, abortStream: boolean, signal?: AbortSignal): AsyncGenerator<SSEEvent> {
  yield ev("run_started", { run_id: MOCK_RUN_ID, question: fixtures.question, resumed: true });
  yield ev("node_started", { node: "ensure_corpus", label: "准备语料", ts: now() });
  await sleep(300, signal);
  yield ev("corpus", { fetched: 18, cached: 6, total: 24, corpus_ready: true });
  await sleep(240, signal);

  yield ev("node_started", { node: "retrieve", label: "多路并行检索", ts: now() });
  for (const [i, q] of planSubQuestions.entries()) {
    const callId = `call_${i}`;
    yield ev("tool_call", {
      node: "retrieve",
      tool: "arxiv_search",
      args: { query: q.text.slice(0, 24), cat: "cs.CL", since: "2020", has_code: i === 2 },
      call_id: callId,
    });
    await sleep(380 + i * 120, signal);
    yield ev("tool_result", { call_id: callId, status: "ok", duration_ms: 380 + i * 120, count: 8 - i });
    yield ev("retrieval", {
      sub_question_id: q.id,
      count: 3,
      hits: [
        { arxiv_id: "2310.11511", title: "Self-RAG: Learning to Retrieve, Generate, and Critique", score: 0.874, section: "§3 Method", source: "arxiv" },
        { arxiv_id: "2401.15884", title: "Corrective Retrieval Augmented Generation", score: 0.812, section: "Abstract", source: "arxiv" },
        { arxiv_id: "2112.01488", title: "ColBERTv2", score: 0.764, section: "§4 Experiments", source: "local_index" },
      ],
    });
    await sleep(180, signal);
  }

  yield ev("node_started", { node: "read", label: "精读与证据抽取", ts: now() });
  await sleep(500, signal);
  yield ev("thought", { node: "read", delta: "去重后 4 篇论文进入抽取，产出 27 条「论断 → 证据」对。" });
  // ★ rescued=0 也照发：这是回归探测，不是装饰
  yield ev("citation_rescue", {
    node: "read",
    checked: 27,
    rescued: 0,
    dropped: 2,
    recovered_by: {},
    drop_reasons: { no_quote_match: 1, quote_not_in_chunk: 1 },
  });
  await sleep(360, signal);

  yield ev("node_started", { node: "compare", label: "结构化对比表抽取", ts: now() });
  await sleep(620, signal);
  yield ev("thought", { node: "compare", delta: `生成 ${mockComparison.length} 行 × 9 列对比表，${countEmptyCells(mockComparison)} 个单元格在原文中确实没有依据，按空处理。` });
  await sleep(300, signal);

  yield ev("node_started", { node: "verify", label: "引用校验", ts: now() });
  await sleep(560, signal);
  yield ev("verification", {
    passed: false,
    score: 0.926,
    unsupported: mockVerification.unsupported,
    revision_count: 0,
  });
  await sleep(320, signal);

  yield ev("node_started", { node: "revise", label: "剔除无支撑论断并重写", ts: now() });
  await sleep(480, signal);
  yield ev("verification", { passed: true, score: 0.963, unsupported: [], revision_count: 1 });
  await sleep(260, signal);

  /* ---- write：增量先经 custom 通道落地，`node_started(write)` 反而后到（边界③） ----
     所以这里**不**先发 node_started：token 的 node 字段就是 "write"，
     前端按 node 归组时不能指望"节点开始"是流式起点。 */
  const sink = new MockTokenSink();
  const abortAt = abortStream ? ABORT_AFTER_CHUNKS : 0;
  for (const delta of tokenChunks) {
    if (!sink.streaming) break; // ★ 关闸之后一条 token 都不再吐（宁少流、不重流）
    for (const e of sink.feed(delta)) yield e;
    if (abortAt > 0 && sink.chunksSent >= abortAt) {
      for (const e of sink.abort(ABORT_REASON)) yield e;
      await sleep(240, signal);
      break;
    }
    if (sink.chunksSent % TOKEN_EVENT_BATCH === 0) await sleep(TOKEN_BATCH_DELAY_MS, signal);
  }

  // 到这里 write 节点跑完，`updates` 的 chunk 才把"节点开始"补上
  yield ev("node_started", { node: "write", label: "生成报告", ts: now() });

  if (sink.closed) {
    // 重试那一轮不再往外吐字，但校验、成稿照常走（边界②）
    yield ev("verification", { passed: true, score: 0.963, unsupported: [], revision_count: 2 });
    await sleep(120, signal);
  }

  /* final_report 的 citations 照的是**事件侧**形状（`app/agents/nodes/writer.py` 的返回）：
     `claim / paper_id / chunk_id(数字) / quote / locator / source / verified`。
     ⚠️ 同一个引用在 REST `/research/{id}/citations` 里叫 `arxiv_id`、还多带
     `note / resolution_how / verify_reanchored` —— 两条路的键名与类型都不是一套（已报后端）。
     这里必须照事件那一份，否则 mock 模式绿灯、真后端把界面字段读成 undefined。 */
  yield ev("final_report", { markdown: finalMarkdown, citations: eventCitations });
  await sleep(200, signal);
  // ★ `total` 与 `by_node` 的每一项都是 `{prompt_tokens, completion_tokens, total_tokens, calls}`
  // 字典，不是裸数字、也不是 `{prompt, completion, total}` 那种短名
  //（`state["token_usage"]` 由 `_common.usage_of` 产出全名，`finalize` 按节点各落一行）；
  // write 的 calls=2 正好对上重试那一轮
  yield ev("token_usage", {
    total: { prompt_tokens: 1_284_500, completion_tokens: 157_800, total_tokens: 1_442_300 },
    cost_estimate: edited ? 0.42 : 0.39,
    by_node: mockTokenByNode,
  });
  await sleep(160, signal);
  yield ev("done", { run_id: MOCK_RUN_ID, status: "ok" });
}

function countEmptyCells(rows: typeof mockComparison): number {
  return rows.reduce((sum, r) => sum + Object.values(r).filter((v) => v === null || v === "").length, 0);
}

export function mockStream(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<SSEEvent> {
  if (path.includes("/resume")) {
    const edited = Array.isArray((body as { sub_questions?: unknown[] })?.sub_questions);
    return phaseTwo(edited, resolveStreamAbort(body), signal);
  }
  return phaseOne(signal);
}

export const mockSessionId = MOCK_SESSION_ID;
