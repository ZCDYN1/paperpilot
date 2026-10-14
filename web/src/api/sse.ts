import { SSE_EVENT_TYPES, type SSEEvent, type SSEEventType } from "@/api/events";

/**
 * POST 版 SSE 客户端。
 *
 * 为什么不用浏览器原生 `EventSource`：它只能发 GET，而后端的核心接口
 * `POST /api/v1/research/stream` 要带一个几百字节的请求体（研究问题 + 过滤条件 +
 * session_id）。所以这里用 `fetch` + `ReadableStream` 自己按 SSE 规范切帧。
 *
 * 帧格式（`app/observability/events.py` 的 `SSEEvent.to_sse`）：
 *   `event: <type>\ndata: <json>\n\n`
 * 两处换行缺一不可；`\r\n` 也要能吃下（代理层可能改写换行）。
 */

const KNOWN = new Set<string>(SSE_EVENT_TYPES);

export interface SseRequest {
  path: string;
  body?: unknown;
  signal?: AbortSignal;
}

/** 解析单帧的 `event:` / `data:` 行；解析不出 JSON 时保留原文，不丢事件 */
function parseFrame(frame: string): SSEEvent | null {
  let name = "message";
  const dataLines: string[] = [];

  for (const rawLine of frame.split(/\r?\n/)) {
    if (!rawLine || rawLine.startsWith(":")) continue; // 空行与注释行（心跳）
    const colon = rawLine.indexOf(":");
    const field = colon === -1 ? rawLine : rawLine.slice(0, colon);
    // 规范允许 `data:x` 与 `data: x` 两种，值要剥掉紧跟冒号的那一个空格
    let value = colon === -1 ? "" : rawLine.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") name = value;
    else if (field === "data") dataLines.push(value);
  }

  if (dataLines.length === 0 && name === "message") return null;
  const text = dataLines.join("\n");

  let data: Record<string, unknown> = {};
  if (text) {
    try {
      data = JSON.parse(text) as Record<string, unknown>;
    } catch {
      // 后端承诺 data 恒为 JSON；真不是 JSON 说明中间有代理截断了，
      // 这时把原文塞进 `_raw` 让界面能显示，而不是让整条流崩掉。
      data = { _raw: text, _parse_error: true };
    }
  }

  return {
    type: (KNOWN.has(name) ? name : "unknown") as SSEEventType | "unknown",
    name,
    data,
    receivedAt: Date.now(),
  };
}

/** 逐条产出事件。用 `for await` 消费，`signal.abort()` 可中途取消。 */
export async function* streamSse({ path, body, signal }: SseRequest): AsyncGenerator<SSEEvent> {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });

  if (!resp.ok) {
    // 流还没开始，所以这里拿得到真实状态码（后端注释里强调过这个区别）
    throw new SseHttpError(resp.status, await safeDetail(resp));
  }
  if (!resp.body) throw new SseHttpError(0, "浏览器未提供 ReadableStream，无法消费 SSE");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // 帧边界是空行；最后一个片段可能不完整，留在 buffer 里等下一批
      let boundary = buffer.search(/\r?\n\r?\n/);
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + buffer.slice(boundary).match(/^\r?\n\r?\n/)![0].length);
        const event = parseFrame(frame);
        if (event) yield event;
        boundary = buffer.search(/\r?\n\r?\n/);
      }
    }
    if (buffer.trim()) {
      const event = parseFrame(buffer);
      if (event) yield event;
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }
}

async function safeDetail(resp: Response): Promise<string> {
  try {
    const json = (await resp.json()) as { detail?: unknown };
    return typeof json.detail === "string" ? json.detail : JSON.stringify(json.detail ?? resp.statusText);
  } catch {
    return resp.statusText;
  }
}

export class SseHttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "SseHttpError";
  }
}
