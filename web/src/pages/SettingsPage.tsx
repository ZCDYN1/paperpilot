import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { CircleCheck, ChevronDown, CircleAlert, Info, PlugZap, RotateCcw, Save, TriangleAlert } from "lucide-react";
import { api } from "@/api/client";
import type {
  ConfigResponse,
  ConfigSource,
  ConfigTestRequest,
  ConfigTestResult,
  ConfigValues,
  ConfigWriteResult,
  SecretState,
} from "@/api/domain";
import { PageHeader } from "@/components/ui/PageHeader";
import { Panel } from "@/components/ui/Panel";
import { GlassCard } from "@/components/ui/GlassCard";
import { NeonButton } from "@/components/ui/NeonButton";
import { Tag } from "@/components/ui/Tag";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { Segmented } from "@/components/ui/Segmented";
import { Stagger, StaggerItem } from "@/components/ui/Stagger";

/**
 * 配置中心：写的是**服务端 .env**，不是浏览器本地设置。
 *
 * 表单形状完全由 GET /config 驱动（options 决定下拉、sources 决定来源标签、
 * restart_required 决定重启提示），页面自身不硬编码任何 provider 列表。
 * 草稿统一用 string 存储：数字字段提交时才转数，provider 取值来自后端 options，
 * 前端类型联合（union）不能挡住在界面上出现的新枚举值。
 */

type FieldKey = keyof ConfigValues;
type Draft = Record<FieldKey, string>;

const SECRET_KEYS: FieldKey[] = ["openai_api_key", "embedding_api_key"];
const NUMBER_KEYS: FieldKey[] = ["llm_temperature", "min_corpus"];
/** 布尔项在表单里以 "true"/"false" 字符串存在 Draft 里，提交前转回布尔 */
const BOOL_KEYS: FieldKey[] = ["llm_disable_thinking"];

const LABELS: Record<FieldKey, string> = {
  llm_provider: "LLM 提供方",
  llm_model: "LLM 模型名",
  llm_temperature: "Temperature",
  llm_structured_method: "结构化输出方式",
  llm_disable_thinking: "结构化调用时关掉思考模式",
  openai_base_url: "OpenAI 接口地址",
  openai_api_key: "OpenAI API Key",
  ollama_base_url: "Ollama 服务地址",
  ollama_model: "Ollama 模型名",
  embedding_provider: "Embedding 提供方",
  embedding_model: "Embedding 模型名",
  embedding_api_base_url: "Embedding 接口地址",
  embedding_api_key: "Embedding API Key",
  rerank_provider: "重排提供方",
  rerank_model: "重排模型",
  min_corpus: "最小语料篇数门槛",
};

function emptyDraft(): Draft {
  return {
    llm_provider: "",
    llm_model: "",
    llm_temperature: "",
    llm_structured_method: "",
    llm_disable_thinking: "",
    openai_base_url: "",
    openai_api_key: "",
    ollama_base_url: "",
    ollama_model: "",
    embedding_provider: "",
    embedding_model: "",
    embedding_api_base_url: "",
    embedding_api_key: "",
    rerank_provider: "",
    rerank_model: "",
    min_corpus: "",
  };
}

function toDraft(values: ConfigValues): Draft {
  const d = emptyDraft();
  (Object.keys(d) as FieldKey[]).forEach((k) => {
    // 密钥永远不回填：后端只给掩码，回填会把掩码当明文提交
    if (SECRET_KEYS.includes(k)) return;
    const v = values[k];
    if (typeof v === "number") d[k] = String(v);
    else if (typeof v === "string") d[k] = v;
    // 布尔必须显式接住：漏掉这一支，界面上会是空串，
    // 于是"后端明明是 false"被渲染成"用户还没选"，一改就误判成改动
    else if (typeof v === "boolean") d[k] = v ? "true" : "false";
  });
  return d;
}

/* ------------------------------------------------------------------ 小组件 */

function SourceTag({ source }: { source: ConfigSource | undefined }) {
  if (!source) return null;
  if (source === "dotenv") return <Tag variant="purple" mono>.env</Tag>;
  if (source === "env") return <Tag variant="amber" mono>环境变量</Tag>;
  return <Tag variant="muted" mono>默认值</Tag>;
}

type InputKind = "text" | "password" | "number";

function FieldRow({
  field,
  config,
  value,
  onChange,
  kind = "text",
  placeholder,
  secret,
  hint,
  error,
  disabled,
}: {
  field: FieldKey;
  config: ConfigResponse;
  value: string;
  onChange: (v: string) => void;
  kind?: InputKind;
  placeholder?: string;
  secret?: SecretState;
  hint?: ReactNode;
  error?: string;
  disabled: boolean;
}) {
  const source = config.sources[field];
  const needsRestart = config.restart_required.includes(field);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-ink">{LABELS[field]}</span>
        <span className="font-mono text-[11px] text-ink-faint">{field}</span>
        <SourceTag source={source} />
        {needsRestart ? <Tag variant="amber">改后需重启</Tag> : null}
        {secret?.set ? <Tag variant="green">已设置</Tag> : null}
      </div>
      <input
        type={kind}
        inputMode={kind === "number" ? "decimal" : undefined}
        value={value}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-[10px] border border-[color:var(--glass-border)] bg-white/[0.03] px-3 py-2 font-mono text-sm text-ink placeholder:text-ink-faint focus-visible:border-[color:var(--accent-cyan-line)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan-dim)] disabled:opacity-50"
      />
      {/* env 优先级坑：写进 .env 也可能被进程环境变量压住，界面上必须说清 */}
      {source === "env" ? (
        <p className="flex items-start gap-1.5 text-xs leading-relaxed text-neon-amber">
          <CircleAlert size={13} className="mt-0.5 shrink-0" aria-hidden />
          <span>
            此项当前来自<b>进程环境变量</b>，优先级高于 .env —— 在这里改了也可能被它覆盖，
            需要同时清理启动环境里对应的变量。
          </span>
        </p>
      ) : null}
      {secret && !secret.set ? (
        <p className="text-xs text-ink-faint">尚未设置。留空提交表示<b>不修改</b>，不是清空。</p>
      ) : null}
      {secret?.set ? (
        <p className="text-xs text-ink-faint">留空提交表示<b>不修改</b>这项目前密钥，不是清空。</p>
      ) : null}
      {hint ? <p className="text-xs leading-relaxed text-ink-faint">{hint}</p> : null}
      {error ? <p className="text-xs text-neon-red">{error}</p> : null}
    </div>
  );
}

/** provider 取值一律来自后端 options，前端不补列表 */
function ProviderPicker({
  field,
  options,
  value,
  onChange,
  disabled,
  ariaLabel,
}: {
  field: FieldKey;
  options: string[];
  value: string;
  onChange: (v: string) => void;
  disabled: boolean;
  ariaLabel: string;
}) {
  return (
    <div className={disabled ? "pointer-events-none opacity-60" : undefined} aria-disabled={disabled || undefined}>
      <Segmented
        ariaLabel={ariaLabel}
        value={value}
        onChange={onChange}
        options={options.map((v) => ({ value: v, label: v }))}
      />
      <span className="sr-only">{LABELS[field]}</span>
    </div>
  );
}

function Collapsible({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-[10px] border border-[color:var(--glass-border)] bg-white/[0.02]">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs text-ink-soft hover:text-ink"
      >
        <ChevronDown size={14} aria-hidden className={open ? "" : "-rotate-90"} />
        {title}
      </button>
      {open ? <div className="flex flex-col gap-4 border-t border-[color:var(--glass-border)] p-3">{children}</div> : null}
    </div>
  );
}

/* ------------------------------------------------------------ 连通性探测 */

/**
 * 能探测的 provider 就是 `options.llm_provider`（后端从 `LLMProvider` 的 Literal
 * 反射出来的那一份），这里**不再抄第二份名单**。
 *
 * 抄过的下场这个项目体验过一次：mock 给 rerank 多写了一个 "api"，
 * 于是界面提供一个必然被拒的选项。取值不在集合里时按钮禁用并说明原因 ——
 * 发出去只会收到一句 `{"code":"unknown_provider"}`，那是把参数校验失败
 * 伪装成网络错误给使用者看。
 */
function LlmConnectivityTest({
  draft,
  secret,
  testable,
  disabled,
}: {
  /** 表单当前状态：探测读的就是它，与已保存的配置可能不一致，这正是重点 */
  draft: Draft;
  /** 后端 `secrets.openai_api_key` 的状态，只用来说清"没填时用的是谁的密钥" */
  secret: SecretState;
  /** 后端反射出来的合法 provider 集合 */
  testable: string[];
  disabled: boolean;
}) {
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<ConfigTestResult | null>(null);
  const [requestError, setRequestError] = useState("");
  const [testedSnapshot, setTestedSnapshot] = useState("");

  const provider = draft.llm_provider;
  const testableNow = testable.includes(provider);

  const openaiUrl = draft.openai_base_url.trim();
  const ollamaUrl = draft.ollama_base_url.trim();
  const typedKey = draft.openai_api_key.trim();

  // 真正参与调用的模型名由 provider 决定：ollama 用 ollama_model，其余用 llm_model
  const modelField = provider === "ollama" ? "ollama_model" : "llm_model";
  const modelName = (provider === "ollama" ? draft.ollama_model : draft.llm_model).trim();

  const snapshot = [provider, openaiUrl, ollamaUrl, modelName, typedKey ? "with-key" : "no-key"].join("|");
  // 结果对应的是测试那一刻的值；之后表单又改过的话，得说清楚它已经过期
  const stale = result !== null && testedSnapshot !== "" && snapshot !== testedSnapshot;

  async function runTest() {
    if (!testableNow) return;
    // provider 的合法集合是运行时从后端拿的，所以这里必须做一次断言式收窄
    const body: ConfigTestRequest = { provider: provider as ConfigTestRequest["provider"] };
    // 空值一律不传这个键：传 "" 会被读成"用默认/已生效的值"，语义上没错，
    // 但省略掉才能和界面显示保持一致 —— 输入框里空的，请求里就没有这一项。
    if (openaiUrl) body.openai_base_url = openaiUrl;
    if (ollamaUrl) body.ollama_base_url = ollamaUrl;
    if (typedKey) body.openai_api_key = typedKey;
    // 模型名一起带上：后端会用一次 1 token 的真实调用判定它能不能用，
    // 比前端拿 /models 列表比对可靠（网关的列表常是展示名，不是调用名）
    if (modelName) body.model = modelName;

    setTesting(true);
    setRequestError("");
    setResult(null);
    try {
      setResult(await api.config.test(body));
      setTestedSnapshot(snapshot);
    } catch (e) {
      setRequestError(e instanceof Error ? e.message : String(e));
    } finally {
      setTesting(false);
    }
  }

  const models = result?.models;
  const check = result?.model_check;
  const hasModelList = models !== undefined && models.length > 0;
  const modelMatched = hasModelList && (models ?? []).some((m) => m.trim() === modelName);

  return (
    <div className="flex flex-col gap-2.5 rounded-[10px] border border-[color:var(--glass-border)] bg-white/[0.02] px-3 py-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-ink">连通性测试</span>
          <span className="font-mono text-[11px] text-ink-faint">POST /api/v1/config/test</span>
        </div>
        <NeonButton variant="secondary" loading={testing} disabled={disabled || !testableNow} onClick={runTest}>
          <PlugZap size={15} aria-hidden />
          {testing ? "正在探测…" : "用当前填写的值探测"}
        </NeonButton>
      </div>

      <p className="text-xs leading-relaxed text-ink-faint">
        探测用的是<b className="text-ink-soft">表单当前填写的值</b>，不需要先保存。
        这里没填密钥时，请求里不带密钥，由后端使用它当前生效的那一个 ——
        界面拿不到明文，也不需要先保存才能测。
      </p>

      {provider !== "" && !testableNow ? (
        <p className="flex items-start gap-1.5 text-xs leading-relaxed text-neon-amber">
          <CircleAlert size={13} className="mt-0.5 shrink-0" aria-hidden />
          <span>
            提供方 <code className="font-mono">{provider}</code> 不在可探测范围内：连通性探测只支持{" "}
            <code className="font-mono">{testable.join(" / ")}</code>。
          </span>
        </p>
      ) : null}

      {provider === "openai" && typedKey === "" && !secret.set ? (
        <p className="flex items-start gap-1.5 text-xs leading-relaxed text-neon-amber">
          <CircleAlert size={13} className="mt-0.5 shrink-0" aria-hidden />
          <span>OpenAI API Key 这里没有填，服务端也没有已生效的密钥 —— 探测会直接回报缺少 Key。</span>
        </p>
      ) : null}

      {provider === "openai" && typedKey === "" && secret.set ? (
        <p className="text-xs leading-relaxed text-ink-faint">
          将使用后端已生效的密钥（<code className="font-mono">{secret.masked}</code>）来探测。
        </p>
      ) : null}

      {requestError ? (
        <div className="flex items-start gap-2 rounded-[10px] border border-neon-red/50 bg-white/[0.05] px-3 py-2.5">
          <TriangleAlert size={15} className="mt-0.5 shrink-0 text-neon-red" aria-hidden />
          <p className="text-sm leading-relaxed text-neon-red">探测请求没有完成：{requestError}</p>
        </div>
      ) : null}

      {result ? (
        result.ok ? (
          <div className="flex flex-col gap-2 rounded-[10px] border border-[color:var(--accent-green)] bg-white/[0.04] px-3 py-2.5">
            <p className="flex items-start gap-1.5 text-sm leading-relaxed text-neon-green">
              <CircleCheck size={15} className="mt-0.5 shrink-0" aria-hidden />
              <span>{result.detail}</span>
            </p>

            {models !== undefined && !hasModelList ? (
              <p className="text-xs leading-relaxed text-ink-soft">
                这个服务没有返回可核对的模型列表（OpenAI 兼容网关经常如此），连通性已经确认，
                模型名是否可用还要靠一次真实调用才能定。
              </p>
            ) : null}

            {hasModelList ? (
              <div className="flex flex-col gap-2">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs text-ink-faint">可用模型（{models?.length}）：</span>
                  {models?.map((m, i) => (
                    <Tag key={`${m}-${i}`} variant={m.trim() === modelName ? "cyan" : "muted"} mono>
                      {m}
                    </Tag>
                  ))}
                </div>

                {modelName === "" ? (
                  <p className="text-xs leading-relaxed text-ink-faint">
                    <code className="font-mono">{modelField}</code> 还没填，可以照上面的列表挑一个。
                  </p>
                ) : check && check.checked ? (
                  /* 后端做过一次真实调用，所以这里照抄它的结论，不再自己猜 */
                  check.ok ? (
                    <p className="flex items-start gap-1.5 text-xs leading-relaxed text-neon-green">
                      <CircleCheck size={13} className="mt-0.5 shrink-0" aria-hidden />
                      <span>{check.detail}</span>
                    </p>
                  ) : (
                    <p className="flex items-start gap-1.5 text-xs leading-relaxed text-neon-amber">
                      <TriangleAlert size={13} className="mt-0.5 shrink-0" aria-hidden />
                      <span>{check.detail}</span>
                    </p>
                  )
                ) : check ? (
                  /* checked=false = 后端没能判定（网络失败 / 上游回了别的错误码）。
                     说"模型名不对"是伪造结论，所以这里只说没判定出来。 */
                  <p className="text-xs leading-relaxed text-ink-faint">{check.detail}</p>
                ) : modelMatched ? (
                  <p className="text-xs leading-relaxed text-neon-green">
                    <code className="font-mono">{modelField}</code> ={" "}
                    <code className="font-mono">{modelName}</code> 在这个服务的模型列表里。
                  </p>
                ) : (
                  /* 走到这里说明后端没做真实调用校验（老版本），只能拿列表比对，
                     而列表对 OpenAI 兼容网关**不可靠**：那里给的是展示名，不是调用名。 */
                  <p className="flex items-start gap-1.5 text-xs leading-relaxed text-neon-amber">
                    <TriangleAlert size={13} className="mt-0.5 shrink-0" aria-hidden />
                    <span>
                      <code className="font-mono">{modelField}</code> ={" "}
                      <code className="font-mono">{modelName}</code> 没有精确出现在它返回的模型列表里。
                      <b>这条只是提示，不一定算错</b> —— 有些 OpenAI 兼容网关的模型列表给的是展示名，
                      而调用要用另一个写法；以一次真实调用为准。
                    </span>
                  </p>
                )}
              </div>
            ) : null}
          </div>
        ) : (
          <div className="flex items-start gap-2 rounded-[10px] border border-neon-red/50 bg-white/[0.05] px-3 py-2.5">
            <TriangleAlert size={15} className="mt-0.5 shrink-0 text-neon-red" aria-hidden />
            <p className="text-sm leading-relaxed text-neon-red">{result.detail}</p>
          </div>
        )
      ) : null}

      {stale ? (
        <p className="text-xs leading-relaxed text-ink-faint">
          上方结果对应的是这次探测时的值；表单之后又改过了，重新探测一次才能反映当前填写的配置。
        </p>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------- 页面 */

export function SettingsPage() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [draft, setDraft] = useState<Draft>(emptyDraft);
  const [confirming, setConfirming] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [writeResult, setWriteResult] = useState<ConfigWriteResult | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setLoadError("");
    api.config
      .get()
      .then((c) => {
        setConfig(c);
        setDraft(toDraft(c.values));
      })
      .catch((e: unknown) => {
        setConfig(null);
        setLoadError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const readOnly = config ? !config.writable : false;
  const initial = useMemo<Draft>(() => (config ? toDraft(config.values) : emptyDraft()), [config]);

  /** 只 diff 改动过的字段；密钥只有用户真的输入了才算改动 */
  const changed = useMemo(
    () =>
      (Object.keys(draft) as FieldKey[]).filter((k) => {
        if (SECRET_KEYS.includes(k)) return draft[k].trim() !== "";
        return draft[k] !== initial[k];
      }),
    [draft, initial],
  );

  const numberErrors = useMemo(() => {
    const errs: Partial<Record<FieldKey, string>> = {};
    NUMBER_KEYS.forEach((k) => {
      if (!changed.includes(k)) return;
      const n = Number(k === "llm_temperature" ? draft[k] : draft[k].trim());
      if (draft[k].trim() === "" || !Number.isFinite(n)) errs[k] = "需要填一个数字";
      else if (k === "min_corpus" && (!Number.isInteger(n) || n < 0)) errs[k] = "需要非负整数";
      else if (k === "llm_temperature" && (n < 0 || n > 2)) errs[k] = "常用范围 0 ~ 2";
    });
    return errs;
  }, [changed, draft]);

  const hasNumberErrors = Object.keys(numberErrors).length > 0;
  const envSourcedFields = useMemo(
    () =>
      config
        ? (Object.keys(LABELS) as FieldKey[]).filter((k) => config.sources[k] === "env")
        : [],
    [config],
  );

  const set = (k: FieldKey) => (v: string) => setDraft((d) => ({ ...d, [k]: v }));

  function buildPayload(): ConfigValues {
    const raw: Record<string, string | number | boolean> = {};
    changed.forEach((k) => {
      if (NUMBER_KEYS.includes(k)) raw[k] = Number(draft[k].trim());
      else if (BOOL_KEYS.includes(k)) raw[k] = draft[k] === "true";
      else raw[k] = draft[k];
    });
    return raw as ConfigValues;
  }

  async function doSave() {
    setSaving(true);
    setSaveError("");
    setWriteResult(null);
    try {
      const result = await api.config.put(buildPayload());
      setWriteResult(result);
      setConfirming(false);
      // 全部落盘才刷新基线；有 rejected 时保留用户输入 —— 静默吞掉填写是最难查的体验
      if (result.ok && result.rejected.length === 0) load();
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
      setConfirming(false);
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col gap-5">
        <Skeleton height={54} />
        <Skeleton height={220} />
        <Skeleton height={220} />
        <Skeleton height={160} />
      </div>
    );
  }

  if (!config) {
    return (
      <div className="flex flex-col gap-5">
        <PageHeader title="配置中心" features="模型 · 接口 · 密钥" description="读取 GET /api/v1/config 失败" />
        <GlassCard>
          <EmptyState
            icon="circle-alert"
            title="配置加载失败"
            description={loadError || "后端不可达。"}
            action={<NeonButton variant="secondary" onClick={load}>重试</NeonButton>}
          />
        </GlassCard>
      </div>
    );
  }

  const llm = draft.llm_provider;
  const emb = draft.embedding_provider;
  const rerank = draft.rerank_provider;
  const disabled = readOnly || saving;

  const restartNow = new Set(config.restart_required);
  const changedRestart = changed.filter((k) => restartNow.has(k));

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="配置中心"
        features="模型 · 接口 · 密钥"
        description="这里修改的是服务端 .env 里的运行配置，会真实影响后端的模型调用 —— 不是浏览器本地设置。写入前自动备份，部分字段改完必须重启后端才生效。"
        actions={
          <NeonButton variant="secondary" onClick={load}>
            <RotateCcw size={15} aria-hidden /> 重新读取
          </NeonButton>
        }
      />

      {readOnly ? (
        <div className="flex items-start gap-2 rounded-lg border border-neon-amber/40 bg-white/[0.04] px-4 py-3 text-sm text-neon-amber">
          <TriangleAlert size={16} className="mt-0.5 shrink-0" aria-hidden />
          <span>
            后端以 <code className="font-mono">CONFIG_WRITABLE=false</code> 运行，配置为<b>只读</b>，表单已禁用。
            要恢复可写：在后端 .env 中把该行删掉或改为 <code className="font-mono">CONFIG_WRITABLE=true</code>，然后重启后端。
          </span>
        </div>
      ) : null}

      {envSourcedFields.length > 0 && !readOnly ? (
        <div className="flex items-start gap-2 rounded-lg border border-neon-amber/40 bg-white/[0.04] px-4 py-3 text-xs leading-relaxed text-ink-soft">
          <CircleAlert size={15} className="mt-0.5 shrink-0 text-neon-amber" aria-hidden />
          <span>
            以下字段的当前值来自<b className="text-neon-amber">进程环境变量</b>：
            <code className="font-mono"> {envSourcedFields.join("、")}</code>。进程环境变量优先级高于 .env，
            在这里保存后<b className="text-neon-amber">可能被它覆盖</b> —— 这是本项目真实踩过的坑，
            改完请同时确认启动环境里没有残留同名变量。
          </span>
        </div>
      ) : null}

      {writeResult ? <WriteResultBlock result={writeResult} /> : null}
      {saveError ? (
        <div className="rounded-lg border border-neon-red/40 bg-white/[0.04] px-4 py-3 text-sm text-neon-red">
          保存失败：{saveError} —— 你的输入仍然保留在表单里，没有丢失。
        </div>
      ) : null}

      <Stagger className="flex flex-col gap-5">
        {/* ------------------------------------------------ LLM */}
        <StaggerItem>
          <Panel title="LLM 生成模型" subtitle="决定研究报告由谁执笔">
            <div className="flex flex-col gap-4">
              <ProviderPicker
                field="llm_provider"
                ariaLabel="LLM 提供方"
                options={config.options.llm_provider}
                value={llm}
                onChange={set("llm_provider")}
                disabled={disabled}
              />

              {llm === "mock" ? (
                <div className="rounded-[10px] border border-[color:var(--accent-green)] bg-white/[0.04] px-3 py-2.5 text-xs leading-relaxed text-ink-soft">
                  <Tag variant="green">mock</Tag> 模式<b>不需要任何 API Key、不需要任何接口地址</b>，
                  模型回复来自内置桩数据，用来演示链路。下方密钥与地址字段都无需填写。
                </div>
              ) : null}

              {llm === "ollama" ? (
                <>
                  <FieldRow field="ollama_base_url" config={config} value={draft.ollama_base_url} onChange={set("ollama_base_url")} disabled={disabled} placeholder="本机 Ollama 服务端口（默认 11434）" hint="指向运行 ollama serve 的机器；跨机部署时改成对应地址。" />
                  <FieldRow field="ollama_model" config={config} value={draft.ollama_model} onChange={set("ollama_model")} disabled={disabled} hint="必须先用 ollama pull 拉过；填了没拉过的名字会在首次调用时报错。" />
                  <Collapsible title="次要字段（openai_*，当前 provider 用不到）">
                    <FieldRow field="openai_base_url" config={config} value={draft.openai_base_url} onChange={set("openai_base_url")} disabled={disabled} />
                    <FieldRow
                      field="openai_api_key"
                      config={config}
                      kind="password"
                      value={draft.openai_api_key}
                      onChange={set("openai_api_key")}
                      disabled={disabled}
                      secret={config.secrets.openai_api_key}
                      placeholder={config.secrets.openai_api_key.masked || "sk-…"}
                    />
                  </Collapsible>
                </>
              ) : null}

              {llm === "openai" ? (
                <>
                  <FieldRow field="openai_base_url" config={config} value={draft.openai_base_url} onChange={set("openai_base_url")} disabled={disabled} hint="填官方或任意 OpenAI 兼容网关的地址。" />
                  <FieldRow
                    field="openai_api_key"
                    config={config}
                    kind="password"
                    value={draft.openai_api_key}
                    onChange={set("openai_api_key")}
                    disabled={disabled}
                    secret={config.secrets.openai_api_key}
                    placeholder={config.secrets.openai_api_key.masked || "粘贴密钥后提交"}
                    hint="后端只回传掩码，界面永远不回填明文密钥。"
                  />
                  <Collapsible title="次要字段（ollama_*，当前 provider 用不到）">
                    <FieldRow field="ollama_base_url" config={config} value={draft.ollama_base_url} onChange={set("ollama_base_url")} disabled={disabled} />
                    <FieldRow field="ollama_model" config={config} value={draft.ollama_model} onChange={set("ollama_model")} disabled={disabled} />
                  </Collapsible>
                </>
              ) : null}

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <FieldRow field="llm_model" config={config} value={draft.llm_model} onChange={set("llm_model")} disabled={disabled} />
                <FieldRow
                  field="llm_temperature"
                  config={config}
                  kind="number"
                  value={draft.llm_temperature}
                  onChange={set("llm_temperature")}
                  disabled={disabled}
                  error={numberErrors.llm_temperature}
                  hint="越低越稳。检索型任务建议 0 ~ 0.3。"
                />
              </div>

              {/* 结构化输出方式：接 OpenAI 兼容网关时这是"能不能跑通"的开关，
                  不接进界面就等于让使用者去猜要往 .env 里塞什么。
                  ⚠️ 只有后端真的回了这两个字段才渲染 —— 界面提供一个后端不接受的字段，
                  就是当初 mock 给 rerank 多写 "api" 那个事故的翻版。 */}
              {typeof config.values.llm_structured_method === "string" ? (
                <>
              <div className="flex flex-col gap-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm text-ink">{LABELS.llm_structured_method}</span>
                  <code className="font-mono text-[11px] text-ink-faint">llm_structured_method</code>
                  <SourceTag source={config.sources.llm_structured_method} />
                </div>
                <ProviderPicker
                  field="llm_structured_method"
                  ariaLabel="结构化输出方式"
                  options={config.options.llm_structured_method}
                  value={draft.llm_structured_method}
                  onChange={set("llm_structured_method")}
                  disabled={disabled}
                />
                <p className="text-xs leading-relaxed text-ink-faint">
                  plan / read / verify / compare 四个节点要靠模型出结构化 JSON。
                  <code className="font-mono">auto</code> 表示不干预、用端点自己的默认，
                  ollama 与 mock 走这条没问题；但不少 OpenAI 兼容网关（DeepSeek 实测）会把它回成
                  <code className="font-mono"> 400 This response_format type is unavailable now</code> ——
                  症状是<b>普通对话一切正常、研究却每个节点都失败</b>。
                  这种端点选 <code className="font-mono">function_calling</code>，
                  并把下面那个开关一起打开。
                </p>
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm text-ink">{LABELS.llm_disable_thinking}</span>
                  <code className="font-mono text-[11px] text-ink-faint">llm_disable_thinking</code>
                  <SourceTag source={config.sources.llm_disable_thinking} />
                </div>
                <ProviderPicker
                  field="llm_disable_thinking"
                  ariaLabel="结构化调用时是否关掉思考模式"
                  options={["true", "false"]}
                  value={draft.llm_disable_thinking}
                  onChange={set("llm_disable_thinking")}
                  disabled={disabled}
                />
                <p className="text-xs leading-relaxed text-ink-faint">
                  只影响<b>结构化调用</b>：思考模式模型（如 deepseek-flash）在那种请求里不接受
                  function_calling 所需的 tool_choice，必须关掉思考才通。
                  写报告那条正文链路不受这个开关影响，不会为了迁就端点把报告质量一起降掉。
                </p>
              </div>
                </>
              ) : null}

              {/* 只有 LLM 这一栏能探测：provider 的合法集合就是 openai / ollama / mock */}
              <LlmConnectivityTest
                draft={draft}
                secret={config.secrets.openai_api_key}
                testable={config.options.llm_provider}
                disabled={saving}
              />
            </div>
          </Panel>
        </StaggerItem>

        {/* ------------------------------------------------ Embedding */}
        <StaggerItem>
          <Panel title="Embedding 向量化" subtitle="决定语料怎么变成向量、检索怎么算相似度">
            <div className="flex flex-col gap-4">
              <ProviderPicker
                field="embedding_provider"
                ariaLabel="Embedding 提供方"
                options={config.options.embedding_provider}
                value={emb}
                onChange={set("embedding_provider")}
                disabled={disabled}
              />

              {emb === "local" ? (
                <div className="flex items-start gap-2 rounded-[10px] border border-neon-amber/40 bg-white/[0.04] px-3 py-2.5 text-xs leading-relaxed text-ink-soft">
                  <TriangleAlert size={14} className="mt-0.5 shrink-0 text-neon-amber" aria-hidden />
                  <span>
                    local 需要安装 extra：<code className="font-mono text-neon-amber">pip install -e ".[local]"</code>，
                    且模型<b>权重体积大</b>（首次运行会下载并缓存在本机）。装不上或磁盘紧张时考虑改用 api 或 mock。
                  </span>
                </div>
              ) : null}
              {emb === "mock" ? (
                <div className="rounded-[10px] border border-[color:var(--accent-green)] bg-white/[0.04] px-3 py-2.5 text-xs leading-relaxed text-ink-soft">
                  <Tag variant="green">mock</Tag> 模式<b>不需要任何 Key 或接口地址</b>，向量为随机占位，只用于跑通流程。
                </div>
              ) : null}

              {emb === "local" || emb === "api" ? (
                <FieldRow
                  field="embedding_model"
                  config={config}
                  value={draft.embedding_model}
                  onChange={set("embedding_model")}
                  disabled={disabled}
                  hint={emb === "api" ? "填远端服务上可用的模型名（如 text-embedding-3-small 这类）。" : "本地模型的仓库名 / 路径，需与已下载权重一致。改模型名会触发重建向量索引（维度可能不同）。"}
                />
              ) : null}

              {emb === "api" ? (
                <>
                  <FieldRow field="embedding_api_base_url" config={config} value={draft.embedding_api_base_url} onChange={set("embedding_api_base_url")} disabled={disabled} hint="OpenAI 兼容的 embeddings 端点地址。" />
                  <FieldRow
                    field="embedding_api_key"
                    config={config}
                    kind="password"
                    value={draft.embedding_api_key}
                    onChange={set("embedding_api_key")}
                    disabled={disabled}
                    secret={config.secrets.embedding_api_key}
                    placeholder={config.secrets.embedding_api_key.masked || "粘贴密钥后提交"}
                  />
                </>
              ) : null}
            </div>
          </Panel>
        </StaggerItem>

        {/* ------------------------------------------------ Rerank */}
        <StaggerItem>
          <Panel title="重排 Rerank" subtitle="召回后精排，直接影响报告引用的质量">
            <div className="flex flex-col gap-4">
              <ProviderPicker
                field="rerank_provider"
                ariaLabel="重排提供方"
                options={config.options.rerank_provider}
                value={rerank}
                onChange={set("rerank_provider")}
                disabled={disabled}
              />
              {rerank === "local" ? (
                <FieldRow
                  field="rerank_model"
                  config={config}
                  value={draft.rerank_model}
                  onChange={set("rerank_model")}
                  disabled={disabled}
                  placeholder="BAAI/bge-reranker-v2-m3 或本地目录"
                  hint="可填 HF 仓库名（首次加载自动下载），也可填已下好的本地目录如 models/bge-reranker-v2-m3。权重缺失时后端不崩，会降级成 lexical 并在日志给出下载命令。"
                />
              ) : null}
              {rerank === "local" ? (
                <p className="text-xs leading-relaxed text-ink-faint">
                  local 是交叉编码器（如 BGE），同样需要 <code className="font-mono">pip install -e ".[local]"</code>{" "}
                  且要下载模型权重；精排会显著增加单条查询延迟（实测百毫秒级 → 秒级）。
                </p>
              ) : null}
              {rerank === "lexical" ? (
                <p className="text-xs leading-relaxed text-ink-faint">
                  lexical 为纯词面降级重排：零依赖、零下载，但收益不稳定 —— 离线消融里它对 nDCG@10 甚至是下降的。
                </p>
              ) : null}
            </div>
          </Panel>
        </StaggerItem>

        {/* ------------------------------------------------ 检索门槛 */}
        <StaggerItem>
          <Panel title="检索门槛" subtitle="语料太少时不硬撑向量检索">
            <div className="max-w-md">
              <FieldRow
                field="min_corpus"
                config={config}
                kind="number"
                value={draft.min_corpus}
                onChange={set("min_corpus")}
                disabled={disabled}
                error={numberErrors.min_corpus}
                hint="语料篇数低于该门槛时，检索链路会走降级路径而不是完整向量检索。改它需要重启后端。"
              />
            </div>
          </Panel>
        </StaggerItem>
      </Stagger>

      {/* ------------------------------------------------ 保存区 */}
      <GlassCard className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2 text-sm text-ink-soft">
            {readOnly ? (
              <span>只读模式，表单已禁用。</span>
            ) : changed.length === 0 ? (
              <span>没有待写入的改动。</span>
            ) : (
              <>
                <Tag variant="cyan" mono>
                  {changed.length} 个字段有改动
                </Tag>
                {changedRestart.length > 0 ? (
                  <Tag variant="amber">其中 {changedRestart.length} 个需重启生效</Tag>
                ) : null}
              </>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {!readOnly && changed.length > 0 ? (
              <NeonButton
                variant="secondary"
                disabled={saving}
                onClick={() => {
                  setConfirming(false);
                  setDraft(initial);
                  setWriteResult(null);
                }}
              >
                放弃修改
              </NeonButton>
            ) : null}
            {!readOnly ? (
              <NeonButton
                loading={saving}
                disabled={changed.length === 0 || hasNumberErrors}
                onClick={() => setConfirming(true)}
              >
                <Save size={15} aria-hidden /> 保存修改
              </NeonButton>
            ) : null}
          </div>
        </div>
        <p className="text-xs leading-relaxed text-ink-faint">
          保存只会把<b>改动过的字段</b>写进服务端 .env；密钥留空代表不修改。
          地址或密钥拿不准时，先用 <b className="text-ink-soft">LLM 生成模型</b> 这一栏底部的连通性测试确认 ——
          它按当前填写的值发一次真实探测，不需要先保存。
        </p>

        {confirming && !readOnly ? (
          <div className="flex flex-col gap-3 rounded-[10px] border border-[color:var(--accent-cyan-line)] bg-[var(--accent-cyan-dim)] p-3">
            <p className="text-sm text-ink">
              确认将以下 <code className="font-mono">{changed.length}</code> 个字段写入服务端 .env：
            </p>
            <ul className="flex flex-col gap-1 text-xs">
              {changed.map((k) => (
                <li key={k} className="flex flex-wrap items-center gap-2">
                  <code className="font-mono text-ink">{k}</code>
                  <span className="text-ink-faint">{LABELS[k]}</span>
                  {restartNow.has(k) ? <Tag variant="amber">需重启后端</Tag> : null}
                  {SECRET_KEYS.includes(k) ? <Tag variant="muted">密钥 · 写入后不可回显</Tag> : null}
                </li>
              ))}
            </ul>
            <div className="flex items-center gap-2">
              <NeonButton loading={saving} disabled={hasNumberErrors} onClick={doSave}>
                确认写入
              </NeonButton>
              <NeonButton variant="secondary" disabled={saving} onClick={() => setConfirming(false)}>
                取消
              </NeonButton>
              {hasNumberErrors ? <span className="text-xs text-neon-red">数字字段还没填对，先修正上方红字。</span> : null}
            </div>
          </div>
        ) : null}
      </GlassCard>
    </div>
  );
}

function WriteResultBlock({ result }: { result: ConfigWriteResult }) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-[color:var(--glass-border-strong)] bg-white/[0.04] px-4 py-3">
      {result.ok && result.rejected.length === 0 ? (
        <p className="text-sm text-neon-green">
          写入成功，共 {result.updated.length} 个字段。
        </p>
      ) : result.ok ? (
        <p className="text-sm text-neon-amber">
          部分写入：成功 {result.updated.length} 个，被拒 {result.rejected.length} 个。
          <span className="ml-1 text-ink-soft">被拒字段的表单输入已保留，不会被悄悄重置回旧值。</span>
        </p>
      ) : (
        <p className="text-sm text-neon-red">
          后端报告写入未完全成功。<span className="ml-1 text-ink-soft">表单输入已保留。</span>
        </p>
      )}

      <p className="text-xs text-ink-soft">
        {result.backup ? (
          <>
            已备份为 <code className="font-mono text-neon-cyan">{result.backup}</code> —— 改坏了可以把这个文件复制回 .env 退回去。
          </>
        ) : (
          <span className="text-ink-faint">后端没有返回备份文件名（backup 为 null），本次写入没有可一键回退的备份。</span>
        )}
      </p>

      {result.restart_required.length > 0 ? (
        <div className="flex items-start gap-2 rounded-[10px] border border-neon-red/50 bg-white/[0.05] px-3 py-2.5">
          <TriangleAlert size={16} className="mt-0.5 shrink-0 text-neon-red" aria-hidden />
          <p className="text-sm leading-relaxed text-neon-red">
            这些改动要<b>重启后端才生效</b>：<code className="font-mono">{result.restart_required.join("、")}</code>。
            现在界面显示的仍然是旧值在跑出来的状态 —— 保存 ≠ 已生效。
          </p>
        </div>
      ) : null}

      {result.shadowed.length > 0 ? (
        <div className="flex items-start gap-2 rounded-[10px] border border-ink-faint/40 bg-white/[0.05] px-3 py-2.5">
          <Info size={16} className="mt-0.5 shrink-0 text-ink-soft" aria-hidden />
          <p className="text-sm leading-relaxed text-ink-soft">
            这几个字段已经写进 <code className="font-mono">.env</code>，但同名
            <b>环境变量</b>优先级更高，所以当前生效的仍是环境里那份：
            <code className="font-mono">{result.shadowed.join("、")}</code>。
            想让它听话，请先 unset 对应环境变量再重启（用 <code className="font-mono">launch.sh</code> /
            <code className="font-mono">启动.bat</code> 起的通常不会有这个问题，多半是容器或 CI 注入的）。
          </p>
        </div>
      ) : null}

      {result.rejected.length > 0 ? (
        <ul className="flex flex-col gap-1.5">
          {result.rejected.map((r) => (
            <li key={r.field} className="flex flex-wrap items-center gap-2 text-sm">
              <Tag variant="red" mono>
                {r.field}
              </Tag>
              <span className="text-ink-soft">{r.reason}</span>
            </li>
          ))}
          <li className="text-xs text-ink-faint">以上字段没有被写入，你的输入原样保留在表单里。</li>
        </ul>
      ) : null}
    </div>
  );
}

export default SettingsPage;
