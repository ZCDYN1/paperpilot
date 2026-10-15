import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { api } from "@/api/client";
import { useResearchStream } from "@/hooks/useResearchStream";

/* ============================================================ 一次研究工作会话
   这里放的是**属于"这次研究"而不属于某个页面**的东西：
   ① SSE 流本身（`useResearchStream`）；② 发起用的表单草稿。

   ★ 为什么必须提到路由之上：
   `App.tsx` 里每个页面都是 `<Route>` 的 children，切换侧栏就是把 ConsolePage **卸载**。
   而 `useResearchStream` 在卸载回调里 `abort()` 掉 fetch —— 后果有两层，都不该发生：
   1. 界面回到初始状态，用户以为研究没了（其实后端还在跑，只是没人读它了）；
   2. 后端看到客户端断开，把这条 run 落成 `cancelled`，**HITL 中断点再也续跑不了**。
   一次研究要跑几分钟、还要在中断点等人确认，"顺手看一眼论文库"不该把它掐死。

   所以流挂在 provider 里：只要应用还开着，它就一直在读；页面切走只是不再渲染它。
   表单草稿同理 —— 用户回来时看到的应该是他离开时的那张表。 */

/** 界面上"分类"下拉里代表"不限"的哨兵值，落到请求体是 `categories: []` */
export const CATEGORY_ALL = "all";
/** 白名单取自 `app/schemas/retrieval.py` 的 `ARXIV_CATEGORIES`，只挑研究最常用的几类 */
export const CATEGORY_CHOICES = ["cs.CL", "cs.LG", "cs.IR", "cs.AI", "cs.CV", "cs.NE", "stat.ML"] as const;

/** `ResearchFilters.year_*` 的 ge/le */
export const YEAR_MIN = 1991;
export const YEAR_MAX = 2100;
/** `ResearchRequest.top_k` 的 ge/le */
export const TOP_K_MIN = 1;
export const TOP_K_MAX = 50;
/** 与后端 `ResearchRequest.top_k` 的 default 一致（留空时不覆盖后端默认） */
export const DEFAULT_TOP_K = 8;

/**
 * ★ `has_code` 是三态不是布尔：
 * true=只召回确认有仓库的；false=只召回**未见**仓库的；null=不过滤。
 * 所以界面用三个显式选项，绝不能用 `checked ? true : false` —— 那样"不勾"会
 * 被后端理解成"我要看没有代码的论文"，跟用户预期完全相反。
 */
export type HasCodeChoice = "all" | "only_with" | "only_without";

export const HAS_CODE_VALUE: Record<HasCodeChoice, boolean | null> = {
  all: null,
  only_with: true,
  only_without: false,
};

/** 界面上的控件 state 一律是"文本框里是什么就是什么"，不做即时强转 */
export type FilterForm = {
  category: string;
  yearFrom: string;
  yearTo: string;
  hasCode: HasCodeChoice;
  language: "zh" | "en";
  topK: string;
  rerank: boolean;
};

export const DEFAULT_FORM: FilterForm = {
  category: "cs.CL",
  yearFrom: "2020",
  yearTo: "",
  hasCode: "all",
  language: "zh",
  topK: String(DEFAULT_TOP_K),
  rerank: true,
};

export const DEFAULT_QUESTION = api.useMock
  ? "检索增强生成在减少大模型幻觉方面有哪些主流方法，各自的代价是什么？"
  : "";

type Workspace = {
  run: ReturnType<typeof useResearchStream>;
  question: string;
  setQuestion: (value: string) => void;
  form: FilterForm;
  setForm: (updater: (prev: FilterForm) => FilterForm) => void;
};

const Context = createContext<Workspace | null>(null);

export function ResearchWorkspaceProvider({ children }: { children: ReactNode }) {
  const run = useResearchStream();
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [form, setForm] = useState<FilterForm>(DEFAULT_FORM);

  const value = useMemo(
    () => ({ run, question, setQuestion, form, setForm }),
    [run, question, form],
  );

  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useResearchWorkspace(): Workspace {
  const value = useContext(Context);
  if (!value) throw new Error("useResearchWorkspace 必须在 <ResearchWorkspaceProvider> 里使用");
  return value;
}
