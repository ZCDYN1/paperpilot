"""成稿与导出层（产品定义 F8「导出 Markdown」）。

`render_run_markdown(run_id)` 从数据库（而非 graph state / checkpointer）渲染
一次运行的完整报告：对比表 + 综述正文 + 参考文献 + 未支撑结论 + 成本耗时 + 固定声明。
挂在 API 层之下、DB 层之上：路由只需要 `GET .../export.md` 一行转发。
"""

from app.export.markdown import RunNotFoundError, render_run_markdown

__all__ = ["RunNotFoundError", "render_run_markdown"]
