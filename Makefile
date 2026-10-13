# PaperPilot — 常用命令速查
# 说明：Windows 上如果没有 make，可直接照抄下面的命令手动执行（或用 scripts/ 下的脚本）。

.PHONY: help install install-local dev-api dev-ui test lint fmt typecheck ablate eval eval-ablation eval-ragas index sample zerokey spotcheck spotcheck-stats docker-build docker-up docker-down clean

help:
	@echo "PaperPilot 常用命令："
	@echo "  make install        安装核心依赖（可编辑安装）"
	@echo "  make install-local  额外安装本地 BGE 依赖（含 torch，体积较大）"
	@echo "  make dev-api        启动 FastAPI（热重载，:8000）"
	@echo "  make dev-ui         启动 Streamlit 面板（:8501）"
	@echo "  make test           运行单元测试"
	@echo "  make lint           代码检查（ruff）"
	@echo "  make fmt            自动格式化（ruff format）"
	@echo "  make typecheck      类型检查（mypy）"
	@echo "  make ablate         跑检索策略消融实验（等价 eval-ablation，直接用 ablation 模块）"
	@echo "  make eval           跑全部评测（检索消融 + RAGAS）"
	@echo "  make eval-ablation  只跑检索消融（只需索引，可完全离线、无需 key）"
	@echo "  make eval-ragas     只跑 RAGAS 生成侧评测（必须有真实 LLM 当裁判）"
	@echo "  make index          从 SQLite 重建检索索引（可重复执行）"
	@echo "  make sample         导入离线样例语料（24 篇真实论文摘要，零联网）"
	@echo "  make zerokey        零 key 冷启动：样例语料 + mock provider 起服务"
	@echo "  make spotcheck      导出引用人工抽检 CSV（逐条填 human_verdict）"
	@echo "  make spotcheck-stats 用已填好的 CSV 出人工口径准确率"
	@echo "  make docker-build   构建镜像"
	@echo "  make docker-up      用 docker compose 起全栈"
	@echo "  make docker-down    停掉并清理容器与 volume"
	@echo "  make clean          清理运行期数据与各类缓存"

install:
	pip install -e ".[dev]"

install-local:
	pip install -e ".[local]"

dev-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-ui:
	streamlit run ui/streamlit_app.py

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff format .

typecheck:
	mypy app

# ---- 评测 ----------------------------------------------------------------
# 统一走 scripts/run_eval.py：它对每套评测都会显式汇报「能不能跑 / 为什么不能 / 可信到几成」，
# 把「不可用」当一等公民处理 —— 直接调底层模块容易在 mock 环境下把没有意义的分数当成结论。
ablate:
	python -m app.eval.ablation

eval:
	python scripts/run_eval.py --suite all

eval-ablation:
	python scripts/run_eval.py --suite ablation

eval-ragas:
	python scripts/run_eval.py --suite ragas

# 重建索引：SQLite 是事实来源，FAISS/BM25 只是可重建的派生数据，所以这个 target 天然幂等 ——
# 换 embedding 模型、改分块策略或索引损坏后，重跑一次即可恢复。
# ⚠️ 必须用**与建索引时相同**的 embedding 配置：否则索引与查询落在不同的向量空间，
# 维度相同的话不会报错、只会静默退化。用 `make dev-api` 起的服务可用 /healthz 的 index.compatible 确认。
index:
	python scripts/build_index.py

# ---- 冷启动与人工抽检 ------------------------------------------------------
# sample：把仓库里自带的 24 篇真实论文摘要灌进（可能还是空的）SQLite。
# 新用户 clone 完 data/ 是空的，不导样例就会逼着第一次提问去联网抓 arXiv。
sample:
	python scripts/load_sample_data.py

# zerokey：零 key、零联网、零模型下载的冷启动。mock 只用于确认链路通，
# 用它跑出来的任何指标都不能当能力证据（详见 README「引用准确率」口径说明）。
zerokey:
	LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock RERANK_ENABLED=false INGEST_ENABLED=false python scripts/load_sample_data.py --force
	LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock RERANK_ENABLED=false INGEST_ENABLED=false uvicorn app.main:app --host 127.0.0.1 --port 8000

# spotcheck：架构文档 §12 要求的人工口径。导出 CSV 后在 human_verdict 列
# 逐条填 supported / unsupported / ambiguous，再用 spotcheck-stats 出一致率。
spotcheck:
	python scripts/spotcheck_citations.py --n 20 --seed 7

spotcheck-stats:
	python scripts/spotcheck_citations.py --from-csv eval_results/citation_spotcheck.csv

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

clean:
	rm -rf data/ .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
