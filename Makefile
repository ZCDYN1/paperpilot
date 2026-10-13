# PaperPilot — 常用命令速查
# 说明：Windows 上如果没有 make，可直接照抄下面的命令手动执行（或用 scripts/ 下的脚本）。

.PHONY: help install install-local dev-api dev-ui test lint fmt typecheck ablate eval docker-build docker-up docker-down clean

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
	@echo "  make ablate         跑检索策略消融实验"
	@echo "  make eval           跑 RAGAS 评测"
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

ablate:
	python -m app.eval.ablation

eval:
	python -m app.eval.ragas_runner

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

clean:
	rm -rf data/ .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
