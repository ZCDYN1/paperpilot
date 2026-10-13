# =============================================================================
# PaperPilot API 镜像
# =============================================================================
FROM python:3.12-slim

# HF_HOME 显式指定：保证模型权重缓存落在挂载点上。
# ⚠️ 该路径必须与 docker-compose.yml 里 api 服务的 volume 挂载点保持一致，
#    否则容器每次重建都会重新下载 ~3GB 的 BGE 权重（而且不会报错，只是变得很慢）。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# faiss 运行时需要 OpenMP（libgomp1），python:slim 镜像里没有，必须显式装
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖、后拷代码：只要 pyproject.toml 不变，改业务代码不会触发依赖重装
COPY pyproject.toml README.md ./
# 造一个占位包，满足构建后端对 packages = ["app"] 的存在性检查
RUN mkdir -p app && touch app/__init__.py

# INSTALL_LOCAL=false 可构建一个不含 torch 的轻量镜像（走 API embedding）
ARG INSTALL_LOCAL=true
RUN pip install --upgrade pip \
    && pip install -e ".[dev]" \
    && if [ "$INSTALL_LOCAL" = "true" ]; then pip install -e ".[local]"; fi

COPY . .

# 运行期数据目录（compose 会用 named volume 覆盖它）
RUN mkdir -p data/faiss data/sqlite

EXPOSE 8000

# 健康检查用 python 而不是 curl：slim 镜像里没有 curl，
# 写 ["CMD","curl",...] 会让容器永远处于 unhealthy 状态。
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz',timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
