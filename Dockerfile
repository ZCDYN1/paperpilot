# syntax=docker/dockerfile:1
# =============================================================================
# PaperPilot API 镜像（内置 React 控制台静态产物）
# =============================================================================

# ---- Stage 1: webbuild —— 构建 React 控制台 ----------------------------------
# Node 构建隔离在独立阶段：node 本身、node_modules、tsc/vite 中间产物都留在
# 这一层，最终镜像只通过 COPY --from 拿走 web/dist 一个目录。
FROM node:22-bookworm-slim AS webbuild

WORKDIR /app/web

# 层顺序为构建缓存设计：
# 1) 先只拷 manifest 两个文件（改动频率最低）→ 只要 package-lock 不变，
#    下面整个 npm ci 层直接命中缓存，改业务代码不会重装几百个包；
# 2) 再 COPY web/ 其余源码（.dockerignore 已排除 web/node_modules 与 web/dist，
#    宿主机产物不会污染容器内构建）→ 改前端代码只会 bust 源码层和其后的 build 层。
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# npm run build = tsc --noEmit && vite build，产出 /app/web/dist。
# vite base 默认 "/"（见 web/vite.config.ts，仅当注入 VITE_BASE_PATH 才变），
# 与 app/main.py 在根路径挂载 /assets + SPA 的约定一致，容器内无需额外配置。
RUN npm run build

# ---- Stage 2: 最终 Python 镜像 ---------------------------------------------
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

# 从 webbuild 阶段只取最终静态产物，落点必须是 ./web/dist ——
# app/main.py 的 _mount_web_ui 用 Path(__file__).parents[1]/"web"/"dist" 定位，
# 而依赖是 `pip install -e` 装回 /app 的，所以 parents[1] 就是 WORKDIR /app。
# 放在 COPY . . 之后：web/dist 已被 .dockerignore 挡在上下文外，这里补上唯一来源，
# 且它只依赖 webbuild 层，改后端代码不会让前端层失效。
COPY --from=webbuild /app/web/dist ./web/dist

# 运行期数据目录（compose 会用 named volume 覆盖它）
RUN mkdir -p data/faiss data/sqlite

EXPOSE 8000

# 健康检查用 python 而不是 curl：slim 镜像里没有 curl，
# 写 ["CMD","curl",...] 会让容器永远处于 unhealthy 状态。
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz',timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
